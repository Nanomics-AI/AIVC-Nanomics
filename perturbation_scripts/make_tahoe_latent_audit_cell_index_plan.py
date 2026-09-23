#!/usr/bin/env python3
"""Create and validate stable cell-index plans for latent Experiment 0."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "hf_data_cache" / "data" / "data"
DEFAULT_COHORT = PROJECT_ROOT / "results" / "tahoe_latent_audit_cohort.csv"
DEFAULT_DMSO_COUNTS = PROJECT_ROOT / "results" / "tahoe_latent_audit_dmso_counts.csv"
DEFAULT_UNIQUE_CELLS = PROJECT_ROOT / "results" / "tahoe_latent_audit_unique_cells.csv"
DEFAULT_SET_INDICES = PROJECT_ROOT / "results" / "tahoe_latent_audit_set_cell_indices.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "results" / "tahoe_latent_audit_cell_index_summary.json"
DEFAULT_LOCAL_MANIFEST = PROJECT_ROOT / "hf_data_cache" / "local_file_manifest.json"

EXPECTED_SHARDS = 3388
SET_SIZE = 256
LOCATOR_VERSION = "tahoe_parquet_row_v1"
SAMPLER_VERSION = "sha256_bottom_k_without_replacement_v1"
READ_COLUMNS = ["plate", "sample", "drug", "cell_line_id", "BARCODE_SUB_LIB_ID"]
COHORT_REQUIRED = {
    "audit_pair_id",
    "comparison_type",
    "condition_id",
    "null_group_id",
    "repeat_index",
    "repeat_base_seed",
    "source_seed",
    "target_seed",
    "set_size",
    "sampling_method",
    "plate",
    "cell_line_id",
    "source_drug",
    "source_samples",
    "source_pool_cell_count",
    "target_drug",
    "target_samples",
    "target_pool_cell_count",
}

AtomicKey = tuple[str, str, str, str]
PoolKey = tuple[str, str, str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class Cell:
    cell_id: str
    locator: str
    shard_path: str
    row_group_index: int
    row_index_in_row_group: int
    row_index_in_shard: int
    plate: str
    sample: str
    drug: str
    cell_line_id: str
    barcode_sub_lib_id: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        return list(reader)


def pool_key(row: dict[str, str], side: str) -> PoolKey:
    samples = tuple(row[f"{side}_samples"].split("|"))
    if not samples or any(not sample for sample in samples):
        raise ValueError(f"Empty {side} sample pool in {row['audit_pair_id']}")
    return (
        row["plate"],
        row["cell_line_id"],
        row[f"{side}_drug"],
        samples,
    )


def pool_id(key: PoolKey) -> str:
    return "|".join((key[0], key[1], key[2], "+".join(key[3])))


def prepare_pool_specs(
    cohort: list[dict[str, str]],
    dmso_counts: dict[AtomicKey, int],
) -> tuple[dict[PoolKey, int], dict[AtomicKey, int]]:
    pools: dict[PoolKey, int] = {}
    for row in cohort:
        if int(row["set_size"]) != SET_SIZE:
            raise ValueError(f"Unexpected set size in {row['audit_pair_id']}")
        if row["sampling_method"] != "repeated_subsampling_without_replacement":
            raise ValueError(f"Unexpected sampling method in {row['audit_pair_id']}")
        for side in ("source", "target"):
            key = pool_key(row, side)
            count = int(row[f"{side}_pool_cell_count"])
            existing = pools.setdefault(key, count)
            if existing != count:
                raise ValueError(f"Inconsistent pool count for {pool_id(key)}")

    atomic_counts: dict[AtomicKey, int] = {}
    for key, pool_count in pools.items():
        plate, cell_line, drug, samples = key
        if drug == "DMSO_TF":
            count = 0
            for sample in samples:
                atomic_key = (plate, sample, drug, cell_line)
                if atomic_key not in dmso_counts:
                    raise ValueError(f"Missing single-sample DMSO count: {atomic_key}")
                atomic_counts[atomic_key] = dmso_counts[atomic_key]
                count += dmso_counts[atomic_key]
        else:
            if len(samples) != 1:
                raise ValueError(f"Treated pool must contain one sample: {pool_id(key)}")
            atomic_key = (plate, samples[0], drug, cell_line)
            existing = atomic_counts.setdefault(atomic_key, pool_count)
            if existing != pool_count:
                raise ValueError(f"Inconsistent treated count: {atomic_key}")
            count = existing
        if count != pool_count:
            raise ValueError(
                f"Atomic counts do not match pool count for {pool_id(key)}: "
                f"{count} != {pool_count}"
            )

    if len(pools) != 96 or len(atomic_counts) != 84:
        raise ValueError(
            f"Unexpected pool shape: {len(pools)} pools, {len(atomic_counts)} atomic groups"
        )
    return pools, atomic_counts


def shortlist_shards(
    data_dir: Path,
    sample_ids: set[str],
    cell_lines: set[str],
) -> tuple[list[Path], int, int]:
    dataset = ds.dataset(str(data_dir), format="parquet", exclude_invalid_files=True)
    if len(dataset.files) != EXPECTED_SHARDS:
        raise ValueError(f"Expected {EXPECTED_SHARDS} shards, got {len(dataset.files)}")
    scanner = dataset.scanner(
        columns=["__filename"],
        filter=(
            ds.field("sample").isin(sorted(sample_ids))
            & ds.field("cell_line_id").isin(sorted(cell_lines))
        ),
    )
    filenames: set[str] = set()
    matched_rows = 0
    for batch in scanner.to_batches():
        matched_rows += batch.num_rows
        filenames.update(batch.column(0).to_pylist())
    return sorted(Path(filename) for filename in filenames), matched_rows, len(dataset.files)


def make_cell(
    path: Path,
    row_group_index: int,
    row_index_in_row_group: int,
    row_index_in_shard: int,
    values: dict[str, str],
) -> Cell:
    shard_path = path.relative_to(PROJECT_ROOT).as_posix()
    locator = (
        f"{shard_path}::row_group={row_group_index}"
        f"::row={row_index_in_row_group}"
    )
    cell_id = hashlib.sha256(f"{LOCATOR_VERSION}|{locator}".encode("utf-8")).hexdigest()
    return Cell(
        cell_id=cell_id,
        locator=locator,
        shard_path=shard_path,
        row_group_index=row_group_index,
        row_index_in_row_group=row_index_in_row_group,
        row_index_in_shard=row_index_in_shard,
        plate=values["plate"],
        sample=values["sample"],
        drug=values["drug"],
        cell_line_id=values["cell_line_id"],
        barcode_sub_lib_id=values["BARCODE_SUB_LIB_ID"],
    )


def scan_atomic_cells(
    paths: list[Path],
    atomic_counts: dict[AtomicKey, int],
) -> tuple[dict[AtomicKey, list[Cell]], list[dict[str, object]]]:
    sample_values = pa.array(sorted({key[1] for key in atomic_counts}))
    cell_line_values = pa.array(sorted({key[3] for key in atomic_counts}))
    cells: dict[AtomicKey, list[Cell]] = defaultdict(list)
    shard_layout = []
    started = time.time()

    for file_index, path in enumerate(paths, start=1):
        parquet = pq.ParquetFile(path)
        missing = set(READ_COLUMNS) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        row_offset = 0
        selected_in_shard = 0

        for row_group_index in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group_index, columns=READ_COLUMNS)
            mask = pc.and_(
                pc.is_in(table["sample"], value_set=sample_values),
                pc.is_in(table["cell_line_id"], value_set=cell_line_values),
            )
            indices = pc.indices_nonzero(mask).to_pylist()
            for row_index in indices:
                values = {
                    column: table[column][row_index].as_py() for column in READ_COLUMNS
                }
                key = (
                    values["plate"],
                    values["sample"],
                    values["drug"],
                    values["cell_line_id"],
                )
                if key not in atomic_counts:
                    raise ValueError(f"Filtered row has an unexpected atomic key: {key}")
                cell = make_cell(
                    path,
                    row_group_index,
                    row_index,
                    row_offset + row_index,
                    values,
                )
                cells[key].append(cell)
                selected_in_shard += 1
            row_offset += table.num_rows

        if selected_in_shard:
            shard_layout.append(
                {
                    "shard_path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "num_rows": parquet.metadata.num_rows,
                    "num_row_groups": parquet.num_row_groups,
                    "selected_rows": selected_in_shard,
                }
            )
        if file_index % 25 == 0 or file_index == len(paths):
            print(
                f"indexed {file_index}/{len(paths)} shortlisted shards, "
                f"selected_rows={sum(len(group) for group in cells.values()):,}, "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    if set(cells) != set(atomic_counts):
        raise ValueError(
            f"Atomic key mismatch: missing={sorted(set(atomic_counts) - set(cells))}, "
            f"unexpected={sorted(set(cells) - set(atomic_counts))}"
        )
    for key, expected in atomic_counts.items():
        observed = len(cells[key])
        if observed != expected:
            raise ValueError(f"Atomic cell count mismatch for {key}: {observed} != {expected}")

    all_cells = [cell for group in cells.values() for cell in group]
    if len({cell.locator for cell in all_cells}) != len(all_cells):
        raise AssertionError("Stable locator is not unique across atomic pools")
    if len({cell.cell_id for cell in all_cells}) != len(all_cells):
        raise AssertionError("cell_id collision detected")
    return cells, shard_layout


def build_pools(
    pool_counts: dict[PoolKey, int],
    atomic_cells: dict[AtomicKey, list[Cell]],
) -> dict[PoolKey, list[Cell]]:
    pools = {}
    for key, expected in pool_counts.items():
        plate, cell_line, drug, samples = key
        pool = [
            cell
            for sample in samples
            for cell in atomic_cells[(plate, sample, drug, cell_line)]
        ]
        pool.sort(key=lambda cell: cell.cell_id)
        if len(pool) != expected or len({cell.cell_id for cell in pool}) != expected:
            raise ValueError(f"Invalid pool membership for {pool_id(key)}")
        pools[key] = pool
    return pools


def stable_subsample(cells: list[Cell], seed: int) -> list[Cell]:
    if len(cells) < SET_SIZE:
        raise ValueError(f"Pool has only {len(cells)} cells")
    seed_bytes = seed.to_bytes(8, "big", signed=False)

    def score(cell: Cell) -> tuple[bytes, str]:
        digest = hashlib.sha256(
            SAMPLER_VERSION.encode("ascii") + seed_bytes + bytes.fromhex(cell.cell_id)
        ).digest()
        return digest, cell.cell_id

    selected = heapq.nsmallest(SET_SIZE, cells, key=score)
    selected.sort(key=lambda cell: cell.cell_id)
    if len(selected) != SET_SIZE or len({cell.cell_id for cell in selected}) != SET_SIZE:
        raise AssertionError("Subsample is not exactly 256 unique cells")
    return selected


def numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def verify_locators(cells: list[Cell]) -> int:
    grouped: dict[str, dict[int, list[Cell]]] = defaultdict(lambda: defaultdict(list))
    for cell in cells:
        grouped[cell.shard_path][cell.row_group_index].append(cell)

    verified = 0
    for shard_path, row_groups in grouped.items():
        parquet = pq.ParquetFile(PROJECT_ROOT / shard_path)
        row_offsets = []
        offset = 0
        for row_group_index in range(parquet.num_row_groups):
            row_offsets.append(offset)
            offset += parquet.metadata.row_group(row_group_index).num_rows
        for row_group_index, expected_cells in row_groups.items():
            table = parquet.read_row_group(row_group_index, columns=READ_COLUMNS)
            for cell in expected_cells:
                row_index = cell.row_index_in_row_group
                if not 0 <= row_index < table.num_rows:
                    raise AssertionError(f"Locator row is out of range: {cell.locator}")
                observed = {
                    column: table[column][row_index].as_py() for column in READ_COLUMNS
                }
                expected = {
                    "plate": cell.plate,
                    "sample": cell.sample,
                    "drug": cell.drug,
                    "cell_line_id": cell.cell_line_id,
                    "BARCODE_SUB_LIB_ID": cell.barcode_sub_lib_id,
                }
                if observed != expected:
                    raise AssertionError(f"Locator metadata mismatch: {cell.locator}")
                if cell.row_index_in_shard != row_offsets[row_group_index] + row_index:
                    raise AssertionError(f"Shard row mismatch: {cell.locator}")
                rebuilt = make_cell(
                    PROJECT_ROOT / cell.shard_path,
                    row_group_index,
                    row_index,
                    cell.row_index_in_shard,
                    observed,
                )
                if rebuilt.cell_id != cell.cell_id or rebuilt.locator != cell.locator:
                    raise AssertionError(f"Locator reconstruction failed: {cell.locator}")
                verified += 1
    return verified


def write_outputs(
    cohort: list[dict[str, str]],
    pools: dict[PoolKey, list[Cell]],
    all_available_cells: list[Cell],
    shard_layout: list[dict[str, object]],
    args: argparse.Namespace,
    total_dataset_shards: int,
    shortlist_rows: int,
) -> dict[str, object]:
    selection_cache: dict[tuple[PoolKey, int], list[Cell]] = {}
    selections = []
    selected_cells: dict[str, Cell] = {}
    usage = Counter()
    usage_source = Counter()
    usage_target = Counter()
    usage_perturbation = Counter()
    usage_null = Counter()
    repeat_sets: dict[tuple[str, str, str], list[set[str]]] = defaultdict(list)
    source_target_overlaps = []

    for row in cohort:
        pair_selections = {}
        for side in ("source", "target"):
            key = pool_key(row, side)
            seed = int(row[f"{side}_seed"])
            selected = selection_cache.setdefault(
                (key, seed), stable_subsample(pools[key], seed)
            )
            ids = [cell.cell_id for cell in selected]
            if len(ids) != SET_SIZE or len(set(ids)) != SET_SIZE:
                raise AssertionError(f"Invalid {side} set: {row['audit_pair_id']}")
            pair_selections[side] = selected
            repeat_sets[(row["comparison_type"], row["condition_id"], side)].append(
                set(ids)
            )
            for cell in selected:
                selected_cells[cell.cell_id] = cell
                usage[cell.cell_id] += 1
                (usage_source if side == "source" else usage_target)[cell.cell_id] += 1
                (
                    usage_perturbation
                    if row["comparison_type"] == "perturbation"
                    else usage_null
                )[cell.cell_id] += 1
            selections.append((row, side, selected))
        overlap = len(
            {cell.cell_id for cell in pair_selections["source"]}
            & {cell.cell_id for cell in pair_selections["target"]}
        )
        source_target_overlaps.append(overlap)
        if overlap:
            raise AssertionError(f"Source/target cell overlap in {row['audit_pair_id']}")

    for key, sets in repeat_sets.items():
        if len(sets) != 5:
            raise AssertionError(f"Expected five repeated sets for {key}, got {len(sets)}")

    unique_cells = sorted(selected_cells.values(), key=lambda cell: cell.cell_id)
    embedding_index = {cell.cell_id: index for index, cell in enumerate(unique_cells)}

    args.unique_cells.parent.mkdir(parents=True, exist_ok=True)
    with args.unique_cells.open("w", encoding="utf-8-sig", newline="") as handle:
        columns = (
            "embedding_index",
            "cell_id",
            "locator_version",
            "cell_locator",
            "shard_path",
            "row_group_index",
            "row_index_in_row_group",
            "row_index_in_shard",
            "plate",
            "sample",
            "drug",
            "cell_line_id",
            "BARCODE_SUB_LIB_ID",
            "usage_count",
            "source_usage_count",
            "target_usage_count",
            "perturbation_usage_count",
            "dmso_null_usage_count",
        )
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for cell in unique_cells:
            writer.writerow(
                {
                    "embedding_index": embedding_index[cell.cell_id],
                    "cell_id": cell.cell_id,
                    "locator_version": LOCATOR_VERSION,
                    "cell_locator": cell.locator,
                    "shard_path": cell.shard_path,
                    "row_group_index": cell.row_group_index,
                    "row_index_in_row_group": cell.row_index_in_row_group,
                    "row_index_in_shard": cell.row_index_in_shard,
                    "plate": cell.plate,
                    "sample": cell.sample,
                    "drug": cell.drug,
                    "cell_line_id": cell.cell_line_id,
                    "BARCODE_SUB_LIB_ID": cell.barcode_sub_lib_id,
                    "usage_count": usage[cell.cell_id],
                    "source_usage_count": usage_source[cell.cell_id],
                    "target_usage_count": usage_target[cell.cell_id],
                    "perturbation_usage_count": usage_perturbation[cell.cell_id],
                    "dmso_null_usage_count": usage_null[cell.cell_id],
                }
            )

    membership_columns = (
        "audit_pair_id",
        "comparison_type",
        "condition_id",
        "null_group_id",
        "repeat_index",
        "repeat_base_seed",
        "side",
        "set_position",
        "sampling_seed",
        "embedding_index",
        "cell_id",
    )
    with args.set_indices.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=membership_columns, lineterminator="\n")
        writer.writeheader()
        for row, side, selected in selections:
            for position, cell in enumerate(selected):
                writer.writerow(
                    {
                        "audit_pair_id": row["audit_pair_id"],
                        "comparison_type": row["comparison_type"],
                        "condition_id": row["condition_id"],
                        "null_group_id": row["null_group_id"],
                        "repeat_index": row["repeat_index"],
                        "repeat_base_seed": row["repeat_base_seed"],
                        "side": side,
                        "set_position": position,
                        "sampling_seed": row[f"{side}_seed"],
                        "embedding_index": embedding_index[cell.cell_id],
                        "cell_id": cell.cell_id,
                    }
                )

    overlap_stats = {}
    for comparison_type in ("perturbation", "dmso_null"):
        for side in ("source", "target"):
            overlaps = []
            for (kind, _, group_side), sets in repeat_sets.items():
                if kind == comparison_type and group_side == side:
                    overlaps.extend(len(left & right) for left, right in combinations(sets, 2))
            overlap_stats[f"{comparison_type}_{side}"] = numeric_summary(overlaps)

    available_ids = {cell.cell_id for cell in all_available_cells}
    if not set(selected_cells).issubset(available_ids):
        raise AssertionError("Selected cell is outside available cohort pools")
    selected_dmso = [cell for cell in unique_cells if cell.drug == "DMSO_TF"]
    selected_treated = [cell for cell in unique_cells if cell.drug != "DMSO_TF"]
    usage_values = list(usage.values())
    layout_lines = [
        f"{row['shard_path']}\t{row['size_bytes']}\t{row['num_rows']}\t{row['num_row_groups']}"
        for row in sorted(shard_layout, key=lambda row: str(row["shard_path"]))
    ]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "cohort_csv": args.cohort.relative_to(PROJECT_ROOT).as_posix(),
            "cohort_sha256": sha256_file(args.cohort),
            "dmso_counts_csv": args.dmso_counts.relative_to(PROJECT_ROOT).as_posix(),
            "dmso_counts_sha256": sha256_file(args.dmso_counts),
            "local_manifest_sha256": sha256_file(args.local_manifest),
        },
        "locator": {
            "version": LOCATOR_VERSION,
            "components": [
                "project-relative parquet shard path",
                "row group index",
                "row index within row group",
            ],
            "cell_id": "SHA256(locator_version + locator)",
            "depends_on_scan_order": False,
            "selected_shard_layout_sha256": hashlib.sha256(
                "\n".join(layout_lines).encode("utf-8")
            ).hexdigest(),
        },
        "sampling": {
            "algorithm": SAMPLER_VERSION,
            "set_size": SET_SIZE,
            "within_set_replacement": False,
            "overlap_between_repeats_allowed": True,
            "ordering_dependency": False,
        },
        "dataset_scan": {
            "dataset_shards": total_dataset_shards,
            "shortlisted_shards": len(shard_layout),
            "shortlist_matched_rows": shortlist_rows,
            "available_unique_pool_cells": len(available_ids),
        },
        "sets": {
            "pair_sets": len(cohort),
            "side_sets": len(selections),
            "total_cell_slots": len(selections) * SET_SIZE,
            "all_sets_exactly_256_unique_cells": True,
            "max_source_target_overlap": max(source_target_overlaps),
        },
        "unique_cell_union": {
            "total": len(unique_cells),
            "dmso": len(selected_dmso),
            "treated": len(selected_treated),
            "fraction_of_available_pools": len(unique_cells) / len(available_ids),
        },
        "reuse": {
            "total_slot_to_unique_cell_ratio": (len(selections) * SET_SIZE) / len(unique_cells),
            "usage_count": numeric_summary(usage_values),
            "used_once": sum(value == 1 for value in usage_values),
            "used_more_than_once": sum(value > 1 for value in usage_values),
            "usage_histogram": {
                str(key): value for key, value in sorted(Counter(usage_values).items())
            },
            "pairwise_repeat_overlap_cells": overlap_stats,
        },
    }

    verified = verify_locators(unique_cells)
    if verified != len(unique_cells):
        raise AssertionError(f"Read-back verified {verified} of {len(unique_cells)} locators")
    summary["locator"]["readback_verified_cells"] = verified
    summary["outputs"] = {
        "unique_cells_csv": args.unique_cells.relative_to(PROJECT_ROOT).as_posix(),
        "unique_cells_sha256": sha256_file(args.unique_cells),
        "set_indices_csv": args.set_indices.relative_to(PROJECT_ROOT).as_posix(),
        "set_indices_sha256": sha256_file(args.set_indices),
    }
    with args.summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--dmso-counts", type=Path, default=DEFAULT_DMSO_COUNTS)
    parser.add_argument("--unique-cells", type=Path, default=DEFAULT_UNIQUE_CELLS)
    parser.add_argument("--set-indices", type=Path, default=DEFAULT_SET_INDICES)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--local-manifest", type=Path, default=DEFAULT_LOCAL_MANIFEST)
    args = parser.parse_args()

    cohort = read_csv(args.cohort, COHORT_REQUIRED)
    if len(cohort) != 360 or len({row["audit_pair_id"] for row in cohort}) != 360:
        raise ValueError("Expected 360 unique cohort pair rows")
    dmso_rows = read_csv(
        args.dmso_counts,
        {"plate", "cell_line_id", "control_sample", "cell_count", "eligible_S256"},
    )
    dmso_counts = {
        (row["plate"], row["control_sample"], "DMSO_TF", row["cell_line_id"]): int(
            row["cell_count"]
        )
        for row in dmso_rows
        if row["eligible_S256"] == "1"
    }
    if len(dmso_counts) != 24:
        raise ValueError(f"Expected 24 eligible DMSO atomic groups, got {len(dmso_counts)}")

    pool_counts, atomic_counts = prepare_pool_specs(cohort, dmso_counts)
    sample_ids = {key[1] for key in atomic_counts}
    cell_lines = {key[3] for key in atomic_counts}
    paths, shortlist_rows, total_dataset_shards = shortlist_shards(
        args.data_dir, sample_ids, cell_lines
    )
    expected_available = sum(atomic_counts.values())
    if shortlist_rows != expected_available:
        raise ValueError(
            f"Predicate-pushdown count mismatch: {shortlist_rows} != {expected_available}"
        )
    print(
        f"shortlisted {len(paths)}/{total_dataset_shards} shards, "
        f"matching_rows={shortlist_rows:,}",
        flush=True,
    )

    atomic_cells, shard_layout = scan_atomic_cells(paths, atomic_counts)
    pools = build_pools(pool_counts, atomic_cells)
    all_available_cells = [cell for group in atomic_cells.values() for cell in group]
    summary = write_outputs(
        cohort,
        pools,
        all_available_cells,
        shard_layout,
        args,
        total_dataset_shards,
        shortlist_rows,
    )
    print(f"Wrote: {args.unique_cells}")
    print(f"Wrote: {args.set_indices}")
    print(f"Wrote: {args.summary}")
    print(f"Unique cells: {summary['unique_cell_union']['total']:,}")
    print(
        "Slot/unique reuse ratio: "
        f"{summary['reuse']['total_slot_to_unique_cell_ratio']:.3f}"
    )
    print("Validation: PASS")


if __name__ == "__main__":
    main()
