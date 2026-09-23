#!/usr/bin/env python3
"""Plan treated-cap512 plus all eligible-DMSO GeneJEPA cache locators."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
DEFAULT_CONDITIONS = RESULTS / "tahoe_experiment1_condition_split_manifest.csv"
DEFAULT_DATA_DIR = PROJECT_ROOT / "hf_data_cache" / "data" / "data"
DEFAULT_LOCAL_MANIFEST = PROJECT_ROOT / "hf_data_cache" / "local_file_manifest.json"
DEFAULT_EXTRACTION_MANIFEST = RESULTS / "tahoe_latent_audit_epoch25_manifest.json"
DEFAULT_OUTPUT_PREFIX = RESULTS / "tahoe_experiment1_cache_cap512_all_dmso"

EXPECTED_SHARDS = 3388
EXPECTED_CONDITIONS = 56993
TREATED_CAP = 512
EMBEDDING_DIM = 768
FLOAT32_BYTES = 4
SEED = 42
WORKERS = 2
LOCATOR_VERSION = "tahoe_parquet_row_v1"
SELECTION_VERSION = "condition_rank_pcg64_without_replacement_v1"
KEY_COLUMNS = ["plate", "sample", "drug", "cell_line_id"]
OWNER_TREATED = 0
OWNER_DMSO = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(kind: str, *values: object) -> str:
    return hashlib.sha256(
        "|".join((kind, *(str(value) for value in values))).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def output_paths(prefix: Path) -> dict[str, Any]:
    stem = str(prefix)
    return {
        "condition_index": Path(stem + "_condition_index.csv"),
        "control_pool_index": Path(stem + "_control_pool_index.csv"),
        "shard_partition": Path(stem + "_shard_partition.csv"),
        "worker_plans": [Path(stem + f"_worker{worker}_plan.parquet") for worker in range(WORKERS)],
        "worker_manifests": [Path(stem + f"_worker{worker}_manifest.json") for worker in range(WORKERS)],
        "worker_progress": [Path(stem + f"_worker{worker}_progress.json") for worker in range(WORKERS)],
        "summary": Path(stem + "_summary.json"),
    }


def read_conditions(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    required = {
        "pair_id",
        "edge_id",
        "split",
        "plate",
        "cell_line_id",
        "drug",
        "dose_uM",
        "treated_samples",
        "treated_cell_count",
        "control_drug",
        "control_samples",
        "control_cell_count",
        "control_pool_id",
    }
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"Condition manifest is missing columns: {sorted(missing)}")
    rows.sort(key=lambda row: row["pair_id"])
    if len(rows) != EXPECTED_CONDITIONS:
        raise ValueError(f"Expected {EXPECTED_CONDITIONS} conditions, found {len(rows)}")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("pair_id is not unique")
    if any(row["control_drug"] != "DMSO_TF" for row in rows):
        raise ValueError("Unexpected control drug")
    return rows, fieldnames


def prepare_control_pools(
    conditions: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    pool_values: dict[str, tuple[str, str, str, int]] = {}
    split_counts: dict[str, Counter[str]] = {}
    condition_counts = Counter()
    for row in conditions:
        pool_id = row["control_pool_id"]
        values = (
            row["plate"],
            row["cell_line_id"],
            row["control_samples"],
            int(row["control_cell_count"]),
        )
        if pool_id in pool_values and pool_values[pool_id] != values:
            raise ValueError(f"Conflicting metadata for control pool {pool_id}")
        pool_values[pool_id] = values
        split_counts.setdefault(pool_id, Counter())[row["split"]] += 1
        condition_counts[pool_id] += 1

    pools = []
    for pool_index, pool_id in enumerate(sorted(pool_values)):
        plate, cell_line, samples, count = pool_values[pool_id]
        pools.append(
            {
                "control_pool_index": pool_index,
                "control_pool_id": pool_id,
                "plate": plate,
                "cell_line_id": cell_line,
                "control_drug": "DMSO_TF",
                "control_samples": samples,
                "available_cell_count": count,
                "cached_cell_count": count,
                "referenced_condition_count": condition_counts[pool_id],
                "train_condition_count": split_counts[pool_id]["train"],
                "val_condition_count": split_counts[pool_id]["val"],
                "test_condition_count": split_counts[pool_id]["test"],
            }
        )
    return pools, {row["control_pool_id"]: row["control_pool_index"] for row in pools}


def prepare_index_ranges(
    conditions: list[dict[str, str]],
    pools: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    available = np.array(
        [int(row["treated_cell_count"]) for row in conditions], dtype=np.int64
    )
    cached = np.minimum(available, TREATED_CAP)
    available_offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(available, dtype=np.int64))
    )
    treated_starts = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(cached, dtype=np.int64))
    )[:-1]
    control_counts = np.array(
        [int(row["cached_cell_count"]) for row in pools], dtype=np.int64
    )
    control_starts = int(cached.sum()) + np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(control_counts, dtype=np.int64))
    )[:-1]
    return available, cached, available_offsets, treated_starts, control_starts


def build_rank_to_embedding(
    conditions: list[dict[str, str]],
    available: np.ndarray,
    cached: np.ndarray,
    available_offsets: np.ndarray,
    treated_starts: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    mapping = np.full(int(available.sum()), -1, dtype=np.int32)
    condition_keys = []
    for index, row in enumerate(conditions):
        condition_key = stable_hash(
            "tahoe_experiment1_cache_condition_v1",
            row["plate"],
            row["cell_line_id"],
            row["drug"],
            row["dose_uM"],
            row["treated_samples"],
        )
        condition_keys.append(condition_key)
        count = int(available[index])
        keep = int(cached[index])
        if keep == count:
            ranks = np.arange(count, dtype=np.int64)
        else:
            seed = int.from_bytes(
                hashlib.sha256(
                    f"{SELECTION_VERSION}|seed={SEED}|{condition_key}".encode("utf-8")
                ).digest()[:8],
                "big",
            )
            ranks = np.sort(
                np.random.Generator(np.random.PCG64(seed)).choice(
                    count, size=keep, replace=False
                )
            )
        flat = int(available_offsets[index]) + ranks
        mapping[flat] = np.arange(
            int(treated_starts[index]),
            int(treated_starts[index]) + keep,
            dtype=np.int32,
        )
    if np.count_nonzero(mapping >= 0) != int(cached.sum()):
        raise AssertionError("Treated rank mapping does not contain the expected cap")
    return mapping, condition_keys


def build_atomic_mapping(
    conditions: list[dict[str, str]],
    pools: list[dict[str, Any]],
) -> pa.Table:
    records: list[dict[str, Any]] = []
    keys = set()
    for condition_index, row in enumerate(conditions):
        for sample in row["treated_samples"].split("|"):
            key = (row["plate"], sample, row["drug"], row["cell_line_id"])
            if key in keys:
                raise ValueError(f"Duplicate treated atomic key: {key}")
            keys.add(key)
            records.append(
                dict(
                    zip(
                        (*KEY_COLUMNS, "owner_type", "owner_index"),
                        (*key, OWNER_TREATED, condition_index),
                        strict=True,
                    )
                )
            )
    for pool in pools:
        for sample in pool["control_samples"].split("|"):
            key = (
                pool["plate"],
                sample,
                "DMSO_TF",
                pool["cell_line_id"],
            )
            if key in keys:
                raise ValueError(f"Duplicate control atomic key: {key}")
            keys.add(key)
            records.append(
                dict(
                    zip(
                        (*KEY_COLUMNS, "owner_type", "owner_index"),
                        (*key, OWNER_DMSO, pool["control_pool_index"]),
                        strict=True,
                    )
                )
            )
    return pa.Table.from_pylist(
        records,
        schema=pa.schema(
            [
                *(pa.field(column, pa.string()) for column in KEY_COLUMNS),
                pa.field("owner_type", pa.int8()),
                pa.field("owner_index", pa.int32()),
            ]
        ),
    )


def local_ranks(
    row_positions: np.ndarray,
    owner_indices: np.ndarray,
    row_indices: np.ndarray,
    seen: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.lexsort((row_indices[row_positions], owner_indices[row_positions]))
    positions = row_positions[order]
    owners = owner_indices[positions]
    starts = np.flatnonzero(
        np.concatenate((np.array([True]), owners[1:] != owners[:-1]))
    )
    counts = np.diff(np.append(starts, len(owners)))
    ranks = seen[owners] + np.arange(len(owners), dtype=np.int64) - np.repeat(
        starts, counts
    )
    seen[owners[starts]] += counts
    return positions, owners, ranks


def scan_and_write_plans(
    *,
    paths: list[Path],
    mapping_table: pa.Table,
    rank_to_embedding: np.ndarray,
    available_offsets: np.ndarray,
    available: np.ndarray,
    cached: np.ndarray,
    control_starts: np.ndarray,
    control_counts: np.ndarray,
    worker_plan_paths: list[Path],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, float]:
    plan_schema = pa.schema(
        [
            pa.field("part_index", pa.int64()),
            pa.field("embedding_index", pa.int64()),
            pa.field("shard_index", pa.int32()),
            pa.field("shard_path", pa.string()),
            pa.field("row_group_index", pa.int16()),
            pa.field("row_index_in_row_group", pa.int32()),
            pa.field("row_index_in_shard", pa.int32()),
            pa.field("owner_type", pa.int8()),
            pa.field("owner_index", pa.int32()),
        ]
    )
    temporary_paths = [path.with_name(path.name + ".tmp") for path in worker_plan_paths]
    writers = [
        pq.ParquetWriter(
            temporary,
            plan_schema,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
            write_statistics=True,
        )
        for temporary in temporary_paths
    ]
    seen_treated = np.zeros(len(available), dtype=np.int64)
    seen_control = np.zeros(len(control_counts), dtype=np.int64)
    worker_rows = np.zeros(WORKERS, dtype=np.int64)
    shard_records = []
    started = time.perf_counter()

    try:
        for shard_index, path in enumerate(paths):
            parquet = pq.ParquetFile(path)
            missing = set(KEY_COLUMNS) - set(parquet.schema_arrow.names)
            if missing:
                raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
            table = parquet.read(columns=KEY_COLUMNS).append_column(
                "row_index_in_shard",
                pa.array(np.arange(parquet.metadata.num_rows, dtype=np.int32)),
            )
            joined = table.join(mapping_table, keys=KEY_COLUMNS, join_type="inner")
            worker_id = shard_index % WORKERS
            part_start = int(worker_rows[worker_id])
            treated_selected = 0
            dmso_selected = 0
            if joined.num_rows:
                row_indices = joined["row_index_in_shard"].to_numpy(
                    zero_copy_only=False
                ).astype(np.int64, copy=False)
                owner_types = joined["owner_type"].to_numpy(
                    zero_copy_only=False
                ).astype(np.int8, copy=False)
                owner_indices = joined["owner_index"].to_numpy(
                    zero_copy_only=False
                ).astype(np.int64, copy=False)

                treated_rows = np.flatnonzero(owner_types == OWNER_TREATED)
                kept_positions: list[np.ndarray] = []
                kept_embeddings: list[np.ndarray] = []
                if len(treated_rows):
                    positions, owners, ranks = local_ranks(
                        treated_rows,
                        owner_indices,
                        row_indices,
                        seen_treated,
                    )
                    if np.any(ranks >= available[owners]):
                        raise AssertionError("Treated scan exceeded expected condition count")
                    flat = available_offsets[owners] + ranks
                    embedding_indices = rank_to_embedding[flat]
                    keep = embedding_indices >= 0
                    kept_positions.append(positions[keep])
                    kept_embeddings.append(embedding_indices[keep].astype(np.int64))
                    treated_selected = int(np.count_nonzero(keep))

                control_rows = np.flatnonzero(owner_types == OWNER_DMSO)
                if len(control_rows):
                    positions, owners, ranks = local_ranks(
                        control_rows,
                        owner_indices,
                        row_indices,
                        seen_control,
                    )
                    if np.any(ranks >= control_counts[owners]):
                        raise AssertionError("DMSO scan exceeded expected pool count")
                    kept_positions.append(positions)
                    kept_embeddings.append(control_starts[owners] + ranks)
                    dmso_selected = len(positions)

                if kept_positions:
                    selected_positions = np.concatenate(kept_positions)
                    embedding_indices = np.concatenate(kept_embeddings)
                    physical_order = np.argsort(
                        row_indices[selected_positions], kind="stable"
                    )
                    selected_positions = selected_positions[physical_order]
                    embedding_indices = embedding_indices[physical_order]
                    selected_rows = row_indices[selected_positions]
                    row_group_ends = np.cumsum(
                        [
                            parquet.metadata.row_group(index).num_rows
                            for index in range(parquet.num_row_groups)
                        ],
                        dtype=np.int64,
                    )
                    row_groups = np.searchsorted(
                        row_group_ends, selected_rows, side="right"
                    ).astype(np.int16)
                    row_group_starts = np.concatenate(
                        (np.zeros(1, dtype=np.int64), row_group_ends[:-1])
                    )
                    rows_in_group = (
                        selected_rows - row_group_starts[row_groups]
                    ).astype(np.int32)
                    count = len(selected_rows)
                    relative = path.relative_to(PROJECT_ROOT).as_posix()
                    plan = pa.Table.from_arrays(
                        [
                            pa.array(
                                np.arange(
                                    part_start,
                                    part_start + count,
                                    dtype=np.int64,
                                )
                            ),
                            pa.array(embedding_indices, type=pa.int64()),
                            pa.array(
                                np.full(count, shard_index, dtype=np.int32)
                            ),
                            pa.array([relative] * count, type=pa.string()),
                            pa.array(row_groups, type=pa.int16()),
                            pa.array(rows_in_group, type=pa.int32()),
                            pa.array(selected_rows.astype(np.int32), type=pa.int32()),
                            pa.array(owner_types[selected_positions], type=pa.int8()),
                            pa.array(
                                owner_indices[selected_positions].astype(np.int32),
                                type=pa.int32(),
                            ),
                        ],
                        schema=plan_schema,
                    )
                    writers[worker_id].write_table(plan, row_group_size=count)
                    worker_rows[worker_id] += count

            shard_records.append(
                {
                    "shard_index": shard_index,
                    "shard_path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "worker_id": worker_id,
                    "source_rows": parquet.metadata.num_rows,
                    "selected_treated_rows": treated_selected,
                    "selected_dmso_rows": dmso_selected,
                    "selected_total_rows": treated_selected + dmso_selected,
                    "worker_part_start": part_start,
                    "worker_part_stop_exclusive": int(worker_rows[worker_id]),
                }
            )
            if (shard_index + 1) % 100 == 0 or shard_index + 1 == len(paths):
                elapsed = time.perf_counter() - started
                print(
                    f"planned {shard_index + 1}/{len(paths)} shards, "
                    f"selected={int(worker_rows.sum()):,}, elapsed={elapsed:.1f}s",
                    flush=True,
                )
    finally:
        for writer in writers:
            writer.close()

    elapsed = time.perf_counter() - started
    if not np.array_equal(seen_treated, available):
        bad = np.flatnonzero(seen_treated != available)
        raise AssertionError(f"Treated count mismatch for {len(bad)} conditions")
    if not np.array_equal(seen_control, control_counts):
        bad = np.flatnonzero(seen_control != control_counts)
        raise AssertionError(f"DMSO count mismatch for {len(bad)} pools")
    if int(worker_rows.sum()) != int(cached.sum() + control_counts.sum()):
        raise AssertionError("Worker row counts do not equal planned cell union")
    for temporary, final in zip(temporary_paths, worker_plan_paths, strict=True):
        temporary.replace(final)
    return shard_records, worker_rows, seen_treated, elapsed


def audit_worker_plans(
    *,
    plan_paths: list[Path],
    total_cells: int,
    treated_starts: np.ndarray,
    cached: np.ndarray,
    control_starts: np.ndarray,
    control_counts: np.ndarray,
) -> dict[str, Any]:
    written = np.zeros(total_cells, dtype=np.bool_)
    treated_observed = np.zeros(len(cached), dtype=np.int64)
    control_observed = np.zeros(len(control_counts), dtype=np.int64)
    worker_summaries = []
    worker_shards = []
    for worker_id, path in enumerate(plan_paths):
        parquet = pq.ParquetFile(path)
        expected_part = 0
        previous_locator_key = -1
        shards = set()
        for batch in parquet.iter_batches(
            batch_size=1_000_000,
            columns=[
                "part_index",
                "embedding_index",
                "shard_index",
                "row_index_in_shard",
                "owner_type",
                "owner_index",
            ],
        ):
            part = batch["part_index"].to_numpy(zero_copy_only=False)
            embedding = batch["embedding_index"].to_numpy(zero_copy_only=False)
            shard = batch["shard_index"].to_numpy(zero_copy_only=False)
            row = batch["row_index_in_shard"].to_numpy(zero_copy_only=False)
            owner_type = batch["owner_type"].to_numpy(zero_copy_only=False)
            owner = batch["owner_index"].to_numpy(zero_copy_only=False).astype(
                np.int64, copy=False
            )
            if not np.array_equal(
                part,
                np.arange(expected_part, expected_part + len(part), dtype=np.int64),
            ):
                raise AssertionError(f"worker{worker_id} part_index is not contiguous")
            expected_part += len(part)
            if np.any((embedding < 0) | (embedding >= total_cells)):
                raise AssertionError("embedding_index is out of range")
            if len(np.unique(embedding)) != len(embedding) or written[embedding].any():
                raise AssertionError("Duplicate embedding_index detected")
            written[embedding] = True
            locator_key = (shard.astype(np.int64) << 32) | row.astype(np.int64)
            if (
                np.any(locator_key[1:] <= locator_key[:-1])
                or int(locator_key[0]) <= previous_locator_key
            ):
                raise AssertionError(f"worker{worker_id} locator order is not strict")
            previous_locator_key = int(locator_key[-1])
            shards.update(np.unique(shard).tolist())

            treated_mask = owner_type == OWNER_TREATED
            if treated_mask.any():
                treated_owner = owner[treated_mask]
                treated_embedding = embedding[treated_mask]
                if np.any(
                    (treated_embedding < treated_starts[treated_owner])
                    | (
                        treated_embedding
                        >= treated_starts[treated_owner] + cached[treated_owner]
                    )
                ):
                    raise AssertionError("Treated embedding range mismatch")
                treated_observed += np.bincount(
                    treated_owner, minlength=len(cached)
                )
            control_mask = owner_type == OWNER_DMSO
            if control_mask.any():
                control_owner = owner[control_mask]
                control_embedding = embedding[control_mask]
                if np.any(
                    (control_embedding < control_starts[control_owner])
                    | (
                        control_embedding
                        >= control_starts[control_owner]
                        + control_counts[control_owner]
                    )
                ):
                    raise AssertionError("DMSO embedding range mismatch")
                control_observed += np.bincount(
                    control_owner, minlength=len(control_counts)
                )

        worker_shards.append(shards)
        worker_summaries.append(
            {
                "worker_id": worker_id,
                "rows": expected_part,
                "shards_with_selected_cells": len(shards),
                "plan_size_bytes": path.stat().st_size,
                "plan_sha256": sha256_file(path),
            }
        )

    if worker_shards[0] & worker_shards[1]:
        raise AssertionError("Worker shard sets overlap")
    if not written.all():
        raise AssertionError(f"Missing {np.count_nonzero(~written)} embedding indices")
    if not np.array_equal(treated_observed, cached):
        raise AssertionError("Treated owner counts do not match cached counts")
    if not np.array_equal(control_observed, control_counts):
        raise AssertionError("DMSO owner counts do not match all eligible cells")
    return {
        "missing_embedding_indices": 0,
        "duplicate_embedding_indices": 0,
        "each_embedding_index_exactly_once": True,
        "worker_cell_overlap": 0,
        "worker_union_cells": int(written.sum()),
        "worker_shard_overlap": 0,
        "strict_locator_order_within_worker": True,
        "owner_counts_match": True,
        "workers": worker_summaries,
    }


def readback_locator_sample(
    *,
    plan_paths: list[Path],
    conditions: list[dict[str, str]],
    pools: list[dict[str, Any]],
    sample_size: int = 512,
) -> int:
    row_groups = [
        (path, row_group)
        for path in plan_paths
        for row_group in range(pq.ParquetFile(path).num_row_groups)
    ]
    chosen = np.unique(
        np.linspace(0, len(row_groups) - 1, min(sample_size, len(row_groups)), dtype=int)
    )
    verified = 0
    for selection in chosen:
        plan_path, plan_row_group = row_groups[int(selection)]
        table = pq.ParquetFile(plan_path).read_row_group(
            plan_row_group,
            columns=[
                "shard_path",
                "row_group_index",
                "row_index_in_row_group",
                "row_index_in_shard",
                "owner_type",
                "owner_index",
            ],
        )
        index = table.num_rows // 2
        shard_path = table["shard_path"][index].as_py()
        row_group = table["row_group_index"][index].as_py()
        row_in_group = table["row_index_in_row_group"][index].as_py()
        row_in_shard = table["row_index_in_shard"][index].as_py()
        owner_type = table["owner_type"][index].as_py()
        owner_index = table["owner_index"][index].as_py()
        source = pq.ParquetFile(PROJECT_ROOT / shard_path)
        source_table = source.read_row_group(row_group, columns=KEY_COLUMNS)
        observed = {
            column: source_table[column][row_in_group].as_py()
            for column in KEY_COLUMNS
        }
        offset = sum(
            source.metadata.row_group(i).num_rows for i in range(row_group)
        )
        if offset + row_in_group != row_in_shard:
            raise AssertionError("Read-back shard row offset mismatch")
        if owner_type == OWNER_TREATED:
            expected = conditions[owner_index]
            valid = (
                observed["plate"] == expected["plate"]
                and observed["cell_line_id"] == expected["cell_line_id"]
                and observed["drug"] == expected["drug"]
                and observed["sample"] in expected["treated_samples"].split("|")
            )
        else:
            expected = pools[owner_index]
            valid = (
                observed["plate"] == expected["plate"]
                and observed["cell_line_id"] == expected["cell_line_id"]
                and observed["drug"] == "DMSO_TF"
                and observed["sample"] in expected["control_samples"].split("|")
            )
        if not valid:
            raise AssertionError("Read-back locator metadata mismatch")
        verified += 1
    return verified


def relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, default=DEFAULT_CONDITIONS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--local-manifest", type=Path, default=DEFAULT_LOCAL_MANIFEST)
    parser.add_argument(
        "--extraction-manifest", type=Path, default=DEFAULT_EXTRACTION_MANIFEST
    )
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    outputs = output_paths(args.output_prefix)
    final_paths = [
        outputs["condition_index"],
        outputs["control_pool_index"],
        outputs["shard_partition"],
        *outputs["worker_plans"],
        *outputs["worker_manifests"],
        *outputs["worker_progress"],
        outputs["summary"],
    ]
    existing = [path for path in final_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Outputs exist; inspect them or pass --overwrite: "
            + ", ".join(map(str, existing))
        )

    started = time.perf_counter()
    conditions, condition_columns = read_conditions(args.conditions)
    pools, pool_lookup = prepare_control_pools(conditions)
    available, cached, available_offsets, treated_starts, control_starts = (
        prepare_index_ranges(conditions, pools)
    )
    control_counts = np.array(
        [row["cached_cell_count"] for row in pools], dtype=np.int64
    )
    total_treated = int(cached.sum())
    total_dmso = int(control_counts.sum())
    total_cells = total_treated + total_dmso
    rank_to_embedding, condition_keys = build_rank_to_embedding(
        conditions,
        available,
        cached,
        available_offsets,
        treated_starts,
    )
    mapping_table = build_atomic_mapping(conditions, pools)
    print(
        f"prepared {len(conditions):,} treated conditions and {len(pools):,} "
        f"eligible DMSO pools; planned cells={total_cells:,}",
        flush=True,
    )

    local_manifest = json.loads(args.local_manifest.read_text(encoding="utf-8"))
    manifest_names = sorted(Path(path).name for path in local_manifest["data_files"])
    paths = sorted(args.data_dir.glob("train-*.parquet"))
    if len(paths) != EXPECTED_SHARDS or [path.name for path in paths] != manifest_names:
        raise ValueError("Local parquet paths do not exactly match local_file_manifest.json")

    shard_records, worker_rows, _seen_treated, scan_seconds = scan_and_write_plans(
        paths=paths,
        mapping_table=mapping_table,
        rank_to_embedding=rank_to_embedding,
        available_offsets=available_offsets,
        available=available,
        cached=cached,
        control_starts=control_starts,
        control_counts=control_counts,
        worker_plan_paths=outputs["worker_plans"],
    )
    del rank_to_embedding, mapping_table

    for index, row in enumerate(conditions):
        pool_index = pool_lookup[row["control_pool_id"]]
        row.update(
            {
                "cache_condition_index": index,
                "condition_key_sha256": condition_keys[index],
                "treated_cache_policy": "min(512, available)",
                "treated_cached_cell_count": int(cached[index]),
                "treated_embedding_start": int(treated_starts[index]),
                "treated_embedding_stop_exclusive": int(
                    treated_starts[index] + cached[index]
                ),
                "control_pool_index": pool_index,
                "control_cache_policy": "all eligible DMSO cells",
                "control_cached_cell_count": int(control_counts[pool_index]),
                "control_embedding_start": int(control_starts[pool_index]),
                "control_embedding_stop_exclusive": int(
                    control_starts[pool_index] + control_counts[pool_index]
                ),
            }
        )
    for pool in pools:
        index = pool["control_pool_index"]
        pool["embedding_start"] = int(control_starts[index])
        pool["embedding_stop_exclusive"] = int(
            control_starts[index] + control_counts[index]
        )

    appended_condition_columns = [
        "cache_condition_index",
        "condition_key_sha256",
        "treated_cache_policy",
        "treated_cached_cell_count",
        "treated_embedding_start",
        "treated_embedding_stop_exclusive",
        "control_pool_index",
        "control_cache_policy",
        "control_cached_cell_count",
        "control_embedding_start",
        "control_embedding_stop_exclusive",
    ]
    write_csv(
        outputs["condition_index"],
        conditions,
        [*condition_columns, *appended_condition_columns],
    )
    pool_columns = list(pools[0])
    write_csv(outputs["control_pool_index"], pools, pool_columns)
    write_csv(
        outputs["shard_partition"],
        shard_records,
        list(shard_records[0]),
    )

    audit_started = time.perf_counter()
    plan_audit = audit_worker_plans(
        plan_paths=outputs["worker_plans"],
        total_cells=total_cells,
        treated_starts=treated_starts,
        cached=cached,
        control_starts=control_starts,
        control_counts=control_counts,
    )
    locator_readback = readback_locator_sample(
        plan_paths=outputs["worker_plans"],
        conditions=conditions,
        pools=pools,
    )
    audit_seconds = time.perf_counter() - audit_started

    extraction = json.loads(args.extraction_manifest.read_text(encoding="utf-8"))
    rate = float(extraction["inference"]["cells_per_second"])
    single_hours = total_cells / rate / 3600
    payload_bytes = total_cells * EMBEDDING_DIM * FLOAT32_BYTES

    for worker_id in range(WORKERS):
        worker_summary = plan_audit["workers"][worker_id]
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "planned_not_started",
            "worker_id": worker_id,
            "device_contract": f"CUDA_VISIBLE_DEVICES={worker_id}; local device cuda:0",
            "distributed": False,
            "plan": {
                "path": relative(outputs["worker_plans"][worker_id]),
                "sha256": worker_summary["plan_sha256"],
                "rows": int(worker_rows[worker_id]),
                "part_index_start": 0,
                "part_index_stop_exclusive": int(worker_rows[worker_id]),
                "physical_order": "shard_path + row_group_index + row_index_in_row_group",
                "scatter_key": "embedding_index",
            },
            "resume": {
                "progress_path": relative(outputs["worker_progress"][worker_id]),
                "completed_bitmap_path": relative(
                    Path(str(args.output_prefix) + f"_worker{worker_id}_completed.npy")
                ),
                "partial_embedding_path": relative(
                    Path(str(args.output_prefix) + f"_worker{worker_id}_embeddings.npy.partial")
                ),
                "final_embedding_path": relative(
                    Path(str(args.output_prefix) + f"_worker{worker_id}_embeddings.npy")
                ),
                "validation_before_resume": [
                    "worker plan SHA-256",
                    "GeneJEPA checkpoint SHA-256",
                    "global stats SHA-256",
                    "part array shape/dtype",
                    "completed bitmap length",
                ],
            },
            "output_contract": {
                "shape": [int(worker_rows[worker_id]), EMBEDDING_DIM],
                "dtype": "float32",
                "one_inference_per_unique_cell": True,
                "no_shared_output_file_between_workers": True,
            },
        }
        write_json(outputs["worker_manifests"][worker_id], manifest)
        write_json(
            outputs["worker_progress"][worker_id],
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "not_started",
                "embedding_extraction_started": False,
                "worker_id": worker_id,
                "plan_sha256": worker_summary["plan_sha256"],
                "processed_cells": 0,
                "total_cells": int(worker_rows[worker_id]),
                "next_part_index": 0,
                "completed_shards": 0,
            },
        )

    index_paths = [
        outputs["condition_index"],
        outputs["control_pool_index"],
        outputs["shard_partition"],
    ]
    plan_bytes = sum(path.stat().st_size for path in outputs["worker_plans"])
    index_bytes = sum(path.stat().st_size for path in index_paths)
    worker_balance = float(worker_rows.max() / worker_rows.min())
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "scope": {
            "locator_plan_generated": True,
            "genejepa_embedding_extraction_run": False,
            "state_training_run": False,
            "expression_columns_read": False,
            "metadata_columns_read": KEY_COLUMNS,
        },
        "inputs": {
            "condition_manifest": relative(args.conditions),
            "condition_manifest_sha256": sha256_file(args.conditions),
            "local_manifest": relative(args.local_manifest),
            "local_manifest_sha256": sha256_file(args.local_manifest),
            "epoch25_extraction_manifest": relative(args.extraction_manifest),
            "epoch25_extraction_manifest_sha256": sha256_file(
                args.extraction_manifest
            ),
            "dataset_shards": len(paths),
        },
        "selection": {
            "seed": SEED,
            "treated": {
                "policy": "min(512, available cells) per eligible condition",
                "conditions": len(conditions),
                "available_cells": int(available.sum()),
                "cached_cells": total_treated,
                "algorithm": SELECTION_VERSION,
                "within_condition_replacement": False,
                "scan_order": "sorted shard path then row index within shard",
                "scan_order_guard": "local_file_manifest SHA-256",
                "numpy_version": np.__version__,
                "conditions_by_split": {
                    split: sum(row["split"] == split for row in conditions)
                    for split in ("train", "val", "test")
                },
                "available_cells_by_split": {
                    split: int(
                        sum(
                            available[index]
                            for index, row in enumerate(conditions)
                            if row["split"] == split
                        )
                    )
                    for split in ("train", "val", "test")
                },
                "cached_cells_by_split": {
                    split: int(
                        sum(
                            cached[index]
                            for index, row in enumerate(conditions)
                            if row["split"] == split
                        )
                    )
                    for split in ("train", "val", "test")
                },
            },
            "dmso": {
                "policy": "all cells in DMSO pools referenced by eligible conditions",
                "control_pools": len(pools),
                "cached_cells": total_dmso,
                "subsampling": False,
                "shared_across_splits": True,
            },
        },
        "embedding_index": {
            "total_cells": total_cells,
            "first": 0,
            "last": total_cells - 1,
            "treated_range": [0, total_treated],
            "dmso_range": [total_treated, total_cells],
            "treated_condition_ranges_contiguous": True,
            "dmso_pool_ranges_contiguous": True,
            "condition_metadata_with_split": relative(outputs["condition_index"]),
            "control_pool_metadata": relative(outputs["control_pool_index"]),
        },
        "locator": {
            "version": LOCATOR_VERSION,
            "canonical_components": [
                "project-relative shard_path",
                "row_group_index",
                "row_index_in_row_group",
            ],
            "depends_on_temporary_global_scan_row": False,
            "readback_verified_sample_cells": locator_readback,
            "plan_format": "Parquet zstd; no CSV/BOM",
        },
        "partition": {
            "workers": WORKERS,
            "algorithm": "sorted shard index modulo 2; whole shard assigned to one worker",
            "worker_rows": worker_rows.tolist(),
            "max_to_min_cell_ratio": worker_balance,
            **plan_audit,
        },
        "storage": {
            "embedding_payload_bytes": payload_bytes,
            "embedding_payload_GiB": payload_bytes / 2**30,
            "locator_plan_parquet_bytes_actual": plan_bytes,
            "locator_plan_parquet_GiB_actual": plan_bytes / 2**30,
            "condition_control_shard_index_bytes_actual": index_bytes,
            "condition_control_shard_index_MiB_actual": index_bytes / 2**20,
            "combined_plan_and_index_bytes_actual": plan_bytes + index_bytes,
            "combined_plan_and_index_GiB_actual": (plan_bytes + index_bytes) / 2**30,
        },
        "extraction_time_estimate": {
            "reference_cells_per_second": rate,
            "single_a6000_hours": single_hours,
            "two_worker_wall_hours": {
                f"{speedup:g}x_effective_speedup": single_hours / speedup
                for speedup in (2.0, 1.7, 1.5, 1.3)
            },
            "qualification": (
                "Two independent workers share parquet storage, so actual speedup can be "
                "below 2x and must be updated from a short extraction pilot."
            ),
        },
        "csv_encoding": "utf-8-sig for human-review CSV; JSON and Parquet have no BOM",
        "timing": {
            "scan_and_plan_seconds": scan_seconds,
            "postwrite_audit_seconds": audit_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "outputs": {},
    }
    for name in ("condition_index", "control_pool_index", "shard_partition"):
        path = outputs[name]
        summary["outputs"][name] = {
            "path": relative(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "encoding": "utf-8-sig",
        }
    for worker_id in range(WORKERS):
        for kind in ("worker_plans", "worker_manifests", "worker_progress"):
            path = outputs[kind][worker_id]
            summary["outputs"][f"worker{worker_id}_{kind.removeprefix('worker_')}"] = {
                "path": relative(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "encoding": "binary Parquet"
                if path.suffix == ".parquet"
                else "UTF-8 without BOM",
            }
    write_json(outputs["summary"], summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
