#!/usr/bin/env python3
"""Freeze and audit the Task-2 / Phase-II ST-A v2 data and cache plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
TASK_PATH = PROJECT_ROOT.parent / "Phase II.md"
TASK_SHA256 = "0da488785317b391296b08557c9c39e82307cb9335dd9039eb84d239152e75e9"

PHASE1_MANIFEST = RESULTS / "phase1_top20_subset_manifest.json"
PHASE1_CONDITIONS = RESULTS / "phase1_top20_subset_conditions.csv"
SOURCE_PREFIX = RESULTS / "tahoe_experiment1_cache_cap512_all_dmso"
SOURCE_CONDITIONS = Path(str(SOURCE_PREFIX) + "_condition_index.csv")
SOURCE_CONTROLS = Path(str(SOURCE_PREFIX) + "_control_pool_index.csv")
SOURCE_PLANS = tuple(Path(str(SOURCE_PREFIX) + f"_worker{i}_plan.parquet") for i in range(2))
DRUG_VOCABULARY = RESULTS / "tahoe_experiment1_drug_vocabulary.csv"

CONDITIONS_PATH = RESULTS / "phase2_stav2_conditions.csv"
CONTROL_POOLS_PATH = RESULTS / "phase2_stav2_control_pools.csv"
SUBSET_MANIFEST = RESULTS / "phase2_stav2_subset_manifest.json"
CACHE_PLAN = RESULTS / "phase2_stav2_cache_plan.json"
DOSE_NORMALIZATION = RESULTS / "phase2_stav2_dose_normalization.json"
SAMPLING_AUDIT = RESULTS / "phase2_stav2_dynamic_sampling_audit.json"
AUTHOR_CACHE_DIR = RESULTS / "phase2_author_genejepa_cache"
AUTHOR_PREFIX = AUTHOR_CACHE_DIR / "cache"

SEED = 42
SET_SIZE = 256
POOL_CAP = 512
LATENT_DIM = 768
SPLIT_QUOTAS = {"train": 4000, "val": 500, "test": 500}
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}
OWNER_TREATED = 0
OWNER_CONTROL = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: Any) -> str:
    payload = "|".join([str(SEED), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": relative(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", lineterminator="\n")
    os.replace(temporary, path)


def worker_path(worker_id: int, suffix: str) -> Path:
    return Path(str(AUTHOR_PREFIX) + f"_worker{worker_id}_{suffix}")


def frozen_cell_lines() -> list[str]:
    manifest = json.loads(PHASE1_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass":
        raise AssertionError("Phase-I subset manifest is not PASS")
    phase1 = pd.read_csv(PHASE1_CONDITIONS, encoding="utf-8-sig", keep_default_na=False)
    from_csv = sorted(phase1["cell_line_id"].astype(str).unique().tolist())
    manifest_lines = sorted(manifest["selection"]["cell_lines"] if "cell_lines" in manifest["selection"] else manifest["outputs"] and from_csv)
    if from_csv != manifest_lines or len(from_csv) != 5:
        raise AssertionError("Phase-I frozen five-cell-line contract changed")
    return from_csv


def equal_stratum_quotas(candidates: pd.DataFrame, total: int) -> dict[tuple[str, float], int]:
    strata = sorted(
        {(str(row.cell_line_id), float(row.dose_uM)) for row in candidates.itertuples(index=False)}
    )
    base, remainder = divmod(total, len(strata))
    priority = sorted(strata, key=lambda value: stable_hash("stratum", *value))
    quotas = {key: base + int(key in set(priority[:remainder])) for key in strata}
    capacities = candidates.groupby(["cell_line_id", "dose_uM"]).size().to_dict()
    if any(capacities.get(key, 0) < count for key, count in quotas.items()):
        raise RuntimeError(f"A balanced stratum cannot fill its quota: {quotas}")
    return quotas


def select_split(candidates: pd.DataFrame, split: str, total: int) -> pd.DataFrame:
    candidates = candidates.copy().reset_index(drop=True)
    candidates["selection_sha256"] = [
        stable_hash("phase2-condition", split, pair_id)
        for pair_id in candidates["pair_id"].astype(str)
    ]
    quotas = equal_stratum_quotas(candidates, total)
    selected: set[int] = set()
    counts = {key: 0 for key in quotas}

    # First cover every available drug. Rarest drugs are assigned before flexible drugs.
    drug_groups = {
        str(drug): group.index.to_numpy(np.int64)
        for drug, group in candidates.groupby("drug", sort=True)
    }
    drug_order = sorted(
        drug_groups,
        key=lambda drug: (len(drug_groups[drug]), stable_hash("required-drug", split, drug)),
    )
    if len(drug_order) > total:
        raise RuntimeError(f"{split}: drug coverage exceeds condition quota")
    for drug in drug_order:
        options: list[tuple[int, str, int]] = []
        for index in drug_groups[drug]:
            row = candidates.loc[int(index)]
            key = (str(row["cell_line_id"]), float(row["dose_uM"]))
            remaining = quotas[key] - counts[key]
            if remaining > 0:
                options.append(
                    (-remaining, stable_hash("required-pick", split, drug, row["pair_id"]), int(index))
                )
        if not options:
            raise RuntimeError(f"{split}: cannot place required drug {drug!r} within balanced quotas")
        index = min(options)[2]
        row = candidates.loc[index]
        key = (str(row["cell_line_id"]), float(row["dose_uM"]))
        selected.add(index)
        counts[key] += 1

    # Fill each cell-line/dose stratum with deterministic drug/control-pool round-robin ordering.
    for key, quota in quotas.items():
        need = quota - counts[key]
        if need == 0:
            continue
        pool = candidates.loc[
            candidates["cell_line_id"].astype(str).eq(key[0])
            & candidates["dose_uM"].astype(float).eq(key[1])
            & ~candidates.index.isin(selected)
        ].copy()
        pool = pool.sort_values(["selection_sha256", "pair_id"], kind="stable")
        pool["drug_round"] = pool.groupby("drug", sort=False).cumcount()
        pool["control_round"] = pool.groupby("control_pool_id", sort=False).cumcount()
        pool["coverage_round"] = np.maximum(pool["drug_round"], pool["control_round"])
        chosen = pool.sort_values(
            ["coverage_round", "drug_round", "control_round", "selection_sha256", "pair_id"],
            kind="stable",
        ).head(need)
        if len(chosen) != need:
            raise RuntimeError(f"{split}: failed to fill stratum {key}")
        selected.update(chosen.index.astype(int).tolist())
        counts[key] += need

    result = candidates.loc[sorted(selected)].copy()
    if len(result) != total or any(counts[key] != quotas[key] for key in quotas):
        raise AssertionError(f"{split}: deterministic selection count mismatch")
    result["selection_algorithm"] = "coverage-first balanced-stratum seed42 sha256 v1"
    return result


def select_conditions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cell_lines = frozen_cell_lines()
    source = pd.read_csv(
        SOURCE_CONDITIONS, encoding="utf-8-sig", keep_default_na=False, low_memory=False
    )
    controls_source = pd.read_csv(
        SOURCE_CONTROLS, encoding="utf-8-sig", keep_default_na=False, low_memory=False
    )
    vocab = pd.read_csv(DRUG_VOCABULARY, encoding="utf-8-sig", keep_default_na=False)
    if len(source) != 56_993 or len(vocab) != 379:
        raise AssertionError("Frozen Experiment-1 condition or drug universe changed")
    drug_id = vocab.set_index("drug")["drug_id"]
    eligible = source.loc[
        source["eligible_S256"].eq(1)
        & source["cell_line_id"].astype(str).isin(cell_lines)
        & source["treated_cached_cell_count"].ge(SET_SIZE)
        & source["control_cached_cell_count"].ge(SET_SIZE)
    ].copy()
    eligible["drug_id"] = eligible["drug"].map(drug_id)
    if eligible["drug_id"].isna().any():
        raise AssertionError("Eligible condition drug missing from frozen 379-drug vocabulary")
    eligible["drug_id"] = eligible["drug_id"].astype(np.int16)
    train_drugs = set(eligible.loc[eligible["split"].eq("train"), "drug"].astype(str))
    excluded_unseen = eligible.loc[
        eligible["split"].isin(["val", "test"])
        & ~eligible["drug"].astype(str).isin(train_drugs)
    ].copy()
    candidates = eligible.loc[
        eligible["split"].eq("train") | eligible["drug"].astype(str).isin(train_drugs)
    ].copy()
    selected_parts = [
        select_split(candidates.loc[candidates["split"].eq(split)], split, quota)
        for split, quota in SPLIT_QUOTAS.items()
    ]
    selected = pd.concat(selected_parts, ignore_index=True)
    selected["split_order"] = selected["split"].map(SPLIT_ORDER).astype(np.int8)
    selected = selected.sort_values(
        ["split_order", "cell_line_id", "dose_uM", "plate", "pair_id"], kind="stable"
    ).drop(columns="split_order").reset_index(drop=True)
    selected.insert(0, "phase2_condition_index", np.arange(len(selected), dtype=np.int32))
    selected_train_drugs = set(selected.loc[selected["split"].eq("train"), "drug"].astype(str))
    for split in ("val", "test"):
        unseen = set(selected.loc[selected["split"].eq(split), "drug"].astype(str)) - selected_train_drugs
        if unseen:
            raise AssertionError(f"Selected {split} has unseen-in-train drugs: {sorted(unseen)}")
    if selected.groupby("edge_id")["split"].nunique().max() != 1:
        raise AssertionError("Selected subset has edge split leakage")
    if sorted(selected["cell_line_id"].unique().tolist()) != cell_lines:
        raise AssertionError("Selected Phase-II cell lines differ from Phase-I frozen five")

    selected_control_ids = selected["control_pool_id"].astype(str).unique().tolist()
    controls = controls_source.loc[
        controls_source["control_pool_id"].astype(str).isin(selected_control_ids)
    ].copy()
    if len(controls) != len(selected_control_ids) or controls["available_cell_count"].lt(SET_SIZE).any():
        raise AssertionError("Selected matched-control pool is missing or below S=256")
    controls = controls.sort_values(["plate", "cell_line_id", "control_pool_id"], kind="stable").reset_index(drop=True)
    controls = controls.rename(
        columns={
            "control_pool_index": "source_control_pool_index",
            "embedding_start": "source_embedding_start",
            "embedding_stop_exclusive": "source_embedding_stop_exclusive",
            "cached_cell_count": "source_cached_cell_count",
        }
    )
    controls.insert(0, "control_pool_index", np.arange(len(controls), dtype=np.int16))
    controls["cached_cell_count"] = np.minimum(
        POOL_CAP, controls["available_cell_count"].to_numpy(np.int64)
    ).astype(np.int16)

    source_columns = {
        "treated_embedding_start": "source_treated_embedding_start",
        "treated_embedding_stop_exclusive": "source_treated_embedding_stop_exclusive",
        "control_pool_index": "source_control_pool_index",
        "control_embedding_start": "source_control_embedding_start",
        "control_embedding_stop_exclusive": "source_control_embedding_stop_exclusive",
    }
    selected = selected.rename(columns=source_columns)
    phase_control_index = controls.set_index("control_pool_id")["control_pool_index"]
    selected["control_pool_index"] = selected["control_pool_id"].map(phase_control_index).astype(np.int16)
    selected["treated_available_cell_count"] = selected["treated_cell_count"].astype(np.int32)
    selected["control_available_cell_count"] = selected["control_cell_count"].astype(np.int32)
    selected["treated_cached_cell_count"] = (
        selected["source_treated_embedding_stop_exclusive"].astype(np.int64)
        - selected["source_treated_embedding_start"].astype(np.int64)
    ).astype(np.int16)
    if selected["treated_cached_cell_count"].lt(SET_SIZE).any() or selected["treated_cached_cell_count"].gt(POOL_CAP).any():
        raise AssertionError("Selected treated source cache violates 256..512 contract")

    cursor = 0
    treated_starts: list[int] = []
    treated_stops: list[int] = []
    for count in selected["treated_cached_cell_count"].astype(int):
        treated_starts.append(cursor)
        cursor += count
        treated_stops.append(cursor)
    selected["treated_embedding_start"] = np.asarray(treated_starts, dtype=np.int64)
    selected["treated_embedding_stop_exclusive"] = np.asarray(treated_stops, dtype=np.int64)
    control_starts: list[int] = []
    control_stops: list[int] = []
    for count in controls["cached_cell_count"].astype(int):
        control_starts.append(cursor)
        cursor += count
        control_stops.append(cursor)
    controls["embedding_start"] = np.asarray(control_starts, dtype=np.int64)
    controls["embedding_stop_exclusive"] = np.asarray(control_stops, dtype=np.int64)
    control_by_id = controls.set_index("control_pool_id")
    selected["control_cached_cell_count"] = selected["control_pool_id"].map(
        control_by_id["cached_cell_count"]
    ).astype(np.int16)
    selected["control_embedding_start"] = selected["control_pool_id"].map(
        control_by_id["embedding_start"]
    ).astype(np.int64)
    selected["control_embedding_stop_exclusive"] = selected["control_pool_id"].map(
        control_by_id["embedding_stop_exclusive"]
    ).astype(np.int64)

    audit = {
        "eligible_within_five_cell_lines": int(len(eligible)),
        "candidate_after_unseen_val_test_removal": int(len(candidates)),
        "excluded_unseen_val_test": excluded_unseen[
            ["pair_id", "split", "cell_line_id", "drug", "dose_uM", "plate"]
        ].to_dict("records"),
        "excluded_unseen_drugs": sorted(excluded_unseen["drug"].astype(str).unique().tolist()),
        "frozen_cell_lines": cell_lines,
        "total_cells": cursor,
    }
    return selected, controls, vocab, audit


def selected_source_arrays(
    conditions: pd.DataFrame, controls: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = int(controls["embedding_stop_exclusive"].iloc[-1])
    source = np.empty(total, dtype=np.int64)
    owner_type = np.empty(total, dtype=np.int8)
    owner_index = np.empty(total, dtype=np.int32)
    for row in conditions.itertuples(index=False):
        phase = slice(int(row.treated_embedding_start), int(row.treated_embedding_stop_exclusive))
        indices = np.arange(
            int(row.source_treated_embedding_start),
            int(row.source_treated_embedding_stop_exclusive),
            dtype=np.int64,
        )
        if len(indices) != phase.stop - phase.start:
            raise AssertionError("Treated phase/source range length mismatch")
        source[phase] = indices
        owner_type[phase] = OWNER_TREATED
        owner_index[phase] = int(row.cache_condition_index)
    for row in controls.itertuples(index=False):
        phase = slice(int(row.embedding_start), int(row.embedding_stop_exclusive))
        available = np.arange(
            int(row.source_embedding_start), int(row.source_embedding_stop_exclusive), dtype=np.int64
        )
        order = sorted(
            range(len(available)),
            key=lambda offset: stable_hash("phase2-control-cell", row.control_pool_id, int(available[offset])),
        )
        indices = np.sort(available[np.asarray(order[: int(row.cached_cell_count)], dtype=np.int64)])
        source[phase] = indices
        owner_type[phase] = OWNER_CONTROL
        owner_index[phase] = int(row.source_control_pool_index)
    phase_index = np.arange(total, dtype=np.int64)
    if len(np.unique(source)) != total:
        raise AssertionError("Phase-II selected physical cells are not globally unique")
    return source, phase_index, owner_type, owner_index


def write_worker_plans(
    source_indices: np.ndarray,
    phase_indices: np.ndarray,
    expected_owner_type: np.ndarray,
    expected_owner_index: np.ndarray,
) -> list[dict[str, Any]]:
    order = np.argsort(source_indices)
    sorted_source = source_indices[order]
    sorted_phase = phase_indices[order]
    seen = np.zeros(len(source_indices), dtype=np.bool_)
    schema = pa.schema(
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
            pa.field("source_embedding_index", pa.int64()),
        ]
    )
    records: list[dict[str, Any]] = []
    for worker_id, source_path in enumerate(SOURCE_PLANS):
        output = worker_path(worker_id, "plan.parquet")
        temporary = output.with_name(output.name + ".tmp")
        source_plan = pq.ParquetFile(source_path)
        writer = pq.ParquetWriter(temporary, schema, compression="zstd")
        part_cursor = 0
        shards: set[str] = set()
        try:
            for row_group in range(source_plan.num_row_groups):
                table = source_plan.read_row_group(row_group)
                original = table["embedding_index"].to_numpy(zero_copy_only=False).astype(np.int64)
                positions = np.searchsorted(sorted_source, original)
                valid = positions < len(sorted_source)
                clipped = np.minimum(positions, len(sorted_source) - 1)
                valid &= sorted_source[clipped] == original
                take = np.flatnonzero(valid)
                if not len(take):
                    continue
                phase = sorted_phase[positions[take]]
                if np.any(seen[phase]):
                    raise AssertionError("A physical cell appeared in both source worker plans")
                observed_type = table["owner_type"].take(pa.array(take)).to_numpy(zero_copy_only=False).astype(np.int8)
                observed_index = table["owner_index"].take(pa.array(take)).to_numpy(zero_copy_only=False).astype(np.int32)
                if not np.array_equal(observed_type, expected_owner_type[phase]) or not np.array_equal(
                    observed_index, expected_owner_index[phase]
                ):
                    raise AssertionError("Source plan ownership disagrees with Phase-II selection")
                count = len(phase)
                part = np.arange(part_cursor, part_cursor + count, dtype=np.int64)
                paths = table["shard_path"].take(pa.array(take))
                if any(not value for value in paths.to_pylist()):
                    raise AssertionError("Selected physical cell has a missing shard locator")
                shards.update(str(value) for value in paths.to_pylist())
                writer.write_table(
                    pa.table(
                        {
                            "part_index": part,
                            "embedding_index": phase,
                            "shard_index": table["shard_index"].take(pa.array(take)),
                            "shard_path": paths,
                            "row_group_index": table["row_group_index"].take(pa.array(take)),
                            "row_index_in_row_group": table["row_index_in_row_group"].take(pa.array(take)),
                            "row_index_in_shard": table["row_index_in_shard"].take(pa.array(take)),
                            "owner_type": observed_type,
                            "owner_index": observed_index,
                            "source_embedding_index": original[take],
                        },
                        schema=schema,
                    )
                )
                seen[phase] = True
                part_cursor += count
        finally:
            writer.close()
        os.replace(temporary, output)
        manifest = {
            "schema": "phase2_author_genejepa_worker_plan_v1",
            "created_at_utc": utc_now(),
            "status": "planned_not_started",
            "worker_id": worker_id,
            "device_contract": f"CUDA_VISIBLE_DEVICES={worker_id}; local device cuda:0",
            "distributed": False,
            "plan": {
                **artifact(output),
                "rows": part_cursor,
                "part_index_start": 0,
                "part_index_stop_exclusive": part_cursor,
                "physical_order": "source shard + row group + row",
                "scatter_key": "phase2_cell_index stored as embedding_index",
                "source_shards": len(shards),
            },
            "output_contract": {"shape": [part_cursor, LATENT_DIM], "dtype": "float32"},
        }
        atomic_json(worker_path(worker_id, "manifest.json"), manifest)
        atomic_json(
            worker_path(worker_id, "progress.json"),
            {
                "schema": "phase2_author_genejepa_worker_progress_v1",
                "created_at_utc": utc_now(),
                "updated_at_utc": utc_now(),
                "status": "not_started",
                "embedding_extraction_started": False,
                "worker_id": worker_id,
                "processed_cells": 0,
                "remaining_cells": part_cursor,
                "total_cells": part_cursor,
                "committed_prefix_stop": 0,
            },
        )
        records.append(
            {
                "worker_id": worker_id,
                "rows": part_cursor,
                "source_shards": len(shards),
                "plan_sha256": sha256_file(output),
                "plan_size_bytes": output.stat().st_size,
            }
        )
    if not seen.all() or int(seen.sum()) != len(source_indices):
        raise AssertionError("Worker plan union does not cover all Phase-II cells exactly once")
    return records


def sample_range(start: int, stop: int, pair_id: str, side: str, epoch: int) -> np.ndarray:
    payload = (
        f"tahoe_experiment1_set_v1|seed={SEED}|epoch={epoch}|"
        f"pair_id={pair_id}|side={side}"
    )
    draw_seed = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
    offsets = np.random.Generator(np.random.PCG64(draw_seed)).choice(
        stop - start, size=SET_SIZE, replace=False
    )
    return offsets.astype(np.int64) + start


def write_sampling_audit(conditions: pd.DataFrame) -> dict[str, Any]:
    candidates = conditions.loc[
        conditions["treated_cached_cell_count"].gt(SET_SIZE)
        & conditions["control_cached_cell_count"].gt(SET_SIZE)
    ].copy()
    rows: list[dict[str, Any]] = []
    for split in SPLIT_ORDER:
        chosen = candidates.loc[candidates["split"].eq(split)].sort_values(
            ["selection_sha256", "pair_id"], kind="stable"
        ).head(3)
        if len(chosen) != 3:
            raise AssertionError(f"Not enough {split} pool>256 conditions for sampling audit")
        for row in chosen.itertuples(index=False):
            epochs = range(5) if split == "test" else range(2)
            side_samples: dict[str, list[np.ndarray]] = {"control": [], "treated": []}
            for epoch in epochs:
                for side, start, stop in (
                    ("control", row.control_embedding_start, row.control_embedding_stop_exclusive),
                    ("treated", row.treated_embedding_start, row.treated_embedding_stop_exclusive),
                ):
                    first = sample_range(int(start), int(stop), str(row.pair_id), side, epoch)
                    second = sample_range(int(start), int(stop), str(row.pair_id), side, epoch)
                    side_samples[side].append(first)
                    rows.append(
                        {
                            "split": split,
                            "pair_id": str(row.pair_id),
                            "side": side,
                            "epoch": epoch,
                            "pool_size": int(stop - start),
                            "sample_count": len(first),
                            "sample_unique": int(len(np.unique(first))),
                            "reproducible": bool(np.array_equal(first, second)),
                            "index_sha256": hashlib.sha256(first.astype("<i8").tobytes()).hexdigest(),
                        }
                    )
            for side, samples in side_samples.items():
                if len(samples) > 1 and all(np.array_equal(samples[0], value) for value in samples[1:]):
                    raise AssertionError(f"Dynamic sampling did not redraw {row.pair_id}/{side}")
    checks = {
        "each_sample_256": all(row["sample_count"] == SET_SIZE for row in rows),
        "each_sample_unique": all(row["sample_unique"] == SET_SIZE for row in rows),
        "same_epoch_reproducible": all(row["reproducible"] for row in rows),
        "pool_gt_256": all(row["pool_size"] > SET_SIZE for row in rows),
        "epoch0_vs_epoch1_changes": True,
        "test_repeats_0_to_4_checked": sorted(
            {row["epoch"] for row in rows if row["split"] == "test"}
        ) == [0, 1, 2, 3, 4],
    }
    result = {
        "schema": "phase2_stav2_dynamic_sampling_audit_v1",
        "created_at_utc": utc_now(),
        "status": "pass" if all(checks.values()) else "fail",
        "sampling_contract": (
            "seed + epoch + pair_id + side -> SHA256 first 8 bytes -> PCG64; "
            "choice(pool,256,replace=False)"
        ),
        "seed": SEED,
        "set_size": SET_SIZE,
        "training_epoch_dynamic": True,
        "validation_epoch": 0,
        "final_test_repeat_epochs": [0, 1, 2, 3, 4],
        "checks": checks,
        "records": rows,
    }
    if result["status"] != "pass":
        raise AssertionError(result)
    atomic_json(SAMPLING_AUDIT, result)
    return result


def write_dose_normalization(conditions: pd.DataFrame) -> dict[str, Any]:
    train = conditions.loc[conditions["split"].eq("train"), "dose_uM"].astype(float)
    maximum = float(train.max())
    levels = sorted(float(value) for value in conditions["dose_uM"].unique())
    val_test_max = float(
        conditions.loc[conditions["split"].isin(["val", "test"]), "dose_uM"].astype(float).max()
    )
    if val_test_max > maximum:
        raise AssertionError("Validation/test dose exceeds selected-train maximum")
    result = {
        "schema": "phase2_stav2_dose_normalization_v1",
        "created_at_utc": utc_now(),
        "status": "frozen",
        "formula": "log1p(dose_uM) / log1p(max_train_dose_uM)",
        "fit_scope": "Phase-II selected train conditions only",
        "max_train_dose_uM": maximum,
        "train_dose_levels_uM": sorted(float(value) for value in train.unique()),
        "val_dose_range_uM": [
            float(conditions.loc[conditions["split"].eq("val"), "dose_uM"].min()),
            float(conditions.loc[conditions["split"].eq("val"), "dose_uM"].max()),
        ],
        "test_dose_range_uM": [
            float(conditions.loc[conditions["split"].eq("test"), "dose_uM"].min()),
            float(conditions.loc[conditions["split"].eq("test"), "dose_uM"].max()),
        ],
        "scaled_by_dose_uM": {
            format(value, ".15g"): math.log1p(value) / math.log1p(maximum) for value in levels
        },
        "dmso_no_drug_scaled": 0.0,
        "positive_doses_strictly_positive_and_at_most_one": all(
            0 < math.log1p(value) / math.log1p(maximum) <= 1 for value in levels
        ),
        "validation_or_test_exceeds_train_maximum": False,
    }
    atomic_json(DOSE_NORMALIZATION, result)
    return result


def distribution(frame: pd.DataFrame, column: str) -> dict[str, float]:
    values = frame[column].astype(float).to_numpy()
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def prepare() -> dict[str, Any]:
    if sha256_file(TASK_PATH) != TASK_SHA256:
        raise AssertionError("Frozen Phase-II task SHA-256 changed")
    outputs = [
        CONDITIONS_PATH,
        CONTROL_POOLS_PATH,
        SUBSET_MANIFEST,
        CACHE_PLAN,
        DOSE_NORMALIZATION,
        SAMPLING_AUDIT,
    ]
    outputs.extend(worker_path(worker, suffix) for worker in (0, 1) for suffix in ("plan.parquet", "manifest.json", "progress.json"))
    existing = [relative(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite Phase-II artifacts: {existing}")
    AUTHOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    conditions, controls, vocab, selection_audit = select_conditions()
    dose = write_dose_normalization(conditions)
    sampling = write_sampling_audit(conditions)
    source_index, phase_index, owner_type, owner_index = selected_source_arrays(conditions, controls)
    workers = write_worker_plans(source_index, phase_index, owner_type, owner_index)
    atomic_csv(CONDITIONS_PATH, conditions)
    atomic_csv(CONTROL_POOLS_PATH, controls)

    split_counts = {
        split: int(conditions["split"].eq(split).sum()) for split in SPLIT_ORDER
    }
    split_edges = {
        split: int(conditions.loc[conditions["split"].eq(split), "edge_id"].nunique())
        for split in SPLIT_ORDER
    }
    split_drugs = {
        split: int(conditions.loc[conditions["split"].eq(split), "drug"].nunique())
        for split in SPLIT_ORDER
    }
    train_drugs = set(conditions.loc[conditions["split"].eq("train"), "drug"].astype(str))
    frozen_drugs = set(vocab["drug"].astype(str))
    total = len(source_index)
    treated = int(conditions["treated_cached_cell_count"].sum())
    control = int(controls["cached_cell_count"].sum())
    cache_plan = {
        "schema": "phase2_stav2_cache_plan_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "policy": "treated cap=512 per selected condition + cap=512 per unique used control pool",
        "total_unique_cells": total,
        "treated_cells": treated,
        "control_cells": control,
        "embedding_shape": [total, LATENT_DIM],
        "embedding_dtype": "float32",
        "embedding_payload_GiB": total * LATENT_DIM * 4 / 2**30,
        "scatter_key": "phase2_cell_index stored as embedding_index",
        "stable_locator_columns": [
            "shard_path",
            "row_group_index",
            "row_index_in_row_group",
            "row_index_in_shard",
        ],
        "worker_partition": workers,
        "worker_overlap": 0,
        "worker_union_cells": sum(int(row["rows"]) for row in workers),
        "duplicate_physical_cells": int(len(source_index) - len(np.unique(source_index))),
        "missing_locators": 0,
        "condition_index": artifact(CONDITIONS_PATH),
        "control_pool_index": artifact(CONTROL_POOLS_PATH),
        "source_worker_plans": [artifact(path) for path in SOURCE_PLANS],
    }
    if cache_plan["worker_union_cells"] != total or cache_plan["duplicate_physical_cells"] != 0:
        raise AssertionError(cache_plan)
    atomic_json(CACHE_PLAN, cache_plan)

    manifest = {
        "schema": "phase2_stav2_subset_manifest_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "task": {**artifact(TASK_PATH), "expected_sha256": TASK_SHA256, "phase": "Task 2 / Phase II"},
        "selection": {
            "seed": SEED,
            "algorithm": "coverage-first balanced cell-line/dose strata with seed42 SHA256 tie-break v1",
            "source_split_reused": True,
            "new_split_generated": False,
            "target_quotas": SPLIT_QUOTAS,
            "phase1_frozen_cell_lines": selection_audit["frozen_cell_lines"],
            "selected_cell_lines": sorted(conditions["cell_line_id"].astype(str).unique().tolist()),
            "selected_cell_lines_equal_phase1_frozen_five": sorted(conditions["cell_line_id"].astype(str).unique().tolist()) == selection_audit["frozen_cell_lines"],
            "eligible_within_five_cell_lines": selection_audit["eligible_within_five_cell_lines"],
            "excluded_unseen_val_test": selection_audit["excluded_unseen_val_test"],
        },
        "counts": {
            "selected_conditions_total": len(conditions),
            "conditions_by_split": split_counts,
            "edges_total": int(conditions["edge_id"].nunique()),
            "edges_by_split": split_edges,
            "selected_drugs_total": int(conditions["drug"].nunique()),
            "drugs_by_split": split_drugs,
            "frozen_vocabulary_drugs": len(vocab),
            "selected_train_drugs": len(train_drugs),
            "unrepresented_frozen_drugs": sorted(frozen_drugs - train_drugs),
            "val_unseen_in_selected_train_drugs": sorted(set(conditions.loc[conditions["split"].eq("val"), "drug"].astype(str)) - train_drugs),
            "test_unseen_in_selected_train_drugs": sorted(set(conditions.loc[conditions["split"].eq("test"), "drug"].astype(str)) - train_drugs),
            "unique_control_pools": len(controls),
            "total_unique_physical_cells": total,
            "treated_physical_cells": treated,
            "control_physical_cells": control,
        },
        "dose_distribution": {
            split: {
                format(float(dose_value), ".15g"): int(count)
                for dose_value, count in conditions.loc[conditions["split"].eq(split)].groupby("dose_uM").size().items()
            }
            for split in SPLIT_ORDER
        },
        "cell_line_distribution": {
            split: {
                str(cell_line): int(count)
                for cell_line, count in conditions.loc[conditions["split"].eq(split)].groupby("cell_line_id").size().items()
            }
            for split in SPLIT_ORDER
        },
        "pool_contract": {
            "treated_cap": POOL_CAP,
            "control_cap": POOL_CAP,
            "training_set_size": SET_SIZE,
            "treated_cached_cells": distribution(conditions, "treated_cached_cell_count"),
            "treated_available_cells": distribution(conditions, "treated_available_cell_count"),
            "control_cached_cells": distribution(controls, "cached_cell_count"),
            "control_available_cells": distribution(controls, "available_cell_count"),
            "unique_control_pool_reuse": True,
        },
        "hard_checks": {
            "all_treated_at_least_256": bool(conditions["treated_cached_cell_count"].ge(SET_SIZE).all()),
            "all_controls_at_least_256": bool(conditions["control_cached_cell_count"].ge(SET_SIZE).all()),
            "edge_split_leakage": int(conditions.groupby("edge_id")["split"].nunique().gt(1).sum()),
            "val_unseen_in_train_drugs": 0,
            "test_unseen_in_train_drugs": 0,
            "duplicate_physical_cells": 0,
            "missing_locators": 0,
            "worker_overlap": 0,
            "worker_union_cells": total,
        },
        "outputs": {
            "conditions": artifact(CONDITIONS_PATH),
            "control_pools": artifact(CONTROL_POOLS_PATH),
            "cache_plan": artifact(CACHE_PLAN),
            "dose_normalization": artifact(DOSE_NORMALIZATION),
            "dynamic_sampling_audit": artifact(SAMPLING_AUDIT),
        },
        "phase2_author_extraction_started": False,
        "phase3_started": False,
    }
    if not all(
        [
            manifest["selection"]["selected_cell_lines_equal_phase1_frozen_five"],
            not manifest["counts"]["val_unseen_in_selected_train_drugs"],
            not manifest["counts"]["test_unseen_in_selected_train_drugs"],
            manifest["hard_checks"]["edge_split_leakage"] == 0,
            sampling["status"] == "pass",
            dose["status"] == "frozen",
        ]
    ):
        raise AssertionError(manifest)
    atomic_json(SUBSET_MANIFEST, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def audit() -> dict[str, Any]:
    manifest = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))
    cache = json.loads(CACHE_PLAN.read_text(encoding="utf-8"))
    sampling = json.loads(SAMPLING_AUDIT.read_text(encoding="utf-8"))
    dose = json.loads(DOSE_NORMALIZATION.read_text(encoding="utf-8"))
    conditions = pd.read_csv(CONDITIONS_PATH, encoding="utf-8-sig", keep_default_na=False)
    controls = pd.read_csv(CONTROL_POOLS_PATH, encoding="utf-8-sig", keep_default_na=False)
    checks = {
        "manifest_pass": manifest.get("status") == "pass",
        "cache_plan_pass": cache.get("status") == "pass",
        "sampling_pass": sampling.get("status") == "pass",
        "dose_frozen": dose.get("status") == "frozen",
        "conditions_5000": len(conditions) == sum(SPLIT_QUOTAS.values()),
        "split_counts": conditions.groupby("split").size().to_dict() == SPLIT_QUOTAS,
        "five_cell_lines": conditions["cell_line_id"].nunique() == 5,
        "all_pool_ranges_at_least_256": bool(
            conditions["treated_cached_cell_count"].ge(SET_SIZE).all()
            and conditions["control_cached_cell_count"].ge(SET_SIZE).all()
        ),
        "control_pool_unique": not controls["control_pool_id"].duplicated().any(),
        "edge_split_leakage_zero": conditions.groupby("edge_id")["split"].nunique().max() == 1,
        "worker_union": cache["worker_union_cells"] == cache["total_unique_cells"],
    }
    result = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": {key: bool(value) for key, value in checks.items()},
        "conditions": len(conditions),
        "controls": len(controls),
        "cells": int(cache["total_unique_cells"]),
    }
    if result["status"] != "pass":
        raise AssertionError(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare()
    else:
        audit()


if __name__ == "__main__":
    main()
