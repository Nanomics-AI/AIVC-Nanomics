#!/usr/bin/env python3
"""Build the train-only Experiment 1 B1 drug-dose mean-delta reference table."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audit_tahoe_experiment1_b1_coverage import (
    AUDIT_JSON_PATH as COVERAGE_AUDIT_PATH,
    CONDITION_COVERAGE_PATH,
    atomic_write_csv,
    atomic_write_text,
    utc_now,
)
from tahoe_experiment1_latent_data import (
    FORMAL_CONDITION_INDEX,
    FORMAL_CONTROL_POOL_INDEX,
    FORMAL_EMBEDDING_MANIFEST,
    FORMAL_EMBEDDINGS,
    LATENT_DIM,
    PROJECT_ROOT,
    RESULTS,
    TahoeExperiment1LatentSetDataset,
    sha256_file,
)


SCRIPT_PATH = Path(__file__).resolve()
DATASET_PATH = PROJECT_ROOT / "perturbation_scripts" / "tahoe_experiment1_latent_data.py"
CONDITION_SHIFT_PATH = RESULTS / "tahoe_experiment1_b1_train_condition_shifts.npy"
CONDITION_METADATA_PATH = RESULTS / "tahoe_experiment1_b1_train_condition_shifts_metadata.csv"
MEAN_DELTA_PATH = RESULTS / "tahoe_experiment1_b1_mean_delta.npy"
MEAN_DELTA_METADATA_PATH = RESULTS / "tahoe_experiment1_b1_mean_delta_metadata.csv"
AUDIT_PATH = RESULTS / "tahoe_experiment1_b1_mean_delta_build_audit.json"
HANDOFF_PATH = RESULTS / "tahoe_experiment1_b1_mean_delta_build_handoff.md"
EXPECTED_TRAIN_CONDITIONS = 45_652
EXPECTED_GROUPS = 1_132
EXPECTED_TEST_CONDITIONS = 5_684


def describe_norms(values: np.ndarray) -> dict[str, float | int]:
    norms = np.linalg.norm(np.asarray(values, dtype=np.float64), axis=1)
    if not norms.size or not np.isfinite(norms).all():
        raise AssertionError("Mean-delta norms are empty or non-finite")
    return {
        "count": int(norms.size),
        "min": float(norms.min()),
        "mean": float(norms.mean()),
        "median": float(np.quantile(norms, 0.50)),
        "p05": float(np.quantile(norms, 0.05)),
        "p95": float(np.quantile(norms, 0.95)),
        "max": float(norms.max()),
    }


def range_centroid(
    embeddings: np.memmap,
    start: int,
    stop: int,
    chunk_cells: int,
) -> np.ndarray:
    count = stop - start
    if count <= 0:
        raise AssertionError("Cannot compute a centroid from an empty embedding range")
    total = np.zeros(LATENT_DIM, dtype=np.float64)
    for chunk_start in range(start, stop, chunk_cells):
        chunk = np.asarray(embeddings[chunk_start : min(chunk_start + chunk_cells, stop)])
        if chunk.dtype != np.float32 or chunk.ndim != 2 or chunk.shape[1] != LATENT_DIM:
            raise AssertionError("Formal cache chunk shape/dtype changed")
        if not np.isfinite(chunk).all():
            raise AssertionError("Non-finite value found in a centroid input range")
        total += chunk.sum(axis=0, dtype=np.float64)
    return total / count


def write_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    output = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=values.shape,
    )
    output[:] = values
    output.flush()
    del output
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_coverage_audit() -> dict[str, Any]:
    coverage = json.loads(COVERAGE_AUDIT_PATH.read_text(encoding="utf-8"))
    counts = coverage.get("counts", {})
    if (
        coverage.get("status") != "pass"
        or counts.get("train_conditions") != EXPECTED_TRAIN_CONDITIONS
        or counts.get("test_conditions") != EXPECTED_TEST_CONDITIONS
        or counts.get("train_unique_drug_dose") != EXPECTED_GROUPS
        or counts.get("test_condition_exact_coverage_rate") != 1.0
        or counts.get("missing_unique_drug_dose") != 0
    ):
        raise AssertionError("B1 coverage audit is not compatible with this build")
    return coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-cells", type=int, default=8192)
    parser.add_argument("--progress-every", type=int, default=2500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_cells < 1 or args.progress_every < 1:
        raise ValueError("chunk-cells and progress-every must be positive")

    started = time.perf_counter()
    coverage_audit = validate_coverage_audit()
    dataset = TahoeExperiment1LatentSetDataset(split=None, seed=42, epoch=0)
    if dataset.embedding_transforms or not isinstance(dataset.embeddings, np.memmap):
        raise AssertionError("B1 requires the raw formal float32 mmap without transforms")

    all_conditions = dataset.all_conditions
    train = (
        all_conditions.loc[all_conditions["split"].eq("train")]
        .sort_values("cache_condition_index", kind="stable")
        .reset_index(drop=True)
    )
    test = all_conditions.loc[all_conditions["split"].eq("test")].copy()
    if len(train) != EXPECTED_TRAIN_CONDITIONS or len(test) != EXPECTED_TEST_CONDITIONS:
        raise AssertionError("Formal train/test condition counts changed")
    if not train["pair_id"].is_unique:
        raise AssertionError("Train condition IDs are not unique")

    train_control_ids = set(train["control_pool_id"])
    controls = (
        dataset.controls.loc[dataset.controls["control_pool_id"].isin(train_control_ids)]
        .sort_values("control_pool_index", kind="stable")
        .reset_index(drop=True)
    )
    if set(controls["control_pool_id"]) != train_control_ids:
        raise AssertionError("A train condition has no control pool")

    control_centroids: dict[str, np.ndarray] = {}
    control_cells_used = 0
    for row in controls.itertuples(index=False):
        start = int(row.embedding_start)
        stop = int(row.embedding_stop_exclusive)
        control_centroids[str(row.control_pool_id)] = range_centroid(
            dataset.embeddings, start, stop, args.chunk_cells
        )
        control_cells_used += stop - start
    print(
        f"control pools={len(control_centroids):,} unique DMSO cells={control_cells_used:,}",
        flush=True,
    )

    condition_temporary = CONDITION_SHIFT_PATH.with_name(CONDITION_SHIFT_PATH.name + ".tmp")
    condition_shifts = np.lib.format.open_memmap(
        condition_temporary,
        mode="w+",
        dtype=np.float32,
        shape=(len(train), LATENT_DIM),
    )
    treated_cells_used = 0
    for condition_shift_row, row in enumerate(train.itertuples(index=False)):
        start = int(row.treated_embedding_start)
        stop = int(row.treated_embedding_stop_exclusive)
        treated_centroid = range_centroid(dataset.embeddings, start, stop, args.chunk_cells)
        delta = treated_centroid - control_centroids[str(row.control_pool_id)]
        if not np.isfinite(delta).all():
            raise AssertionError(f"Non-finite condition shift: {row.pair_id}")
        condition_shifts[condition_shift_row] = delta.astype(np.float32)
        treated_cells_used += stop - start
        if (condition_shift_row + 1) % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"train conditions={condition_shift_row + 1:,}/{len(train):,} "
                f"rate={(condition_shift_row + 1) / elapsed:.2f}/s",
                flush=True,
            )
    condition_shifts.flush()
    del condition_shifts
    with condition_temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    condition_temporary.replace(CONDITION_SHIFT_PATH)

    condition_metadata = pd.DataFrame(
        {
            "condition_shift_row": np.arange(len(train), dtype=np.int64),
            "condition_id": train["pair_id"].astype(str),
            "cell_line_id": train["cell_line_id"].astype(str),
            "drug": train["drug"].astype(str),
            "dose_uM": train["dose_uM"].astype(np.float64),
            "control_pool_id": train["control_pool_id"].astype(str),
            "treated_cached_cell_count": train["treated_cached_cell_count"].astype(np.int64),
            "control_cached_cell_count": train["control_cached_cell_count"].astype(np.int64),
        }
    )
    atomic_write_csv(CONDITION_METADATA_PATH, condition_metadata)

    condition_shifts = np.load(CONDITION_SHIFT_PATH, mmap_mode="r")
    group_items = list(train.groupby(["drug", "dose_uM"], sort=True).indices.items())
    if len(group_items) != EXPECTED_GROUPS:
        raise AssertionError("Unexpected train (drug, dose_uM) group count")
    mean_delta = np.empty((len(group_items), LATENT_DIM), dtype=np.float32)
    mean_metadata_rows: list[dict[str, Any]] = []
    for mean_delta_row, ((drug, dose_uM), indices) in enumerate(group_items):
        condition_indices = np.asarray(indices, dtype=np.int64)
        group_values = np.asarray(condition_shifts[condition_indices], dtype=np.float32)
        group_mean = group_values.mean(axis=0, dtype=np.float64)
        mean_delta[mean_delta_row] = group_mean.astype(np.float32)
        mean_metadata_rows.append(
            {
                "mean_delta_row": mean_delta_row,
                "drug": str(drug),
                "dose_uM": float(dose_uM),
                "train_condition_count": int(len(condition_indices)),
            }
        )
    mean_metadata = pd.DataFrame(mean_metadata_rows)
    write_npy(MEAN_DELTA_PATH, mean_delta)
    atomic_write_csv(MEAN_DELTA_METADATA_PATH, mean_metadata)

    if condition_shifts.shape != (EXPECTED_TRAIN_CONDITIONS, LATENT_DIM):
        raise AssertionError("Condition-shift shape changed")
    if condition_shifts.dtype != np.float32 or not np.isfinite(condition_shifts).all():
        raise AssertionError("Condition shifts are not finite float32")
    if mean_delta.shape != (EXPECTED_GROUPS, LATENT_DIM):
        raise AssertionError("Mean-delta shape changed")
    if mean_delta.dtype != np.float32 or not np.isfinite(mean_delta).all():
        raise AssertionError("Mean delta is not finite float32")
    if not ((condition_shifts < 0).any() and (condition_shifts > 0).any()):
        raise AssertionError("Signed condition shifts were not preserved")
    if not ((mean_delta < 0).any() and (mean_delta > 0).any()):
        raise AssertionError("Signed mean deltas were not preserved")
    if int(mean_metadata["train_condition_count"].sum()) != EXPECTED_TRAIN_CONDITIONS:
        raise AssertionError("Train condition weights do not sum to the train condition count")

    test_mapping = test[["pair_id", "drug", "dose_uM"]].merge(
        mean_metadata[["drug", "dose_uM", "mean_delta_row"]],
        on=["drug", "dose_uM"],
        how="left",
        validate="many_to_one",
    )
    if len(test_mapping) != EXPECTED_TEST_CONDITIONS or test_mapping["mean_delta_row"].isna().any():
        raise AssertionError("A test condition does not map uniquely to a train mean-delta row")

    norm_summary = describe_norms(mean_delta)
    elapsed = time.perf_counter() - started
    extraction_manifest = json.loads(FORMAL_EMBEDDING_MANIFEST.read_text(encoding="utf-8"))
    audit = {
        "schema": "tahoe_experiment1_b1_mean_delta_build_audit_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "scope": {
            "fit_split": "train only",
            "test_prediction_generated": False,
            "energy_distance_computed": False,
            "b2_or_state_or_st_run": False,
        },
        "fitting_rule": {
            "condition_shift": "Delta_condition = centroid(all cached treated cells) - centroid(all cached matched DMSO cells)",
            "centroid_sampling": "all cached cells; no S=256 subsampling",
            "mean_delta_group_key": ["drug", "dose_uM"],
            "mean_delta_aggregation": "equal arithmetic mean of train condition shifts",
            "condition_weight": 1.0,
            "cell_count_weighting": False,
            "latent_preprocessing": [],
            "dmso_residualization": False,
            "fallbacks": [],
        },
        "inputs": {
            "formal_embedding": {
                "path": FORMAL_EMBEDDINGS.relative_to(PROJECT_ROOT).as_posix(),
                "sha256_from_pass_manifest": extraction_manifest["output"]["sha256"],
                "sha256_recomputed": False,
                "mmap": True,
            },
            "formal_embedding_manifest": {
                "path": FORMAL_EMBEDDING_MANIFEST.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(FORMAL_EMBEDDING_MANIFEST),
            },
            "condition_index": {
                "path": FORMAL_CONDITION_INDEX.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(FORMAL_CONDITION_INDEX),
            },
            "control_pool_index": {
                "path": FORMAL_CONTROL_POOL_INDEX.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(FORMAL_CONTROL_POOL_INDEX),
            },
            "coverage_audit": {
                "path": COVERAGE_AUDIT_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(COVERAGE_AUDIT_PATH),
            },
            "condition_coverage": {
                "path": CONDITION_COVERAGE_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(CONDITION_COVERAGE_PATH),
            },
            "dataset_code": {
                "path": DATASET_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(DATASET_PATH),
            },
        },
        "counts": {
            "train_conditions": len(train),
            "train_unique_drug_dose": len(mean_metadata),
            "train_control_pools_reused": len(control_centroids),
            "unique_train_treated_cells_used": treated_cells_used,
            "unique_matched_dmso_cells_used": control_cells_used,
            "test_conditions_uniquely_mapped": len(test_mapping),
            "test_coverage_rate": float(test_mapping["mean_delta_row"].notna().mean()),
        },
        "condition_shift": {
            "shape": list(condition_shifts.shape),
            "dtype": str(condition_shifts.dtype),
            "finite": True,
            "min": float(condition_shifts.min()),
            "max": float(condition_shifts.max()),
            "negative_ratio": float(np.mean(condition_shifts < 0)),
        },
        "mean_delta": {
            "shape": list(mean_delta.shape),
            "dtype": str(mean_delta.dtype),
            "finite": True,
            "min": float(mean_delta.min()),
            "max": float(mean_delta.max()),
            "negative_ratio": float(np.mean(mean_delta < 0)),
            "norm": norm_summary,
        },
        "checks": {
            "train_conditions_only": "pass",
            "condition_count_45652": "pass",
            "drug_dose_groups_1132": "pass",
            "float32_and_finite": "pass",
            "signed_values_preserved": "pass",
            "no_latent_preprocessing": "pass",
            "no_dmso_residualization": "pass",
            "macro_average_equal_condition_weights": "pass",
            "val_test_treated_cells_excluded": "pass",
            "test_exact_mapping_100_percent": "pass",
        },
        "outputs": {
            "condition_shift": {
                "path": CONDITION_SHIFT_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(CONDITION_SHIFT_PATH),
                "shape": list(condition_shifts.shape),
                "dtype": str(condition_shifts.dtype),
            },
            "condition_metadata": {
                "path": CONDITION_METADATA_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(CONDITION_METADATA_PATH),
                "rows": len(condition_metadata),
                "encoding": "utf-8-sig",
            },
            "mean_delta": {
                "path": MEAN_DELTA_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(MEAN_DELTA_PATH),
                "shape": list(mean_delta.shape),
                "dtype": str(mean_delta.dtype),
            },
            "mean_delta_metadata": {
                "path": MEAN_DELTA_METADATA_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(MEAN_DELTA_METADATA_PATH),
                "rows": len(mean_metadata),
                "encoding": "utf-8-sig",
            },
            "audit_json": AUDIT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "handoff_md": HANDOFF_PATH.relative_to(PROJECT_ROOT).as_posix(),
        },
        "runtime": {"elapsed_seconds": elapsed, "chunk_cells": args.chunk_cells},
        "implementation": {
            "script": SCRIPT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "issues": [],
    }
    atomic_write_text(AUDIT_PATH, json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    handoff = f"""# Tahoe Experiment 1 B1 train-only mean-delta build

- Status: **PASS**
- Condition shifts: **{list(condition_shifts.shape)} float32**, all finite and signed.
- Exact train drug-dose means: **{list(mean_delta.shape)} float32**, all finite and signed.
- Test exact mapping: **{len(test_mapping):,}/{len(test):,} (100%)**.
- Energy/test prediction/B2/STATE/ST: **not run**.

Each train condition uses all cached treated cells and all cells in its matched DMSO
pool. The final exact `(drug, dose_uM)` mean gives every condition one equal vote;
there is no cell-count weighting, latent preprocessing, DMSO residualization, or
fallback.

## Mean-delta norm sanity check

| min | mean | median | p05 | p95 | max |
|---:|---:|---:|---:|---:|---:|
| {norm_summary['min']:.8g} | {norm_summary['mean']:.8g} | {norm_summary['median']:.8g} | {norm_summary['p05']:.8g} | {norm_summary['p95']:.8g} | {norm_summary['max']:.8g} |

## Outputs

- `{CONDITION_SHIFT_PATH.relative_to(PROJECT_ROOT).as_posix()}`
- `{CONDITION_METADATA_PATH.relative_to(PROJECT_ROOT).as_posix()}`
- `{MEAN_DELTA_PATH.relative_to(PROJECT_ROOT).as_posix()}`
- `{MEAN_DELTA_METADATA_PATH.relative_to(PROJECT_ROOT).as_posix()}`
- `{AUDIT_PATH.relative_to(PROJECT_ROOT).as_posix()}`
- `{HANDOFF_PATH.relative_to(PROJECT_ROOT).as_posix()}`
"""
    atomic_write_text(HANDOFF_PATH, handoff)
    print(
        json.dumps(
            {
                "status": "pass",
                "condition_shift_shape": list(condition_shifts.shape),
                "mean_delta_shape": list(mean_delta.shape),
                "mean_delta_norm": norm_summary,
                "test_coverage_rate": 1.0,
                "audit": AUDIT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
