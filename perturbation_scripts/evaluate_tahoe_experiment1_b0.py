#!/usr/bin/env python3
"""Evaluate the frozen Experiment 1 B0 identity baseline on the test split."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from geomloss import SamplesLoss

from tahoe_experiment1_latent_data import (
    PROJECT_ROOT,
    RESULTS,
    SET_SIZE,
    TahoeExperiment1LatentSetDataset,
    make_dataloader,
    sha256_file,
)


PROTOCOL_PATH = RESULTS / "tahoe_experiment1_evaluation_sampling_protocol.json"
DATASET_PATH = PROJECT_ROOT / "perturbation_scripts" / "tahoe_experiment1_latent_data.py"
STATE_MODEL_PATH = PROJECT_ROOT.parent / "state-main" / "src" / "state" / "tx" / "models" / "state_transition.py"
SCRIPT_PATH = Path(__file__).resolve()
RAW_PATH = RESULTS / "tahoe_experiment1_b0_identity_condition_repeat.csv"
EDGE_PATH = RESULTS / "tahoe_experiment1_b0_identity_edge.csv"
RESULT_PATH = RESULTS / "tahoe_experiment1_b0_identity_result.json"
HANDOFF_PATH = RESULTS / "tahoe_experiment1_b0_identity_handoff.md"
EXPECTED_TEST_CONDITIONS = 5_684
EXPECTED_TEST_EDGES = 1_717
EXPECTED_REPEAT_COUNT = 5
EXPECTED_RAW_ROWS = EXPECTED_TEST_CONDITIONS * EXPECTED_REPEAT_COUNT
ENERGY_BLUR = 0.05


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding=encoding, newline="\n")
    temporary.replace(path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
        float_format="%.10g",
    )
    temporary.replace(path)


def describe(values: pd.Series | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise AssertionError("Descriptive input is empty or non-finite")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.quantile(array, 0.50)),
        "std": float(array.std(ddof=0)),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def grouped_describe(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(column, sort=True):
        row: dict[str, Any] = {
            column: float(value) if column == "dose_uM" else str(value)
        }
        row.update(describe(group["energy_distance"]))
        rows.append(row)
    return rows


def load_protocol() -> tuple[dict[str, Any], str]:
    protocol_sha = sha256_file(PROTOCOL_PATH)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    expected = {
        "status": "frozen",
        "S": SET_SIZE,
        "evaluation_repeat_count": EXPECTED_REPEAT_COUNT,
        "evaluation_base_seed": 42,
        "evaluation_repeat_epochs": [0, 1, 2, 3, 4],
        "shuffle": False,
        "drop_last": False,
        "all-models-share-identical-evaluation-indices": True,
        "experiment0_sampling_path_used": False,
        "experiment0_BASE_SEEDS_used": False,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise AssertionError(f"Frozen evaluation protocol mismatch: {key}")
    sampling = protocol.get("sampling_method", {})
    if (
        sampling.get("within_set_replacement") is not False
        or sampling.get("cross_repeat_overlap_allowed") is not True
        or protocol.get("dataset_sampling_rule", {}).get("version")
        != "tahoe_experiment1_set_v1"
    ):
        raise AssertionError("Frozen sampling method changed")
    dataset_sha = sha256_file(DATASET_PATH)
    if protocol.get("tahoe_experiment1_latent_data.py_sha256") != dataset_sha:
        raise AssertionError("Dataset code no longer matches the frozen protocol")
    return protocol, protocol_sha


def assert_without_replacement(indices: torch.Tensor, side: str) -> None:
    ordered = torch.sort(indices, dim=1).values
    if torch.any(ordered[:, 1:] == ordered[:, :-1]):
        raise AssertionError(f"Within-set replacement detected on {side}")


def evaluate_raw(
    dataset: TahoeExperiment1LatentSetDataset,
    repeat_epochs: list[int],
    base_seed: int,
    batch_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metric = SamplesLoss(loss="energy", blur=ENERGY_BLUR)
    metadata = dataset.conditions.set_index("pair_id", verify_integrity=True)
    records: list[dict[str, Any]] = []
    batch_count = 0
    started = time.perf_counter()

    with torch.inference_mode():
        for repeat_epoch in repeat_epochs:
            dataset.set_epoch(repeat_epoch)
            loader = make_dataloader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                seed=base_seed,
                drop_last=False,
            )
            if loader.drop_last or type(loader.sampler).__name__ != "SequentialSampler":
                raise AssertionError("Evaluation DataLoader is shuffled or drop_last=True")
            observed_condition_ids: list[str] = []
            for batch in loader:
                condition_ids = list(batch["condition_id"])
                observed_condition_ids.extend(condition_ids)
                if set(batch["split"]) != {"test"}:
                    raise AssertionError("Non-test condition entered B0 evaluation")
                assert_without_replacement(batch["source_embedding_index"], "control")
                assert_without_replacement(batch["target_embedding_index"], "treated")

                control = batch["ctrl_cell_emb"].to(device, non_blocking=True)
                treated = batch["pert_cell_emb"].to(device, non_blocking=True)
                if control.dtype != torch.float32 or treated.dtype != torch.float32:
                    raise AssertionError("B0 inputs must remain float32")
                if control.shape[1:] != (SET_SIZE, 768) or treated.shape != control.shape:
                    raise AssertionError("Unexpected B0 tensor shape")
                values = metric(control, treated)
                if values.shape != (len(condition_ids),) or not torch.isfinite(values).all():
                    raise AssertionError("Energy distance output is not finite per condition")

                for condition_id, value in zip(
                    condition_ids, values.detach().cpu().numpy(), strict=True
                ):
                    row = metadata.loc[condition_id]
                    records.append(
                        {
                            "condition_id": condition_id,
                            "edge_id": str(row["edge_id"]),
                            "cell_line_id": str(row["cell_line_id"]),
                            "drug": str(row["drug"]),
                            "dose_uM": float(row["dose_uM"]),
                            "plate": str(row["plate"]),
                            "repeat_epoch": int(repeat_epoch),
                            "energy_distance": float(value),
                            "_cache_condition_index": int(row["cache_condition_index"]),
                        }
                    )
                batch_count += 1

            expected_order = dataset.conditions["pair_id"].astype(str).tolist()
            if observed_condition_ids != expected_order:
                raise AssertionError(f"repeat_epoch={repeat_epoch} was shuffled or dropped")
            print(
                f"repeat_epoch={repeat_epoch} conditions={len(observed_condition_ids)}/{len(dataset)}",
                flush=True,
            )

    raw = pd.DataFrame.from_records(records).sort_values(
        ["_cache_condition_index", "repeat_epoch"], kind="stable"
    )
    elapsed = time.perf_counter() - started
    return raw, {
        "device": str(device),
        "batch_size": batch_size,
        "batches": batch_count,
        "elapsed_seconds": elapsed,
        "condition_repeat_rows_per_second": len(raw) / elapsed,
        "gpu_peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else None
        ),
        "gpu_peak_reserved_gib": (
            torch.cuda.max_memory_reserved(device) / 2**30
            if device.type == "cuda"
            else None
        ),
    }


def aggregate(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    plate_keys = [
        "condition_id",
        "edge_id",
        "cell_line_id",
        "drug",
        "dose_uM",
        "plate",
    ]
    plate_condition = (
        raw.groupby(plate_keys, as_index=False, sort=False)["energy_distance"].mean()
    )

    dose_keys = ["edge_id", "cell_line_id", "drug", "dose_uM"]
    replicate_frames: list[pd.DataFrame] = []
    paired_groups = 0
    paired_conditions = 0
    for key, group in plate_condition.groupby(dose_keys, sort=False):
        plates = set(group["plate"])
        paired = group["plate"].isin(["plate6", "plate14"])
        if float(key[-1]) == 5.0 and {"plate6", "plate14"} <= plates:
            pair = group.loc[paired]
            if len(pair) != 2:
                raise AssertionError(f"Ambiguous plate6/plate14 replicate group: {key}")
            replicate_frames.append(
                pd.DataFrame(
                    [
                        {
                            **dict(zip(dose_keys, key, strict=True)),
                            "biological_replicate_id": "plate6|plate14",
                            "energy_distance": float(pair["energy_distance"].mean()),
                            "plate_condition_count": 2,
                        }
                    ]
                )
            )
            paired_groups += 1
            paired_conditions += 2
            group = group.loc[~paired]
        if not group.empty:
            remainder = group[dose_keys + ["condition_id", "energy_distance"]].copy()
            remainder = remainder.rename(columns={"condition_id": "biological_replicate_id"})
            remainder["plate_condition_count"] = 1
            replicate_frames.append(remainder)

    biological_replicate = pd.concat(replicate_frames, ignore_index=True)
    dose = biological_replicate.groupby(dose_keys, as_index=False, sort=False).agg(
        energy_distance=("energy_distance", "mean"),
        biological_replicate_count=("biological_replicate_id", "size"),
        plate_condition_count=("plate_condition_count", "sum"),
    )
    edge_keys = ["edge_id", "cell_line_id", "drug"]
    edge = dose.groupby(edge_keys, as_index=False, sort=False).agg(
        energy_distance=("energy_distance", "mean"),
        dose_count=("dose_uM", "size"),
        biological_replicate_count=("biological_replicate_count", "sum"),
        plate_condition_count=("plate_condition_count", "sum"),
    )
    edge = edge.sort_values("edge_id", kind="stable").reset_index(drop=True)
    details = {
        "plate_condition": plate_condition,
        "biological_replicate": biological_replicate,
        "dose": dose,
    }
    details["replicate_audit"] = pd.DataFrame(
        [
            {
                "plate6_plate14_groups": paired_groups,
                "plate6_plate14_conditions": paired_conditions,
            }
        ]
    )
    return edge, details


def validate_results(
    dataset: TahoeExperiment1LatentSetDataset,
    protocol: dict[str, Any],
    protocol_sha_before: str,
    raw: pd.DataFrame,
    edge: pd.DataFrame,
    details: dict[str, pd.DataFrame],
) -> dict[str, str]:
    counts = raw.groupby("condition_id")["repeat_epoch"].agg(["size", "nunique"])
    expected_epochs = set(protocol["evaluation_repeat_epochs"])
    observed_epochs = raw.groupby("condition_id")["repeat_epoch"].agg(set)
    test_condition_ids = set(dataset.conditions["pair_id"].astype(str))
    test_edge_ids = set(dataset.conditions["edge_id"].astype(str))
    if len(dataset) != EXPECTED_TEST_CONDITIONS or test_condition_ids != set(raw["condition_id"]):
        raise AssertionError("Not all formal test conditions were evaluated")
    if len(raw) != EXPECTED_RAW_ROWS:
        raise AssertionError(f"Expected {EXPECTED_RAW_ROWS} raw rows, found {len(raw)}")
    if not ((counts["size"] == EXPECTED_REPEAT_COUNT) & (counts["nunique"] == EXPECTED_REPEAT_COUNT)).all():
        raise AssertionError("A test condition does not have exactly five unique repeats")
    if not observed_epochs.map(lambda value: value == expected_epochs).all():
        raise AssertionError("A test condition has the wrong repeat epochs")
    if raw.duplicated(["condition_id", "repeat_epoch"]).any():
        raise AssertionError("Duplicate condition/repeat result")
    if not np.isfinite(raw["energy_distance"].to_numpy(np.float64)).all():
        raise AssertionError("Non-finite Energy distance found")
    if len(edge) != EXPECTED_TEST_EDGES or set(edge["edge_id"]) != test_edge_ids:
        raise AssertionError("Not all formal test edges reached final aggregation")
    if dataset.embedding_transforms:
        raise AssertionError("Unexpected latent preprocessing configured")
    if any(dataset.edge_overlap.values()):
        raise AssertionError("Frozen split has edge leakage")
    if len(details["plate_condition"]) != EXPECTED_TEST_CONDITIONS:
        raise AssertionError("Plate-condition aggregation changed condition grain")
    if sha256_file(PROTOCOL_PATH) != protocol_sha_before:
        raise AssertionError("Frozen evaluation protocol changed during B0")
    if sha256_file(DATASET_PATH) != protocol["tahoe_experiment1_latent_data.py_sha256"]:
        raise AssertionError("Frozen Dataset changed during B0")
    return {
        "all_5684_test_conditions_evaluated": "pass",
        "exactly_5_repeats_per_condition": "pass",
        "shuffle_false": "pass",
        "drop_last_false": "pass",
        "train_val_conditions_excluded": "pass",
        "all_1717_test_edges_aggregated": "pass",
        "energy_distance_finite": "pass",
        "latent_preprocessing_reapplied": "pass: none",
        "frozen_evaluation_protocol_unchanged": "pass",
        "parameters_trained": "pass: none",
        "cell_count_weighting": "pass: none",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Reuse the existing raw condition/repeat CSV without recomputing Energy distance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not args.aggregate_only:
        torch.cuda.reset_peak_memory_stats(device)

    protocol, protocol_sha = load_protocol()
    dataset = TahoeExperiment1LatentSetDataset(
        split="test",
        seed=int(protocol["evaluation_base_seed"]),
        epoch=int(protocol["evaluation_repeat_epochs"][0]),
    )
    previous_result: dict[str, Any] | None = None
    previous_edge: pd.DataFrame | None = None
    raw_sha_before: str | None = None
    if args.aggregate_only:
        if not RAW_PATH.exists() or not EDGE_PATH.exists() or not RESULT_PATH.exists():
            raise FileNotFoundError("Aggregate-only mode requires the existing B0 outputs")
        raw_sha_before = sha256_file(RAW_PATH)
        previous_result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        previous_edge = pd.read_csv(
            EDGE_PATH,
            usecols=["edge_id", "energy_distance"],
            keep_default_na=False,
            encoding="utf-8-sig",
        )
        raw = pd.read_csv(RAW_PATH, keep_default_na=False, encoding="utf-8-sig")
        required_raw_columns = {
            "condition_id",
            "edge_id",
            "cell_line_id",
            "drug",
            "dose_uM",
            "plate",
            "repeat_epoch",
            "energy_distance",
        }
        if set(raw.columns) != required_raw_columns:
            raise AssertionError("Existing raw B0 CSV schema changed")
        runtime = previous_result["runtime"]
    else:
        raw, runtime = evaluate_raw(
            dataset,
            list(protocol["evaluation_repeat_epochs"]),
            int(protocol["evaluation_base_seed"]),
            args.batch_size,
            device,
        )
    aggregation_started = time.perf_counter()
    edge, details = aggregate(raw)
    checks = validate_results(dataset, protocol, protocol_sha, raw, edge, details)
    aggregation_elapsed = time.perf_counter() - aggregation_started

    comparison: dict[str, Any] | None = None
    if args.aggregate_only:
        assert previous_edge is not None and raw_sha_before is not None
        compared = edge[["edge_id", "energy_distance"]].merge(
            previous_edge,
            on="edge_id",
            how="inner",
            validate="one_to_one",
            suffixes=("_corrected", "_previous"),
        )
        if len(compared) != EXPECTED_TEST_EDGES:
            raise AssertionError("Previous and corrected edge results do not align")
        absolute_difference = np.abs(
            compared["energy_distance_corrected"].to_numpy(np.float64)
            - compared["energy_distance_previous"].to_numpy(np.float64)
        )
        comparison = {
            "previous_edge_csv_sha256": sha256_file(EDGE_PATH),
            "max_edge_level_absolute_difference": float(absolute_difference.max()),
        }
        if sha256_file(RAW_PATH) != raw_sha_before:
            raise AssertionError("Raw B0 CSV changed during aggregate-only correction")
        checks["raw_energy_distance_recomputed"] = "pass: no"
        checks["raw_csv_unchanged"] = "pass"
    else:
        output_raw = raw.drop(columns="_cache_condition_index")
        atomic_write_csv(RAW_PATH, output_raw)
    atomic_write_csv(EDGE_PATH, edge)
    metric_source_path = Path(inspect.getfile(SamplesLoss)).resolve()
    replicate_audit = details["replicate_audit"].iloc[0]
    result = {
        "schema": "tahoe_experiment1_b0_identity_result_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "run_mode": (
            "aggregate_only_from_existing_raw" if args.aggregate_only else "full_b0_evaluation"
        ),
        "energy_distance_recomputed": not args.aggregate_only,
        "baseline": {"name": "B0 Identity", "definition": "Zpred = Zctrl"},
        "evaluation_protocol": {
            "path": PROTOCOL_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": protocol_sha,
            "S": protocol["S"],
            "repeat_count": protocol["evaluation_repeat_count"],
            "base_seed": protocol["evaluation_base_seed"],
            "repeat_epochs": protocol["evaluation_repeat_epochs"],
            "shuffle": protocol["shuffle"],
            "drop_last": protocol["drop_last"],
        },
        "dataset": {
            "class": "TahoeExperiment1LatentSetDataset",
            "path": DATASET_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(DATASET_PATH),
            "sampling_rule_version": protocol["dataset_sampling_rule"]["version"],
            "embedding_transforms": [],
        },
        "metric": {
            "implementation": "geomloss.SamplesLoss",
            "configuration": {"loss": "energy", "blur": ENERGY_BLUR},
            "geomloss_version": importlib.metadata.version("geomloss"),
            "source_path": str(metric_source_path),
            "source_sha256": sha256_file(metric_source_path),
            "state_reference_path": str(STATE_MODEL_PATH),
            "state_reference_sha256": sha256_file(STATE_MODEL_PATH),
            "evaluation_script": SCRIPT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "evaluation_script_sha256": sha256_file(SCRIPT_PATH),
            "raw_evaluation_script_sha256": (
                previous_result["metric"]["evaluation_script_sha256"]
                if previous_result is not None
                else sha256_file(SCRIPT_PATH)
            ),
        },
        "counts": {
            "test_conditions": len(dataset),
            "test_edges": int(dataset.conditions["edge_id"].nunique()),
            "repeat_count": len(protocol["evaluation_repeat_epochs"]),
            "expected_raw_rows": EXPECTED_RAW_ROWS,
            "observed_raw_rows": len(raw),
            "plate_level_conditions": len(details["plate_condition"]),
            "biological_replicate_units": len(details["biological_replicate"]),
            "plate6_plate14_replicate_groups": int(replicate_audit["plate6_plate14_groups"]),
            "dose_level_units": len(details["dose"]),
            "final_edge_count": len(edge),
        },
        "aggregation": {
            "hierarchy": [
                "repeat",
                "plate-level condition",
                "biological replicate",
                "(cell_line_id, drug, dose_uM)",
                "equal dose average",
                "(cell_line_id, drug) edge",
            ],
            "plate6_plate14_rule": "only dose_uM == 5.0 high-dose groups are paired",
            "weights": "equal within each hierarchy level; no cell-count weighting",
        },
        "b0_edge_level_energy_distance": describe(edge["energy_distance"]),
        "stratified_summary": {
            "by_dose": grouped_describe(details["dose"], "dose_uM"),
            "by_cell_line": grouped_describe(edge, "cell_line_id"),
            "by_drug": grouped_describe(edge, "drug"),
        },
        "runtime": runtime,
        "current_run": {
            "aggregation_elapsed_seconds": aggregation_elapsed,
            "energy_distance_recomputed": not args.aggregate_only,
        },
        "comparison_to_previous_result": comparison,
        "checks": checks,
        "outputs": {
            "condition_repeat_csv": {
                "path": RAW_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(RAW_PATH),
            },
            "edge_csv": {
                "path": EDGE_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(EDGE_PATH),
            },
            "result_json": RESULT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "handoff_md": HANDOFF_PATH.relative_to(PROJECT_ROOT).as_posix(),
        },
        "issues": [],
    }
    atomic_write_text(
        RESULT_PATH,
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )

    summary = result["b0_edge_level_energy_distance"]
    handoff = f"""# Tahoe Experiment 1 B0 Identity baseline

- Status: **PASS**
- Definition: `Zpred = Zctrl`
- Scope: formal test split only; no B1/B2/STATE/ST and no parameter training.
- Run mode: **{'aggregation only; existing raw Energy rows reused' if args.aggregate_only else 'full B0 evaluation'}**
- Raw condition/repeat rows: **{len(raw):,}** (`{len(dataset):,} conditions x {EXPECTED_REPEAT_COUNT} repeats`)
- Final `(cell_line_id, drug)` edges: **{len(edge):,}**
- Energy distance finite: **yes**
- Plate6/plate14 high-dose replicate groups: **{int(replicate_audit['plate6_plate14_groups']):,}**

## Edge-level Energy distance

| count | mean | median | p05 | p95 |
|---:|---:|---:|---:|---:|
| {summary['count']:,} | {summary['mean']:.8g} | {summary['median']:.8g} | {summary['p05']:.8g} | {summary['p95']:.8g} |

Aggregation is equal-weighted at each frozen level: repeat -> plate-level condition ->
biological replicate -> `(cell_line_id, drug, dose_uM)` -> dose average -> edge.
Plate6/plate14 are one replicate unit only when `dose_uM == 5.0` and both plates
occur in the same dose group. No cell-count weighting or latent preprocessing was used.

## Outputs

- `{RAW_PATH.relative_to(PROJECT_ROOT).as_posix()}`
- `{EDGE_PATH.relative_to(PROJECT_ROOT).as_posix()}`
- `{RESULT_PATH.relative_to(PROJECT_ROOT).as_posix()}`
- `{HANDOFF_PATH.relative_to(PROJECT_ROOT).as_posix()}`
"""
    atomic_write_text(HANDOFF_PATH, handoff)
    print(json.dumps({
        "status": "pass",
        "raw_rows": len(raw),
        "final_edges": len(edge),
        "edge_summary": summary,
        "result": RESULT_PATH.relative_to(PROJECT_ROOT).as_posix(),
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
