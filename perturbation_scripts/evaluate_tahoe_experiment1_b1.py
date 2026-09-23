#!/usr/bin/env python3
"""Evaluate the frozen Experiment 1 B1 drug-dose mean-shift baseline."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from geomloss import SamplesLoss

from evaluate_tahoe_experiment1_b0 import (
    DATASET_PATH,
    ENERGY_BLUR,
    EXPECTED_RAW_ROWS,
    EXPECTED_REPEAT_COUNT,
    EXPECTED_TEST_CONDITIONS,
    EXPECTED_TEST_EDGES,
    PROTOCOL_PATH,
    STATE_MODEL_PATH,
    aggregate,
    assert_without_replacement,
    atomic_write_csv,
    atomic_write_text,
    describe,
    load_protocol,
    utc_now,
)
from freeze_tahoe_experiment1_evaluation_sampling import (
    AUDIT_PATH as SAMPLING_AUDIT_PATH,
    index_sha256,
    selected_positions,
)
from tahoe_experiment1_latent_data import (
    LATENT_DIM,
    PROJECT_ROOT,
    RESULTS,
    SET_SIZE,
    TahoeExperiment1LatentSetDataset,
    make_dataloader,
    sha256_file,
)


SCRIPT_PATH = Path(__file__).resolve()
B0_SCRIPT_PATH = PROJECT_ROOT / "perturbation_scripts" / "evaluate_tahoe_experiment1_b0.py"
B0_RAW_PATH = RESULTS / "tahoe_experiment1_b0_identity_condition_repeat.csv"
B0_EDGE_PATH = RESULTS / "tahoe_experiment1_b0_identity_edge.csv"
B0_RESULT_PATH = RESULTS / "tahoe_experiment1_b0_identity_result.json"
MEAN_DELTA_PATH = RESULTS / "tahoe_experiment1_b1_mean_delta.npy"
MEAN_DELTA_METADATA_PATH = RESULTS / "tahoe_experiment1_b1_mean_delta_metadata.csv"
MEAN_DELTA_AUDIT_PATH = RESULTS / "tahoe_experiment1_b1_mean_delta_build_audit.json"
CONDITION_SHIFT_METADATA_PATH = (
    RESULTS / "tahoe_experiment1_b1_train_condition_shifts_metadata.csv"
)
RAW_PATH = RESULTS / "tahoe_experiment1_b1_mean_shift_condition_repeat_paired.csv"
EDGE_PATH = RESULTS / "tahoe_experiment1_b1_mean_shift_edge_comparison.csv"
RESULT_PATH = RESULTS / "tahoe_experiment1_b1_mean_shift_result.json"
HANDOFF_PATH = RESULTS / "tahoe_experiment1_b1_mean_shift_handoff.md"
EXPECTED_MEAN_DELTA_ROWS = 1_132
EXPECTED_HIGH_DOSE_REPLICATE_GROUPS = 397
EPS = float(np.finfo(np.float64).eps)


def load_mean_delta(
    dataset: TahoeExperiment1LatentSetDataset,
) -> tuple[np.memmap, pd.DataFrame, dict[str, int], dict[str, Any]]:
    build_audit = json.loads(MEAN_DELTA_AUDIT_PATH.read_text(encoding="utf-8"))
    if build_audit.get("status") != "pass":
        raise AssertionError("The train-only mean-delta build is not recorded as pass")
    if (
        build_audit.get("scope", {}).get("fit_split") != "train only"
        or build_audit.get("counts", {}).get("train_conditions") != 45_652
        or build_audit.get("counts", {}).get("train_unique_drug_dose")
        != EXPECTED_MEAN_DELTA_ROWS
        or build_audit.get("checks", {}).get("val_test_treated_cells_excluded") != "pass"
        or build_audit.get("checks", {}).get("macro_average_equal_condition_weights")
        != "pass"
    ):
        raise AssertionError("The frozen train-only mean-delta fitting contract changed")
    if sha256_file(MEAN_DELTA_PATH) != build_audit["outputs"]["mean_delta"]["sha256"]:
        raise AssertionError("Mean-delta NPY SHA-256 changed")
    if (
        sha256_file(MEAN_DELTA_METADATA_PATH)
        != build_audit["outputs"]["mean_delta_metadata"]["sha256"]
    ):
        raise AssertionError("Mean-delta metadata SHA-256 changed")
    if (
        sha256_file(CONDITION_SHIFT_METADATA_PATH)
        != build_audit["outputs"]["condition_metadata"]["sha256"]
    ):
        raise AssertionError("Train condition-shift metadata SHA-256 changed")

    mean_delta = np.load(MEAN_DELTA_PATH, mmap_mode="r")
    metadata = pd.read_csv(
        MEAN_DELTA_METADATA_PATH,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    if (
        not isinstance(mean_delta, np.memmap)
        or mean_delta.shape != (EXPECTED_MEAN_DELTA_ROWS, LATENT_DIM)
        or mean_delta.dtype != np.float32
        or not np.isfinite(mean_delta).all()
        or not (mean_delta < 0).any()
        or not (mean_delta > 0).any()
    ):
        raise AssertionError("Mean-delta array is not the frozen signed finite float32 table")
    if (
        len(metadata) != EXPECTED_MEAN_DELTA_ROWS
        or metadata.duplicated(["drug", "dose_uM"]).any()
        or not np.array_equal(
            metadata["mean_delta_row"].to_numpy(np.int64),
            np.arange(EXPECTED_MEAN_DELTA_ROWS, dtype=np.int64),
        )
        or int(metadata["train_condition_count"].sum()) != 45_652
    ):
        raise AssertionError("Mean-delta metadata grain or row mapping changed")

    train = dataset.all_conditions.loc[dataset.all_conditions["split"].eq("train")]
    train_group_counts = (
        train.groupby(["drug", "dose_uM"], sort=False)
        .size()
        .rename("expected_train_condition_count")
        .reset_index()
    )
    group_check = metadata.merge(
        train_group_counts,
        on=["drug", "dose_uM"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if (
        len(group_check) != EXPECTED_MEAN_DELTA_ROWS
        or not group_check["_merge"].eq("both").all()
        or not group_check["train_condition_count"].eq(
            group_check["expected_train_condition_count"]
        ).all()
    ):
        raise AssertionError("Mean-delta groups do not exactly match train conditions")

    fit_conditions = pd.read_csv(
        CONDITION_SHIFT_METADATA_PATH,
        usecols=["condition_id"],
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    train_ids = set(train["pair_id"].astype(str))
    if (
        len(fit_conditions) != 45_652
        or fit_conditions["condition_id"].duplicated().any()
        or set(fit_conditions["condition_id"].astype(str)) != train_ids
    ):
        raise AssertionError("Mean-delta fit condition IDs are not exactly the train split")

    mapping = dataset.conditions[["pair_id", "drug", "dose_uM"]].merge(
        metadata[["drug", "dose_uM", "mean_delta_row"]],
        on=["drug", "dose_uM"],
        how="left",
        validate="many_to_one",
    )
    if len(mapping) != EXPECTED_TEST_CONDITIONS or mapping["mean_delta_row"].isna().any():
        raise AssertionError("A test condition has no unique exact train mean-delta row")
    row_by_condition = dict(
        zip(
            mapping["pair_id"].astype(str),
            mapping["mean_delta_row"].astype(np.int64),
            strict=True,
        )
    )
    return mean_delta, metadata, row_by_condition, build_audit


def load_b0_reference(
    protocol_sha: str,
    dataset_sha: str,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    result = json.loads(B0_RESULT_PATH.read_text(encoding="utf-8"))
    raw_sha = sha256_file(B0_RAW_PATH)
    if (
        result.get("status") != "pass"
        or result.get("counts", {}).get("observed_raw_rows") != EXPECTED_RAW_ROWS
        or result.get("counts", {}).get("final_edge_count") != EXPECTED_TEST_EDGES
        or result.get("counts", {}).get("plate6_plate14_replicate_groups")
        != EXPECTED_HIGH_DOSE_REPLICATE_GROUPS
        or result.get("aggregation", {}).get("plate6_plate14_rule")
        != "only dose_uM == 5.0 high-dose groups are paired"
        or result.get("evaluation_protocol", {}).get("sha256") != protocol_sha
        or result.get("dataset", {}).get("sha256") != dataset_sha
        or result.get("metric", {}).get("configuration")
        != {"loss": "energy", "blur": ENERGY_BLUR}
        or result.get("outputs", {}).get("condition_repeat_csv", {}).get("sha256")
        != raw_sha
    ):
        raise AssertionError("Formal B0 reference is incompatible with B1 evaluation")
    raw = pd.read_csv(B0_RAW_PATH, keep_default_na=False, encoding="utf-8-sig")
    expected_columns = {
        "condition_id",
        "edge_id",
        "cell_line_id",
        "drug",
        "dose_uM",
        "plate",
        "repeat_epoch",
        "energy_distance",
    }
    if (
        set(raw.columns) != expected_columns
        or len(raw) != EXPECTED_RAW_ROWS
        or raw.duplicated(["condition_id", "repeat_epoch"]).any()
        or not np.isfinite(raw["energy_distance"].to_numpy(np.float64)).all()
    ):
        raise AssertionError("Formal B0 raw condition/repeat artifact is invalid")
    return raw, result, raw_sha


def evaluate_b1(
    dataset: TahoeExperiment1LatentSetDataset,
    repeat_epochs: list[int],
    base_seed: int,
    batch_size: int,
    device: torch.device,
    mean_delta: np.memmap,
    row_by_condition: dict[str, int],
    sampling_audit: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metric = SamplesLoss(loss="energy", blur=ENERGY_BLUR)
    metadata = dataset.conditions.set_index("pair_id", verify_integrity=True)
    records: list[dict[str, Any]] = []
    sample_positions = selected_positions(len(dataset))
    sample_condition_ids = dataset.conditions.iloc[sample_positions]["pair_id"].astype(str).tolist()
    expected_repeat_audits = {
        int(row["repeat_epoch"]): row
        for row in sampling_audit["results"]["test"]["repeat_results"]
    }
    index_spotchecks: list[dict[str, Any]] = []
    batch_count = 0
    prediction_min = float("inf")
    prediction_max = float("-inf")
    control_unchanged = True
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
            sampled_indices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for batch in loader:
                condition_ids = list(batch["condition_id"])
                observed_condition_ids.extend(condition_ids)
                if set(batch["split"]) != {"test"}:
                    raise AssertionError("Non-test condition entered B1 evaluation")
                assert_without_replacement(batch["source_embedding_index"], "control")
                assert_without_replacement(batch["target_embedding_index"], "treated")
                for offset, condition_id in enumerate(condition_ids):
                    if condition_id in sample_condition_ids:
                        sampled_indices[condition_id] = (
                            batch["source_embedding_index"][offset].numpy().copy(),
                            batch["target_embedding_index"][offset].numpy().copy(),
                        )

                control = batch["ctrl_cell_emb"].to(device, non_blocking=True)
                treated = batch["pert_cell_emb"].to(device, non_blocking=True)
                if control.dtype != torch.float32 or treated.dtype != torch.float32:
                    raise AssertionError("B1 inputs must remain float32")
                if control.shape[1:] != (SET_SIZE, LATENT_DIM) or treated.shape != control.shape:
                    raise AssertionError("Unexpected B1 input tensor shape")
                delta_rows = np.fromiter(
                    (row_by_condition[condition_id] for condition_id in condition_ids),
                    dtype=np.int64,
                    count=len(condition_ids),
                )
                shifts = torch.from_numpy(
                    np.ascontiguousarray(mean_delta[delta_rows], dtype=np.float32)
                ).to(device, non_blocking=True)
                if shifts.shape != (len(condition_ids), LATENT_DIM):
                    raise AssertionError("Unexpected B1 mean-delta tensor shape")
                first_batch = batch_count == 0
                control_snapshot = control.clone() if first_batch else None
                predicted = control + shifts[:, None, :]
                if predicted.shape != control.shape or predicted.dtype != torch.float32:
                    raise AssertionError("Unexpected B1 predicted tensor shape or dtype")
                if first_batch:
                    assert control_snapshot is not None
                    control_unchanged = bool(torch.equal(control, control_snapshot))
                    if not control_unchanged or not torch.isfinite(predicted).all():
                        raise AssertionError("B1 addition mutated control or produced non-finite values")
                    prediction_min = float(predicted.min())
                    prediction_max = float(predicted.max())
                values = metric(predicted, treated)
                if values.shape != (len(condition_ids),) or not torch.isfinite(values).all():
                    raise AssertionError("B1 Energy distance is not finite per condition")

                for condition_id, delta_row, value in zip(
                    condition_ids,
                    delta_rows,
                    values.detach().cpu().numpy(),
                    strict=True,
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
                            "mean_delta_row": int(delta_row),
                            "b1_energy": float(value),
                            "_cache_condition_index": int(row["cache_condition_index"]),
                        }
                    )
                batch_count += 1

            expected_order = dataset.conditions["pair_id"].astype(str).tolist()
            if observed_condition_ids != expected_order:
                raise AssertionError(f"repeat_epoch={repeat_epoch} was shuffled or dropped")
            if set(sampled_indices) != set(sample_condition_ids):
                raise AssertionError("Sampling fingerprint spot-check conditions are incomplete")
            sample_source = np.stack(
                [sampled_indices[condition_id][0] for condition_id in sample_condition_ids]
            )
            sample_target = np.stack(
                [sampled_indices[condition_id][1] for condition_id in sample_condition_ids]
            )
            observed_digest = index_sha256(sample_condition_ids, sample_source, sample_target)
            expected_digest = expected_repeat_audits[repeat_epoch][
                "independent_dataset_a_indices_sha256"
            ]
            if observed_digest != expected_digest:
                raise AssertionError("B1 sampled indices differ from the frozen audit")
            index_spotchecks.append(
                {
                    "repeat_epoch": int(repeat_epoch),
                    "condition_ids": sample_condition_ids,
                    "indices_sha256": observed_digest,
                    "matches_frozen_sampling_audit": True,
                }
            )
            print(
                f"repeat_epoch={repeat_epoch} conditions={len(observed_condition_ids)}/{len(dataset)}",
                flush=True,
            )

    raw = pd.DataFrame.from_records(records).sort_values(
        ["_cache_condition_index", "repeat_epoch"], kind="stable"
    )
    elapsed = time.perf_counter() - started
    if not (prediction_min < 0 < prediction_max):
        raise AssertionError("B1 predicted latents did not preserve signed values")
    return raw, {
        "device": str(device),
        "batch_size": batch_size,
        "batches": batch_count,
        "elapsed_seconds": elapsed,
        "condition_repeat_rows_per_second": len(raw) / elapsed,
        "gpu_peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
        ),
        "gpu_peak_reserved_gib": (
            torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else None
        ),
        "prediction_min": prediction_min,
        "prediction_max": prediction_max,
        "control_tensor_unchanged_after_out_of_place_addition": control_unchanged,
        "sampling_index_spotchecks": index_spotchecks,
    }


def pair_with_b0(b1_raw: pd.DataFrame, b0_raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["condition_id", "repeat_epoch"]
    metadata_columns = ["edge_id", "cell_line_id", "drug", "dose_uM", "plate"]
    paired = b1_raw.merge(
        b0_raw,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_b1", "_b0"),
    )
    if len(paired) != EXPECTED_RAW_ROWS:
        raise AssertionError("B0/B1 condition-repeat pairing is incomplete")
    for column in metadata_columns:
        left = paired[f"{column}_b1"]
        right = paired[f"{column}_b0"]
        if column == "dose_uM":
            same = np.array_equal(left.to_numpy(np.float64), right.to_numpy(np.float64))
        else:
            same = left.astype(str).eq(right.astype(str)).all()
        if not same:
            raise AssertionError(f"B0/B1 paired metadata mismatch: {column}")
        paired[column] = left

    paired = paired.rename(columns={"energy_distance": "b0_energy"})
    paired["delta_ed"] = paired["b0_energy"] - paired["b1_energy"]
    paired["energy_gain"] = paired["delta_ed"] / (paired["b0_energy"] + EPS)
    paired = paired.sort_values(
        ["_cache_condition_index", "repeat_epoch"], kind="stable"
    ).reset_index(drop=True)
    output_columns = [
        "condition_id",
        "edge_id",
        "cell_line_id",
        "drug",
        "dose_uM",
        "plate",
        "repeat_epoch",
        "mean_delta_row",
        "b0_energy",
        "b1_energy",
        "delta_ed",
        "energy_gain",
    ]
    output = paired[output_columns]
    if not np.isfinite(
        output[["b0_energy", "b1_energy", "delta_ed", "energy_gain"]].to_numpy(
            np.float64
        )
    ).all():
        raise AssertionError("A paired B0/B1 metric is non-finite")
    return output


def metric_frame(raw: pd.DataFrame, column: str) -> pd.DataFrame:
    columns = [
        "condition_id",
        "edge_id",
        "cell_line_id",
        "drug",
        "dose_uM",
        "plate",
        "repeat_epoch",
        column,
    ]
    return raw[columns].rename(columns={column: "energy_distance"})


def add_comparison_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["delta_ed"] = result["b0_energy"] - result["b1_energy"]
    result["energy_gain"] = result["delta_ed"] / (result["b0_energy"] + EPS)
    result["b1_beats_b0"] = result["b1_energy"] < result["b0_energy"]
    return result


def aggregate_paired(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    b0_edge, b0_details = aggregate(metric_frame(raw, "b0_energy"))
    b1_edge, b1_details = aggregate(metric_frame(raw, "b1_energy"))
    edge_keys = ["edge_id", "cell_line_id", "drug"]
    edge = b0_edge.merge(
        b1_edge,
        on=edge_keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_b0", "_b1"),
    ).rename(
        columns={
            "energy_distance_b0": "b0_energy",
            "energy_distance_b1": "b1_energy",
        }
    )
    for column in ("dose_count", "biological_replicate_count", "plate_condition_count"):
        if not edge[f"{column}_b0"].eq(edge[f"{column}_b1"]).all():
            raise AssertionError(f"B0/B1 aggregation units differ: {column}")
        edge[column] = edge[f"{column}_b0"]
        edge = edge.drop(columns=[f"{column}_b0", f"{column}_b1"])
    edge = add_comparison_columns(edge).sort_values("edge_id", kind="stable").reset_index(
        drop=True
    )

    dose_keys = ["edge_id", "cell_line_id", "drug", "dose_uM"]
    dose = b0_details["dose"].merge(
        b1_details["dose"],
        on=dose_keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_b0", "_b1"),
    ).rename(
        columns={
            "energy_distance_b0": "b0_energy",
            "energy_distance_b1": "b1_energy",
        }
    )
    dose = add_comparison_columns(dose)
    return edge, dose, b0_details, b1_details


def grouped_comparison(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(column, sort=True):
        rows.append(
            {
                column: float(value) if column == "dose_uM" else str(value),
                "count": int(len(group)),
                "b0_mean_energy": float(group["b0_energy"].mean()),
                "b1_mean_energy": float(group["b1_energy"].mean()),
                "mean_delta_ed": float(group["delta_ed"].mean()),
                "mean_energy_gain": float(group["energy_gain"].mean()),
                "median_energy_gain": float(group["energy_gain"].median()),
                "fraction_b1_beats_b0": float(group["b1_beats_b0"].mean()),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    protocol, protocol_sha = load_protocol()
    dataset_sha = sha256_file(DATASET_PATH)
    sampling_audit = json.loads(SAMPLING_AUDIT_PATH.read_text(encoding="utf-8"))
    if (
        sampling_audit.get("status") != "pass"
        or sampling_audit.get("protocol", {}).get("sha256") != protocol_sha
        or sampling_audit.get("results", {}).get("test", {}).get(
            "all_repeat_indices_exactly_reproduced"
        )
        is not True
    ):
        raise AssertionError("Frozen evaluation sampling audit is incompatible")

    dataset = TahoeExperiment1LatentSetDataset(
        split="test",
        seed=int(protocol["evaluation_base_seed"]),
        epoch=int(protocol["evaluation_repeat_epochs"][0]),
    )
    mean_delta, mean_metadata, row_by_condition, build_audit = load_mean_delta(dataset)
    b0_raw, b0_result, b0_raw_sha = load_b0_reference(protocol_sha, dataset_sha)
    b1_raw, runtime = evaluate_b1(
        dataset,
        list(protocol["evaluation_repeat_epochs"]),
        int(protocol["evaluation_base_seed"]),
        args.batch_size,
        device,
        mean_delta,
        row_by_condition,
        sampling_audit,
    )
    paired = pair_with_b0(b1_raw, b0_raw)
    edge, dose, b0_details, b1_details = aggregate_paired(paired)

    official_b0_edge = pd.read_csv(
        B0_EDGE_PATH,
        usecols=["edge_id", "energy_distance"],
        keep_default_na=False,
        encoding="utf-8-sig",
    ).rename(columns={"energy_distance": "official_b0_energy"})
    b0_check = edge[["edge_id", "b0_energy"]].merge(
        official_b0_edge,
        on="edge_id",
        how="inner",
        validate="one_to_one",
    )
    b0_edge_max_abs_difference = float(
        np.max(np.abs(b0_check["b0_energy"] - b0_check["official_b0_energy"]))
    )
    if len(b0_check) != EXPECTED_TEST_EDGES or b0_edge_max_abs_difference > 5e-10:
        raise AssertionError("Reaggregated B0 edge values differ from the formal B0 artifact")

    repeats = paired.groupby("condition_id")["repeat_epoch"].agg(["size", "nunique"])
    expected_epochs = set(protocol["evaluation_repeat_epochs"])
    observed_epochs = paired.groupby("condition_id")["repeat_epoch"].agg(set)
    b0_replicate_groups = int(
        b0_details["replicate_audit"].iloc[0]["plate6_plate14_groups"]
    )
    b1_replicate_groups = int(
        b1_details["replicate_audit"].iloc[0]["plate6_plate14_groups"]
    )
    if (
        len(dataset) != EXPECTED_TEST_CONDITIONS
        or len(paired) != EXPECTED_RAW_ROWS
        or paired.duplicated(["condition_id", "repeat_epoch"]).any()
        or not ((repeats["size"] == EXPECTED_REPEAT_COUNT) & (repeats["nunique"] == EXPECTED_REPEAT_COUNT)).all()
        or not observed_epochs.map(lambda value: value == expected_epochs).all()
        or len(edge) != EXPECTED_TEST_EDGES
        or set(edge["edge_id"]) != set(dataset.conditions["edge_id"])
        or b0_replicate_groups != EXPECTED_HIGH_DOSE_REPLICATE_GROUPS
        or b1_replicate_groups != EXPECTED_HIGH_DOSE_REPLICATE_GROUPS
        or dataset.embedding_transforms
        or any(dataset.edge_overlap.values())
        or sha256_file(PROTOCOL_PATH) != protocol_sha
        or sha256_file(DATASET_PATH) != dataset_sha
        or sha256_file(B0_RAW_PATH) != b0_raw_sha
        or sha256_file(MEAN_DELTA_PATH)
        != build_audit["outputs"]["mean_delta"]["sha256"]
    ):
        raise AssertionError("Final B1 evaluation audit failed")

    atomic_write_csv(RAW_PATH, paired)
    atomic_write_csv(EDGE_PATH, edge)
    metric_source_path = Path(inspect.getfile(SamplesLoss)).resolve()
    edge_summary = {
        "b0_energy": describe(edge["b0_energy"]),
        "b1_energy": describe(edge["b1_energy"]),
        "delta_ed": describe(edge["delta_ed"]),
        "energy_gain": describe(edge["energy_gain"]),
        "fraction_b1_beats_b0": float(edge["b1_beats_b0"].mean()),
    }
    result = {
        "schema": "tahoe_experiment1_b1_mean_shift_result_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "baseline": {
            "name": "B1 Drug-dose mean shift",
            "definition": "Zpred = Zctrl + mean_delta(drug, dose_uM)[None, :]",
            "mean_delta_fit": "train-only equal-weight macro-average of condition shifts",
            "parameters_trained_during_evaluation": False,
            "fallbacks": [],
        },
        "evaluation_protocol": {
            "path": PROTOCOL_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": protocol_sha,
            "S": protocol["S"],
            "repeat_count": protocol["evaluation_repeat_count"],
            "base_seed": protocol["evaluation_base_seed"],
            "repeat_epochs": protocol["evaluation_repeat_epochs"],
            "shuffle": protocol["shuffle"],
            "drop_last": protocol["drop_last"],
            "sampling_rule": protocol["dataset_sampling_rule"],
            "sampling_reproducibility_audit": {
                "path": SAMPLING_AUDIT_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(SAMPLING_AUDIT_PATH),
            },
            "identical_to_b0_basis": (
                "same protocol SHA, Dataset SHA, seed, repeat epochs, pair_id, side, "
                "and model-independent deterministic sampler"
            ),
        },
        "dataset": {
            "class": "TahoeExperiment1LatentSetDataset",
            "path": DATASET_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": dataset_sha,
            "embedding_transforms": [],
        },
        "mean_delta_reference": {
            "npy_path": MEAN_DELTA_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "npy_sha256": sha256_file(MEAN_DELTA_PATH),
            "metadata_path": MEAN_DELTA_METADATA_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "metadata_sha256": sha256_file(MEAN_DELTA_METADATA_PATH),
            "build_audit_path": MEAN_DELTA_AUDIT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "build_audit_sha256": sha256_file(MEAN_DELTA_AUDIT_PATH),
            "shape": list(mean_delta.shape),
            "dtype": str(mean_delta.dtype),
            "exact_groups": len(mean_metadata),
            "test_mapping_coverage": 1.0,
            "fit_split": "train only",
        },
        "b0_reference": {
            "raw_path": B0_RAW_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "raw_sha256": b0_raw_sha,
            "edge_path": B0_EDGE_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "edge_sha256": sha256_file(B0_EDGE_PATH),
            "result_path": B0_RESULT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "result_sha256": sha256_file(B0_RESULT_PATH),
            "raw_evaluation_script_sha256": b0_result["metric"][
                "raw_evaluation_script_sha256"
            ],
            "reaggregated_edge_max_abs_difference": b0_edge_max_abs_difference,
        },
        "metric": {
            "implementation": "geomloss.SamplesLoss",
            "configuration": {"loss": "energy", "blur": ENERGY_BLUR},
            "geomloss_version": importlib.metadata.version("geomloss"),
            "source_path": str(metric_source_path),
            "source_sha256": sha256_file(metric_source_path),
            "state_reference_path": str(STATE_MODEL_PATH),
            "state_reference_sha256": sha256_file(STATE_MODEL_PATH),
            "b0_aggregation_implementation": B0_SCRIPT_PATH.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "b0_aggregation_implementation_sha256": sha256_file(B0_SCRIPT_PATH),
            "evaluation_script": SCRIPT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "evaluation_script_sha256": sha256_file(SCRIPT_PATH),
        },
        "counts": {
            "test_conditions": len(dataset),
            "test_edges": int(dataset.conditions["edge_id"].nunique()),
            "repeat_count": len(protocol["evaluation_repeat_epochs"]),
            "expected_raw_rows": EXPECTED_RAW_ROWS,
            "observed_raw_rows": len(paired),
            "paired_b0_rows": len(paired),
            "mean_delta_mapped_conditions": len(row_by_condition),
            "plate_level_conditions": len(b1_details["plate_condition"]),
            "biological_replicate_units": len(b1_details["biological_replicate"]),
            "plate6_plate14_replicate_groups": b1_replicate_groups,
            "dose_level_units": len(b1_details["dose"]),
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
        "comparison_formulas": {
            "delta_ed": "b0_energy - b1_energy",
            "energy_gain": "(b0_energy - b1_energy) / (b0_energy + eps)",
            "eps": EPS,
            "positive_energy_gain_means": "B1 is better than B0",
            "b1_beats_b0": "b1_edge_energy < b0_edge_energy",
        },
        "edge_level": edge_summary,
        "stratified_summary": {
            "by_dose": grouped_comparison(dose, "dose_uM"),
            "by_drug": grouped_comparison(edge, "drug"),
            "by_cell_line": grouped_comparison(edge, "cell_line_id"),
        },
        "runtime": runtime,
        "checks": {
            "all_5684_test_conditions_evaluated": "pass",
            "exactly_5_repeats_per_condition": "pass",
            "raw_rows_28420": "pass",
            "all_b0_rows_uniquely_paired": "pass",
            "mean_delta_exact_mapping_100_percent": "pass",
            "all_1717_test_edges_aggregated": "pass",
            "plate6_plate14_high_dose_groups_397": "pass",
            "energy_distance_finite": "pass",
            "b0_b1_identical_sampling": "pass",
            "train_only_mean_delta": "pass",
            "val_test_fitting_leakage": "pass: none",
            "parameters_trained": "pass: none",
            "fallbacks": "pass: none",
            "latent_preprocessing": "pass: none",
            "control_embedding_mutated": "pass: no",
        },
        "outputs": {
            "condition_repeat_paired_csv": {
                "path": RAW_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(RAW_PATH),
            },
            "edge_comparison_csv": {
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

    comparison = result["edge_level"]
    handoff = f"""# Tahoe Experiment 1 B1 Drug-dose mean-shift baseline

- Status: **PASS**
- Prediction: `Zpred = Zctrl + mean_delta(drug, dose_uM)[None, :]`
- Scope: formal test split only; no B2/STATE/ST and no parameter training.
- Raw/paired condition-repeat rows: **{len(paired):,}/{len(paired):,}**
- Final `(cell_line_id, drug)` edges: **{len(edge):,}**
- Plate6/plate14 high-dose replicate groups: **{b1_replicate_groups:,}**
- Exact train mean-delta mapping: **{len(row_by_condition):,}/{len(dataset):,} (100%)**

## Edge-level comparison

| metric | value |
|---|---:|
| B0 mean Energy | {comparison['b0_energy']['mean']:.10g} |
| B1 mean Energy | {comparison['b1_energy']['mean']:.10g} |
| B1 median Energy | {comparison['b1_energy']['median']:.10g} |
| B1 p05 / p95 Energy | {comparison['b1_energy']['p05']:.10g} / {comparison['b1_energy']['p95']:.10g} |
| mean / median DeltaED | {comparison['delta_ed']['mean']:.10g} / {comparison['delta_ed']['median']:.10g} |
| mean / median EnergyGain | {comparison['energy_gain']['mean']:.10g} / {comparison['energy_gain']['median']:.10g} |
| p05 / p95 EnergyGain | {comparison['energy_gain']['p05']:.10g} / {comparison['energy_gain']['p95']:.10g} |
| fraction edges B1 beats B0 | {comparison['fraction_b1_beats_b0']:.10g} |

The frozen B0 Energy implementation and corrected aggregation hierarchy were reused.
B0 and B1 share the same deterministic Dataset sampler, protocol SHA, Dataset SHA,
seed, repeat epochs, condition IDs, and sides. No latent preprocessing, fallback,
test-time fitting, cell-count weighting, or parameter training was used.

## Outputs

- `{RAW_PATH.relative_to(PROJECT_ROOT).as_posix()}`
- `{EDGE_PATH.relative_to(PROJECT_ROOT).as_posix()}`
- `{RESULT_PATH.relative_to(PROJECT_ROOT).as_posix()}`
- `{HANDOFF_PATH.relative_to(PROJECT_ROOT).as_posix()}`
"""
    atomic_write_text(HANDOFF_PATH, handoff)
    print(
        json.dumps(
            {
                "status": "pass",
                "raw_rows": len(paired),
                "paired_rows": len(paired),
                "final_edges": len(edge),
                "edge_level": edge_summary,
                "result": RESULT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
