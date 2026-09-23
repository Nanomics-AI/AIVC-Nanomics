#!/usr/bin/env python3
"""Evaluate the frozen B2-v2 best checkpoint on the formal test repeats."""

from __future__ import annotations

import argparse
import hashlib
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
from evaluate_tahoe_experiment1_b1 import (
    B0_EDGE_PATH,
    B0_RAW_PATH,
    B0_RESULT_PATH,
    EDGE_PATH as B1_EDGE_PATH,
    EPS,
    EXPECTED_HIGH_DOSE_REPLICATE_GROUPS,
    RAW_PATH as B1_RAW_PATH,
    RESULT_PATH as B1_RESULT_PATH,
    load_b0_reference,
)
from freeze_tahoe_experiment1_evaluation_sampling import (
    AUDIT_PATH as SAMPLING_AUDIT_PATH,
    index_sha256,
    selected_positions,
)
from tahoe_experiment1_latent_data import (
    LATENT_DIM,
    PERT_DIM,
    PROJECT_ROOT,
    RESULTS,
    SET_SIZE,
    TahoeExperiment1LatentSetDataset,
    make_dataloader,
    sha256_file,
)
from train_tahoe_experiment1_b2 import (
    B2PooledMLP,
    SCRIPT_PATH as MODEL_PATH,
    atomic_write_json,
)


SCRIPT_PATH = Path(__file__).resolve()
TASK_PATH = PROJECT_ROOT.parent / "当前任务.txt"
CHECKPOINT_PATH = (
    RESULTS / "tahoe_experiment1_b2_v2_formal_checkpoints" / "b2_v2_best.pt"
)
B2_PROTOCOL_PATH = RESULTS / "tahoe_experiment1_b2_v2_training_protocol.json"
B2_TRAINING_RESULT_PATH = RESULTS / "tahoe_experiment1_b2_v2_formal_training_result.json"
RAW_PATH = RESULTS / "tahoe_experiment1_b2_v2_test_condition_repeat_paired.csv"
EDGE_PATH = RESULTS / "tahoe_experiment1_b2_v2_test_edge_comparison.csv"
RESULT_PATH = RESULTS / "tahoe_experiment1_b2_v2_test_result.json"
HANDOFF_PATH = RESULTS / "tahoe_experiment1_b2_v2_test_handoff.md"

EXPECTED_TASK_SHA256 = "0b170c50eeecd4322b848652c5220d24eeb86543f64643f383db469aaccf3f21"
EXPECTED_CHECKPOINT_SHA256 = "8d23da4e6760e1165eeab216eeb0cda22111cfcd735af4aa1884599209fbcdd3"
EXPECTED_B2_PROTOCOL_SHA256 = "d49932b5e7765627eb4e0681e48af5f3053a03b16ce18eb721ad65fc0e0045c0"
EXPECTED_BEST_VAL_ENERGY = 0.032510743475582186


def model_state_sha256(model: B2PooledMLP) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def load_best_checkpoint(device: torch.device) -> tuple[B2PooledMLP, dict[str, Any], str]:
    checkpoint_sha = sha256_file(CHECKPOINT_PATH)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise AssertionError("B2-v2 best checkpoint SHA-256 changed")
    if sha256_file(B2_PROTOCOL_PATH) != EXPECTED_B2_PROTOCOL_SHA256:
        raise AssertionError("B2-v2 frozen training protocol SHA-256 changed")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    expected = {
        "format": "tahoe_experiment1_b2_v2_formal_v1",
        "checkpoint_kind": "best",
        "epoch": 10,
        "global_step": 1958,
        "best_val_edge_energy": EXPECTED_BEST_VAL_ENERGY,
        "fresh_initialization": True,
        "b2_v1_or_smoke_checkpoint_loaded": False,
        "b2_v2_protocol_sha256": EXPECTED_B2_PROTOCOL_SHA256,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise AssertionError(f"B2-v2 best checkpoint mismatch for {key}")
    if "model_state" not in checkpoint or len(checkpoint["model_state"]) != 6:
        raise AssertionError("B2-v2 best checkpoint has an invalid model state")
    if not all(torch.isfinite(value).all() for value in checkpoint["model_state"].values()):
        raise AssertionError("B2-v2 best checkpoint contains non-finite parameters")

    training_result = json.loads(B2_TRAINING_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        training_result.get("status") != "pass"
        or training_result.get("best_epoch") != 10
        or training_result.get("best_val_edge_energy") != EXPECTED_BEST_VAL_ENERGY
        or training_result.get("checkpoint", {}).get("best", {}).get("sha256")
        != checkpoint_sha
        or training_result.get("test_dataset_constructed") is not False
    ):
        raise AssertionError("B2-v2 training result does not identify the frozen best checkpoint")

    model = B2PooledMLP()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint, checkpoint_sha


def load_b1_reference(protocol_sha: str, dataset_sha: str) -> tuple[pd.DataFrame, dict[str, Any], str]:
    result = json.loads(B1_RESULT_PATH.read_text(encoding="utf-8"))
    raw_sha = sha256_file(B1_RAW_PATH)
    if (
        result.get("status") != "pass"
        or result.get("counts", {}).get("observed_raw_rows") != EXPECTED_RAW_ROWS
        or result.get("counts", {}).get("final_edge_count") != EXPECTED_TEST_EDGES
        or result.get("counts", {}).get("plate6_plate14_replicate_groups")
        != EXPECTED_HIGH_DOSE_REPLICATE_GROUPS
        or result.get("evaluation_protocol", {}).get("sha256") != protocol_sha
        or result.get("dataset", {}).get("sha256") != dataset_sha
        or result.get("metric", {}).get("configuration")
        != {"loss": "energy", "blur": ENERGY_BLUR}
        or result.get("outputs", {}).get("condition_repeat_paired_csv", {}).get("sha256")
        != raw_sha
    ):
        raise AssertionError("Formal B1 reference is incompatible with B2-v2 evaluation")
    raw = pd.read_csv(B1_RAW_PATH, keep_default_na=False, encoding="utf-8-sig")
    required = {
        "condition_id",
        "edge_id",
        "cell_line_id",
        "drug",
        "dose_uM",
        "plate",
        "repeat_epoch",
        "b0_energy",
        "b1_energy",
    }
    if (
        not required <= set(raw.columns)
        or len(raw) != EXPECTED_RAW_ROWS
        or raw.duplicated(["condition_id", "repeat_epoch"]).any()
        or not np.isfinite(raw[["b0_energy", "b1_energy"]].to_numpy(np.float64)).all()
    ):
        raise AssertionError("Formal B1 paired condition/repeat artifact is invalid")
    return raw, result, raw_sha


def evaluate_b2(
    model: B2PooledMLP,
    dataset: TahoeExperiment1LatentSetDataset,
    repeat_epochs: list[int],
    base_seed: int,
    batch_size: int,
    device: torch.device,
    sampling_audit: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metric = SamplesLoss(loss="energy", blur=ENERGY_BLUR)
    metadata = dataset.conditions.set_index("pair_id", verify_integrity=True)
    records: list[dict[str, Any]] = []
    sample_positions = selected_positions(len(dataset))
    sample_condition_ids = (
        dataset.conditions.iloc[sample_positions]["pair_id"].astype(str).tolist()
    )
    expected_repeat_audits = {
        int(row["repeat_epoch"]): row
        for row in sampling_audit["results"]["test"]["repeat_results"]
    }
    index_spotchecks: list[dict[str, Any]] = []
    prediction_min = float("inf")
    prediction_max = float("-inf")
    batch_count = 0
    started = time.perf_counter()

    model.eval()
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
                raise AssertionError("B2-v2 test DataLoader was shuffled or dropped rows")
            observed_condition_ids: list[str] = []
            sampled_indices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for batch in loader:
                condition_ids = list(batch["condition_id"])
                observed_condition_ids.extend(condition_ids)
                if set(batch["split"]) != {"test"}:
                    raise AssertionError("Non-test condition entered B2-v2 evaluation")
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
                perturbation = batch["pert_emb"].to(device, non_blocking=True)
                if (
                    control.dtype != torch.float32
                    or treated.dtype != torch.float32
                    or perturbation.dtype != torch.float32
                    or control.shape[1:] != (SET_SIZE, LATENT_DIM)
                    or treated.shape != control.shape
                    or perturbation.shape != (len(condition_ids), SET_SIZE, PERT_DIM)
                ):
                    raise AssertionError("B2-v2 test input tensor contract changed")
                prediction = model(
                    {"ctrl_cell_emb": control, "pert_emb": perturbation}
                )
                if (
                    prediction.shape != control.shape
                    or prediction.dtype != torch.float32
                    or not torch.isfinite(prediction).all()
                ):
                    raise AssertionError("B2-v2 prediction is invalid")
                if batch_count == 0:
                    prediction_min = float(prediction.min())
                    prediction_max = float(prediction.max())
                values = metric(prediction, treated)
                if values.shape != (len(condition_ids),) or not torch.isfinite(values).all():
                    raise AssertionError("B2-v2 Energy is non-finite or has the wrong shape")
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
                            "b2_energy": float(value),
                            "_cache_condition_index": int(row["cache_condition_index"]),
                        }
                    )
                batch_count += 1

            expected_order = dataset.conditions["pair_id"].astype(str).tolist()
            if observed_condition_ids != expected_order:
                raise AssertionError(f"repeat_epoch={repeat_epoch} was shuffled or dropped")
            if set(sampled_indices) != set(sample_condition_ids):
                raise AssertionError("A frozen sampling spot-check condition was not observed")
            sample_source = np.stack(
                [sampled_indices[condition_id][0] for condition_id in sample_condition_ids]
            )
            sample_target = np.stack(
                [sampled_indices[condition_id][1] for condition_id in sample_condition_ids]
            )
            observed_digest = index_sha256(
                sample_condition_ids, sample_source, sample_target
            )
            expected_digest = expected_repeat_audits[repeat_epoch][
                "independent_dataset_a_indices_sha256"
            ]
            if observed_digest != expected_digest:
                raise AssertionError("B2-v2 sampled indices differ from the frozen audit")
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
        raise AssertionError("B2-v2 predicted latent did not preserve signed values")
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
        "prediction_min": prediction_min,
        "prediction_max": prediction_max,
        "sampling_index_spotchecks": index_spotchecks,
    }


def pair_all(
    b2_raw: pd.DataFrame, b1_raw: pd.DataFrame, b0_raw: pd.DataFrame
) -> pd.DataFrame:
    keys = ["condition_id", "repeat_epoch"]
    metadata_columns = ["edge_id", "cell_line_id", "drug", "dose_uM", "plate"]
    paired = b1_raw.merge(
        b2_raw,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_b1ref", "_b2"),
    )
    if len(paired) != EXPECTED_RAW_ROWS:
        raise AssertionError("B1/B2 condition-repeat pairing is incomplete")
    for column in metadata_columns:
        left = paired[f"{column}_b1ref"]
        right = paired[f"{column}_b2"]
        same = (
            np.array_equal(left.to_numpy(np.float64), right.to_numpy(np.float64))
            if column == "dose_uM"
            else left.astype(str).eq(right.astype(str)).all()
        )
        if not same:
            raise AssertionError(f"B1/B2 paired metadata mismatch: {column}")
        paired[column] = left

    b0_check = paired[keys + metadata_columns + ["b0_energy"]].merge(
        b0_raw,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_paired", "_b0"),
    )
    if len(b0_check) != EXPECTED_RAW_ROWS:
        raise AssertionError("B0/B1/B2 condition-repeat pairing is incomplete")
    for column in metadata_columns:
        left = b0_check[f"{column}_paired"]
        right = b0_check[f"{column}_b0"]
        same = (
            np.array_equal(left.to_numpy(np.float64), right.to_numpy(np.float64))
            if column == "dose_uM"
            else left.astype(str).eq(right.astype(str)).all()
        )
        if not same:
            raise AssertionError(f"B0/B1/B2 paired metadata mismatch: {column}")
    if not np.array_equal(
        b0_check["b0_energy"].to_numpy(np.float64),
        b0_check["energy_distance"].to_numpy(np.float64),
    ):
        raise AssertionError("B1 embedded B0 energies differ from formal B0 raw rows")

    output = paired[
        keys + metadata_columns + ["b0_energy", "b1_energy", "b2_energy"]
    ].copy()
    output["delta_b2_vs_b0"] = output["b0_energy"] - output["b2_energy"]
    output["gain_b2_vs_b0"] = output["delta_b2_vs_b0"] / (
        output["b0_energy"] + EPS
    )
    output["delta_b2_vs_b1"] = output["b1_energy"] - output["b2_energy"]
    output["gain_b2_vs_b1"] = output["delta_b2_vs_b1"] / (
        output["b1_energy"] + EPS
    )
    output = output[
        [
            "condition_id",
            "edge_id",
            "cell_line_id",
            "drug",
            "dose_uM",
            "plate",
            "repeat_epoch",
            "b0_energy",
            "b1_energy",
            "b2_energy",
            "delta_b2_vs_b0",
            "gain_b2_vs_b0",
            "delta_b2_vs_b1",
            "gain_b2_vs_b1",
        ]
    ]
    if not np.isfinite(output.select_dtypes(include=[np.number]).to_numpy()).all():
        raise AssertionError("A paired B0/B1/B2 numeric value is non-finite")
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


def aggregate_all(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
    aggregated: dict[str, tuple[pd.DataFrame, dict[str, pd.DataFrame]]] = {
        name: aggregate(metric_frame(raw, f"{name}_energy"))
        for name in ("b0", "b1", "b2")
    }
    b0_edge, b0_details = aggregated["b0"]
    edge = b0_edge.rename(columns={"energy_distance": "b0_energy"}).copy()
    count_columns = ["dose_count", "biological_replicate_count", "plate_condition_count"]
    for name in ("b1", "b2"):
        candidate, _details = aggregated[name]
        for column in ["edge_id", "cell_line_id", "drug", *count_columns]:
            if not edge[column].equals(candidate[column]):
                raise AssertionError(f"{name.upper()} edge aggregation units differ: {column}")
        edge[f"{name}_energy"] = candidate["energy_distance"]
    edge["delta_b2_vs_b0"] = edge["b0_energy"] - edge["b2_energy"]
    edge["gain_b2_vs_b0"] = edge["delta_b2_vs_b0"] / (edge["b0_energy"] + EPS)
    edge["b2_beats_b0"] = edge["b2_energy"] < edge["b0_energy"]
    edge["delta_b2_vs_b1"] = edge["b1_energy"] - edge["b2_energy"]
    edge["gain_b2_vs_b1"] = edge["delta_b2_vs_b1"] / (edge["b1_energy"] + EPS)
    edge["b2_beats_b1"] = edge["b2_energy"] < edge["b1_energy"]

    b0_dose = b0_details["dose"]
    dose = b0_dose.rename(columns={"energy_distance": "b0_energy"}).copy()
    dose_keys = ["edge_id", "cell_line_id", "drug", "dose_uM"]
    for name in ("b1", "b2"):
        candidate = aggregated[name][1]["dose"]
        for column in [*dose_keys, "biological_replicate_count", "plate_condition_count"]:
            if not dose[column].equals(candidate[column]):
                raise AssertionError(f"{name.upper()} dose aggregation units differ: {column}")
        dose[f"{name}_energy"] = candidate["energy_distance"]
    dose["delta_b2_vs_b1"] = dose["b1_energy"] - dose["b2_energy"]
    dose["b2_beats_b1"] = dose["b2_energy"] < dose["b1_energy"]
    details = {name: value[1] for name, value in aggregated.items()}
    return edge, dose, details


def grouped_b1_b2(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(column, sort=True):
        rows.append(
            {
                column: float(value) if column == "dose_uM" else str(value),
                "count": int(len(group)),
                "b1_mean_energy": float(group["b1_energy"].mean()),
                "b2_mean_energy": float(group["b2_energy"].mean()),
                "mean_delta_b2_vs_b1": float(group["delta_b2_vs_b1"].mean()),
                "fraction_b2_beats_b1": float(group["b2_beats_b1"].mean()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if any(path.exists() for path in (RAW_PATH, EDGE_PATH, RESULT_PATH, HANDOFF_PATH)):
        raise FileExistsError("A formal B2-v2 test output already exists; refusing to overwrite")
    if sha256_file(TASK_PATH) != EXPECTED_TASK_SHA256:
        raise AssertionError("当前任务.txt changed before B2-v2 test evaluation")
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
    if len(dataset) != EXPECTED_TEST_CONDITIONS or dataset.embedding_transforms:
        raise AssertionError("Formal raw-latent test Dataset changed")

    model, checkpoint, checkpoint_sha = load_best_checkpoint(device)
    parameter_sha_before = model_state_sha256(model)
    b0_raw, b0_result, b0_raw_sha = load_b0_reference(protocol_sha, dataset_sha)
    b1_raw, b1_result, b1_raw_sha = load_b1_reference(protocol_sha, dataset_sha)
    b2_raw, runtime = evaluate_b2(
        model,
        dataset,
        list(protocol["evaluation_repeat_epochs"]),
        int(protocol["evaluation_base_seed"]),
        args.batch_size,
        device,
        sampling_audit,
    )
    parameter_sha_after = model_state_sha256(model)
    if parameter_sha_after != parameter_sha_before:
        raise AssertionError("B2-v2 parameters changed during test evaluation")

    paired = pair_all(b2_raw, b1_raw, b0_raw)
    edge, dose, details = aggregate_all(paired)
    repeats = paired.groupby("condition_id")["repeat_epoch"].agg(["size", "nunique"])
    expected_epochs = set(protocol["evaluation_repeat_epochs"])
    observed_epochs = paired.groupby("condition_id")["repeat_epoch"].agg(set)
    replicate_groups = {
        name: int(value["replicate_audit"].iloc[0]["plate6_plate14_groups"])
        for name, value in details.items()
    }
    if (
        len(paired) != EXPECTED_RAW_ROWS
        or paired.duplicated(["condition_id", "repeat_epoch"]).any()
        or not ((repeats["size"] == EXPECTED_REPEAT_COUNT) & (repeats["nunique"] == EXPECTED_REPEAT_COUNT)).all()
        or not observed_epochs.map(lambda value: value == expected_epochs).all()
        or len(edge) != EXPECTED_TEST_EDGES
        or set(edge["edge_id"]) != set(dataset.conditions["edge_id"])
        or set(replicate_groups.values()) != {EXPECTED_HIGH_DOSE_REPLICATE_GROUPS}
        or not np.isfinite(
            edge[
                [
                    "b0_energy",
                    "b1_energy",
                    "b2_energy",
                    "delta_b2_vs_b0",
                    "gain_b2_vs_b0",
                    "delta_b2_vs_b1",
                    "gain_b2_vs_b1",
                ]
            ].to_numpy(np.float64)
        ).all()
        or sha256_file(CHECKPOINT_PATH) != checkpoint_sha
        or sha256_file(PROTOCOL_PATH) != protocol_sha
        or sha256_file(DATASET_PATH) != dataset_sha
        or sha256_file(B0_RAW_PATH) != b0_raw_sha
        or sha256_file(B1_RAW_PATH) != b1_raw_sha
    ):
        raise AssertionError("Final B2-v2 formal test audit failed")

    atomic_write_csv(RAW_PATH, paired)
    atomic_write_csv(EDGE_PATH, edge)
    metric_source_path = Path(inspect.getfile(SamplesLoss)).resolve()
    edge_summary = {
        "b0_energy": describe(edge["b0_energy"]),
        "b1_energy": describe(edge["b1_energy"]),
        "b2_energy": describe(edge["b2_energy"]),
        "b2_vs_b0_delta_ed": describe(edge["delta_b2_vs_b0"]),
        "b2_vs_b0_energy_gain": describe(edge["gain_b2_vs_b0"]),
        "fraction_b2_beats_b0": float(edge["b2_beats_b0"].mean()),
        "b2_vs_b1_delta_ed": describe(edge["delta_b2_vs_b1"]),
        "b2_vs_b1_relative_gain": describe(edge["gain_b2_vs_b1"]),
        "fraction_b2_beats_b1": float(edge["b2_beats_b1"].mean()),
    }
    result = {
        "schema": "tahoe_experiment1_b2_v2_test_result_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "scope": "formal B2-v2 best-checkpoint test evaluation; no fitting or STATE/ST",
        "checkpoint": {
            "path": CHECKPOINT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": checkpoint_sha,
            "format": checkpoint["format"],
            "checkpoint_kind": checkpoint["checkpoint_kind"],
            "epoch": checkpoint["epoch"],
            "global_step": checkpoint["global_step"],
            "best_val_edge_energy": checkpoint["best_val_edge_energy"],
            "fresh_initialization": checkpoint["fresh_initialization"],
            "b2_v1_or_smoke_checkpoint_loaded": checkpoint[
                "b2_v1_or_smoke_checkpoint_loaded"
            ],
            "model_state_sha256_before_evaluation": parameter_sha_before,
            "model_state_sha256_after_evaluation": parameter_sha_after,
            "selected_using_test": False,
        },
        "b2_v2_training_protocol": {
            "path": B2_PROTOCOL_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(B2_PROTOCOL_PATH),
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
        },
        "dataset": {
            "class": "TahoeExperiment1LatentSetDataset",
            "path": DATASET_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": dataset_sha,
            "embedding_transforms": [],
            "raw_signed_latent": True,
        },
        "model": {
            "implementation": MODEL_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "implementation_sha256": sha256_file(MODEL_PATH),
            "prediction": "raw_Zctrl + Delta_pred[:,None,:]",
            "condition_shift_target_loaded_during_test": False,
            "test_mse_computed": False,
        },
        "metric": {
            "implementation": "geomloss.SamplesLoss",
            "configuration": {"loss": "energy", "blur": ENERGY_BLUR},
            "geomloss_version": importlib.metadata.version("geomloss"),
            "source_path": str(metric_source_path),
            "source_sha256": sha256_file(metric_source_path),
            "state_reference_path": str(STATE_MODEL_PATH),
            "state_reference_sha256": sha256_file(STATE_MODEL_PATH),
            "aggregation_implementation": "perturbation_scripts/evaluate_tahoe_experiment1_b0.py",
            "aggregation_implementation_sha256": sha256_file(
                PROJECT_ROOT / "perturbation_scripts" / "evaluate_tahoe_experiment1_b0.py"
            ),
            "evaluation_script": SCRIPT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "evaluation_script_sha256": sha256_file(SCRIPT_PATH),
        },
        "references": {
            "b0": {
                "raw_path": B0_RAW_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "raw_sha256": b0_raw_sha,
                "edge_path": B0_EDGE_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "edge_sha256": sha256_file(B0_EDGE_PATH),
                "result_path": B0_RESULT_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "result_sha256": sha256_file(B0_RESULT_PATH),
                "status": b0_result["status"],
            },
            "b1": {
                "raw_path": B1_RAW_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "raw_sha256": b1_raw_sha,
                "edge_path": B1_EDGE_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "edge_sha256": sha256_file(B1_EDGE_PATH),
                "result_path": B1_RESULT_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "result_sha256": sha256_file(B1_RESULT_PATH),
                "status": b1_result["status"],
            },
        },
        "counts": {
            "test_conditions": len(dataset),
            "test_edges": int(dataset.conditions["edge_id"].nunique()),
            "repeat_count": len(protocol["evaluation_repeat_epochs"]),
            "expected_raw_rows": EXPECTED_RAW_ROWS,
            "observed_raw_rows": len(paired),
            "uniquely_paired_b0_b1_b2_rows": len(paired),
            "plate6_plate14_replicate_groups": replicate_groups,
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
            "delta_b2_vs_b0": "b0_energy - b2_energy",
            "gain_b2_vs_b0": "(b0_energy - b2_energy) / (b0_energy + eps)",
            "delta_b2_vs_b1": "b1_energy - b2_energy",
            "gain_b2_vs_b1": "(b1_energy - b2_energy) / (b1_energy + eps)",
            "eps": EPS,
            "positive_means": "B2 is better",
        },
        "edge_level": edge_summary,
        "stratified_summary": {
            "by_dose": grouped_b1_b2(dose, "dose_uM"),
            "by_drug": grouped_b1_b2(edge, "drug"),
            "by_cell_line": grouped_b1_b2(edge, "cell_line_id"),
        },
        "runtime": runtime,
        "checks": {
            "checkpoint_sha_correct": "pass",
            "checkpoint_epoch_10_best": "pass",
            "checkpoint_selection_on_test": "pass: none",
            "test_conditions_5684": "pass",
            "five_repeats_per_condition": "pass",
            "raw_rows_28420": "pass",
            "b0_b1_b2_unique_pairing_28420": "pass",
            "identical_frozen_sampling": "pass",
            "final_edges_1717": "pass",
            "plate6_plate14_high_dose_groups_397": "pass",
            "energy_all_finite": "pass",
            "latent_transforms": "pass: none",
            "condition_shift_target_used": "pass: no",
            "test_mse_or_fitting": "pass: none",
            "fallback": "pass: none",
            "parameter_update": "pass: none",
            "model_eval_and_inference_mode": "pass",
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
        "blockers": [],
        "warnings": [],
    }
    atomic_write_text(
        HANDOFF_PATH,
        f"""# Tahoe Experiment 1 B2-v2 formal test handoff

- Status: **PASS**
- Checkpoint: epoch **10**, `{checkpoint_sha}`
- Paired condition-repeat rows: **{len(paired):,}**
- Final test edges: **{len(edge):,}**
- B0 / B1 / B2 mean Energy: **{edge_summary['b0_energy']['mean']:.10f} / {edge_summary['b1_energy']['mean']:.10f} / {edge_summary['b2_energy']['mean']:.10f}**
- Fraction B2 beats B0 / B1: **{edge_summary['fraction_b2_beats_b0']:.6f} / {edge_summary['fraction_b2_beats_b1']:.6f}**
- No training, checkpoint search, target loading, test fitting, or STATE/ST was performed.
""",
    )
    atomic_write_json(RESULT_PATH, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
