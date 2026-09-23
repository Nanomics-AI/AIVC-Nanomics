#!/usr/bin/env python3
"""Evaluate frozen ST-A/ST-R best checkpoints on formal raw test repeats only."""

from __future__ import annotations

import argparse
import gc
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

from evaluate_tahoe_experiment1_b0 import (
    ENERGY_BLUR,
    EXPECTED_RAW_ROWS,
    EXPECTED_REPEAT_COUNT,
    EXPECTED_TEST_CONDITIONS,
    PROTOCOL_PATH as EVALUATION_PROTOCOL_PATH,
    assert_without_replacement,
    atomic_write_csv,
    atomic_write_text,
    utc_now,
)
from freeze_tahoe_experiment1_evaluation_sampling import (
    AUDIT_PATH as SAMPLING_AUDIT_PATH,
    index_sha256,
    selected_positions,
)
from run_tahoe_experiment1_st_formal import (
    DATASET_PATH,
    DATASET_SHA256,
    FORMAL_CHECKPOINT_ROOT,
    PROTOCOL_PATH as ST_PROTOCOL_PATH,
    STATE_MODEL_PATH,
    STATE_SHA256,
    TEST_CONDITIONS,
    build_frozen_model,
    ensure_protocol,
    forward_energy,
    model_fingerprint,
    move_batch,
    validate_checkpoint_payload,
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
from train_tahoe_experiment1_b2 import atomic_write_json


SCRIPT_PATH = Path(__file__).resolve()
TASK_PATH = PROJECT_ROOT.parent / "当前任务.txt"
RAW_PATHS = {
    "st-a": RESULTS / "tahoe_experiment1_st_a_test_condition_repeat.csv",
    "st-r": RESULTS / "tahoe_experiment1_st_r_test_condition_repeat.csv",
}
AUDIT_PATH = RESULTS / "tahoe_experiment1_st_test_raw_audit.json"
HANDOFF_PATH = RESULTS / "tahoe_experiment1_st_test_raw_handoff.md"

EXPECTED_TASK_SHA256 = "aedc6c34a38da5564e7b3b203f404e849847d69c5cc65bde68c9af8935b331a1"
EXPECTED_ST_PROTOCOL_SHA256 = "3630a8e527aac812b1bb182c35d310b71122dca6257b63f109ac0ae0ad5b581b"
EXPECTED_EVALUATION_PROTOCOL_SHA256 = "66ddf436acb54a9c662553f48df3af382d57b947a3a796173ee5cc6fb30555a1"
EXPECTED_SAMPLING_AUDIT_SHA256 = "09f675d15942ffc3bf06465b37f20b94809e34593c28c27fd67f5891505fbbbd"
HISTORICAL_EVALUATION_DATASET_SHA256 = "9f482ef606d91d10eaf28511bf6793d69832f08cac7d7bb2249210f56d49ae4d"

CHECKPOINTS = {
    "st-a": {
        "path": FORMAL_CHECKPOINT_ROOT / "st-a" / "best.pt",
        "sha256": "9bd0f2719f42dac4fa5a9aabc4fd2bb242a442fb652ae90a8ee52fa54ca03652",
        "epoch": 28,
        "best_val_edge_energy": 0.021047738211574886,
        "training_result": RESULTS / "tahoe_experiment1_st_st-a_formal_training_result.json",
    },
    "st-r": {
        "path": FORMAL_CHECKPOINT_ROOT / "st-r" / "best.pt",
        "sha256": "357fd6ec00473f5a6f2b82f5693c58365ea08f0af30500d092d354d0fbefd3ab",
        "epoch": 26,
        "best_val_edge_energy": 0.027806505977352507,
        "training_result": RESULTS / "tahoe_experiment1_st_st-r_formal_training_result.json",
    },
}

REFERENCES = {
    "B0": {
        "raw": RESULTS / "tahoe_experiment1_b0_identity_condition_repeat.csv",
        "result": RESULTS / "tahoe_experiment1_b0_identity_result.json",
        "output_key": "condition_repeat_csv",
    },
    "B1": {
        "raw": RESULTS / "tahoe_experiment1_b1_mean_shift_condition_repeat_paired.csv",
        "result": RESULTS / "tahoe_experiment1_b1_mean_shift_result.json",
        "output_key": "condition_repeat_paired_csv",
    },
    "B2": {
        "raw": RESULTS / "tahoe_experiment1_b2_v2_test_condition_repeat_paired.csv",
        "result": RESULTS / "tahoe_experiment1_b2_v2_test_result.json",
        "output_key": "condition_repeat_paired_csv",
    },
}

KEY_COLUMNS = [
    "condition_id",
    "edge_id",
    "cell_line_id",
    "drug",
    "dose_uM",
    "plate",
    "repeat_epoch",
]
CSV_COLUMNS = [*KEY_COLUMNS, "energy_distance"]


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return "../" + resolved.relative_to(PROJECT_ROOT.parent.resolve()).as_posix()


def canonical_keys(frame: pd.DataFrame) -> pd.DataFrame:
    if not set(KEY_COLUMNS) <= set(frame.columns):
        raise AssertionError("A raw result is missing condition/repeat key columns")
    return frame[KEY_COLUMNS].sort_values(
        ["condition_id", "repeat_epoch"], kind="stable"
    ).reset_index(drop=True)


def load_sampling_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(EVALUATION_PROTOCOL_PATH) != EXPECTED_EVALUATION_PROTOCOL_SHA256:
        raise AssertionError("Frozen evaluation sampling protocol SHA-256 changed")
    if sha256_file(SAMPLING_AUDIT_PATH) != EXPECTED_SAMPLING_AUDIT_SHA256:
        raise AssertionError("Frozen sampling reproducibility audit SHA-256 changed")
    protocol = json.loads(EVALUATION_PROTOCOL_PATH.read_text(encoding="utf-8"))
    expected = {
        "status": "frozen",
        "S": SET_SIZE,
        "evaluation_repeat_count": EXPECTED_REPEAT_COUNT,
        "evaluation_base_seed": 42,
        "evaluation_repeat_epochs": [0, 1, 2, 3, 4],
        "shuffle": False,
        "drop_last": False,
        "all-models-share-identical-evaluation-indices": True,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise AssertionError(f"Frozen evaluation sampling mismatch: {key}")
    if (
        protocol.get("tahoe_experiment1_latent_data.py_sha256")
        != HISTORICAL_EVALUATION_DATASET_SHA256
        or protocol.get("dataset_sampling_rule", {}).get("version")
        != "tahoe_experiment1_set_v1"
        or protocol.get("sampling_method", {}).get("within_set_replacement") is not False
    ):
        raise AssertionError("Frozen evaluation sampling provenance changed")

    method_source = inspect.getsource(TahoeExperiment1LatentSetDataset._sample_range)
    required_terms = (
        "tahoe_experiment1_set_v1",
        "seed={self.seed}",
        "epoch={self.epoch}",
        "pair_id={pair_id}",
        "side={side}",
        "np.random.PCG64",
        "replace=False",
    )
    if any(term not in method_source for term in required_terms):
        raise AssertionError("Current Dataset sampling rule differs from the frozen rule")
    if sha256_file(DATASET_PATH) != DATASET_SHA256:
        raise AssertionError("Current v2 Dataset SHA-256 changed")

    sampling_audit = json.loads(SAMPLING_AUDIT_PATH.read_text(encoding="utf-8"))
    test_audit = sampling_audit.get("results", {}).get("test", {})
    if (
        sampling_audit.get("status") != "pass"
        or sampling_audit.get("protocol", {}).get("sha256")
        != EXPECTED_EVALUATION_PROTOCOL_SHA256
        or test_audit.get("all_repeat_indices_exactly_reproduced") is not True
        or test_audit.get("repeat_epochs") != [0, 1, 2, 3, 4]
    ):
        raise AssertionError("Frozen sampling reproducibility audit is incompatible")
    return protocol, sampling_audit


def expected_spotchecks(sampling_audit: dict[str, Any]) -> dict[int, str]:
    return {
        int(row["repeat_epoch"]): str(row["independent_dataset_a_indices_sha256"])
        for row in sampling_audit["results"]["test"]["repeat_results"]
    }


def load_reference_keys(
    sampling_audit: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    expected_spots = expected_spotchecks(sampling_audit)
    reference_keys: pd.DataFrame | None = None
    records: dict[str, Any] = {}
    for name, paths in REFERENCES.items():
        result = json.loads(paths["result"].read_text(encoding="utf-8"))
        raw_sha = sha256_file(paths["raw"])
        declared = result.get("outputs", {}).get(paths["output_key"], {})
        if (
            result.get("status") != "pass"
            or result.get("evaluation_protocol", {}).get("sha256")
            != EXPECTED_EVALUATION_PROTOCOL_SHA256
            or result.get("dataset", {}).get("sha256")
            != HISTORICAL_EVALUATION_DATASET_SHA256
            or result.get("counts", {}).get("observed_raw_rows") != EXPECTED_RAW_ROWS
            or declared.get("sha256") != raw_sha
        ):
            raise AssertionError(f"Frozen {name} raw reference provenance is incompatible")
        raw = pd.read_csv(paths["raw"], encoding="utf-8-sig", keep_default_na=False)
        keys = canonical_keys(raw)
        if len(keys) != EXPECTED_RAW_ROWS or keys.duplicated(
            ["condition_id", "repeat_epoch"]
        ).any():
            raise AssertionError(f"Frozen {name} raw reference keys are invalid")
        if reference_keys is None:
            reference_keys = keys
        elif not keys.equals(reference_keys):
            raise AssertionError(f"Frozen {name} row keys differ from B0")

        spotchecks = result.get("runtime", {}).get("sampling_index_spotchecks")
        if name in {"B1", "B2"}:
            observed_spots = {
                int(row["repeat_epoch"]): str(row["indices_sha256"])
                for row in spotchecks or []
            }
            if observed_spots != expected_spots:
                raise AssertionError(f"Frozen {name} sampling spotchecks changed")
        records[name] = {
            "raw_path": relative(paths["raw"]),
            "raw_sha256": raw_sha,
            "result_path": relative(paths["result"]),
            "result_sha256": sha256_file(paths["result"]),
            "rows": len(raw),
            "row_keys_match_B0": True,
            "evaluation_protocol_sha256": EXPECTED_EVALUATION_PROTOCOL_SHA256,
            "sampling_spotchecks_match_frozen_audit": (
                True if name in {"B1", "B2"} else "not stored; protocol and row keys verified"
            ),
        }
    assert reference_keys is not None
    return reference_keys, records


def load_checkpoint(
    variant: str, device: torch.device, protocol_sha: str
) -> tuple[torch.nn.Module, dict[str, Any]]:
    spec = CHECKPOINTS[variant]
    checkpoint_sha = sha256_file(spec["path"])
    if checkpoint_sha != spec["sha256"]:
        raise AssertionError(f"{variant} best checkpoint SHA-256 changed")
    payload = torch.load(spec["path"], map_location="cpu", weights_only=False)
    validate_checkpoint_payload(
        payload, variant=variant, kind="best", protocol_sha256=protocol_sha
    )
    if (
        payload.get("checkpoint_kind") != "best"
        or payload.get("variant") != variant
        or payload.get("epoch") != spec["epoch"]
        or payload.get("best_epoch") != spec["epoch"]
        or payload.get("best_val_edge_energy") != spec["best_val_edge_energy"]
        or payload.get("current_val_edge_energy") != spec["best_val_edge_energy"]
    ):
        raise AssertionError(f"{variant} checkpoint is not the frozen best epoch")

    training_result = json.loads(spec["training_result"].read_text(encoding="utf-8"))
    if (
        training_result.get("status") != "pass"
        or training_result.get("variant") != variant
        or training_result.get("best_epoch") != spec["epoch"]
        or training_result.get("best_val_edge_energy") != spec["best_val_edge_energy"]
        or training_result.get("checkpoints", {}).get("best", {}).get("sha256")
        != checkpoint_sha
        or training_result.get("scope") != "formal train+val only; test not constructed"
    ):
        raise AssertionError(f"{variant} training result does not identify the frozen best")

    model, model_kwargs = build_frozen_model(variant)
    model.load_state_dict(payload["model_state"], strict=True)
    before = model_fingerprint(model)
    checkpoint_record = {
        "path": relative(spec["path"]),
        "sha256": checkpoint_sha,
        "checkpoint_kind": "best",
        "variant": variant,
        "epoch": int(payload["epoch"]),
        "best_epoch": int(payload["best_epoch"]),
        "best_val_edge_energy": float(payload["best_val_edge_energy"]),
        "global_step": int(payload["global_step"]),
        "formal_protocol_sha256": str(payload["formal_protocol_sha256"]),
        "model_kwargs": model_kwargs,
        "model_state_sha256_before_evaluation": before,
        "selected_using_test": False,
        "last_checkpoint_loaded": False,
        "training_result_path": relative(spec["training_result"]),
        "training_result_sha256": sha256_file(spec["training_result"]),
    }
    del payload
    gc.collect()
    model.to(device)
    model.eval()
    if model.training or any(module.training for module in model.modules()):
        raise AssertionError(f"{variant} model.eval() did not cover all modules")
    return model, checkpoint_record


def update_index_digest(
    digest: Any,
    condition_ids: list[str],
    source: np.ndarray,
    target: np.ndarray,
) -> None:
    for condition_id, control_indices, treated_indices in zip(
        condition_ids, source, target, strict=True
    ):
        digest.update(condition_id.encode("utf-8"))
        digest.update(b"\0control\0")
        digest.update(np.asarray(control_indices, dtype="<i8").tobytes())
        digest.update(b"\0treated\0")
        digest.update(np.asarray(treated_indices, dtype="<i8").tobytes())


def evaluate_variant(
    variant: str,
    model: torch.nn.Module,
    dataset: TahoeExperiment1LatentSetDataset,
    protocol: dict[str, Any],
    sampling_audit: dict[str, Any],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = dataset.conditions.set_index("pair_id", verify_integrity=True)
    expected_order = dataset.conditions["pair_id"].astype(str).tolist()
    positions = selected_positions(len(dataset))
    spot_condition_ids = dataset.conditions.iloc[positions]["pair_id"].astype(str).tolist()
    expected_spots = expected_spotchecks(sampling_audit)
    repeat_fingerprints: dict[str, str] = {}
    spotcheck_records: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    forward_audit: dict[str, Any] = {}
    prediction_min = float("inf")
    prediction_max = float("-inf")
    batches = 0
    started = time.perf_counter()

    model.eval()
    with torch.inference_mode():
        for repeat_epoch in protocol["evaluation_repeat_epochs"]:
            dataset.set_epoch(int(repeat_epoch))
            loader = make_dataloader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                seed=int(protocol["evaluation_base_seed"]),
                drop_last=False,
                num_workers=num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=False,
                prefetch_factor=2 if num_workers else None,
            )
            if loader.drop_last or type(loader.sampler).__name__ != "SequentialSampler":
                raise AssertionError("Test DataLoader shuffled or dropped conditions")
            observed_condition_ids: list[str] = []
            sampled_spots: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            digest = hashlib.sha256()

            for batch in loader:
                condition_ids = [str(value) for value in batch["condition_id"]]
                observed_condition_ids.extend(condition_ids)
                if set(batch["split"]) != {"test"}:
                    raise AssertionError("A non-test condition entered ST evaluation")
                assert_without_replacement(batch["source_embedding_index"], "control")
                assert_without_replacement(batch["target_embedding_index"], "treated")
                source = batch["source_embedding_index"].numpy()
                target = batch["target_embedding_index"].numpy()
                update_index_digest(digest, condition_ids, source, target)
                for offset, condition_id in enumerate(condition_ids):
                    if condition_id in spot_condition_ids:
                        sampled_spots[condition_id] = (
                            source[offset].copy(),
                            target[offset].copy(),
                        )

                moved = move_batch(batch, device)
                prediction, loss, per_set = forward_energy(
                    model,
                    moved,
                    forward_audit=forward_audit if not forward_audit else None,
                )
                if (
                    prediction.shape != (len(condition_ids), SET_SIZE, LATENT_DIM)
                    or not torch.isfinite(prediction).all()
                    or not torch.isfinite(loss)
                    or per_set.shape != (len(condition_ids),)
                    or not torch.isfinite(per_set).all()
                ):
                    raise AssertionError(f"{variant} prediction or Energy is invalid")
                if batches == 0:
                    prediction_min = float(prediction.float().min())
                    prediction_max = float(prediction.float().max())

                for condition_id, energy in zip(
                    condition_ids, per_set.detach().cpu().numpy(), strict=True
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
                            "energy_distance": float(energy),
                            "_cache_condition_index": int(row["cache_condition_index"]),
                        }
                    )
                batches += 1

            if observed_condition_ids != expected_order:
                raise AssertionError(f"{variant}/repeat={repeat_epoch} shuffled or dropped rows")
            repeat_fingerprints[str(repeat_epoch)] = digest.hexdigest()
            if set(sampled_spots) != set(spot_condition_ids):
                raise AssertionError("Frozen sampling spotcheck conditions were not observed")
            spot_source = np.stack(
                [sampled_spots[condition_id][0] for condition_id in spot_condition_ids]
            )
            spot_target = np.stack(
                [sampled_spots[condition_id][1] for condition_id in spot_condition_ids]
            )
            observed_spot = index_sha256(spot_condition_ids, spot_source, spot_target)
            if observed_spot != expected_spots[int(repeat_epoch)]:
                raise AssertionError("ST sampled indices differ from frozen evaluation audit")
            spotcheck_records.append(
                {
                    "repeat_epoch": int(repeat_epoch),
                    "condition_ids": spot_condition_ids,
                    "indices_sha256": observed_spot,
                    "matches_frozen_sampling_audit": True,
                }
            )
            print(
                f"{variant} repeat_epoch={repeat_epoch} "
                f"conditions={len(observed_condition_ids)}/{len(dataset)}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    raw = pd.DataFrame.from_records(records).sort_values(
        ["_cache_condition_index", "repeat_epoch"], kind="stable"
    )
    raw = raw[CSV_COLUMNS].reset_index(drop=True)
    if not prediction_min < 0 < prediction_max:
        raise AssertionError(f"{variant} first-batch prediction did not preserve signed values")
    combined = hashlib.sha256()
    for repeat_epoch in protocol["evaluation_repeat_epochs"]:
        combined.update(str(repeat_epoch).encode("ascii"))
        combined.update(bytes.fromhex(repeat_fingerprints[str(repeat_epoch)]))
    return raw, {
        "device": str(device),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "batches": batches,
        "elapsed_seconds": elapsed,
        "condition_repeat_rows_per_second": len(raw) / elapsed,
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "prediction_min_first_batch": prediction_min,
        "prediction_max_first_batch": prediction_max,
        "forward_precision_audit": forward_audit,
        "full_sampling_index_sha256_by_repeat": repeat_fingerprints,
        "combined_full_sampling_index_sha256": combined.hexdigest(),
        "sampling_index_spotchecks": spotcheck_records,
    }


def validate_raw(raw: pd.DataFrame, protocol: dict[str, Any], variant: str) -> dict[str, Any]:
    repeat_counts = raw.groupby("condition_id", sort=False)["repeat_epoch"].agg(
        ["size", "nunique"]
    )
    observed_epochs = raw.groupby("condition_id", sort=False)["repeat_epoch"].agg(set)
    expected_epochs = set(protocol["evaluation_repeat_epochs"])
    energy = raw["energy_distance"].to_numpy(np.float64)
    if (
        len(raw) != EXPECTED_RAW_ROWS
        or raw["condition_id"].nunique() != EXPECTED_TEST_CONDITIONS
        or raw.duplicated(["condition_id", "repeat_epoch"]).any()
        or not ((repeat_counts["size"] == 5) & (repeat_counts["nunique"] == 5)).all()
        or not observed_epochs.map(lambda value: value == expected_epochs).all()
        or not np.isfinite(energy).all()
    ):
        raise AssertionError(f"{variant} raw condition/repeat validation failed")
    return {
        "rows": len(raw),
        "unique_conditions": int(raw["condition_id"].nunique()),
        "unique_edges_without_aggregation": int(raw["edge_id"].nunique()),
        "repeats_per_condition_min": int(repeat_counts["size"].min()),
        "repeats_per_condition_max": int(repeat_counts["size"].max()),
        "repeat_epochs": sorted(int(value) for value in raw["repeat_epoch"].unique()),
        "duplicate_condition_repeat_keys": int(
            raw.duplicated(["condition_id", "repeat_epoch"]).sum()
        ),
        "energy_finite": True,
        "mean_raw_condition_repeat_energy": float(energy.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers cannot be negative")
    if not torch.cuda.is_available():
        raise RuntimeError("Formal ST raw evaluation requires CUDA")
    output_paths = [*RAW_PATHS.values(), AUDIT_PATH, HANDOFF_PATH]
    if any(path.exists() for path in output_paths):
        raise FileExistsError("A formal ST raw test output already exists; refusing to overwrite")
    if sha256_file(TASK_PATH) != EXPECTED_TASK_SHA256:
        raise AssertionError("当前任务.txt changed before formal ST raw evaluation")

    st_protocol, st_protocol_sha = ensure_protocol(create=False)
    if st_protocol_sha != EXPECTED_ST_PROTOCOL_SHA256:
        raise AssertionError("Frozen ST v2 protocol SHA-256 changed")
    protocol, sampling_audit = load_sampling_contract()
    reference_keys, references = load_reference_keys(sampling_audit)
    reference_hashes_before = {
        name: sha256_file(paths["raw"]) for name, paths in REFERENCES.items()
    }

    dataset = TahoeExperiment1LatentSetDataset(
        split="test",
        seed=int(protocol["evaluation_base_seed"]),
        epoch=int(protocol["evaluation_repeat_epochs"][0]),
    )
    if (
        len(dataset) != TEST_CONDITIONS
        or TEST_CONDITIONS != EXPECTED_TEST_CONDITIONS
        or not dataset.conditions["split"].eq("test").all()
        or dataset.embedding_transforms
        or dataset.edge_overlap != {"train_val": 0, "train_test": 0, "val_test": 0}
    ):
        raise AssertionError("Formal test-only raw-latent Dataset contract changed")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    variant_results: dict[str, pd.DataFrame] = {}
    checkpoints: dict[str, Any] = {}
    runtimes: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    for variant in ("st-a", "st-r"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model, checkpoint = load_checkpoint(variant, device, st_protocol_sha)
        raw, runtime = evaluate_variant(
            variant,
            model,
            dataset,
            protocol,
            sampling_audit,
            args.batch_size,
            args.num_workers,
            device,
        )
        checkpoint["model_state_sha256_after_evaluation"] = model_fingerprint(model)
        checkpoint["parameters_unchanged"] = (
            checkpoint["model_state_sha256_after_evaluation"]
            == checkpoint["model_state_sha256_before_evaluation"]
        )
        if not checkpoint["parameters_unchanged"]:
            raise AssertionError(f"{variant} parameters changed during inference")
        variant_results[variant] = raw
        checkpoints[variant] = checkpoint
        runtimes[variant] = runtime
        summaries[variant] = validate_raw(raw, protocol, variant)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    a_keys = canonical_keys(variant_results["st-a"])
    r_keys = canonical_keys(variant_results["st-r"])
    if not a_keys.equals(r_keys) or not a_keys.equals(reference_keys):
        raise AssertionError("ST-A/ST-R/B0/B1/B2 row keys are not exactly identical")
    a_fingerprints = runtimes["st-a"]["full_sampling_index_sha256_by_repeat"]
    r_fingerprints = runtimes["st-r"]["full_sampling_index_sha256_by_repeat"]
    if a_fingerprints != r_fingerprints:
        raise AssertionError("ST-A/ST-R full sampled indices differ")
    reference_hashes_after = {
        name: sha256_file(paths["raw"]) for name, paths in REFERENCES.items()
    }
    if reference_hashes_after != reference_hashes_before:
        raise AssertionError("A frozen B0/B1/B2 raw artifact changed during ST evaluation")
    if sha256_file(TASK_PATH) != EXPECTED_TASK_SHA256:
        raise AssertionError("当前任务.txt changed during formal ST raw evaluation")

    for variant in ("st-a", "st-r"):
        atomic_write_csv(RAW_PATHS[variant], variant_results[variant])
        written = pd.read_csv(
            RAW_PATHS[variant], encoding="utf-8-sig", keep_default_na=False
        )
        validate_raw(written, protocol, variant)
        if not canonical_keys(written).equals(reference_keys):
            raise AssertionError(f"Written {variant} CSV row keys changed")

    warnings = [
        "The frozen evaluation protocol records the historical whole-file Dataset SHA "
        f"{HISTORICAL_EVALUATION_DATASET_SHA256}; formal ST v2 freezes the current Dataset SHA "
        f"{DATASET_SHA256}. The sampler version/rule and every frozen 3-condition x 5-repeat "
        "index spotcheck match exactly; B0/B1/B2 row keys also match. No protocol file was modified."
    ]
    audit = {
        "schema": "tahoe_experiment1_st_test_raw_audit_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "scope": "frozen ST-A/ST-R best-checkpoint raw test Energy only; no aggregation or baseline recomputation",
        "task": {"path": relative(TASK_PATH), "sha256": EXPECTED_TASK_SHA256},
        "st_training_protocol": {
            "path": relative(ST_PROTOCOL_PATH),
            "sha256": st_protocol_sha,
            "schema": st_protocol["schema"],
        },
        "evaluation_sampling_protocol": {
            "path": relative(EVALUATION_PROTOCOL_PATH),
            "sha256": EXPECTED_EVALUATION_PROTOCOL_SHA256,
            "S": protocol["S"],
            "base_seed": protocol["evaluation_base_seed"],
            "repeat_count": protocol["evaluation_repeat_count"],
            "repeat_epochs": protocol["evaluation_repeat_epochs"],
            "shuffle": protocol["shuffle"],
            "drop_last": protocol["drop_last"],
            "sampling_rule": protocol["dataset_sampling_rule"],
            "reproducibility_audit": {
                "path": relative(SAMPLING_AUDIT_PATH),
                "sha256": EXPECTED_SAMPLING_AUDIT_SHA256,
            },
        },
        "dataset": {
            "class": "TahoeExperiment1LatentSetDataset",
            "path": relative(DATASET_PATH),
            "current_sha256": DATASET_SHA256,
            "historical_evaluation_protocol_sha256": HISTORICAL_EVALUATION_DATASET_SHA256,
            "split": "test",
            "conditions": len(dataset),
            "train_or_val_dataset_constructed": False,
            "embedding_transforms": [],
            "raw_signed_latent": True,
        },
        "checkpoints": checkpoints,
        "model_contract": {
            "st-a": "Zpred = project_out(ST_hidden)",
            "st-r": "Zpred = raw_Zctrl + project_out(ST_hidden)",
            "model_eval": True,
            "torch_inference_mode": True,
            "final_activation": "identity",
            "prohibited_latent_transforms_applied": [],
            "state_model_path": relative(STATE_MODEL_PATH),
            "state_model_sha256": STATE_SHA256,
        },
        "metric": {
            "implementation": "StateTransitionPerturbationModel._compute_distribution_loss -> geomloss.SamplesLoss",
            "configuration": {"loss": "energy", "blur": ENERGY_BLUR},
            "geomloss_version": importlib.metadata.version("geomloss"),
            "energy_inputs_dtype": "float32",
            "model_forward_autocast": "bfloat16",
            "forward_implementation": relative(
                PROJECT_ROOT / "perturbation_scripts" / "run_tahoe_experiment1_st_formal.py"
            ),
            "forward_implementation_sha256": sha256_file(
                PROJECT_ROOT / "perturbation_scripts" / "run_tahoe_experiment1_st_formal.py"
            ),
            "evaluation_script": relative(SCRIPT_PATH),
            "evaluation_script_sha256": sha256_file(SCRIPT_PATH),
        },
        "raw_results": summaries,
        "sampling": {
            "st_a_st_r_full_indices_exact_match": True,
            "full_sampling_index_sha256_by_repeat": a_fingerprints,
            "combined_full_sampling_index_sha256": runtimes["st-a"][
                "combined_full_sampling_index_sha256"
            ],
            "frozen_spotcheck_fingerprints_match": True,
            "b0_b1_b2_row_keys_exact_match": True,
            "b1_b2_stored_sampling_spotchecks_match_frozen_audit": True,
            "b0_sampling_evidence": "same frozen protocol plus exact raw row keys; B0 did not store index fingerprints",
        },
        "references_read_only": references,
        "runtime": runtimes,
        "execution": {
            "single_process": True,
            "single_gpu": True,
            "ddp": False,
            "optimizer_constructed": False,
            "backward_called": False,
            "parameter_update": False,
            "test_time_fitting_or_calibration": False,
            "test_selected_epoch": False,
            "edge_aggregation_run": False,
            "b0_b1_b2_energy_recomputed": False,
        },
        "checks": {
            "st_a_checkpoint_sha_and_epoch": "pass",
            "st_r_checkpoint_sha_and_epoch": "pass",
            "st_a_rows_28420": "pass",
            "st_r_rows_28420": "pass",
            "unique_test_conditions_5684": "pass",
            "five_repeats_each": "pass",
            "repeat_epochs_0_to_4": "pass",
            "duplicate_condition_repeat_keys": "pass: none",
            "all_energy_finite": "pass",
            "st_a_st_r_row_keys_exact_match": "pass",
            "st_a_st_r_sampling_indices_exact_match": "pass",
            "frozen_sampling_basis_match": "pass",
            "test_dataset_only": "pass",
            "training_or_parameter_update": "pass: none",
            "final_edge_aggregation": "pass: not run",
        },
        "outputs": {
            "st_a_condition_repeat_csv": {
                "path": relative(RAW_PATHS["st-a"]),
                "sha256": sha256_file(RAW_PATHS["st-a"]),
            },
            "st_r_condition_repeat_csv": {
                "path": relative(RAW_PATHS["st-r"]),
                "sha256": sha256_file(RAW_PATHS["st-r"]),
            },
            "audit_json": relative(AUDIT_PATH),
            "handoff_md": relative(HANDOFF_PATH),
        },
        "warnings": warnings,
        "blockers": [],
    }
    atomic_write_json(AUDIT_PATH, audit)
    handoff = f"""# Experiment 1 ST raw test handoff

Status: **PASS**

- ST-A: {summaries['st-a']['rows']:,} condition-repeat rows; mean raw Energy = {summaries['st-a']['mean_raw_condition_repeat_energy']:.12g}
- ST-R: {summaries['st-r']['rows']:,} condition-repeat rows; mean raw Energy = {summaries['st-r']['mean_raw_condition_repeat_energy']:.12g}
- Test conditions: {summaries['st-a']['unique_conditions']:,}; repeats per condition: 5; repeat epochs: `[0,1,2,3,4]`
- ST-A/ST-R row keys and full sampled cell-index fingerprints match exactly.
- Frozen sampling spotchecks and B0/B1/B2 raw row keys match.
- Both best checkpoint SHA-256 values and epochs were verified; parameters were unchanged.
- No train/val fitting, optimizer, backward, baseline recomputation, or edge aggregation was run.

The audit warning about the historical versus current whole-file Dataset SHA is recorded in `{relative(AUDIT_PATH)}`; exact sampler fingerprints pass, so it is not a blocker.
"""
    atomic_write_text(HANDOFF_PATH, handoff)
    print(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
