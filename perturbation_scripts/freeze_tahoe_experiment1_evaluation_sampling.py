#!/usr/bin/env python3
"""Freeze and audit the Experiment 1 evaluation sampling protocol only."""

from __future__ import annotations

import hashlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Subset

from tahoe_experiment1_latent_data import (
    PROJECT_ROOT,
    RESULTS,
    SET_SIZE,
    TahoeExperiment1LatentSetDataset,
    make_dataloader,
    sha256_file,
)


PROTOCOL_PATH = RESULTS / "tahoe_experiment1_evaluation_sampling_protocol.json"
AUDIT_PATH = RESULTS / "tahoe_experiment1_evaluation_sampling_reproducibility_audit.json"
DATASET_PATH = PROJECT_ROOT / "perturbation_scripts" / "tahoe_experiment1_latent_data.py"
REPEAT_COUNT = 5
BASE_SEED = 42
REPEAT_EPOCHS = [0, 1, 2, 3, 4]
AUDIT_POSITIONS_PER_SPLIT = 3
AUDIT_BATCH_SIZE = 2
SAMPLING_VERSION = "tahoe_experiment1_set_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def build_protocol(frozen_at_utc: str) -> dict[str, Any]:
    if SET_SIZE != 256 or REPEAT_COUNT != len(REPEAT_EPOCHS):
        raise AssertionError("Evaluation sampling constants are inconsistent")
    method_source = inspect.getsource(TahoeExperiment1LatentSetDataset._sample_range)
    required_source_terms = (
        SAMPLING_VERSION,
        "seed={self.seed}",
        "epoch={self.epoch}",
        "pair_id={pair_id}",
        "side={side}",
        "np.random.PCG64",
        "replace=False",
    )
    if any(term not in method_source for term in required_source_terms):
        raise AssertionError("Existing Dataset sampling implementation changed")
    return {
        "schema": "tahoe_experiment1_evaluation_sampling_protocol_v1",
        "status": "frozen",
        "frozen_at_utc": frozen_at_utc,
        "scope": ["val", "test"],
        "S": SET_SIZE,
        "evaluation_repeat_count": REPEAT_COUNT,
        "evaluation_base_seed": BASE_SEED,
        "evaluation_repeat_epochs": REPEAT_EPOCHS,
        "sampling_method": {
            "name": "deterministic_repeated_subsampling_without_replacement",
            "within_set_replacement": False,
            "cross_repeat_overlap_allowed": True,
            "large_cell_index_manifest_materialized": False,
        },
        "dataset_sampling_rule": {
            "class": "TahoeExperiment1LatentSetDataset",
            "version": SAMPLING_VERSION,
            "identity_fields": ["seed", "epoch", "pair_id", "side"],
            "derivation": "sha256(version|seed|epoch|pair_id|side)[:8] big-endian -> PCG64 -> choice(size=256, replace=False)",
            "implementation": DATASET_PATH.relative_to(PROJECT_ROOT).as_posix(),
        },
        "tahoe_experiment1_latent_data.py_sha256": sha256_file(DATASET_PATH),
        "shuffle": False,
        "drop_last": False,
        "all-models-share-identical-evaluation-indices": True,
        "models": ["B0", "B1", "B2", "ST-A", "ST-R"],
        "model_identity_is_not_part_of_sampling_key": True,
        "training_evaluation_decoupling": {
            "evaluation_epochs_are_protocol_repeat_ids": True,
            "evaluation_epochs_are_not_training_epochs": True,
            "training_may_call_dataset_set_epoch_training_epoch": True,
        },
        "experiment0_sampling_path_used": False,
        "experiment0_BASE_SEEDS_used": False,
    }


def freeze_protocol() -> dict[str, Any]:
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        expected = build_protocol(str(existing.get("frozen_at_utc", "")))
        if existing != expected:
            raise AssertionError("Frozen evaluation protocol differs from current code/constants")
        return existing
    protocol = build_protocol(utc_now())
    atomic_write_json(PROTOCOL_PATH, protocol)
    return protocol


def selected_positions(length: int) -> list[int]:
    if length < AUDIT_POSITIONS_PER_SPLIT:
        raise AssertionError("Evaluation split is too small for reproducibility audit")
    return [0, length // 2, length - 1]


def collect_indices(
    dataset: TahoeExperiment1LatentSetDataset,
    positions: list[int],
) -> dict[str, Any]:
    loader = make_dataloader(
        Subset(dataset, positions),
        batch_size=AUDIT_BATCH_SIZE,
        shuffle=False,
        seed=BASE_SEED,
        drop_last=False,
    )
    condition_ids: list[str] = []
    source: list[np.ndarray] = []
    target: list[np.ndarray] = []
    batch_sizes: list[int] = []
    for batch in loader:
        batch_sizes.append(len(batch["condition_id"]))
        condition_ids.extend(batch["condition_id"])
        source.append(batch["source_embedding_index"].numpy())
        target.append(batch["target_embedding_index"].numpy())
    return {
        "condition_ids": condition_ids,
        "source": np.concatenate(source),
        "target": np.concatenate(target),
        "batch_sizes": batch_sizes,
    }


def index_sha256(condition_ids: list[str], source: np.ndarray, target: np.ndarray) -> str:
    digest = hashlib.sha256()
    for condition_id, control_indices, treated_indices in zip(
        condition_ids, source, target, strict=True
    ):
        digest.update(condition_id.encode("utf-8"))
        digest.update(b"\0control\0")
        digest.update(np.asarray(control_indices, dtype="<i8").tobytes())
        digest.update(b"\0treated\0")
        digest.update(np.asarray(treated_indices, dtype="<i8").tobytes())
    return digest.hexdigest()


def audit_split(split: str) -> dict[str, Any]:
    first = TahoeExperiment1LatentSetDataset(split=split, seed=BASE_SEED, epoch=0)
    second = TahoeExperiment1LatentSetDataset(split=split, seed=BASE_SEED, epoch=0)
    positions = selected_positions(len(first))
    expected_condition_ids = first.conditions.iloc[positions]["pair_id"].astype(str).tolist()
    if expected_condition_ids != second.conditions.iloc[positions]["pair_id"].astype(str).tolist():
        raise AssertionError(f"{split} independent Dataset condition order changed")

    repeat_rows = []
    first_repeat_sets: dict[str, dict[str, list[str]]] = {
        condition_id: {"control": [], "treated": []}
        for condition_id in expected_condition_ids
    }
    for repeat_epoch in REPEAT_EPOCHS:
        first.set_epoch(repeat_epoch)
        second.set_epoch(repeat_epoch)
        left = collect_indices(first, positions)
        right = collect_indices(second, positions)
        if left["condition_ids"] != expected_condition_ids or right["condition_ids"] != expected_condition_ids:
            raise AssertionError(f"{split}/epoch{repeat_epoch} was shuffled or dropped")
        if left["batch_sizes"] != [2, 1] or right["batch_sizes"] != [2, 1]:
            raise AssertionError(f"{split}/epoch{repeat_epoch} drop_last=False contract failed")
        for side in ("source", "target"):
            if not np.array_equal(left[side], right[side]):
                raise AssertionError(
                    f"{split}/epoch{repeat_epoch}/{side} differs across Dataset constructions"
                )
            if any(len(np.unique(indices)) != SET_SIZE for indices in left[side]):
                raise AssertionError(f"{split}/epoch{repeat_epoch}/{side} sampled with replacement")
        for condition_id, source, target in zip(
            expected_condition_ids, left["source"], left["target"], strict=True
        ):
            first_repeat_sets[condition_id]["control"].append(
                hashlib.sha256(np.asarray(source, dtype="<i8").tobytes()).hexdigest()
            )
            first_repeat_sets[condition_id]["treated"].append(
                hashlib.sha256(np.asarray(target, dtype="<i8").tobytes()).hexdigest()
            )
        left_sha = index_sha256(left["condition_ids"], left["source"], left["target"])
        right_sha = index_sha256(right["condition_ids"], right["source"], right["target"])
        if left_sha != right_sha:
            raise AssertionError(f"{split}/epoch{repeat_epoch} index fingerprints differ")
        repeat_rows.append(
            {
                "repeat_epoch": repeat_epoch,
                "independent_dataset_a_indices_sha256": left_sha,
                "independent_dataset_b_indices_sha256": right_sha,
                "exact_match": True,
                "condition_count": len(expected_condition_ids),
                "control_set_size": SET_SIZE,
                "treated_set_size": SET_SIZE,
                "within_set_replacement": False,
                "batch_sizes_with_drop_last_false": left["batch_sizes"],
            }
        )

    return {
        "split": split,
        "selection": "positions [0, floor(N/2), N-1] in frozen split order",
        "positions": positions,
        "condition_ids": expected_condition_ids,
        "independent_dataset_constructions": 2,
        "repeat_epochs": REPEAT_EPOCHS,
        "all_repeat_indices_exactly_reproduced": True,
        "shuffle_false_verified": True,
        "drop_last_false_verified": True,
        "repeat_results": repeat_rows,
        "distinct_repeat_set_fingerprints": {
            condition_id: {
                side: len(set(fingerprints))
                for side, fingerprints in sides.items()
            }
            for condition_id, sides in first_repeat_sets.items()
        },
    }


def main() -> None:
    protocol = freeze_protocol()
    audit = {
        "schema": "tahoe_experiment1_evaluation_sampling_reproducibility_audit_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "protocol": {
            "path": PROTOCOL_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(PROTOCOL_PATH),
            "schema": protocol["schema"],
        },
        "scope": {
            "splits": ["val", "test"],
            "conditions_per_split": AUDIT_POSITIONS_PER_SPLIT,
            "repeat_count": REPEAT_COUNT,
            "dataset_constructions_per_split": 2,
            "metrics_run": False,
            "baselines_run": False,
            "state_or_st_run": False,
            "large_cell_index_manifest_generated": False,
        },
        "results": {split: audit_split(split) for split in ("val", "test")},
        "checks": {
            "existing_dataset_sampler_reused": "pass",
            "epoch_0_through_4": "pass",
            "independent_dataset_exact_index_reproduction": "pass",
            "within_set_without_replacement": "pass",
            "shuffle_false": "pass",
            "drop_last_false": "pass",
            "model_independent_sampling_key": "pass",
        },
        "issues": [],
    }
    atomic_write_json(AUDIT_PATH, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
