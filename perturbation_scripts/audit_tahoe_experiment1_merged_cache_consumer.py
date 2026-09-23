#!/usr/bin/env python3
"""Audit the formal merged-cache Dataset/DataLoader without training a model."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset

from tahoe_experiment1_latent_data import (
    EDGE_SPLIT_MANIFEST,
    FORMAL_CACHE_SUMMARY,
    FORMAL_CONDITION_INDEX,
    FORMAL_CONTROL_POOL_INDEX,
    FORMAL_EMBEDDING_MANIFEST,
    FORMAL_EMBEDDINGS,
    LATENT_DIM,
    PERT_DIM,
    PROJECT_ROOT,
    RESULTS,
    SET_SIZE,
    TahoeExperiment1LatentSetDataset,
    make_dataloader,
    sha256_file,
)


DEFAULT_JSON = RESULTS / "tahoe_experiment1_merged_cache_consumer_audit.json"
DEFAULT_MD = RESULTS / "tahoe_experiment1_merged_cache_consumer_audit.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def stable_positions(length: int, count: int) -> list[int]:
    if not 1 <= count <= length:
        raise ValueError(f"Cannot select {count} conditions from {length}")
    return np.linspace(0, length - 1, count, dtype=np.int64).tolist()


def audit_split(
    split: str,
    *,
    conditions_per_split: int,
    seed: int,
) -> tuple[dict[str, Any], TahoeExperiment1LatentSetDataset]:
    dataset = TahoeExperiment1LatentSetDataset(split=split, seed=seed, epoch=0)
    positions = stable_positions(len(dataset), conditions_per_split)
    loader = make_dataloader(
        Subset(dataset, positions),
        batch_size=len(positions),
        shuffle=False,
        seed=seed,
        drop_last=False,
    )
    batch = next(iter(loader))
    batch_size = len(positions)
    expected_shapes = {
        "ctrl_cell_emb": (batch_size, SET_SIZE, LATENT_DIM),
        "pert_cell_emb": (batch_size, SET_SIZE, LATENT_DIM),
        "pert_emb": (batch_size, SET_SIZE, PERT_DIM),
        "source_embedding_index": (batch_size, SET_SIZE),
        "target_embedding_index": (batch_size, SET_SIZE),
    }
    for key, expected in expected_shapes.items():
        if tuple(batch[key].shape) != expected:
            raise AssertionError(f"{split}/{key} shape {tuple(batch[key].shape)} != {expected}")
    for key in ("ctrl_cell_emb", "pert_cell_emb", "pert_emb"):
        if batch[key].dtype != torch.float32 or not bool(torch.isfinite(batch[key]).all()):
            raise AssertionError(f"{split}/{key} is not finite float32")

    source_indices = batch["source_embedding_index"].numpy()
    target_indices = batch["target_embedding_index"].numpy()
    if source_indices.dtype != np.int64 or target_indices.dtype != np.int64:
        raise AssertionError(f"{split} embedding indices must be int64")
    for side, indices in (("control", source_indices), ("treated", target_indices)):
        if indices.min() < 0 or indices.max() >= len(dataset.embeddings):
            raise AssertionError(f"{split}/{side} contains an out-of-range embedding_index")
        if any(len(np.unique(row)) != SET_SIZE for row in indices):
            raise AssertionError(f"{split}/{side} sampled with within-set replacement")
    if any(np.intersect1d(source, target).size for source, target in zip(source_indices, target_indices, strict=True)):
        raise AssertionError(f"{split} control/treated indices overlap")

    ctrl = batch["ctrl_cell_emb"].numpy()
    target = batch["pert_cell_emb"].numpy()
    ctrl_identity = max(
        float(np.abs(ctrl[index] - np.asarray(dataset.embeddings[rows])).max())
        for index, rows in enumerate(source_indices)
    )
    target_identity = max(
        float(np.abs(target[index] - np.asarray(dataset.embeddings[rows])).max())
        for index, rows in enumerate(target_indices)
    )
    if ctrl_identity != 0 or target_identity != 0:
        raise AssertionError(f"{split} DataLoader changed raw cached latent values")
    if not ((ctrl < 0).any() and (ctrl > 0).any() and (target < 0).any() and (target > 0).any()):
        raise AssertionError(f"{split} sampled latents do not preserve signed values")

    perturbation = batch["pert_emb"]
    if not torch.equal(perturbation, perturbation[:, :1, :].expand_as(perturbation)):
        raise AssertionError(f"{split} perturbation vector varies within a set")
    one_hot = perturbation[:, 0, : PERT_DIM - 1]
    if not bool(((one_hot == 0) | (one_hot == 1)).all()):
        raise AssertionError(f"{split} drug feature is not binary one-hot")
    if not torch.equal(one_hot.sum(dim=1), torch.ones(batch_size)):
        raise AssertionError(f"{split} drug one-hot does not contain exactly one 1")
    observed_drug_ids = one_hot.argmax(dim=1).numpy()
    expected_drug_ids = batch["drug_id"].numpy()
    if not np.array_equal(observed_drug_ids, expected_drug_ids):
        raise AssertionError(f"{split} one-hot index disagrees with frozen drug_id")
    for drug, drug_id in zip(batch["drug"], expected_drug_ids, strict=True):
        if dataset.featurizer.drug_to_id[drug] != int(drug_id):
            raise AssertionError(f"{split} vocabulary lookup changed for {drug}")

    expected_dose = torch.tensor(
        [
            np.float32(
                (math.log10(float(value)) - dataset.featurizer.log10_mean)
                / dataset.featurizer.log10_std
            )
            for value in batch["dose_uM"].numpy()
        ],
        dtype=torch.float32,
    )
    if not torch.equal(perturbation[:, 0, PERT_DIM - 1], expected_dose):
        raise AssertionError(f"{split} dose feature disagrees with train-only transform")

    for offset, position in enumerate(positions):
        row = dataset.conditions.iloc[position]
        for field, expected in (
            ("condition_id", str(row["pair_id"])),
            ("edge_id", str(row["edge_id"])),
            ("split", split),
            ("drug", str(row["drug"])),
        ):
            if batch[field][offset] != expected:
                raise AssertionError(f"{split}/{field} disagrees with frozen condition metadata")

    epoch0_source = source_indices.copy()
    epoch0_target = target_indices.copy()
    dataset.set_epoch(1)
    changed_control = 0
    changed_treated = 0
    for offset, position in enumerate(positions):
        row = dataset.conditions.iloc[position]
        redrawn = dataset[position]
        redraw_source = redrawn["source_embedding_index"].numpy()
        redraw_target = redrawn["target_embedding_index"].numpy()
        if len(np.unique(redraw_source)) != SET_SIZE or len(np.unique(redraw_target)) != SET_SIZE:
            raise AssertionError(f"{split} epoch-1 redraw contains a duplicate cell")
        if int(row["control_cached_cell_count"]) > SET_SIZE:
            changed_control += int(not np.array_equal(redraw_source, epoch0_source[offset]))
        if int(row["treated_cached_cell_count"]) > SET_SIZE:
            changed_treated += int(not np.array_equal(redraw_target, epoch0_target[offset]))
    dataset.set_epoch(0)
    for offset, position in enumerate(positions):
        repeated = dataset[position]
        if not np.array_equal(repeated["source_embedding_index"].numpy(), epoch0_source[offset]):
            raise AssertionError(f"{split} epoch-0 control draw is not reproducible")
        if not np.array_equal(repeated["target_embedding_index"].numpy(), epoch0_target[offset]):
            raise AssertionError(f"{split} epoch-0 treated draw is not reproducible")

    eligible_control_redraws = int(
        (dataset.conditions.iloc[positions]["control_cached_cell_count"] > SET_SIZE).sum()
    )
    eligible_treated_redraws = int(
        (dataset.conditions.iloc[positions]["treated_cached_cell_count"] > SET_SIZE).sum()
    )
    if changed_control != eligible_control_redraws or changed_treated != eligible_treated_redraws:
        raise AssertionError(f"{split} epoch change did not redraw every eligible population")

    return {
        "dataset_conditions": len(dataset),
        "audited_conditions": batch_size,
        "condition_ids": list(batch["condition_id"]),
        "edge_ids": list(batch["edge_id"]),
        "drugs": list(batch["drug"]),
        "doses_uM": [float(value) for value in batch["dose_uM"]],
        "tensor_shapes": {key: list(value) for key, value in expected_shapes.items()},
        "tensor_dtype": "float32",
        "finite": True,
        "signed": True,
        "control_negative_ratio": float((batch["ctrl_cell_emb"] < 0).float().mean()),
        "treated_negative_ratio": float((batch["pert_cell_emb"] < 0).float().mean()),
        "control_set_unique_cells": SET_SIZE,
        "treated_set_unique_cells": SET_SIZE,
        "control_treated_overlap": 0,
        "embedding_index_min": int(min(source_indices.min(), target_indices.min())),
        "embedding_index_max": int(max(source_indices.max(), target_indices.max())),
        "raw_cache_identity_max_abs": {
            "control": ctrl_identity,
            "treated": target_identity,
        },
        "drug_one_hot_ones_per_set": 1,
        "drug_vocabulary_index_match": True,
        "dose_feature_train_only_match": True,
        "perturbation_constant_within_set": True,
        "epoch0_reproducible": True,
        "epoch1_redraw": {
            "eligible_control_conditions": eligible_control_redraws,
            "changed_control_conditions": changed_control,
            "eligible_treated_conditions": eligible_treated_redraws,
            "changed_treated_conditions": changed_treated,
        },
    }, dataset


def build_markdown(result: dict[str, Any]) -> str:
    rows = []
    for split in ("train", "val", "test"):
        audit = result["sampled_dataloader_audit"][split]
        rows.append(
            f"| {split} | {audit['dataset_conditions']:,} | {audit['audited_conditions']} | "
            f"{audit['tensor_shapes']['ctrl_cell_emb']} | {audit['tensor_shapes']['pert_cell_emb']} | "
            f"{audit['tensor_shapes']['pert_emb']} | PASS |"
        )
    return "\n".join(
        [
            "# Tahoe Experiment 1 merged-cache consumer / DataLoader audit",
            "",
            f"- Status: **{result['status'].upper()}**",
            f"- Created (UTC): `{result['created_at_utc']}`",
            "- Scope: formal merged cache consumer only; no B0/B1/B2, baseline, STATE, or ST run.",
            "",
            "## Contract",
            "",
            "`formal merged cache -> condition/control pool -> global embedding_index -> S=256 set`",
            "",
            "| Split | Dataset conditions | Audited | ctrl | treated | perturbation | Result |",
            "|---|---:|---:|---|---|---|---|",
            *rows,
            "",
            "## Key findings",
            "",
            f"- Merged mmap: `{result['cache']['shape']}`, `{result['cache']['dtype']}`, row equals global `embedding_index`.",
            "- Returned latent tensors are byte-for-byte equal to the selected cache rows; no latent transform was applied.",
            "- Every control and treated set contains exactly 256 unique cells sampled without replacement.",
            "- Signed values and finite float32 tensors are preserved in train, val, and test samples.",
            "- Perturbation is 379-d exact drug one-hot plus train-only standardized log10 dose, repeated across all 256 cells.",
            "- Frozen `(cell_line_id, drug)` edges do not cross train/val/test.",
            "- Exact DMSO pools are intentionally reusable across splits under the frozen cache policy; treated cells/edges remain split-exclusive.",
            "",
            "## Blockers",
            "",
            "None for the next Experiment 1 stage.",
            "",
        ]
    )


def run_audit(conditions_per_split: int, seed: int) -> dict[str, Any]:
    sampled: dict[str, Any] = {}
    reference_dataset: TahoeExperiment1LatentSetDataset | None = None
    for split in ("train", "val", "test"):
        sampled[split], dataset = audit_split(
            split,
            conditions_per_split=conditions_per_split,
            seed=seed,
        )
        if reference_dataset is None:
            reference_dataset = dataset
    assert reference_dataset is not None

    conditions = reference_dataset.all_conditions
    controls = reference_dataset.controls
    shared_control_pools = int(
        (conditions.groupby("control_pool_id")["split"].nunique() > 1).sum()
    )
    scripts = [
        Path(__file__),
        PROJECT_ROOT / "perturbation_scripts" / "tahoe_experiment1_latent_data.py",
        PROJECT_ROOT / "perturbation_scripts" / "prepare_tahoe_experiment1_manifests.py",
        PROJECT_ROOT / "perturbation_scripts" / "plan_tahoe_experiment1_full_cache.py",
        PROJECT_ROOT / "perturbation_scripts" / "merge_tahoe_experiment1_full_cache.py",
        PROJECT_ROOT / "perturbation_scripts" / "smoke_tahoe_experiment1_state.py",
        PROJECT_ROOT / "perturbation_scripts" / "benchmark_tahoe_experiment1_state_throughput.py",
    ]
    return {
        "created_at_utc": utc_now(),
        "status": "pass",
        "scope": {
            "consumer_dataloader_audit": True,
            "baseline_run": False,
            "state_or_st_run": False,
            "genejepa_inference_run": False,
            "expression_preprocessing_run": False,
        },
        "contract": "formal merged cache -> condition/control pool -> global embedding_index -> S=256 set",
        "audited_scripts": [
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in scripts
        ],
        "cache": {
            "path": FORMAL_EMBEDDINGS.relative_to(PROJECT_ROOT).as_posix(),
            "access": "numpy mmap_mode='r'",
            "is_memmap": isinstance(reference_dataset.embeddings, np.memmap),
            "shape": list(reference_dataset.embeddings.shape),
            "dtype": str(reference_dataset.embeddings.dtype),
            "row_equals_global_embedding_index": True,
            "manifest_global_finite": True,
            "sha256_from_pass_manifest": reference_dataset.provenance[
                "merged_embedding_sha256_from_pass_manifest"
            ],
            "sha256_recomputed_by_consumer": False,
            "latent_transforms": list(reference_dataset.embedding_transforms),
        },
        "index_universe": {
            "conditions": len(conditions),
            "control_pools": len(controls),
            "treated_cells": int(conditions["treated_cached_cell_count"].sum()),
            "dmso_cells": int(controls["cached_cell_count"].sum()),
            "total_cells": len(reference_dataset.embeddings),
            "split_condition_counts": reference_dataset.split_counts,
            "edge_counts": {
                split: int(conditions.loc[conditions["split"] == split, "edge_id"].nunique())
                for split in ("train", "val", "test")
            },
            "treated_edge_overlap": reference_dataset.edge_overlap,
            "shared_dmso_pools_across_splits": shared_control_pools,
            "shared_dmso_qualification": "Expected under the frozen global DMSO reuse policy; treated edges/cells remain split-exclusive.",
        },
        "dose_normalization": reference_dataset.dose_statistics,
        "perturbation": {
            "dimensions": PERT_DIM,
            "drug_one_hot_dimensions": PERT_DIM - 1,
            "dose_dimension": PERT_DIM - 1,
            "replicated_cells": SET_SIZE,
        },
        "sampled_dataloader_audit": sampled,
        "checks": {
            "small_input_manifest_sha256": "pass",
            "formal_cache_manifest": "pass",
            "condition_to_control_pool_ranges": "pass",
            "embedding_index_bounds": "pass",
            "within_set_without_replacement": "pass",
            "raw_cache_value_identity": "pass",
            "finite_signed_float32": "pass",
            "drug_vocabulary_and_one_hot": "pass",
            "train_only_dose_statistics": "pass",
            "frozen_split_match": "pass",
            "treated_edge_leakage": "pass",
        },
        "blockers": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions-per-split", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    if args.conditions_per_split < 2:
        parser.error("--conditions-per-split must be at least 2")
    result = run_audit(args.conditions_per_split, args.seed)
    atomic_write_text(
        args.json_output,
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    atomic_write_text(args.md_output, build_markdown(result))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
