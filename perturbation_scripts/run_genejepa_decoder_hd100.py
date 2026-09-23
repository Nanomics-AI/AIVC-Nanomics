#!/usr/bin/env python3
"""Freeze, smoke-test, and train the train-only high-detection Top100 Decoder."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from state.tx.models.base import LatentToGeneDecoder

import run_genejepa_decoder_v1 as decoder_v1
from tahoe_decoder_v1_data import (
    FORMAL_EMBEDDING_MANIFEST,
    FORMAL_EMBEDDINGS,
    GENE_METADATA,
    LATENT_DIM,
    PROJECT_ROOT,
    RESULTS,
    TahoeDecoderV1PairedDataset,
    atomic_write_json,
    display_path,
    sha256_file,
    utc_now,
)


SCRIPT_PATH = Path(__file__).resolve()
TASK_PATH = PROJECT_ROOT.parent / "当前任务.txt"
SPARSITY_STATS_PATH = RESULTS / "genejepa_decoder_gene_sparsity_train_stats.csv"
BASE_PANEL_PATH = RESULTS / "genejepa_decoder_v1_gene_panel.csv"
BASE_CONFIG_PATH = RESULTS / "genejepa_decoder_v1_training_config.json"
BASE_PARTITION_PATH = RESULTS / "genejepa_decoder_v1_train_partition_audit.json"
DESCRIPTOR_COUNTS_PATH = RESULTS / "genejepa_decoder_v1_descriptor_counts.npz"

PANEL_PATH = RESULTS / "genejepa_decoder_hd100_gene_panel.csv"
PANEL_SUMMARY_PATH = RESULTS / "genejepa_decoder_hd100_gene_panel.json"
TRAINING_CONFIG_PATH = RESULTS / "genejepa_decoder_hd100_training_config.json"
STATIC_SMOKE_PATH = RESULTS / "genejepa_decoder_hd100_static_smoke.json"
DDP_SMOKE_PATH = RESULTS / "genejepa_decoder_hd100_exact_config_smoke.json"
SMOKE_CHECKPOINT_DIR = RESULTS / "genejepa_decoder_hd100_smoke_checkpoints"
FORMAL_CHECKPOINT_DIR = RESULTS / "genejepa_decoder_hd100_checkpoints"
SMOKE_TENSORBOARD = RESULTS / "tensorboard/genejepa_decoder_hd100/exact_config_smoke"
FORMAL_TENSORBOARD = RESULTS / "tensorboard/genejepa_decoder_hd100/formal"
FORMAL_RESULT_PATH = RESULTS / "genejepa_decoder_hd100_training_result.json"

TASK_SHA256 = "0f5752d1f553a14b8f9b501d193d5870a616931a4f94e9eb91a9c942b3ef52a2"
SPARSITY_STATS_SHA256 = "54dac1e1bd593ca7fe58174b5e65f3c66ab457512643d15c2ae00cc8e2b46528"
BASE_PANEL_SHA256 = "c4f79cfb0a37ec278d04c3f0045fa568872961445a9f251fff37dfa3d38d790e"
GENE_DIM = 100
EXPECTED_PARAMETER_COUNT = 2_418_276
STATIC_SMOKE_CELLS = 8

_BASE_CHECKPOINT_PAYLOAD = decoder_v1.checkpoint_payload
_BASE_VALIDATE_CHECKPOINT = decoder_v1.validate_checkpoint


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": display_path(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", lineterminator="\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_panel_from_stats() -> pd.DataFrame:
    if sha256_file(SPARSITY_STATS_PATH) != SPARSITY_STATS_SHA256:
        raise AssertionError("Frozen train-treated sparsity statistics changed")
    if sha256_file(BASE_PANEL_PATH) != BASE_PANEL_SHA256:
        raise AssertionError("Frozen Decoder v1 Top5000 panel changed")
    stats = pd.read_csv(SPARSITY_STATS_PATH, encoding="utf-8-sig", keep_default_na=False)
    required = {
        "genejepa_index",
        "gene_symbol",
        "ensembl_id",
        "token_id",
        "mean_logcp10k",
        "variance_logcp10k",
        "detection_rate",
        "rank_by_detection",
        "current_top5000",
        "current_panel_rank",
    }
    missing = required - set(stats.columns)
    if missing or len(stats) != 62_710:
        raise AssertionError(f"Sparsity statistics schema/rows changed: {sorted(missing)}")
    gene_index = stats["genejepa_index"].to_numpy(np.int64)
    detection = stats["detection_rate"].to_numpy(np.float64)
    mean = stats["mean_logcp10k"].to_numpy(np.float64)
    expected_order = np.lexsort((gene_index, -mean, -detection))
    recorded_rank = stats["rank_by_detection"].to_numpy(np.int64)
    expected_rank = np.empty(len(stats), dtype=np.int64)
    expected_rank[expected_order] = np.arange(1, len(stats) + 1)
    if not np.array_equal(recorded_rank, expected_rank):
        raise AssertionError("rank_by_detection disagrees with the frozen ranking definition")

    selected = stats.iloc[expected_order[:GENE_DIM]].copy().reset_index(drop=True)
    if not np.array_equal(selected["rank_by_detection"].to_numpy(np.int64), np.arange(1, 101)):
        raise AssertionError("Selected HD100 ranks are not exactly 1..100")
    if not selected["current_top5000"].astype(int).eq(1).all():
        raise AssertionError("HD100 is not a subset of the frozen variance Top5000")
    current_ranks = selected["current_panel_rank"].to_numpy(np.int64)
    if len(np.unique(current_ranks)) != GENE_DIM or np.any((current_ranks < 0) | (current_ranks >= 5000)):
        raise AssertionError("HD100 current_panel_rank mapping is invalid")

    base_panel = pd.read_csv(BASE_PANEL_PATH, encoding="utf-8-sig", keep_default_na=False)
    base_panel = base_panel.sort_values("panel_rank", kind="stable").reset_index(drop=True)
    if not np.array_equal(
        selected["genejepa_index"].to_numpy(np.int64),
        base_panel.iloc[current_ranks]["genejepa_index"].to_numpy(np.int64),
    ):
        raise AssertionError("HD100 current_panel_rank does not map to the same v1 genes")

    output = selected[
        [
            "genejepa_index",
            "gene_symbol",
            "ensembl_id",
            "token_id",
            "detection_rate",
            "mean_logcp10k",
            "variance_logcp10k",
            "current_panel_rank",
        ]
    ].copy()
    output.insert(0, "panel_rank", np.arange(GENE_DIM, dtype=np.int32))
    return output


def freeze_panel() -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = selected_panel_from_stats()
    if PANEL_PATH.exists():
        existing = pd.read_csv(PANEL_PATH, encoding="utf-8-sig", keep_default_na=False)
        if list(existing.columns) != list(selected.columns) or len(existing) != GENE_DIM:
            raise AssertionError("Existing HD100 panel has the wrong schema or row count")
        for column in selected.columns:
            if column in {"detection_rate", "mean_logcp10k", "variance_logcp10k"}:
                if not np.allclose(
                    existing[column].to_numpy(np.float64),
                    selected[column].to_numpy(np.float64),
                    rtol=1e-14,
                    atol=1e-15,
                ):
                    raise AssertionError(f"Existing HD100 panel differs in {column}")
            elif not existing[column].astype(str).equals(selected[column].astype(str)):
                raise AssertionError(f"Existing HD100 panel differs in {column}")
    else:
        atomic_write_csv(PANEL_PATH, selected)

    payload = {
        "schema": "genejepa_decoder_hd100_gene_panel_v1",
        "created_at_utc": utc_now(),
        "status": "frozen",
        "selection_scope": "Experiment 1 train-treated physical cells only",
        "ranking": [
            "detection_rate descending",
            "mean_logcp10k descending",
            "genejepa_index ascending",
        ],
        "genes": GENE_DIM,
        "panel_rank": "zero-based 0..99",
        "source": artifact(SPARSITY_STATS_PATH),
        "current_variance_top5000": artifact(BASE_PANEL_PATH),
        "subset_assertion": {
            "HD100_subset_of_current_Top5000": True,
            "overlap": 100,
            "expected": 100,
            "current_panel_rank_recorded": True,
        },
        "output": artifact(PANEL_PATH),
        "selection_used_val_test_or_ARC7": False,
    }
    if PANEL_SUMMARY_PATH.exists():
        existing = read_json(PANEL_SUMMARY_PATH)
        for value in (existing, payload):
            value.pop("created_at_utc", None)
        if existing != payload:
            raise AssertionError("Existing HD100 panel summary differs from current inputs")
    else:
        atomic_write_json(PANEL_SUMMARY_PATH, payload)
    return selected, read_json(PANEL_SUMMARY_PATH)


def build_decoder() -> LatentToGeneDecoder:
    decoder_v1.seed_everything(decoder_v1.SEED)
    model = LatentToGeneDecoder(
        latent_dim=LATENT_DIM,
        gene_dim=GENE_DIM,
        hidden_dims=[1024, 1024, 512],
        dropout=0.1,
        residual_decoder=False,
    )
    if not isinstance(model.decoder[-1], nn.ReLU):
        raise AssertionError("STATE decoder no longer has its expected terminal ReLU")
    model.decoder[-1] = nn.Softplus()
    assert_decoder(model)
    return model


def assert_decoder(model: LatentToGeneDecoder) -> None:
    expected_types = [
        nn.Linear,
        nn.LayerNorm,
        nn.GELU,
        nn.Dropout,
        nn.Linear,
        nn.LayerNorm,
        nn.GELU,
        nn.Dropout,
        nn.Linear,
        nn.LayerNorm,
        nn.GELU,
        nn.Dropout,
        nn.Linear,
        nn.Softplus,
    ]
    modules = list(model.decoder)
    if model.residual_decoder or len(modules) != len(expected_types) or any(
        not isinstance(module, expected) for module, expected in zip(modules, expected_types)
    ):
        raise AssertionError("HD100 Decoder module sequence changed")
    dimensions = [
        (module.in_features, module.out_features)
        for module in modules
        if isinstance(module, nn.Linear)
    ]
    if dimensions != [(768, 1024), (1024, 1024), (1024, 512), (512, 100)]:
        raise AssertionError(f"HD100 Decoder dimensions changed: {dimensions}")
    if any(module.p != 0.1 for module in modules if isinstance(module, nn.Dropout)):
        raise AssertionError("HD100 Decoder dropout changed")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise AssertionError(f"HD100 parameter count changed: {parameter_count}")


def model_contract() -> dict[str, Any]:
    model = build_decoder()
    result = {
        "source_class": "state.tx.models.base.LatentToGeneDecoder",
        "source_path": display_path(decoder_v1.STATE_DECODER_PATH),
        "source_sha256": sha256_file(decoder_v1.STATE_DECODER_PATH),
        "dimensions": [768, 1024, 1024, 512, 100],
        "hidden_block": "Linear -> LayerNorm -> GELU -> Dropout(0.1)",
        "residual_decoder": False,
        "final_activation": "Softplus(beta=1, threshold=20)",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "initialization": "random from scratch, seed=42; no Decoder v1 weights loaded",
    }
    del model
    return result


def configure_decoder_v1_runtime(panel_sha256: str) -> None:
    """Point the proven v1 training engine at the isolated HD100 contract."""
    decoder_v1.SCRIPT_PATH = SCRIPT_PATH
    decoder_v1.TASK_PATH = TASK_PATH
    decoder_v1.PANEL_PATH = PANEL_PATH
    decoder_v1.PANEL_SHA256 = panel_sha256
    decoder_v1.GENE_DIM = GENE_DIM
    decoder_v1.EXPECTED_PARAMETER_COUNT = EXPECTED_PARAMETER_COUNT
    decoder_v1.TRAINING_CONFIG_PATH = TRAINING_CONFIG_PATH
    decoder_v1.SMOKE_RESULT_PATH = DDP_SMOKE_PATH
    decoder_v1.SMOKE_CHECKPOINT_DIR = SMOKE_CHECKPOINT_DIR
    decoder_v1.FORMAL_CHECKPOINT_DIR = FORMAL_CHECKPOINT_DIR
    decoder_v1.SMOKE_TENSORBOARD = SMOKE_TENSORBOARD
    decoder_v1.FORMAL_TENSORBOARD = FORMAL_TENSORBOARD
    decoder_v1.FORMAL_RESULT_PATH = FORMAL_RESULT_PATH
    decoder_v1.build_decoder = build_decoder
    decoder_v1.assert_decoder = assert_decoder
    decoder_v1.checkpoint_payload = checkpoint_payload
    decoder_v1.validate_checkpoint = validate_checkpoint
    decoder_v1.load_runtime_contract = load_runtime_contract


def checkpoint_payload(**kwargs: Any) -> dict[str, Any]:
    payload = _BASE_CHECKPOINT_PAYLOAD(**kwargs)
    payload["schema"] = "genejepa_decoder_hd100_checkpoint_v1"
    payload["experiment"] = "train-only high-detection Top100 Decoder"
    payload["random_initialization_from_scratch"] = True
    payload["decoder_v1_weights_loaded"] = False
    return payload


def validate_checkpoint(payload: dict[str, Any], **kwargs: Any) -> None:
    if payload.get("schema") != "genejepa_decoder_hd100_checkpoint_v1":
        raise AssertionError("Checkpoint is not an HD100 checkpoint")
    if payload.get("random_initialization_from_scratch") is not True or payload.get(
        "decoder_v1_weights_loaded"
    ) is not False:
        raise AssertionError("HD100 checkpoint initialization provenance changed")
    compatible = dict(payload)
    compatible["schema"] = "genejepa_decoder_v1_checkpoint_v1"
    _BASE_VALIDATE_CHECKPOINT(compatible, **kwargs)


def common_input_audit() -> dict[str, Any]:
    if sha256_file(TASK_PATH) != TASK_SHA256:
        raise AssertionError("当前任务.txt changed before HD100 preparation")
    # Run this before adapting decoder_v1 module globals. It revalidates every
    # frozen cache/dataset input used by the existing production runner.
    inputs = decoder_v1.frozen_inputs()
    base_config = read_json(BASE_CONFIG_PATH)
    partition = read_json(BASE_PARTITION_PATH)
    if base_config.get("status") != "frozen" or partition.get("status") != "pass":
        raise AssertionError("Decoder v1 training/partition contract is not frozen and PASS")
    if sha256_file(decoder_v1.DATASET_PATH) != base_config["data"]["dataset_sha256"]:
        raise AssertionError("Frozen Decoder Dataset helper changed")
    if partition["descriptor_counts"]["train_cells"] != decoder_v1.TRAIN_CELLS:
        raise AssertionError("Frozen train-treated cell count changed")
    return inputs


def training_config(panel_sha256: str) -> dict[str, Any]:
    return {
        "schema": "genejepa_decoder_hd100_training_config_v1",
        "created_at_utc": utc_now(),
        "status": "frozen",
        "experiment": "high-detection Top100 extreme diagnostic",
        "base_scientific_contract": artifact(decoder_v1.CONTRACT_PATH),
        "panel": {
            **artifact(PANEL_PATH),
            "summary": artifact(PANEL_SUMMARY_PATH),
            "genes": GENE_DIM,
            "selection_scope": "train-treated only",
            "ranking": "detection_rate desc, mean_logcp10k desc, genejepa_index asc",
            "subset_of_decoder_v1_top5000": True,
        },
        "model": model_contract(),
        "data": {
            "dataset": "unchanged TahoeDecoderV1PairedDataset + exact v1 train ownership adapter",
            "dataset_path": display_path(decoder_v1.DATASET_PATH),
            "dataset_sha256": sha256_file(decoder_v1.DATASET_PATH),
            "latent": "raw signed frozen GeneJEPA Epoch25 EMA-teacher [768]",
            "target": "full mapped counts -> CP10000 denominator over all mapped genes -> log1p -> HD100 select",
            "CP10000_denominator_gene_universe": 62_710,
            "train_treated_cells": decoder_v1.TRAIN_CELLS,
            "val_treated_cells": decoder_v1.VAL_CELLS,
            "test_treated_cells": decoder_v1.TEST_CELLS,
            "DMSO_used": False,
            "test_dataset_constructed": False,
            "descriptor_counts": artifact(DESCRIPTOR_COUNTS_PATH),
            "partition_audit_reused": artifact(BASE_PARTITION_PATH),
        },
        "training": {
            "world_size": 2,
            "batch_size_per_gpu": 1024,
            "global_batch": 2048,
            "gradient_accumulation": 1,
            "precision": "BF16 autocast forward; FP32 prediction/target MSE",
            "optimizer": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "gradient_clip_norm": 1.0,
            "scheduler": None,
            "seed": 42,
            "max_epochs": 10,
            "early_stopping_patience_full_val_epochs": 3,
            "steps_per_epoch": 11_202,
            "used_cells_per_epoch": 22_941_696,
            "global_tail_dropped": 240,
            "random_initialization_from_scratch": True,
            "Decoder_v1_weights_loaded": False,
        },
        "dataloader": {
            "workers_per_rank": 4,
            "pin_memory": True,
            "prefetch_factor": 2,
            "persistent_workers": False,
        },
        "validation": {
            "full_validation_cells": 2_841_724,
            "criterion": "global FP64 SSE / global cell-by-gene element count",
            "full_validation_elements": 2_841_724 * GENE_DIM,
            "shuffle": False,
            "drop_last": False,
            "strict_minimum_checkpoint": True,
        },
        "checkpoint": {
            "best": display_path(FORMAL_CHECKPOINT_DIR / "best.pt"),
            "last": display_path(FORMAL_CHECKPOINT_DIR / "last.pt"),
            "resume": "complete epoch boundaries",
        },
        "implementation": {
            "runner_path": display_path(SCRIPT_PATH),
            "runner_sha256": sha256_file(SCRIPT_PATH),
            "reused_training_engine": artifact(Path(decoder_v1.__file__)),
            "task_sha256": sha256_file(TASK_PATH),
        },
        "formal_training_gate": {
            "artifact": display_path(DDP_SMOKE_PATH),
            "required": "status=pass and ready_for_formal_training=true",
        },
        "panel_sha256_redundant_check": panel_sha256,
    }


def prepare() -> dict[str, Any]:
    common_input_audit()
    panel, panel_summary = freeze_panel()
    panel_sha = panel_summary["output"]["sha256"]
    configure_decoder_v1_runtime(panel_sha)
    config = training_config(panel_sha)
    if TRAINING_CONFIG_PATH.exists():
        existing = read_json(TRAINING_CONFIG_PATH)
        for value in (existing, config):
            value.pop("created_at_utc", None)
        if existing != config:
            raise AssertionError("Existing HD100 training config differs from current contract")
    else:
        atomic_write_json(TRAINING_CONFIG_PATH, config)
    result = {
        "status": "pass",
        "scope": "panel/config freeze only; no GPU training, test data, ARC7, or inference",
        "panel": artifact(PANEL_PATH),
        "panel_summary": artifact(PANEL_SUMMARY_PATH),
        "training_config": artifact(TRAINING_CONFIG_PATH),
        "panel_rows": len(panel),
        "Top100_subset_of_current_Top5000": True,
        "overlap": 100,
        "ready_for_static_smoke": True,
        "ready_for_DDP_smoke": False,
        "formal_training_started": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def load_runtime_contract() -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    if not TRAINING_CONFIG_PATH.is_file() or not PANEL_PATH.is_file():
        raise FileNotFoundError("Run the HD100 prepare command first")
    config = read_json(TRAINING_CONFIG_PATH)
    panel_sha = sha256_file(PANEL_PATH)
    expected = {
        "status": "frozen",
        "panel_sha": panel_sha,
        "runner_sha": sha256_file(SCRIPT_PATH),
        "dataset_sha": sha256_file(decoder_v1.DATASET_PATH),
        "task_sha": sha256_file(TASK_PATH),
        "descriptor_counts_sha": sha256_file(DESCRIPTOR_COUNTS_PATH),
        "partition_sha": sha256_file(BASE_PARTITION_PATH),
    }
    observed = {
        "status": config.get("status"),
        "panel_sha": config["panel"]["sha256"],
        "runner_sha": config["implementation"]["runner_sha256"],
        "dataset_sha": config["data"]["dataset_sha256"],
        "task_sha": config["implementation"]["task_sha256"],
        "descriptor_counts_sha": config["data"]["descriptor_counts"]["sha256"],
        "partition_sha": config["data"]["partition_audit_reused"]["sha256"],
    }
    if observed != expected:
        raise AssertionError(f"HD100 runtime contract changed: {observed} != {expected}")
    with np.load(DESCRIPTOR_COUNTS_PATH, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    inputs = {"cache_summary": read_json(decoder_v1.CACHE_PLAN_SUMMARY)}
    decoder_v1.validate_descriptor_counts(arrays, inputs)
    return config, arrays, sha256_file(TRAINING_CONFIG_PATH)


def hidden_initialization_audit(model: nn.Module) -> bool:
    decoder_v1.seed_everything(decoder_v1.SEED)
    reference = LatentToGeneDecoder(
        latent_dim=LATENT_DIM,
        gene_dim=5000,
        hidden_dims=[1024, 1024, 512],
        dropout=0.1,
        residual_decoder=False,
    )
    reference.decoder[-1] = nn.Softplus()
    model_state = model.state_dict()
    reference_state = reference.state_dict()
    shared = [key for key in model_state if not key.startswith("decoder.12.")]
    exact = bool(shared) and all(torch.equal(model_state[key], reference_state[key]) for key in shared)
    del reference
    if not exact:
        raise AssertionError("Same-seed hidden-layer initialization differs from Decoder v1")
    return exact


def static_smoke() -> dict[str, Any]:
    panel, panel_summary = freeze_panel()
    configure_decoder_v1_runtime(panel_summary["output"]["sha256"])
    load_runtime_contract()
    hd_dataset = TahoeDecoderV1PairedDataset(
        split="train", panel_path=PANEL_PATH, shuffle_shards=False, max_cells=STATIC_SMOKE_CELLS
    )
    old_dataset = TahoeDecoderV1PairedDataset(
        split="train",
        panel_path=BASE_PANEL_PATH,
        shuffle_shards=False,
        max_cells=STATIC_SMOKE_CELLS,
    )
    hd_rows = list(hd_dataset)
    old_rows = list(old_dataset)
    if len(hd_rows) != STATIC_SMOKE_CELLS or len(old_rows) != STATIC_SMOKE_CELLS:
        raise AssertionError("Static smoke did not retrieve the requested real cells")
    hd_indices = np.asarray([row["global_embedding_index"] for row in hd_rows], dtype=np.int64)
    old_indices = np.asarray([row["global_embedding_index"] for row in old_rows], dtype=np.int64)
    if not np.array_equal(hd_indices, old_indices):
        raise AssertionError("HD100 and old-panel smoke cells differ")
    latent = torch.stack([row["latent"] for row in hd_rows])
    target = torch.stack([row["target_expression"] for row in hd_rows])
    old_target = torch.stack([row["target_expression"] for row in old_rows])
    current_ranks = torch.from_numpy(panel["current_panel_rank"].to_numpy(np.int64))
    target_subset_exact = torch.equal(target, old_target.index_select(1, current_ranks))
    if latent.shape != (STATIC_SMOKE_CELLS, 768) or target.shape != (
        STATIC_SMOKE_CELLS,
        100,
    ):
        raise AssertionError("Static smoke latent/target shape changed")
    if not target_subset_exact:
        raise AssertionError("HD100 target is not the exact v1 target column subset")
    model = build_decoder().eval()
    hidden_initialization_exact = hidden_initialization_audit(model)
    with torch.inference_mode():
        prediction = model(latent)
        loss = torch.mean((prediction.float() - target.float()).square())
    if (
        prediction.shape != target.shape
        or not torch.isfinite(prediction).all()
        or torch.any(prediction < 0)
        or not torch.isfinite(target).all()
        or torch.any(target < 0)
        or not torch.isfinite(loss)
    ):
        raise AssertionError("HD100 static forward/target/loss contract failed")
    result = {
        "schema": "genejepa_decoder_hd100_static_smoke_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "scope": "eight real train cells; no fitting, validation, test data, ST-A, or ARC7",
        "panel_rows": len(panel),
        "panel_subset_overlap": 100,
        "cells": STATIC_SMOKE_CELLS,
        "embedding_indices": hd_indices.tolist(),
        "latent_shape": list(latent.shape),
        "target_shape": list(target.shape),
        "prediction_shape": list(prediction.shape),
        "target_exactly_equals_same_cells_v1_5000_target_at_current_panel_rank": True,
        "CP10000_denominator_remains_full_mapped_gene_universe": True,
        "prediction_finite": True,
        "prediction_nonnegative": True,
        "target_finite": True,
        "target_nonnegative": True,
        "loss_finite": True,
        "initial_mse": float(loss),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "same_seed_hidden_initialization_matches_5000_decoder": hidden_initialization_exact,
        "random_initialization_from_scratch": True,
        "decoder_v1_weights_loaded": False,
        "provenance": {
            "panel": artifact(PANEL_PATH),
            "panel_summary": artifact(PANEL_SUMMARY_PATH),
            "training_config": artifact(TRAINING_CONFIG_PATH),
            "dataset": artifact(decoder_v1.DATASET_PATH),
            "runner": artifact(SCRIPT_PATH),
        },
        "ready_for_DDP_smoke": True,
        "formal_training_started": False,
    }
    atomic_write_json(STATIC_SMOKE_PATH, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def rewrite_smoke_result(result: dict[str, Any] | None) -> None:
    if result is None:
        return
    result["schema"] = "genejepa_decoder_hd100_exact_config_smoke_v1"
    result["experiment"] = "train-only high-detection Top100 Decoder"
    result["random_initialization_from_scratch"] = True
    result["decoder_v1_weights_loaded"] = False
    result["static_smoke"] = artifact(STATIC_SMOKE_PATH)
    atomic_write_json(DDP_SMOKE_PATH, result)


def rewrite_formal_result(result: dict[str, Any] | None) -> None:
    if result is None:
        return
    result["schema"] = "genejepa_decoder_hd100_training_result_v1"
    result["experiment"] = "train-only high-detection Top100 extreme diagnostic"
    result["panel"] = artifact(PANEL_PATH)
    result["random_initialization_from_scratch"] = True
    result["decoder_v1_weights_loaded"] = False
    result["comparison_guardrail"] = (
        "HD100 val MSE and Decoder v1 5000-gene val MSE use different target distributions; "
        "do not compare them as model quality."
    )
    atomic_write_json(FORMAL_RESULT_PATH, result)


def configure_for_runtime() -> None:
    if not PANEL_SUMMARY_PATH.is_file() or not STATIC_SMOKE_PATH.is_file():
        raise FileNotFoundError("Run prepare and static-smoke before any GPU command")
    panel_summary = read_json(PANEL_SUMMARY_PATH)
    if read_json(STATIC_SMOKE_PATH).get("status") != "pass":
        raise AssertionError("HD100 static smoke is not PASS")
    configure_decoder_v1_runtime(panel_summary["output"]["sha256"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Freeze the train-only HD100 panel and training config")
    commands.add_parser("static-smoke", help="Run the short CPU real-target/model smoke")
    smoke = commands.add_parser("smoke", help="Run the proven two-GPU exact-config smoke")
    smoke.add_argument("--resume", action="store_true")
    smoke.add_argument("--train-steps", type=int)
    smoke.add_argument("--val-batches", type=int, default=4)
    formal = commands.add_parser("run", help="Run formal HD100 training")
    formal.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError("Run prepare with one ordinary Python process")
        prepare()
    elif args.command == "static-smoke":
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError("Run static-smoke with one ordinary Python process")
        static_smoke()
    elif args.command == "smoke":
        configure_for_runtime()
        steps = args.train_steps if args.train_steps is not None else (2 if args.resume else 8)
        result = decoder_v1.run_smoke(
            resume=args.resume,
            train_steps=steps,
            val_batches=args.val_batches,
            fallback_512=False,
        )
        rewrite_smoke_result(result)
    elif args.command == "run":
        configure_for_runtime()
        result = decoder_v1.run_formal(resume=args.resume, fallback_512=False)
        rewrite_formal_result(result)


if __name__ == "__main__":
    main()
