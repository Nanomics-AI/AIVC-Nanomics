#!/usr/bin/env python3
"""Prepare, smoke-test, and train HD100 on Author GeneJEPA epoch49 latents.

This is a thin adapter around the proven HD100/DDP training engine.  It changes
only the latent cache and keeps the target builder, model, optimizer, and exact
DDP ownership contract unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import IterableDataset

import author_genejepa_epoch49_hd100_cache as author_cache
import run_genejepa_decoder_hd100 as hd100
import run_genejepa_decoder_v1 as engine
import tahoe_decoder_v1_data as data


PROJECT_ROOT = data.PROJECT_ROOT
RESULTS = data.RESULTS
SCRIPT_PATH = Path(__file__).resolve()
TASK_PATH = PROJECT_ROOT.parent / "当前任务.txt"

PANEL_PATH = RESULTS / "genejepa_decoder_hd100_gene_panel.csv"
PANEL_SUMMARY_PATH = RESULTS / "genejepa_decoder_hd100_gene_panel.json"
CACHE_EMBEDDINGS = author_cache.MERGED_EMBEDDINGS
CACHE_MANIFEST = author_cache.FINAL_MANIFEST
CACHE_SUMMARY = author_cache.PLAN_SUMMARY
CACHE_PLANS = tuple(author_cache.worker_paths(index)["plan"] for index in (0, 1))
CONDITION_INDEX = author_cache.CONDITION_INDEX

CONTRACT_PATH = RESULTS / "author_genejepa_hd100_contract.json"
DESCRIPTOR_COUNTS_PATH = RESULTS / "author_genejepa_hd100_descriptor_counts.npz"
PARTITION_AUDIT_PATH = RESULTS / "author_genejepa_hd100_train_partition_audit.json"
TRAINING_CONFIG_PATH = RESULTS / "author_genejepa_hd100_training_config.json"
STATIC_SMOKE_PATH = RESULTS / "author_genejepa_hd100_static_smoke.json"
DDP_SMOKE_PATH = RESULTS / "author_genejepa_hd100_exact_config_smoke.json"
SMOKE_CHECKPOINT_DIR = RESULTS / "author_genejepa_hd100_smoke_checkpoints"
FORMAL_CHECKPOINT_DIR = RESULTS / "author_genejepa_hd100_checkpoints"
SMOKE_TENSORBOARD = RESULTS / "tensorboard/author_genejepa_hd100/exact_config_smoke"
FORMAL_TENSORBOARD = RESULTS / "tensorboard/author_genejepa_hd100/formal"
FORMAL_RESULT_PATH = RESULTS / "author_genejepa_hd100_training_result.json"

TRAIN_CELLS = author_cache.TRAIN_CELLS
VAL_CELLS = author_cache.VAL_CELLS
ARC7_TREATED_CELLS = author_cache.ARC7_TREATED_CELLS
ARC7_CONTROL_CELLS = author_cache.ARC7_CONTROL_CELLS
TOTAL_CELLS = author_cache.EXPECTED_TOTAL
GENE_DIM = 100
EXPECTED_PARAMETER_COUNT = hd100.EXPECTED_PARAMETER_COUNT

_BASE_CHECKPOINT_PAYLOAD = engine.checkpoint_payload
_BASE_VALIDATE_CHECKPOINT = engine.validate_checkpoint


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    return engine.artifact_record(path, hash_file=hash_file)


def write_once(path: Path, payload: dict[str, Any]) -> None:
    if path.is_file():
        existing = read_json(path)
        left = json.loads(json.dumps(existing))
        right = json.loads(json.dumps(payload))
        left.pop("created_at_utc", None)
        right.pop("created_at_utc", None)
        if left != right:
            raise AssertionError(f"Existing frozen artifact differs: {data.display_path(path)}")
        return
    data.atomic_write_json(path, payload)


def validate_panel() -> str:
    if not PANEL_PATH.is_file() or not PANEL_SUMMARY_PATH.is_file():
        raise FileNotFoundError("Frozen HD100 panel/summary is missing")
    summary = read_json(PANEL_SUMMARY_PATH)
    panel_sha = data.sha256_file(PANEL_PATH)
    if (
        summary.get("status") != "frozen"
        or int(summary.get("genes", -1)) != GENE_DIM
        or summary.get("output", {}).get("sha256") != panel_sha
    ):
        raise AssertionError("Frozen HD100 panel contract changed")
    return panel_sha


def validate_cache() -> dict[str, Any]:
    required = [CACHE_EMBEDDINGS, CACHE_MANIFEST, CACHE_SUMMARY, *CACHE_PLANS]
    missing = [data.display_path(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Author latent cache is incomplete: {missing}")
    manifest = read_json(CACHE_MANIFEST)
    summary = read_json(CACHE_SUMMARY)
    output = manifest.get("output", {})
    hard = manifest.get("hard_checks", {})
    expected_hard = {
        "train_cells": TRAIN_CELLS,
        "val_cells": VAL_CELLS,
        "arc7_c_cells": author_cache.ARC7_CELLS,
        "embedding_dim": data.LATENT_DIM,
        "dtype": "float32",
        "all_finite": True,
        "missing": 0,
        "duplicate": 0,
        "worker_overlap": 0,
    }
    if (
        manifest.get("status") != "pass"
        or summary.get("status") != "pass"
        or output.get("shape") != [TOTAL_CELLS, data.LATENT_DIM]
        or output.get("dtype") != "float32"
        or output.get("row_equals_global_embedding_index") is not True
        or hard != expected_hard
        or int(summary["embedding_index"]["total_cells"]) != TOTAL_CELLS
        or CACHE_EMBEDDINGS.stat().st_size != int(output["size_bytes"])
    ):
        raise AssertionError("Author latent cache manifest/index contract failed")
    for plan, worker in zip(CACHE_PLANS, summary["partition"]["workers"], strict=True):
        if (
            plan.stat().st_size != int(worker["plan_size_bytes"])
            or data.sha256_file(plan) != worker["plan_sha256"]
        ):
            raise AssertionError(f"Author cache plan changed: {data.display_path(plan)}")
    return {"manifest": manifest, "summary": summary}


class AuthorTahoeDecoderDataset(data.TahoeDecoderV1PairedDataset):
    """The existing paired Dataset with the Author-only merged cache contract."""

    def __init__(
        self,
        *,
        split: str,
        panel_path: Path,
        embeddings_path: Path = CACHE_EMBEDDINGS,
        embedding_manifest_path: Path = CACHE_MANIFEST,
        cache_summary_path: Path = CACHE_SUMMARY,
        condition_index_path: Path = CONDITION_INDEX,
        plan_paths: tuple[Path, ...] = CACHE_PLANS,
        gene_metadata_path: Path = data.GENE_METADATA,
        shuffle_shards: bool = False,
        seed: int = 42,
        epoch: int = 0,
        max_cells: int | None = None,
        max_cells_per_shard: int | None = None,
        max_cells_per_source_row_group: int | None = None,
    ) -> None:
        IterableDataset.__init__(self)
        if split not in data.SPLIT_TO_CODE:
            raise ValueError("split must be train, val, or test")
        if epoch < 0 or (max_cells is not None and max_cells < 1):
            raise ValueError("epoch/max_cells is invalid")
        self.split = split
        self.panel_path = Path(panel_path)
        self.embeddings_path = Path(embeddings_path)
        self.plan_paths = tuple(Path(path) for path in plan_paths)
        self.shuffle_shards = bool(shuffle_shards)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.max_cells = max_cells
        self.max_cells_per_shard = max_cells_per_shard
        self.max_cells_per_source_row_group = max_cells_per_source_row_group
        self._embeddings: np.memmap | None = None

        manifest = read_json(Path(embedding_manifest_path))
        summary = read_json(Path(cache_summary_path))
        output = manifest["output"]
        hard = manifest["hard_checks"]
        if (
            manifest.get("status") != "pass"
            or hard.get("missing") != 0
            or hard.get("duplicate") != 0
            or hard.get("worker_overlap") != 0
            or hard.get("all_finite") is not True
            or output.get("row_equals_global_embedding_index") is not True
            or tuple(output.get("shape", ())) != (TOTAL_CELLS, data.LATENT_DIM)
            or output.get("dtype") != "float32"
            or self.embeddings_path.stat().st_size != int(output["size_bytes"])
            or int(summary["embedding_index"]["total_cells"]) != TOTAL_CELLS
        ):
            raise AssertionError("Author merged cache has not passed its frozen contract")

        self.conditions = data.load_conditions(Path(condition_index_path))
        self.condition_records = self.conditions.to_dict("records")
        self.condition_split_codes = self.conditions["split"].map(data.SPLIT_TO_CODE).to_numpy(
            np.int8
        )
        self.genes, self.token_lookup = data.load_gene_universe(Path(gene_metadata_path))
        self.panel_indices = data.load_panel(self.panel_path, self.genes)
        self.panel_lookup = data.build_panel_lookup(
            self.panel_indices, vocabulary_size=len(self.genes)
        )
        self.descriptors = data.plan_descriptors(self.plan_paths)
        expected = {"train": TRAIN_CELLS, "val": VAL_CELLS, "test": ARC7_TREATED_CELLS}
        self.expected_cells = expected[split]
        self.provenance = {
            "cache_sha256_from_pass_manifest": output["sha256"],
            "cache_hash_recomputed": False,
            "row_equals_global_embedding_index": True,
            "condition_index_sha256": data.sha256_file(Path(condition_index_path)),
            "gene_metadata_sha256": data.sha256_file(Path(gene_metadata_path)),
            "panel_sha256": data.sha256_file(self.panel_path),
            "latent_source": "Author GeneJEPA epoch49 EMA teacher",
            "latent_transforms": [],
            "target_transform": "full mapped library -> CP10000 -> log1p -> HD100 select",
        }


class AuthorExactTrainDataset(engine.ExactTrainTahoeDecoderDataset):
    """Exact old DDP stream ownership over the new Author cache."""

    def __init__(
        self,
        descriptor_train_counts: np.ndarray,
        *,
        batch_size: int,
        gradient_accumulation: int,
    ) -> None:
        AuthorTahoeDecoderDataset.__init__(
            self, split="train", panel_path=PANEL_PATH, shuffle_shards=False
        )
        self.descriptor_train_counts = np.asarray(descriptor_train_counts, dtype=np.int64)
        if self.descriptor_train_counts.shape != (len(self.descriptors),):
            raise AssertionError("Train descriptor counts do not match Author cache descriptors")
        self.batch_size = int(batch_size)
        self.gradient_accumulation = int(gradient_accumulation)
        self.layout = engine.partition_layout(self.batch_size, self.gradient_accumulation)


def build_decoder() -> torch.nn.Module:
    model = hd100.build_decoder()
    hd100.assert_decoder(model)
    return model


def checkpoint_payload(**kwargs: Any) -> dict[str, Any]:
    payload = _BASE_CHECKPOINT_PAYLOAD(**kwargs)
    payload["schema"] = "author_genejepa_hd100_checkpoint_v1"
    payload["experiment"] = "Author GeneJEPA epoch49 EMA teacher -> HD100"
    payload["latent_cache_manifest_sha256"] = data.sha256_file(CACHE_MANIFEST)
    payload["author_genejepa_checkpoint_sha256"] = author_cache.AUTHOR_CHECKPOINT_SHA256
    payload["random_initialization_from_scratch"] = True
    payload["old_decoder_weights_loaded"] = False
    return payload


def validate_checkpoint(payload: dict[str, Any], **kwargs: Any) -> None:
    if payload.get("schema") != "author_genejepa_hd100_checkpoint_v1":
        raise AssertionError("Checkpoint is not an Author-GeneJEPA HD100 checkpoint")
    if (
        payload.get("latent_cache_manifest_sha256") != data.sha256_file(CACHE_MANIFEST)
        or payload.get("author_genejepa_checkpoint_sha256")
        != author_cache.AUTHOR_CHECKPOINT_SHA256
        or payload.get("random_initialization_from_scratch") is not True
        or payload.get("old_decoder_weights_loaded") is not False
    ):
        raise AssertionError("Author-HD100 checkpoint provenance changed")
    compatible = dict(payload)
    compatible["schema"] = "genejepa_decoder_v1_checkpoint_v1"
    _BASE_VALIDATE_CHECKPOINT(compatible, **kwargs)


def configure_engine() -> None:
    engine.SCRIPT_PATH = SCRIPT_PATH
    engine.TASK_PATH = TASK_PATH
    engine.CONTRACT_PATH = CONTRACT_PATH
    engine.PANEL_PATH = PANEL_PATH
    engine.PANEL_SHA256 = data.sha256_file(PANEL_PATH)
    engine.PANEL_AUDIT_PATH = PANEL_SUMMARY_PATH
    engine.DATASET_PATH = SCRIPT_PATH
    engine.CACHE_PLAN_SUMMARY = CACHE_SUMMARY
    engine.CONDITION_INDEX = CONDITION_INDEX
    engine.FORMAL_EMBEDDINGS = CACHE_EMBEDDINGS
    engine.FORMAL_EMBEDDING_MANIFEST = CACHE_MANIFEST
    engine.FORMAL_PLANS = CACHE_PLANS
    engine.DESCRIPTOR_COUNTS_PATH = DESCRIPTOR_COUNTS_PATH
    engine.PARTITION_AUDIT_PATH = PARTITION_AUDIT_PATH
    engine.TRAINING_CONFIG_PATH = TRAINING_CONFIG_PATH
    engine.SMOKE_RESULT_PATH = DDP_SMOKE_PATH
    engine.SMOKE_CHECKPOINT_DIR = SMOKE_CHECKPOINT_DIR
    engine.FORMAL_CHECKPOINT_DIR = FORMAL_CHECKPOINT_DIR
    engine.SMOKE_TENSORBOARD = SMOKE_TENSORBOARD
    engine.FORMAL_TENSORBOARD = FORMAL_TENSORBOARD
    engine.FORMAL_RESULT_PATH = FORMAL_RESULT_PATH
    engine.GENE_DIM = GENE_DIM
    engine.EXPECTED_PARAMETER_COUNT = EXPECTED_PARAMETER_COUNT
    engine.TRAIN_CELLS = TRAIN_CELLS
    engine.VAL_CELLS = VAL_CELLS
    engine.TEST_CELLS = ARC7_TREATED_CELLS
    engine.TREATED_CELLS = TRAIN_CELLS + VAL_CELLS + ARC7_TREATED_CELLS
    engine.DMSO_CELLS = ARC7_CONTROL_CELLS
    engine.TahoeDecoderV1PairedDataset = AuthorTahoeDecoderDataset
    engine.ExactTrainTahoeDecoderDataset = AuthorExactTrainDataset
    engine.build_decoder = build_decoder
    engine.assert_decoder = hd100.assert_decoder
    engine.checkpoint_payload = checkpoint_payload
    engine.validate_checkpoint = validate_checkpoint
    engine.load_runtime_contract = load_runtime_contract


def contract_payload(panel_sha: str) -> dict[str, Any]:
    return {
        "schema": "author_genejepa_hd100_contract_v1",
        "created_at_utc": data.utc_now(),
        "status": "frozen",
        "experiment": "Author epoch49 full-size GeneJEPA backbone replacement",
        "author_genejepa": {
            "checkpoint": artifact(author_cache.AUTHOR_CHECKPOINT),
            "author_code_commit": author_cache.EXPECTED_AUTHOR_COMMIT,
            "use_teacher": True,
            "branch": "teacher_encoder.ema_model",
            "embedding_dim": data.LATENT_DIM,
            "output": "raw signed float32",
            "normalization_stats_policy": author_cache.NORMALIZATION_POLICY,
        },
        "cache": artifact(CACHE_MANIFEST),
        "panel": {**artifact(PANEL_PATH), "genes": GENE_DIM},
        "model": hd100.model_contract(),
        "training_target": "raw Tahoe counts -> full mapped CP10000 denominator -> log1p -> frozen HD100",
        "random_initialization_from_scratch": True,
        "old_decoder_weights_loaded": False,
        "task": artifact(TASK_PATH),
        "panel_sha256_redundant_check": panel_sha,
    }


def training_config_payload(partition: dict[str, Any], panel_sha: str) -> dict[str, Any]:
    return {
        "schema": "author_genejepa_hd100_training_config_v1",
        "created_at_utc": data.utc_now(),
        "status": "frozen",
        "experiment": "Author GeneJEPA epoch49 EMA teacher -> independently trained HD100",
        "scientific_contract": artifact(CONTRACT_PATH),
        "author_genejepa": {
            "checkpoint": artifact(author_cache.AUTHOR_CHECKPOINT),
            "author_code_commit": author_cache.EXPECTED_AUTHOR_COMMIT,
            "use_teacher": True,
            "branch": "teacher_encoder.ema_model",
        },
        "latent_cache": artifact(CACHE_MANIFEST),
        "panel": {**artifact(PANEL_PATH), "genes": GENE_DIM},
        "model": hd100.model_contract(),
        "data": {
            "dataset": "existing Tahoe target path + exact train ownership adapter",
            "dataset_path": data.display_path(SCRIPT_PATH),
            "dataset_sha256": data.sha256_file(SCRIPT_PATH),
            "latent": "raw signed Author epoch49 EMA-teacher [768]",
            "latent_transforms": [],
            "target": "full mapped counts -> CP10000 denominator -> log1p -> frozen HD100",
            "train_treated_cells": TRAIN_CELLS,
            "val_treated_cells": VAL_CELLS,
            "arc7_treated_cells_cached_but_not_used_for_training": ARC7_TREATED_CELLS,
            "DMSO_used_for_training": False,
            "descriptor_counts": artifact(DESCRIPTOR_COUNTS_PATH),
            "partition_audit": artifact(PARTITION_AUDIT_PATH),
        },
        "training": {
            "world_size": 2,
            "batch_size_per_gpu": 1024,
            "global_batch": 2048,
            "precision": "BF16 autocast forward; FP32 MSE",
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
            "old_decoder_weights_loaded": False,
        },
        "dataloader": {
            "workers_per_rank": 4,
            "pin_memory": True,
            "prefetch_factor": 2,
            "persistent_workers": False,
        },
        "validation": {
            "cells": VAL_CELLS,
            "criterion": "global FP64 SSE / global cell-by-gene element count",
            "shuffle": False,
            "drop_last": False,
            "strict_minimum_checkpoint": True,
        },
        "partition": {
            "audit": artifact(PARTITION_AUDIT_PATH),
            "rank_parity": partition["rank_parity"],
        },
        "checkpoint": {
            "best": data.display_path(FORMAL_CHECKPOINT_DIR / "best.pt"),
            "last": data.display_path(FORMAL_CHECKPOINT_DIR / "last.pt"),
            "resume": "complete epoch boundaries",
        },
        "implementation": {
            "runner_path": data.display_path(SCRIPT_PATH),
            "runner_sha256": data.sha256_file(SCRIPT_PATH),
            "reused_training_engine": artifact(Path(engine.__file__)),
            "reused_target_builder": artifact(Path(data.__file__)),
            "task_sha256": data.sha256_file(TASK_PATH),
        },
        "formal_training_gate": {
            "artifact": data.display_path(DDP_SMOKE_PATH),
            "required": "status=pass and ready_for_formal_training=true",
        },
        "panel_sha256_redundant_check": panel_sha,
    }


def prepare() -> dict[str, Any]:
    panel_sha = validate_panel()
    cache = validate_cache()
    configure_engine()
    write_once(CONTRACT_PATH, contract_payload(panel_sha))
    inputs = {"cache_summary": cache["summary"]}
    arrays, scan_seconds = engine.ensure_descriptor_counts(inputs)
    partition = engine.partition_audit(arrays, scan_seconds)
    write_once(TRAINING_CONFIG_PATH, training_config_payload(partition, panel_sha))
    result = {
        "status": "pass",
        "scope": "Author-HD100 trainer preparation only; no GPU training or ARC7 inference",
        "contract": artifact(CONTRACT_PATH),
        "training_config": artifact(TRAINING_CONFIG_PATH),
        "descriptor_counts": artifact(DESCRIPTOR_COUNTS_PATH),
        "partition_audit": artifact(PARTITION_AUDIT_PATH),
        "cache_manifest": artifact(CACHE_MANIFEST),
        "panel": artifact(PANEL_PATH),
        "ready_for_static_smoke": True,
        "formal_training_started": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def load_runtime_contract() -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    validate_panel()
    cache = validate_cache()
    required = [CONTRACT_PATH, TRAINING_CONFIG_PATH, DESCRIPTOR_COUNTS_PATH, PARTITION_AUDIT_PATH]
    missing = [data.display_path(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Run Author-HD100 prepare first: {missing}")
    config = read_json(TRAINING_CONFIG_PATH)
    expected = {
        "status": "frozen",
        "contract": data.sha256_file(CONTRACT_PATH),
        "cache": data.sha256_file(CACHE_MANIFEST),
        "panel": data.sha256_file(PANEL_PATH),
        "runner": data.sha256_file(SCRIPT_PATH),
        "descriptor": data.sha256_file(DESCRIPTOR_COUNTS_PATH),
        "partition": data.sha256_file(PARTITION_AUDIT_PATH),
    }
    observed = {
        "status": config.get("status"),
        "contract": config["scientific_contract"]["sha256"],
        "cache": config["latent_cache"]["sha256"],
        "panel": config["panel"]["sha256"],
        "runner": config["implementation"]["runner_sha256"],
        "descriptor": config["data"]["descriptor_counts"]["sha256"],
        "partition": config["data"]["partition_audit"]["sha256"],
    }
    if observed != expected:
        raise AssertionError(f"Author-HD100 runtime contract changed: {observed} != {expected}")
    with np.load(DESCRIPTOR_COUNTS_PATH, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    engine.validate_descriptor_counts(arrays, {"cache_summary": cache["summary"]})
    return config, arrays, data.sha256_file(TRAINING_CONFIG_PATH)


def static_smoke() -> dict[str, Any]:
    configure_engine()
    load_runtime_contract()
    dataset = AuthorTahoeDecoderDataset(
        split="train", panel_path=PANEL_PATH, shuffle_shards=False, max_cells=8
    )
    rows = list(dataset)
    if len(rows) != 8:
        raise AssertionError("Author-HD100 static smoke did not return eight cells")
    indices = np.asarray([row["global_embedding_index"] for row in rows], dtype=np.int64)
    latent = torch.stack([row["latent"] for row in rows])
    target = torch.stack([row["target_expression"] for row in rows])
    model = build_decoder().eval()
    with torch.inference_mode():
        prediction = model(latent)
        loss = torch.mean((prediction.float() - target.float()).square())
    if (
        latent.shape != (8, data.LATENT_DIM)
        or target.shape != (8, GENE_DIM)
        or prediction.shape != target.shape
        or not torch.isfinite(latent).all()
        or not torch.any(latent < 0)
        or not torch.any(latent > 0)
        or not torch.isfinite(target).all()
        or torch.any(target < 0)
        or not torch.isfinite(prediction).all()
        or torch.any(prediction < 0)
        or not torch.isfinite(loss)
        or np.any((indices < 0) | (indices >= TRAIN_CELLS))
    ):
        raise AssertionError("Author-HD100 real-batch static smoke failed")
    result = {
        "schema": "author_genejepa_hd100_static_smoke_v1",
        "created_at_utc": data.utc_now(),
        "status": "pass",
        "scope": "eight real train cells; no fitting, validation, ARC7, DE, or metrics",
        "embedding_indices": indices.tolist(),
        "latent_shape": list(latent.shape),
        "target_shape": list(target.shape),
        "prediction_shape": list(prediction.shape),
        "latent_dtype": str(latent.dtype),
        "latent_finite": True,
        "latent_signed": True,
        "prediction_finite_nonnegative": True,
        "initial_mse": float(loss),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "random_initialization_from_scratch": True,
        "old_decoder_weights_loaded": False,
        "provenance": {
            "cache_manifest": artifact(CACHE_MANIFEST),
            "panel": artifact(PANEL_PATH),
            "training_config": artifact(TRAINING_CONFIG_PATH),
            "runner": artifact(SCRIPT_PATH),
        },
        "ready_for_DDP_smoke": True,
        "formal_training_started": False,
    }
    data.atomic_write_json(STATIC_SMOKE_PATH, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def rewrite_smoke_result(result: dict[str, Any] | None) -> None:
    if result is None:
        return
    result["schema"] = "author_genejepa_hd100_exact_config_smoke_v1"
    result["experiment"] = "Author GeneJEPA epoch49 EMA teacher -> HD100"
    result["author_genejepa_checkpoint"] = artifact(author_cache.AUTHOR_CHECKPOINT)
    result["author_latent_cache_manifest"] = artifact(CACHE_MANIFEST)
    result["static_smoke"] = artifact(STATIC_SMOKE_PATH)
    result["random_initialization_from_scratch"] = True
    result["old_decoder_weights_loaded"] = False
    data.atomic_write_json(DDP_SMOKE_PATH, result)


def rewrite_formal_result(result: dict[str, Any] | None) -> None:
    if result is None:
        return
    result["schema"] = "author_genejepa_hd100_training_result_v1"
    result["experiment"] = "Author GeneJEPA epoch49 EMA teacher -> independently trained HD100"
    result["author_genejepa_checkpoint"] = artifact(author_cache.AUTHOR_CHECKPOINT)
    result["author_latent_cache_manifest"] = artifact(CACHE_MANIFEST)
    result["panel"] = artifact(PANEL_PATH)
    result["random_initialization_from_scratch"] = True
    result["old_decoder_weights_loaded"] = False
    result["comparison_guardrail"] = (
        "This is a backbone-level comparison: half-size Epoch25 versus full-size final Epoch49, "
        "not a checkpoint-weight-only ablation."
    )
    data.atomic_write_json(FORMAL_RESULT_PATH, result)


def configure_for_runtime() -> None:
    configure_engine()
    if not STATIC_SMOKE_PATH.is_file() or read_json(STATIC_SMOKE_PATH).get("status") != "pass":
        raise FileNotFoundError("Run prepare and static-smoke before the DDP commands")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("static-smoke")
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--resume", action="store_true")
    smoke.add_argument("--train-steps", type=int)
    smoke.add_argument("--val-batches", type=int, default=4)
    formal = commands.add_parser("run")
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
        result = engine.run_smoke(
            resume=args.resume,
            train_steps=steps,
            val_batches=args.val_batches,
            fallback_512=False,
        )
        rewrite_smoke_result(result)
    elif args.command == "run":
        configure_for_runtime()
        result = engine.run_formal(resume=args.resume, fallback_512=False)
        rewrite_formal_result(result)


if __name__ == "__main__":
    main()
