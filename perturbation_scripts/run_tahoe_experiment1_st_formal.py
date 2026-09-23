#!/usr/bin/env python3
"""Freeze, smoke, or run formal Experiment 1 ST-A/ST-R training."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from geomloss import SamplesLoss
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from evaluate_tahoe_experiment1_b0 import aggregate
from smoke_tahoe_experiment1_state import build_model
from tahoe_experiment1_latent_data import (
    LATENT_DIM,
    PERT_DIM,
    PROJECT_ROOT,
    SET_SIZE,
    TahoeExperiment1LatentSetDataset,
    make_dataloader,
)
from train_tahoe_experiment1_b2 import (
    assert_unique_set_indices,
    atomic_write_json,
    atomic_write_text,
    seed_everything,
    train_redraw_audit,
)


RESULTS = PROJECT_ROOT / "results"
SCRIPT_PATH = Path(__file__).resolve()
TASK_PATH = PROJECT_ROOT.parent / "当前任务.txt"
STATE_MODEL_PATH = (
    PROJECT_ROOT.parent / "state-main" / "src" / "state" / "tx" / "models" / "state_transition.py"
)
DATASET_PATH = PROJECT_ROOT / "perturbation_scripts" / "tahoe_experiment1_latent_data.py"
MODEL_BUILDER_PATH = PROJECT_ROOT / "perturbation_scripts" / "smoke_tahoe_experiment1_state.py"
AGGREGATION_PATH = PROJECT_ROOT / "perturbation_scripts" / "evaluate_tahoe_experiment1_b0.py"
READINESS_PATH = RESULTS / "tahoe_experiment1_st_formal_training_readiness.json"
PROTOCOL_V1_PATH = RESULTS / "tahoe_experiment1_st_training_protocol.json"
PROTOCOL_PATH = RESULTS / "tahoe_experiment1_st_training_protocol_v2.json"
SMOKE_PATH = RESULTS / "tahoe_experiment1_st_final_config_smoke.json"
SMOKE_HANDOFF_PATH = RESULTS / "tahoe_experiment1_st_final_config_smoke.md"
TENSORBOARD_ROOT = RESULTS / "tensorboard" / "tahoe_experiment1_st"
FORMAL_CHECKPOINT_ROOT = RESULTS / "tahoe_experiment1_st_formal_checkpoints_v2"

FROZEN_AT_UTC = "2026-09-10T08:51:54.4272959+00:00"
TASK_SHA256 = "bb5763abb85e60d9553fe5ff362f6f2afcadf49f6ed8f75406800935e3ec284f"
STATE_SHA256 = "af6959663585ce141bd6701492fd2ccc36a0ff86e7927c87bf23b449210d4b5e"
DATASET_SHA256 = "e0cb346797a3517e7d4aeddc4a40d0560f11f483520da8a43f0bc808f14e641d"
MODEL_BUILDER_SHA256 = "f34ec72b3070c7cf31b2fe77181600cc65e1d8bff5b3ec2bab7fa43df2c33d5e"
AGGREGATION_SHA256 = "e35dc32ace9eb941175f57cef4e31836f25d76f459631ea6f3c45874bb7af900"
READINESS_SHA256 = "095fea7daf4e50d426613b0fb8aba94b989ea9a8f581373e208af59e6a2ff9eb"

TRAIN_CONDITIONS = 45_652
VAL_CONDITIONS = 5_657
TEST_CONDITIONS = 5_684
TRAIN_EDGES = 13_740
VAL_EDGES = 1_717
WORLD_SIZE = 2
BATCH_SIZE_PER_GPU = 64
MICROBATCH_GLOBAL_SIZE = BATCH_SIZE_PER_GPU * WORLD_SIZE
GRADIENT_ACCUMULATION_STEPS = 2
EFFECTIVE_GLOBAL_BATCH_SIZE = MICROBATCH_GLOBAL_SIZE * GRADIENT_ACCUMULATION_STEPS
NUM_WORKERS_PER_RANK = 2
PIN_MEMORY = True
PERSISTENT_WORKERS = False
PREFETCH_FACTOR = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
TRAINING_SEED = 42
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
ENERGY_BLUR = 0.05
SMOKE_TRAIN_CONDITIONS = 1_280
SMOKE_VAL_CONDITIONS = 64
SMOKE_MICROBATCHES = 10
SMOKE_OPTIMIZER_UPDATES = 5
EXPECTED_PARAMETERS = 101_560_320
EXPECTED_TRAINABLE_PARAMETERS = 76_984_320


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return "../" + resolved.relative_to(PROJECT_ROOT.parent.resolve()).as_posix()


def artifact_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": relative(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def tensorboard_environment_audit() -> dict[str, Any]:
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        from tensorboard.backend.event_processing.event_accumulator import (  # noqa: F401
            EventAccumulator,
        )
    except (ImportError, ModuleNotFoundError) as error:
        return {
            "status": "fail",
            "available": False,
            "writer": "torch.utils.tensorboard.SummaryWriter",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return {
        "status": "pass",
        "available": True,
        "writer": "torch.utils.tensorboard.SummaryWriter",
        "tensorboard_version": importlib.metadata.version("tensorboard"),
    }


def require_tensorboard() -> tuple[Any, Any, dict[str, Any]]:
    audit = tensorboard_environment_audit()
    if not audit["available"]:
        raise RuntimeError(f"TensorBoard dependency blocker: {audit['error']}")
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    return SummaryWriter, EventAccumulator, audit


def write_blocked_smoke_outputs(protocol_sha256: str, audit: dict[str, Any]) -> None:
    blocker = {
        "component": "TensorBoard",
        "required_import": "torch.utils.tensorboard.SummaryWriter",
        "reason": audit["error"],
        "environment_was_modified": False,
    }
    payload = {
        "schema": "tahoe_experiment1_st_final_config_smoke_v2",
        "created_at_utc": utc_now(),
        "status": "fail",
        "scope": "exact-config smoke preflight",
        "protocol": {"path": relative(PROTOCOL_PATH), "sha256": protocol_sha256},
        "variants": {
            "st-a": {"status": "not_run", "reason": "TensorBoard dependency blocker"},
            "st-r": {"status": "not_run", "reason": "TensorBoard dependency blocker"},
        },
        "tensorboard": audit,
        "blockers": [blocker],
        "scientific_result": False,
        "formal_training_started": False,
        "test_used": False,
        "test_dataset_constructed": False,
        "deepspeed": False,
    }
    atomic_write_json(SMOKE_PATH, payload)
    atomic_write_text(
        SMOKE_HANDOFF_PATH,
        f"""# Tahoe Experiment 1 ST final-config smoke

Status: **FAIL / BLOCKED**

The v2 protocol is frozen, but neither ST-A nor ST-R exact-config smoke was started because
`torch.utils.tensorboard.SummaryWriter` cannot import: `{audit['error']}`.

The environment was not modified. Formal training and test were not started.

Protocol: `{relative(PROTOCOL_PATH)}` (`{protocol_sha256}`)
""",
    )


def verify_source_provenance(*, verify_task: bool) -> None:
    expected = {
        STATE_MODEL_PATH: STATE_SHA256,
        DATASET_PATH: DATASET_SHA256,
        MODEL_BUILDER_PATH: MODEL_BUILDER_SHA256,
        AGGREGATION_PATH: AGGREGATION_SHA256,
        READINESS_PATH: READINESS_SHA256,
    }
    if verify_task:
        expected[TASK_PATH] = TASK_SHA256
    for path, digest in expected.items():
        observed = sha256_file(path)
        if observed != digest:
            raise AssertionError(f"Frozen source SHA changed: {path}: {observed} != {digest}")


def optimizer_contract() -> dict[str, Any]:
    return {
        "class": "torch.optim.AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "scheduler": None,
    }


def protocol_payload() -> dict[str, Any]:
    return {
        "schema": "tahoe_experiment1_st_training_protocol_v2",
        "protocol_version": "v2",
        "status": "frozen",
        "frozen_at_utc": FROZEN_AT_UTC,
        "source_task": {"path": relative(TASK_PATH), "sha256": TASK_SHA256},
        "variants": {
            "st-a": {
                "predict_residual": False,
                "formula": "Zpred = project_out(ST_hidden)",
            },
            "st-r": {
                "predict_residual": True,
                "residual_mode": "output",
                "formula": "Zpred = raw_Zctrl + project_out(ST_hidden)",
            },
            "only_controlled_difference": "absolute vs raw output-space residual",
        },
        "model": {
            "class": "StateTransitionPerturbationModel",
            "input_dim": LATENT_DIM,
            "output_dim": LATENT_DIM,
            "cell_set_len": SET_SIZE,
            "pert_dim": PERT_DIM,
            "use_basal_projection": True,
            "basal_encoder": "Sequential(Linear(768,768)); no activation/dropout",
            "perturbation_encoder": "Sequential(Linear(380,768)); no activation/dropout",
            "project_out": "Sequential(Linear(768,768)); no activation/dropout",
            "n_encoder_layers": 1,
            "n_decoder_layers": 1,
            "final_activation": "identity",
            "output_space": "embedding",
            "embed_key": "X_genejepa_epoch25",
            "transformer": {
                "class": "LlamaBidirectionalModel",
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_hidden_layers": 8,
                "num_attention_heads": 12,
                "num_key_value_heads": 12,
                "head_dim": 64,
                "max_position_embeddings": 256,
                "attention_dropout": 0.0,
                "hidden_dropout": 0.0,
                "layer_norm_eps": 1e-6,
                "use_cache": False,
                "use_rotary_embeddings": False,
                "hidden_act": "silu",
            },
            "disabled": [
                "gene/count decoder",
                "batch encoder",
                "batch predictor",
                "batch token",
                "confidence token",
                "LoRA",
                "log1p_from_raw_counts",
                "auxiliary loss",
            ],
            "parameter_count": EXPECTED_PARAMETERS,
            "trainable_parameter_count": EXPECTED_TRAINABLE_PARAMETERS,
        },
        "data": {
            "dataset_class": "TahoeExperiment1LatentSetDataset",
            "dataset_sha256": DATASET_SHA256,
            "train_conditions": TRAIN_CONDITIONS,
            "val_conditions": VAL_CONDITIONS,
            "test_conditions": TEST_CONDITIONS,
            "train_edges": TRAIN_EDGES,
            "val_edges": VAL_EDGES,
            "input": {
                "ctrl_cell_emb": ["B", SET_SIZE, LATENT_DIM],
                "pert_emb": ["B", SET_SIZE, PERT_DIM],
            },
            "target": {"pert_cell_emb": ["B", SET_SIZE, LATENT_DIM]},
            "target_semantics": "raw real treated frozen GeneJEPA Epoch25 latent",
            "latent_transforms": [],
            "prohibited": [
                "centering",
                "whitening",
                "L2 normalization",
                "latent standardization",
                "DMSO residualization",
                "clipping",
                "ReLU",
                "condition-shift MSE",
                "B1/B2 supervision",
            ],
        },
        "loss": {
            "class": "geomloss.SamplesLoss",
            "loss": "energy",
            "blur": ENERGY_BLUR,
            "reduction": "one Energy value per set, then batch mean",
            "auxiliary_losses": [],
        },
        "optimizer": optimizer_contract(),
        "training": {
            "training_seed": TRAINING_SEED,
            "same_seed_for_both_variants": True,
            "linear_lr_scaling": False,
            "precision": "bfloat16_autocast",
            "autocast": True,
            "autocast_dtype": "bfloat16",
            "energy_dtype": "float32",
            "grad_scaler": False,
            "tf32": False,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "validation_frequency": "once after every epoch",
            "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
            "world_size": WORLD_SIZE,
            "microbatch_global_size": MICROBATCH_GLOBAL_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_global_batch_size": EFFECTIVE_GLOBAL_BATCH_SIZE,
            "st_a_and_st_r_share_all_non_variant_contract": True,
        },
        "dataloader": {
            "num_workers_per_rank": NUM_WORKERS_PER_RANK,
            "total_train_workers": NUM_WORKERS_PER_RANK * WORLD_SIZE,
            "pin_memory": PIN_MEMORY,
            "persistent_workers": PERSISTENT_WORKERS,
            "prefetch_factor": PREFETCH_FACTOR,
            "gpu_transfer_non_blocking": True,
        },
        "train_sampling": {
            "conditions": TRAIN_CONDITIONS,
            "set_size": SET_SIZE,
            "dataset_set_epoch": "training_epoch",
            "sampler_set_epoch": "training_epoch",
            "shuffle": True,
            "drop_last": True,
            "microbatches_per_epoch": 356,
            "optimizer_updates_per_epoch": 178,
            "maximum_optimizer_updates": 5_340,
            "conditions_consumed_per_epoch": 45_568,
            "conditions_dropped_per_epoch": 84,
            "dropped_conditions_are_not_padded": True,
        },
        "validation": {
            "conditions": VAL_CONDITIONS,
            "set_size": SET_SIZE,
            "dataset_epoch": 0,
            "shuffle": False,
            "drop_last": False,
            "execution": "rank0 exact full validation; rank1 waits",
            "unique_conditions_exactly_once": True,
            "metric": "val edge-level mean Energy",
            "aggregation": [
                "condition",
                "biological replicate",
                "(cell_line_id,drug,dose_uM)",
                "equal dose average",
                "(cell_line_id,drug) edge",
            ],
            "plate6_plate14_rule": "pair only dose_uM == 5.0 when both plates exist",
            "checkpoint_selection": "strict minimum val edge-level mean Energy; ties keep earlier epoch",
            "early_stop": "five consecutive epochs without strict improvement",
        },
        "distributed": {
            "launcher": "torchrun --standalone --nproc_per_node=2",
            "backend": "nccl",
            "implementation": "native PyTorch DistributedDataParallel",
            "deepspeed": False,
            "zero_stage": None,
            "CUDA_VISIBLE_DEVICES": "0,1",
            "NCCL_CUMEM_ENABLE": "0",
            "NCCL_CUMEM_HOST_ENABLE": "0",
        },
        "checkpoint": {
            "format": "plain PyTorch .pt v2; no Lightning or DeepSpeed checkpoint machinery",
            "files_per_variant": ["best.pt", "last.pt"],
            "atomic_save": True,
            "resume_source": "same variant last.pt only",
            "separate_variant_directories": True,
            "saved_fields": [
                "variant",
                "protocol_version",
                "model_state",
                "optimizer_state",
                "epoch",
                "global_step",
                "current_val_edge_energy",
                "best_val_edge_energy",
                "best_epoch",
                "no_improvement_epochs",
                "training_seed",
                "formal_protocol_sha256",
                "precision",
                "energy_dtype",
                "num_workers_per_rank",
                "pin_memory",
                "persistent_workers",
                "prefetch_factor",
                "batch_size_per_gpu",
                "world_size",
                "gradient_accumulation_steps",
                "effective_global_batch_size",
                "deepspeed",
                "tensorboard_log_dir",
                "state_implementation_sha256",
                "dataset_sha256",
                "runner_sha256",
                "model_kwargs",
                "optimizer_contract",
                "history",
                "per-rank Python/NumPy/torch CPU/torch CUDA RNG states",
            ],
        },
        "integration_smoke": {
            "train_conditions": SMOKE_TRAIN_CONDITIONS,
            "val_conditions": SMOKE_VAL_CONDITIONS,
            "microbatches_per_variant": SMOKE_MICROBATCHES,
            "optimizer_updates_per_variant": SMOKE_OPTIMIZER_UPDATES,
            "world_size": WORLD_SIZE,
            "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "precision": "BF16 model forward + FP32 Energy",
            "scientific_result": False,
        },
        "tensorboard": {
            "writer": "torch.utils.tensorboard.SummaryWriter",
            "rank0_only": True,
            "root": relative(TENSORBOARD_ROOT),
            "formal_subdirectories": ["formal/st-a", "formal/st-r"],
            "smoke_subdirectories": ["smoke/st-a", "smoke/st-r"],
            "training_test_curves_prohibited": True,
            "required_tags": [
                "train/energy_step",
                "train/energy_epoch",
                "val/condition_energy",
                "val/edge_energy",
                "train/lr",
                "train/grad_norm_before_clip",
                "train/grad_norm_after_clip",
                "system/conditions_per_second",
                "system/peak_allocated_gib",
                "system/peak_reserved_gib",
                "best/val_edge_energy",
                "best/epoch",
            ],
        },
        "test_policy": {
            "test_dataset_constructed_during_training": False,
            "test_used_for_hyperparameters_or_checkpoint_selection": False,
        },
        "historical_baselines_reference_only": {
            "B0_test_mean_energy": 0.0359202899,
            "B1_test_mean_energy": 0.0335430391,
            "B2_v2_test_mean_energy": 0.0327345830,
            "used_for_training_decisions": False,
        },
        "implementation": {
            "formal_runner": artifact_record(SCRIPT_PATH),
            "historical_v1_protocol": artifact_record(PROTOCOL_V1_PATH),
            "state_model": artifact_record(STATE_MODEL_PATH),
            "dataset": artifact_record(DATASET_PATH),
            "accepted_reference_model_builder": artifact_record(MODEL_BUILDER_PATH),
            "frozen_aggregation": artifact_record(AGGREGATION_PATH),
            "readiness_audit": artifact_record(READINESS_PATH),
            "generic_state_yaml_used": False,
        },
    }


def ensure_protocol(*, create: bool, rewrite: bool = False) -> tuple[dict[str, Any], str]:
    expected = protocol_payload()
    if rewrite or (create and not PROTOCOL_PATH.exists()):
        atomic_write_json(PROTOCOL_PATH, expected)
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(f"Run prepare first: {PROTOCOL_PATH}")
    observed = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if observed != expected:
        raise AssertionError("Frozen ST protocol or one of its implementation files changed")
    return observed, sha256_file(PROTOCOL_PATH)


def _assert_single_linear(module: nn.Module, in_features: int, out_features: int, name: str) -> None:
    if not isinstance(module, nn.Sequential) or len(module) != 1:
        raise AssertionError(f"{name} is not a one-layer Sequential")
    layer = module[0]
    if not isinstance(layer, nn.Linear) or (layer.in_features, layer.out_features) != (
        in_features,
        out_features,
    ):
        raise AssertionError(f"{name} Linear dimensions changed")


def build_frozen_model(variant: str) -> tuple[nn.Module, dict[str, Any]]:
    seed_everything(TRAINING_SEED)
    model, kwargs = build_model(variant)
    if (model.input_dim, model.output_dim, model.cell_sentence_len, model.pert_dim) != (
        LATENT_DIM,
        LATENT_DIM,
        SET_SIZE,
        PERT_DIM,
    ):
        raise AssertionError("ST model dimensions changed")
    _assert_single_linear(model.basal_encoder, 768, 768, "basal_encoder")
    _assert_single_linear(model.pert_encoder, 380, 768, "pert_encoder")
    _assert_single_linear(model.project_out, 768, 768, "project_out")
    if type(model.transformer_backbone).__name__ != "LlamaBidirectionalModel":
        raise AssertionError("Formal ST requires LlamaBidirectionalModel")
    config = model.transformer_backbone.config
    expected_config = {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 8,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,
        "head_dim": 64,
        "max_position_embeddings": 256,
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
        "layer_norm_eps": 1e-6,
        "use_cache": False,
        "use_rotary_embeddings": False,
        "hidden_act": "silu",
    }
    for name, expected in expected_config.items():
        if getattr(config, name, None) != expected:
            raise AssertionError(f"Llama config changed: {name}")
    if (
        model.final_activation_name != "identity"
        or model.apply_output_relu
        or model.output_space != "embedding"
        or model.gene_decoder is not None
        or model.batch_encoder is not None
        or model.batch_predictor
        or model.use_batch_token
        or model.confidence_token is not None
        or model.regularization != 0.0
        or model.mmd_num_chunks != 1
        or model.randomize_mmd_chunks
    ):
        raise AssertionError("Disabled heads/transforms or signed-output contract changed")
    if not isinstance(model.loss_fn, SamplesLoss) or model.distributional_loss != "energy":
        raise AssertionError("Formal ST requires real geomloss Energy")
    if model.loss_fn.loss != "energy" or model.loss_fn.blur != ENERGY_BLUR:
        raise AssertionError("Energy configuration changed")
    if variant == "st-a" and model.predict_residual:
        raise AssertionError("ST-A unexpectedly predicts a residual")
    if variant == "st-r" and (
        not model.predict_residual or model.residual_mode != "output"
    ):
        raise AssertionError("ST-R is not the raw output-space residual")
    parameters = sum(value.numel() for value in model.parameters())
    trainable = sum(value.numel() for value in model.parameters() if value.requires_grad)
    if (parameters, trainable) != (EXPECTED_PARAMETERS, EXPECTED_TRAINABLE_PARAMETERS):
        raise AssertionError(f"ST parameter count changed: {(parameters, trainable)}")
    effective_kwargs = copy.deepcopy(kwargs)
    effective_kwargs["effective_hidden_act"] = config.hidden_act
    return model, effective_kwargs


def model_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def make_datasets(
    *, train_limit: int | None = None, val_limit: int | None = None
) -> tuple[TahoeExperiment1LatentSetDataset, TahoeExperiment1LatentSetDataset]:
    train = TahoeExperiment1LatentSetDataset(split="train", seed=TRAINING_SEED, epoch=0)
    val = TahoeExperiment1LatentSetDataset(split="val", seed=TRAINING_SEED, epoch=0)
    expected = {"train": TRAIN_CONDITIONS, "val": VAL_CONDITIONS, "test": TEST_CONDITIONS}
    if train.split_counts != expected or val.split_counts != expected:
        raise AssertionError("Frozen split counts changed")
    if train.embedding_transforms or val.embedding_transforms:
        raise AssertionError("Formal raw latent Dataset unexpectedly applies transforms")
    if train_limit is not None:
        train.conditions = train.conditions.iloc[:train_limit].copy().reset_index(drop=True)
    if val_limit is not None:
        val.conditions = val.conditions.iloc[:val_limit].copy().reset_index(drop=True)
    if not train.conditions["split"].eq("train").all() or not val.conditions["split"].eq("val").all():
        raise AssertionError("A Dataset contains the wrong split")
    return train, val


def training_shape() -> dict[str, int]:
    samples_per_rank = TRAIN_CONDITIONS // WORLD_SIZE
    microbatches = samples_per_rank // BATCH_SIZE_PER_GPU
    optimizer_updates = microbatches // GRADIENT_ACCUMULATION_STEPS
    consumed = microbatches * BATCH_SIZE_PER_GPU * WORLD_SIZE
    return {
        "samples_per_rank_before_dataloader_drop_last": samples_per_rank,
        "microbatches_per_epoch": microbatches,
        "optimizer_updates_per_epoch": optimizer_updates,
        "maximum_optimizer_updates": optimizer_updates * MAX_EPOCHS,
        "conditions_consumed_per_epoch": consumed,
        "conditions_dropped_per_epoch": TRAIN_CONDITIONS - consumed,
    }


def prepare(*, rewrite: bool) -> dict[str, Any]:
    verify_source_provenance(verify_task=True)
    train, val = make_datasets()
    variant_audits: dict[str, Any] = {}
    fingerprints: dict[str, str] = {}
    for variant in ("st-a", "st-r"):
        model, kwargs = build_frozen_model(variant)
        fingerprints[variant] = model_fingerprint(model)
        variant_audits[variant] = {
            "parameters": sum(value.numel() for value in model.parameters()),
            "trainable_parameters": sum(
                value.numel() for value in model.parameters() if value.requires_grad
            ),
            "initial_state_sha256": fingerprints[variant],
            "model_kwargs": kwargs,
        }
        del model
        gc.collect()
    if fingerprints["st-a"] != fingerprints["st-r"]:
        raise AssertionError("ST-A/ST-R seed=42 initial parameters are not identical")
    protocol, protocol_sha = ensure_protocol(create=True, rewrite=rewrite)
    tensorboard_audit = tensorboard_environment_audit()
    if not tensorboard_audit["available"]:
        write_blocked_smoke_outputs(protocol_sha, tensorboard_audit)
    result = {
        "created_at_utc": utc_now(),
        "status": "pass",
        "scope": "formal ST contract preparation only; no optimizer step and no test Dataset",
        "protocol": {"path": relative(PROTOCOL_PATH), "sha256": protocol_sha},
        "train_conditions": len(train),
        "val_conditions": len(val),
        "test_dataset_constructed": False,
        "variant_audits": variant_audits,
        "identical_seed42_initialization": True,
        "training_shape": training_shape(),
        "tensorboard_environment": tensorboard_audit,
        "exact_config_smoke_blocked": not tensorboard_audit["available"],
        "generic_state_yaml_used": protocol["implementation"]["generic_state_yaml_used"],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in ("ctrl_cell_emb", "pert_cell_emb", "pert_emb")
    }


def forward_energy(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    forward_audit: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    core = model.module if isinstance(model, DistributedDataParallel) else model
    batch_size = batch["ctrl_cell_emb"].shape[0]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        autocast_enabled = torch.is_autocast_enabled("cuda")
        prediction = model(batch).reshape(batch_size, SET_SIZE, LATENT_DIM)
    prediction_for_loss = prediction.float()
    target_for_loss = batch["pert_cell_emb"].reshape(batch_size, SET_SIZE, LATENT_DIM).float()
    if forward_audit is not None:
        forward_audit["autocast_enabled_inside_model_context"] = autocast_enabled
        forward_audit["autocast_dtype"] = "bfloat16"
        forward_audit["prediction_dtype"] = str(prediction.dtype).removeprefix("torch.")
        forward_audit["energy_prediction_dtype"] = str(
            prediction_for_loss.dtype
        ).removeprefix("torch.")
        forward_audit["energy_target_dtype"] = str(target_for_loss.dtype).removeprefix("torch.")
    if prediction_for_loss.dtype != torch.float32 or target_for_loss.dtype != torch.float32:
        raise AssertionError("Formal Energy inputs must be explicit float32 tensors")
    per_set = core._compute_distribution_loss(prediction_for_loss, target_for_loss).reshape(-1)
    if per_set.shape != (batch_size,):
        raise AssertionError(f"Energy did not return one value per set: {per_set.shape}")
    return prediction, per_set.mean(), per_set


@torch.no_grad()
def validate_exact(
    model: nn.Module,
    dataset: TahoeExperiment1LatentSetDataset,
    device: torch.device,
) -> dict[str, Any]:
    dataset.set_epoch(0)
    loader = make_dataloader(
        dataset,
        batch_size=BATCH_SIZE_PER_GPU,
        shuffle=False,
        seed=TRAINING_SEED,
        drop_last=False,
        num_workers=NUM_WORKERS_PER_RANK,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
    )
    if loader.drop_last or type(loader.sampler).__name__ != "SequentialSampler":
        raise AssertionError("Validation must be sequential with drop_last=False")
    metadata = dataset.conditions.set_index("pair_id")
    records: list[dict[str, Any]] = []
    condition_ids: list[str] = []
    energies: list[float] = []
    index_digest = hashlib.sha256()
    was_training = model.training
    model.eval()
    for raw_batch in loader:
        if set(raw_batch["split"]) != {"val"}:
            raise AssertionError("A non-val condition entered validation")
        assert_unique_set_indices(raw_batch["source_embedding_index"], "control")
        assert_unique_set_indices(raw_batch["target_embedding_index"], "treated")
        index_digest.update(
            np.asarray(raw_batch["source_embedding_index"].numpy(), dtype="<i8").tobytes()
        )
        index_digest.update(
            np.asarray(raw_batch["target_embedding_index"].numpy(), dtype="<i8").tobytes()
        )
        batch = move_batch(raw_batch, device)
        prediction, _loss, values = forward_energy(model, batch)
        if (
            tuple(prediction.shape[1:]) != (SET_SIZE, LATENT_DIM)
            or not torch.isfinite(prediction).all()
            or not torch.isfinite(values).all()
        ):
            raise AssertionError("Validation prediction/Energy is invalid")
        values_cpu = values.detach().cpu().numpy().astype(np.float64, copy=False)
        energies.extend(values_cpu.tolist())
        for condition_id, value in zip(raw_batch["condition_id"], values_cpu, strict=True):
            condition_id = str(condition_id)
            condition_ids.append(condition_id)
            row = metadata.loc[condition_id]
            records.append(
                {
                    "condition_id": condition_id,
                    "edge_id": str(row["edge_id"]),
                    "cell_line_id": str(row["cell_line_id"]),
                    "drug": str(row["drug"]),
                    "dose_uM": float(row["dose_uM"]),
                    "plate": str(row["plate"]),
                    "repeat_epoch": 0,
                    "energy_distance": float(value),
                }
            )
    if was_training:
        model.train()
    expected_ids = dataset.conditions["pair_id"].astype(str).tolist()
    if condition_ids != expected_ids or len(set(condition_ids)) != len(dataset):
        raise AssertionError("Validation did not evaluate each selected condition exactly once")
    raw = pd.DataFrame.from_records(records)
    edge, details = aggregate(raw)
    energy_array = np.asarray(energies, dtype="<f8")
    edge_mean = float(edge["energy_distance"].mean())
    if not np.isfinite(energy_array).all() or not math.isfinite(edge_mean):
        raise AssertionError("Validation Energy is non-finite")
    return {
        "condition_count": len(condition_ids),
        "unique_condition_count": len(set(condition_ids)),
        "edge_count": len(edge),
        "condition_ids_sha256": hashlib.sha256(
            "\n".join(condition_ids).encode("utf-8")
        ).hexdigest(),
        "cell_index_sha256": index_digest.hexdigest(),
        "condition_energy_sha256": hashlib.sha256(energy_array.tobytes()).hexdigest(),
        "condition_energy_mean": float(energy_array.mean()),
        "edge_level_mean_energy": edge_mean,
        "plate6_plate14_replicate_groups": int(
            details["replicate_audit"].iloc[0]["plate6_plate14_groups"]
        ),
        "dataset_epoch": dataset.epoch,
        "shuffle": False,
        "drop_last": False,
        "exact_once": True,
        "finite": True,
        "condition_energy_values": energy_array.tolist(),
    }


def validation_reproducibility(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    index_keys = (
        "condition_count",
        "unique_condition_count",
        "condition_ids_sha256",
        "cell_index_sha256",
    )
    indices_equal = all(first[key] == second[key] for key in index_keys)
    first_energy = np.asarray(first["condition_energy_values"], dtype=np.float64)
    second_energy = np.asarray(second["condition_energy_values"], dtype=np.float64)
    absolute = np.abs(first_energy - second_energy)
    max_absolute = float(absolute.max(initial=0.0))
    energy_close = bool(np.allclose(first_energy, second_energy, rtol=1e-6, atol=1e-7))
    edge_absolute = abs(first["edge_level_mean_energy"] - second["edge_level_mean_energy"])
    if not indices_equal or not energy_close or not math.isclose(
        first["edge_level_mean_energy"],
        second["edge_level_mean_energy"],
        rel_tol=1e-6,
        abs_tol=1e-7,
    ):
        raise AssertionError("Fixed BF16 validation exceeded the frozen reproducibility tolerance")
    return {
        "status": "pass",
        "indices_identical": indices_equal,
        "condition_order_identical": indices_equal,
        "rtol": 1e-6,
        "atol": 1e-7,
        "condition_energy_within_tolerance": energy_close,
        "condition_energy_max_absolute_difference": max_absolute,
        "edge_energy_max_absolute_difference": float(edge_absolute),
    }


def public_validation(result: dict[str, Any]) -> dict[str, Any]:
    clean = dict(result)
    clean.pop("condition_energy_values", None)
    return clean


def gradient_norm(model: nn.Module) -> float:
    squares = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is not None:
            squares += parameter.grad.detach().double().square().sum()
    return float(squares.sqrt())


def dataloader_runtime_audit(
    loader: Any, iterator: Any, raw_batch: dict[str, Any]
) -> dict[str, Any]:
    workers = list(getattr(iterator, "_workers", []))
    pids = [int(worker.pid) for worker in workers if worker.pid is not None]
    alive = [bool(worker.is_alive()) for worker in workers]
    pinned = all(
        raw_batch[key].is_pinned()
        for key in ("ctrl_cell_emb", "pert_cell_emb", "pert_emb")
    )
    audit = {
        "configured_num_workers": loader.num_workers,
        "active_worker_process_count": len(pids),
        "worker_pids": pids,
        "all_worker_processes_alive": bool(alive) and all(alive),
        "pin_memory": loader.pin_memory,
        "batch_tensors_pinned": pinned,
        "persistent_workers": loader.persistent_workers,
        "prefetch_factor": loader.prefetch_factor,
    }
    expected = {
        "configured_num_workers": NUM_WORKERS_PER_RANK,
        "active_worker_process_count": NUM_WORKERS_PER_RANK,
        "all_worker_processes_alive": True,
        "pin_memory": PIN_MEMORY,
        "batch_tensors_pinned": True,
        "persistent_workers": PERSISTENT_WORKERS,
        "prefetch_factor": PREFETCH_FACTOR,
    }
    for key, value in expected.items():
        if audit[key] != value:
            raise AssertionError(f"DataLoader runtime contract failed: {key}={audit[key]!r}")
    return audit


def tensorboard_event_audit(log_dir: Path, event_accumulator_class: Any) -> dict[str, Any]:
    event_files = sorted(log_dir.glob("events.out.tfevents.*"))
    if not event_files or any(path.stat().st_size <= 0 for path in event_files):
        raise AssertionError("TensorBoard event file was not created or is empty")
    accumulator = event_accumulator_class(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = sorted(accumulator.Tags().get("scalars", []))
    required = {
        "train/energy_step",
        "train/energy_epoch",
        "val/condition_energy",
        "val/edge_energy",
        "train/lr",
        "train/grad_norm_before_clip",
        "train/grad_norm_after_clip",
    }
    missing = sorted(required - set(tags))
    test_tags = [tag for tag in tags if tag.startswith("test/")]
    if missing or test_tags:
        raise AssertionError(f"TensorBoard tag audit failed: missing={missing}, test={test_tags}")
    return {
        "status": "pass",
        "log_dir": relative(log_dir),
        "event_files": [artifact_record(path) for path in event_files],
        "scalar_tags": tags,
        "required_scalar_tags_present": True,
        "test_tags": test_tags,
    }


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": [value.cpu() for value in torch.cuda.get_rng_state_all()],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def gather_rng_states(world_size: int) -> list[dict[str, Any]]:
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, capture_rng_state())
    if any(value is None for value in gathered):
        raise AssertionError("Failed to gather per-rank RNG state")
    return [value for value in gathered if value is not None]


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def checkpoint_paths(root: Path, variant: str) -> tuple[Path, Path]:
    directory = root / variant
    return directory / "best.pt", directory / "last.pt"


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    kind: str,
    variant: str,
    epoch: int,
    global_step: int,
    current_val_edge_energy: float,
    best_val_edge_energy: float,
    best_epoch: int,
    no_improvement_epochs: int,
    protocol_sha256: str,
    model_kwargs: dict[str, Any],
    history: list[dict[str, Any]],
    rng_states_by_rank: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = {
        "format": "tahoe_experiment1_st_formal_v2",
        "protocol_version": "v2",
        "checkpoint_kind": kind,
        "variant": variant,
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state": cpu_tree(optimizer.state_dict()),
        "epoch": epoch,
        "global_step": global_step,
        "current_val_edge_energy": current_val_edge_energy,
        "best_val_edge_energy": best_val_edge_energy,
        "best_epoch": best_epoch,
        "no_improvement_epochs": no_improvement_epochs,
        "training_seed": TRAINING_SEED,
        "precision": "bfloat16_autocast",
        "energy_dtype": "float32",
        "grad_scaler": False,
        "num_workers_per_rank": NUM_WORKERS_PER_RANK,
        "pin_memory": PIN_MEMORY,
        "persistent_workers": PERSISTENT_WORKERS,
        "prefetch_factor": PREFETCH_FACTOR,
        "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
        "world_size": WORLD_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_global_batch_size": EFFECTIVE_GLOBAL_BATCH_SIZE,
        "deepspeed": False,
        "tensorboard_log_dir": relative(TENSORBOARD_ROOT / "formal" / variant),
        "formal_protocol_sha256": protocol_sha256,
        "state_implementation_sha256": STATE_SHA256,
        "dataset_sha256": DATASET_SHA256,
        "runner_sha256": sha256_file(SCRIPT_PATH),
        "model_kwargs": model_kwargs,
        "optimizer_contract": optimizer_contract(),
        "history": history,
        "rng_states_by_rank": rng_states_by_rank,
    }
    torch.save(payload, temporary)
    temporary.replace(path)


def validate_checkpoint_payload(
    payload: dict[str, Any], *, variant: str, kind: str, protocol_sha256: str
) -> None:
    expected = {
        "format": "tahoe_experiment1_st_formal_v2",
        "protocol_version": "v2",
        "checkpoint_kind": kind,
        "variant": variant,
        "training_seed": TRAINING_SEED,
        "precision": "bfloat16_autocast",
        "energy_dtype": "float32",
        "grad_scaler": False,
        "num_workers_per_rank": NUM_WORKERS_PER_RANK,
        "pin_memory": PIN_MEMORY,
        "persistent_workers": PERSISTENT_WORKERS,
        "prefetch_factor": PREFETCH_FACTOR,
        "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
        "world_size": WORLD_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_global_batch_size": EFFECTIVE_GLOBAL_BATCH_SIZE,
        "deepspeed": False,
        "tensorboard_log_dir": relative(TENSORBOARD_ROOT / "formal" / variant),
        "formal_protocol_sha256": protocol_sha256,
        "state_implementation_sha256": STATE_SHA256,
        "dataset_sha256": DATASET_SHA256,
        "runner_sha256": sha256_file(SCRIPT_PATH),
        "optimizer_contract": optimizer_contract(),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise AssertionError(f"Checkpoint mismatch: {key}")
    required = (
        "model_state",
        "optimizer_state",
        "epoch",
        "global_step",
        "current_val_edge_energy",
        "best_val_edge_energy",
        "best_epoch",
        "no_improvement_epochs",
        "model_kwargs",
        "history",
        "rng_states_by_rank",
    )
    if any(key not in payload for key in required) or len(payload["rng_states_by_rank"]) != WORLD_SIZE:
        raise AssertionError("Checkpoint is incomplete")


def load_last_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    variant: str,
    protocol_sha256: str,
    rank: int,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint_payload(payload, variant=variant, kind="last", protocol_sha256=protocol_sha256)
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    restore_rng_state(payload["rng_states_by_rank"][rank])
    return payload


def checkpoint_roundtrip(path: Path, *, variant: str, protocol_sha256: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint_payload(payload, variant=variant, kind="last", protocol_sha256=protocol_sha256)
    reloaded, _kwargs = build_frozen_model(variant)
    reloaded.load_state_dict(payload["model_state"], strict=True)
    optimizer = torch.optim.AdamW(
        (value for value in reloaded.parameters() if value.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    optimizer.load_state_dict(payload["optimizer_state"])
    exact = all(
        torch.equal(value.detach().cpu(), payload["model_state"][name])
        for name, value in reloaded.state_dict().items()
    )
    if not exact:
        raise AssertionError("Checkpoint model round-trip changed values")
    result = {
        "status": "pass",
        "path": relative(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "model_state_strict_load": True,
        "model_state_exact": True,
        "optimizer_state_loaded": True,
        "rng_state_fields_present": True,
    }
    del optimizer, reloaded, payload
    gc.collect()
    return result


def parameter_snapshots(model: nn.Module) -> dict[str, torch.Tensor]:
    names = ("basal_encoder.0.weight", "pert_encoder.0.weight", "project_out.0.weight")
    parameters = dict(model.named_parameters())
    return {name: parameters[name].detach().cpu().clone() for name in names}


def parameter_update_audit(model: nn.Module, before: dict[str, torch.Tensor]) -> dict[str, Any]:
    parameters = dict(model.named_parameters())
    changes = {
        name: float((parameters[name].detach().cpu() - initial).abs().max())
        for name, initial in before.items()
    }
    changed = [name for name, value in changes.items() if value > 0]
    if not changed or not all(math.isfinite(value) for value in changes.values()):
        raise AssertionError("ST parameters did not update")
    return {"status": "pass", "changed_tensors": changed, "max_abs_change": changes}


def ddp_parameter_sync_error(model: nn.Module, world_size: int) -> float:
    moments = torch.zeros(3, device=next(model.parameters()).device, dtype=torch.float64)
    for parameter in model.parameters():
        values = parameter.detach().double()
        moments[0] += values.sum()
        moments[1] += values.square().sum()
        moments[2] += values.abs().sum()
    gathered = [torch.empty_like(moments) for _ in range(world_size)]
    dist.all_gather(gathered, moments)
    return max(float((value - gathered[0]).abs().max()) for value in gathered)


def setup_distributed() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != WORLD_SIZE:
        raise RuntimeError("ST smoke/run requires exactly two torchrun processes")
    if os.environ.get("NCCL_CUMEM_ENABLE") != "0" or os.environ.get(
        "NCCL_CUMEM_HOST_ENABLE"
    ) != "0":
        raise RuntimeError("Set both NCCL CUMEM workaround variables to 0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("Exactly two CUDA GPUs must be visible")
    torch.cuda.set_device(local_rank)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group(backend="nccl")
    return rank, local_rank, torch.device("cuda", local_rank)


def environment_payload(device: torch.device, rank: int, local_rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "local_rank": local_rank,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "geomloss": importlib.metadata.version("geomloss"),
        "precision": "bfloat16_autocast",
        "autocast": True,
        "autocast_dtype": "bfloat16",
        "energy_dtype": "float32",
        "grad_scaler": False,
        "tf32": False,
        "deepspeed": False,
        "nccl_cumem_enable": os.environ.get("NCCL_CUMEM_ENABLE"),
        "nccl_cumem_host_enable": os.environ.get("NCCL_CUMEM_HOST_ENABLE"),
    }


def smoke_output_guard(variant: str, *, overwrite: bool) -> Path:
    existing_variant = False
    if SMOKE_PATH.exists():
        payload = json.loads(SMOKE_PATH.read_text(encoding="utf-8"))
        existing_variant = payload.get("variants", {}).get(variant, {}).get("status") == "pass"
    log_dir = TENSORBOARD_ROOT / "smoke" / variant
    occupied = list(log_dir.glob("events.out.tfevents.*")) if log_dir.exists() else []
    if (existing_variant or occupied) and not overwrite:
        raise FileExistsError(f"Final-config smoke output already exists for {variant}; use --overwrite")
    return log_dir


def update_smoke_outputs(variant_result: dict[str, Any], *, protocol_sha256: str) -> None:
    payload: dict[str, Any]
    if SMOKE_PATH.exists():
        payload = json.loads(SMOKE_PATH.read_text(encoding="utf-8"))
        if payload.get("protocol", {}).get("sha256") != protocol_sha256:
            raise AssertionError("Existing smoke JSON belongs to another protocol")
    else:
        payload = {
            "schema": "tahoe_experiment1_st_final_config_smoke_v2",
            "created_at_utc": utc_now(),
            "status": "incomplete",
            "scope": "final v2 exact-config two-GPU smoke; no formal training and no test",
            "protocol": {"path": relative(PROTOCOL_PATH), "sha256": protocol_sha256},
            "dataset": {"path": relative(DATASET_PATH), "sha256": DATASET_SHA256},
            "state_model": {"path": relative(STATE_MODEL_PATH), "sha256": STATE_SHA256},
            "variants": {},
            "test_dataset_constructed": False,
            "scientific_result": False,
        }
    payload["schema"] = "tahoe_experiment1_st_final_config_smoke_v2"
    payload["scope"] = "final v2 exact-config two-GPU smoke; no formal training and no test"
    payload["dataset"] = {"path": relative(DATASET_PATH), "sha256": DATASET_SHA256}
    payload["state_model"] = {"path": relative(STATE_MODEL_PATH), "sha256": STATE_SHA256}
    payload["updated_at_utc"] = utc_now()
    payload["variants"][variant_result["variant"]] = variant_result
    complete = all(
        payload["variants"].get(name, {}).get("status") == "pass"
        for name in ("st-a", "st-r")
    )
    payload["status"] = (
        "pass"
        if complete
        else "incomplete"
    )
    payload["blockers"] = [] if complete else ["other exact-config variant smoke pending"]
    payload["scientific_result"] = False
    payload["formal_training_started"] = False
    payload["test_used"] = False
    payload["test_dataset_constructed"] = False
    payload["deepspeed"] = False
    atomic_write_json(SMOKE_PATH, payload)
    rows = []
    for variant in ("st-a", "st-r"):
        result = payload["variants"].get(variant)
        if result is None or result.get("status") != "pass":
            rows.append(f"| {variant} | pending | - | - | - |")
        else:
            runtime = result["runtime"]
            rows.append(
                f"| {variant} | {result['status']} | {runtime['mean_optimizer_update_seconds']:.6f} | "
                f"{runtime['global_conditions_per_second']:.3f} | "
                f"{runtime['peak_allocated_gib_max_across_ranks']:.3f} |"
            )
    handoff = f"""# Tahoe Experiment 1 ST final-config smoke

Status: **{payload['status'].upper()}**

This is an exact engineering-config smoke, not a scientific training result. It directly uses the current
`TahoeExperiment1LatentSetDataset` SHA `{DATASET_SHA256}` and never constructs test.

| variant | status | mean optimizer-update seconds | global conditions/s | peak allocated GiB/rank |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Protocol: `{relative(PROTOCOL_PATH)}` (`{protocol_sha256}`)

Formal full training was not started.
"""
    atomic_write_text(SMOKE_HANDOFF_PATH, handoff)


def run_smoke(variant: str, *, overwrite: bool) -> dict[str, Any] | None:
    summary_writer_class, event_accumulator_class, tensorboard_environment = require_tensorboard()
    rank, local_rank, device = setup_distributed()
    started = time.perf_counter()
    writer: Any | None = None
    try:
        verify_source_provenance(verify_task=False)
        _protocol, protocol_sha = ensure_protocol(create=False)
        if rank == 0:
            log_dir = smoke_output_guard(variant, overwrite=overwrite)
            log_dir.mkdir(parents=True, exist_ok=True)
            writer = summary_writer_class(log_dir=str(log_dir))
        else:
            log_dir = TENSORBOARD_ROOT / "smoke" / variant
        dist.barrier()

        train_dataset, val_dataset = make_datasets(
            train_limit=SMOKE_TRAIN_CONDITIONS, val_limit=SMOKE_VAL_CONDITIONS
        )
        redraw = train_redraw_audit(train_dataset)
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=WORLD_SIZE,
            rank=rank,
            shuffle=True,
            seed=TRAINING_SEED,
            drop_last=True,
        )
        loader = make_dataloader(
            train_dataset,
            batch_size=BATCH_SIZE_PER_GPU,
            shuffle=False,
            seed=TRAINING_SEED,
            sampler=sampler,
            drop_last=True,
            num_workers=NUM_WORKERS_PER_RANK,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
            prefetch_factor=PREFETCH_FACTOR,
        )
        if len(loader) != SMOKE_MICROBATCHES:
            raise AssertionError("1,280-condition smoke must have 10 microbatches per rank")

        raw_model, model_kwargs = build_frozen_model(variant)
        initial_sha = model_fingerprint(raw_model)
        before = parameter_snapshots(raw_model)
        raw_model.to(device)
        project_out_dtypes: list[str] = []
        hook = raw_model.project_out[0].register_forward_hook(
            lambda _module, _inputs, output: project_out_dtypes.append(
                str(output.dtype).removeprefix("torch.")
            )
        )
        optimizer = torch.optim.AdamW(
            (value for value in raw_model.parameters() if value.requires_grad),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        model = DistributedDataParallel(raw_model, device_ids=[local_rank])
        torch.cuda.reset_peak_memory_stats(device)

        microbatch_losses: list[float] = []
        update_losses: list[float] = []
        update_times: list[float] = []
        grad_norms_before: list[float] = []
        grad_norms_after: list[float] = []
        contract: dict[str, Any] | None = None
        loader_audit: dict[str, Any] | None = None
        forward_audit: dict[str, Any] = {}
        no_sync_microbatches = 0
        synchronized_microbatches = 0
        clipping_calls = 0
        optimizer_updates = 0
        epoch = 0
        train_dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        iterator = iter(loader)
        optimizer.zero_grad(set_to_none=True)
        window_started = time.perf_counter()
        window_losses: list[float] = []
        for microbatch_index in range(SMOKE_MICROBATCHES):
            torch.cuda.synchronize(device)
            raw_batch = next(iterator)
            if loader_audit is None:
                loader_audit = dataloader_runtime_audit(loader, iterator, raw_batch)
            if set(raw_batch["split"]) != {"train"}:
                raise AssertionError("A non-train condition entered ST fitting")
            assert_unique_set_indices(raw_batch["source_embedding_index"], "control")
            assert_unique_set_indices(raw_batch["target_embedding_index"], "treated")
            batch = move_batch(raw_batch, device)
            accumulation_boundary = (microbatch_index + 1) % GRADIENT_ACCUMULATION_STEPS == 0
            synchronization = nullcontext() if accumulation_boundary else model.no_sync()
            with synchronization:
                prediction, loss, _per_set = forward_energy(
                    model, batch, forward_audit=forward_audit
                )
                if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
                    raise AssertionError("ST prediction/Energy became non-finite")
                (loss / GRADIENT_ACCUMULATION_STEPS).backward()
            signed = bool((prediction < 0).any() and (prediction > 0).any())
            signed_tensor = torch.tensor(int(signed), device=device)
            dist.all_reduce(signed_tensor, op=dist.ReduceOp.MIN)
            if not int(signed_tensor):
                raise AssertionError("A rank lost signed ST output")
            report_loss = loss.detach().clone()
            dist.all_reduce(report_loss, op=dist.ReduceOp.SUM)
            report_loss /= WORLD_SIZE
            microbatch_loss = float(report_loss)
            microbatch_losses.append(microbatch_loss)
            window_losses.append(microbatch_loss)
            if accumulation_boundary:
                synchronized_microbatches += 1
                if any(
                    value.grad is not None and not torch.isfinite(value.grad).all()
                    for value in model.parameters()
                ):
                    raise AssertionError("ST gradient became non-finite")
                before_clip = gradient_norm(model)
                returned_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), GRADIENT_CLIP_NORM, error_if_nonfinite=True
                    )
                )
                after_clip = gradient_norm(model)
                if not math.isclose(before_clip, returned_norm, rel_tol=1e-5, abs_tol=1e-7):
                    raise AssertionError("Gradient norm audit disagrees with clip_grad_norm_")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                clipping_calls += 1
                optimizer_updates += 1
                torch.cuda.synchronize(device)
                elapsed = torch.tensor(time.perf_counter() - window_started, device=device)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                update_times.append(float(elapsed))
                update_energy = float(np.mean(window_losses))
                update_losses.append(update_energy)
                grad_norms_before.append(before_clip)
                grad_norms_after.append(after_clip)
                if rank == 0:
                    writer.add_scalar("train/energy_step", update_energy, optimizer_updates)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], optimizer_updates)
                    writer.add_scalar(
                        "train/grad_norm_before_clip", before_clip, optimizer_updates
                    )
                    writer.add_scalar(
                        "train/grad_norm_after_clip", after_clip, optimizer_updates
                    )
                window_losses = []
                window_started = time.perf_counter()
            else:
                no_sync_microbatches += 1
            if contract is None:
                contract = {
                    "ctrl_shape": list(batch["ctrl_cell_emb"].shape),
                    "target_shape": list(batch["pert_cell_emb"].shape),
                    "pert_shape": list(batch["pert_emb"].shape),
                    "output_shape": list(prediction.shape),
                    "dtype": str(prediction.dtype).removeprefix("torch."),
                    "finite": True,
                    "signed": True,
                    "negative_ratio": float((prediction < 0).float().mean()),
                    "minimum": float(prediction.min()),
                    "maximum": float(prediction.max()),
                }
        if contract is None or loader_audit is None:
            raise AssertionError("No exact-config smoke microbatch ran")
        if (
            len(microbatch_losses) != SMOKE_MICROBATCHES
            or optimizer_updates != SMOKE_OPTIMIZER_UPDATES
            or no_sync_microbatches != SMOKE_OPTIMIZER_UPDATES
            or synchronized_microbatches != SMOKE_OPTIMIZER_UPDATES
            or clipping_calls != SMOKE_OPTIMIZER_UPDATES
        ):
            raise AssertionError("Gradient accumulation counts changed")
        if not project_out_dtypes or set(project_out_dtypes) != {"bfloat16"}:
            raise AssertionError(f"project_out did not execute in BF16: {set(project_out_dtypes)}")
        update = parameter_update_audit(raw_model, before)
        sync_error = ddp_parameter_sync_error(raw_model, WORLD_SIZE)
        if sync_error != 0.0:
            raise AssertionError(f"DDP parameters differ across ranks: {sync_error}")

        dist.barrier()
        first_val = validate_exact(raw_model, val_dataset, device) if rank == 0 else None
        second_val = validate_exact(raw_model, val_dataset, device) if rank == 0 else None
        reproducibility = (
            validation_reproducibility(first_val, second_val) if rank == 0 else None
        )
        val_edge_tensor = torch.tensor(
            first_val["edge_level_mean_energy"] if rank == 0 else 0.0,
            device=device,
            dtype=torch.float64,
        )
        dist.broadcast(val_edge_tensor, src=0)
        if rank == 0:
            writer.add_scalar("train/energy_epoch", float(np.mean(microbatch_losses)), 0)
            writer.add_scalar("val/condition_energy", first_val["condition_energy_mean"], 0)
            writer.add_scalar("val/edge_energy", first_val["edge_level_mean_energy"], 0)
            writer.add_scalar("best/val_edge_energy", first_val["edge_level_mean_energy"], 0)
            writer.add_scalar("best/epoch", 0, 0)
        dist.barrier()

        local_runtime = {
            **environment_payload(device, rank, local_rank),
            "wall_seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        }
        runtimes: list[dict[str, Any] | None] = [None] * WORLD_SIZE
        dist.all_gather_object(runtimes, local_runtime)
        complete_runtimes = [value for value in runtimes if value is not None]
        if len(complete_runtimes) != WORLD_SIZE:
            raise AssertionError("Both DDP ranks did not report runtime")
        loader_audits: list[dict[str, Any] | None] = [None] * WORLD_SIZE
        dist.all_gather_object(loader_audits, loader_audit)
        if rank != 0:
            dist.barrier()
            return None
        assert first_val is not None and second_val is not None and reproducibility is not None
        mean_update = float(np.mean(update_times))
        writer.add_scalar(
            "system/conditions_per_second", EFFECTIVE_GLOBAL_BATCH_SIZE / mean_update, 0
        )
        writer.add_scalar(
            "system/peak_allocated_gib",
            max(value["peak_allocated_gib"] for value in complete_runtimes),
            0,
        )
        writer.add_scalar(
            "system/peak_reserved_gib",
            max(value["peak_reserved_gib"] for value in complete_runtimes),
            0,
        )
        writer.flush()
        writer.close()
        writer = None
        tensorboard = tensorboard_event_audit(log_dir, event_accumulator_class)
        clean_first_val = public_validation(first_val)
        clean_second_val = public_validation(second_val)
        result = {
            "status": "pass",
            "variant": variant,
            "microbatches": SMOKE_MICROBATCHES,
            "optimizer_updates": optimizer_updates,
            "train_conditions_selected": len(train_dataset),
            "val_conditions_selected": len(val_dataset),
            "parameter_count": EXPECTED_PARAMETERS,
            "trainable_parameter_count": EXPECTED_TRAINABLE_PARAMETERS,
            "initial_state_sha256": initial_sha,
            "model_kwargs": model_kwargs,
            "data_contract": contract,
            "dataloader": {
                "status": "pass",
                "per_rank": [value for value in loader_audits if value is not None],
                "total_active_worker_processes": sum(
                    value["active_worker_process_count"]
                    for value in loader_audits
                    if value is not None
                ),
            },
            "dataset_set_epoch_redraw": redraw,
            "precision": {
                **forward_audit,
                "project_out_forward_dtypes": sorted(set(project_out_dtypes)),
                "project_out_executed_in_bfloat16": True,
                "energy_inputs_explicit_float32": True,
                "grad_scaler": False,
                "tf32": False,
            },
            "gradient_accumulation": {
                "steps": GRADIENT_ACCUMULATION_STEPS,
                "microbatches": len(microbatch_losses),
                "optimizer_updates": optimizer_updates,
                "no_sync_microbatches": no_sync_microbatches,
                "synchronized_microbatches": synchronized_microbatches,
                "clip_calls": clipping_calls,
                "loss_divided_for_backward": True,
                "reported_energy_uses_unscaled_loss": True,
            },
            "training": {
                "energy_first_microbatch": microbatch_losses[0],
                "energy_last_microbatch": microbatch_losses[-1],
                "energy_epoch_condition_weighted_mean": float(np.mean(microbatch_losses)),
                "energy_all_finite": all(math.isfinite(value) for value in microbatch_losses),
                "backward_all_finite": True,
                "gradient_norm_before_clip_min": min(grad_norms_before),
                "gradient_norm_before_clip_max": max(grad_norms_before),
                "gradient_norm_after_clip_min": min(grad_norms_after),
                "gradient_norm_after_clip_max": max(grad_norms_after),
                "gradient_clip_norm": GRADIENT_CLIP_NORM,
                "clip_only_at_accumulation_boundary": True,
                "parameter_update": update,
                "epochs_touched": [0],
            },
            "ddp": {
                "world_size": WORLD_SIZE,
                "parameter_moment_max_abs_error": sync_error,
                "synchronized": True,
            },
            "fixed_validation": {
                "run_1": clean_first_val,
                "run_2": clean_second_val,
                "reproducibility": reproducibility,
                "rank0_only": True,
                "rank1_waited": True,
            },
            "tensorboard": tensorboard,
            "runtime": {
                "mean_optimizer_update_seconds": mean_update,
                "global_conditions_per_second": EFFECTIVE_GLOBAL_BATCH_SIZE / mean_update,
                "global_cells_per_second": EFFECTIVE_GLOBAL_BATCH_SIZE * SET_SIZE / mean_update,
                "peak_allocated_gib_max_across_ranks": max(
                    value["peak_allocated_gib"] for value in complete_runtimes
                ),
                "peak_reserved_gib_max_across_ranks": max(
                    value["peak_reserved_gib"] for value in complete_runtimes
                ),
                "rank": complete_runtimes,
                "step_timing_includes_dataloader_fetch_and_h2d": True,
            },
            "current_dataset_sha256_directly_covered": DATASET_SHA256,
            "tensorboard_environment": tensorboard_environment,
            "test_dataset_constructed": False,
            "test_used": False,
            "formal_training_started": False,
            "deepspeed": False,
            "scientific_result": False,
            "blockers": [],
        }
        update_smoke_outputs(result, protocol_sha256=protocol_sha)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        dist.barrier()
        return result
    finally:
        if writer is not None:
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def formal_artifact_guard(variant: str, *, resume_last: bool) -> tuple[Path, Path]:
    best, last = checkpoint_paths(FORMAL_CHECKPOINT_ROOT, variant)
    if resume_last:
        if not last.is_file():
            raise FileNotFoundError(f"No same-variant last checkpoint: {last}")
    elif best.exists() or last.exists():
        raise FileExistsError(f"Formal {variant} checkpoints already exist; use --resume-last")
    return best, last


def run_formal(variant: str, *, resume_last: bool) -> dict[str, Any] | None:
    summary_writer_class, _event_accumulator_class, _tensorboard_environment = require_tensorboard()
    rank, local_rank, device = setup_distributed()
    started = time.perf_counter()
    writer: Any | None = None
    try:
        verify_source_provenance(verify_task=False)
        _protocol, protocol_sha = ensure_protocol(create=False)
        if rank == 0:
            best_path, last_path = formal_artifact_guard(variant, resume_last=resume_last)
        else:
            best_path, last_path = checkpoint_paths(FORMAL_CHECKPOINT_ROOT, variant)
        dist.barrier()

        train_dataset, val_dataset = make_datasets()
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=WORLD_SIZE,
            rank=rank,
            shuffle=True,
            seed=TRAINING_SEED,
            drop_last=True,
        )
        loader = make_dataloader(
            train_dataset,
            batch_size=BATCH_SIZE_PER_GPU,
            shuffle=False,
            seed=TRAINING_SEED,
            sampler=sampler,
            drop_last=True,
            num_workers=NUM_WORKERS_PER_RANK,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
            prefetch_factor=PREFETCH_FACTOR,
        )
        if len(loader) != training_shape()["microbatches_per_epoch"]:
            raise AssertionError("Formal train DataLoader shape changed")
        raw_model, model_kwargs = build_frozen_model(variant)
        raw_model.to(device)
        optimizer = torch.optim.AdamW(
            (value for value in raw_model.parameters() if value.requires_grad),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        if resume_last:
            checkpoint = load_last_checkpoint(
                last_path,
                raw_model,
                optimizer,
                variant=variant,
                protocol_sha256=protocol_sha,
                rank=rank,
            )
            start_epoch = int(checkpoint["epoch"]) + 1
            global_step = int(checkpoint["global_step"])
            best_metric = float(checkpoint["best_val_edge_energy"])
            best_epoch = int(checkpoint["best_epoch"])
            no_improvement = int(checkpoint["no_improvement_epochs"])
            history = list(checkpoint["history"])
            if start_epoch >= MAX_EPOCHS or no_improvement >= EARLY_STOPPING_PATIENCE:
                raise RuntimeError("The same-variant last checkpoint is already terminal")
        else:
            start_epoch = 0
            global_step = 0
            best_metric = float("inf")
            best_epoch = -1
            no_improvement = 0
            history: list[dict[str, Any]] = []
        log_dir = TENSORBOARD_ROOT / "formal" / variant
        if rank == 0:
            log_dir.mkdir(parents=True, exist_ok=True)
            writer = summary_writer_class(
                log_dir=str(log_dir), purge_step=global_step if resume_last else None
            )
        model = DistributedDataParallel(raw_model, device_ids=[local_rank])
        torch.cuda.reset_peak_memory_stats(device)
        stop_reason = "max_epochs"
        for epoch in range(start_epoch, MAX_EPOCHS):
            train_dataset.set_epoch(epoch)
            sampler.set_epoch(epoch)
            if train_dataset.epoch != epoch or sampler.epoch != epoch:
                raise AssertionError("Dataset/sampler epoch was not updated")
            model.train()
            epoch_energy_weighted_sum = 0.0
            epoch_condition_count = 0
            epoch_microbatches = 0
            epoch_updates = 0
            window_losses: list[float] = []
            optimizer.zero_grad(set_to_none=True)
            epoch_started = time.perf_counter()
            for microbatch_index, raw_batch in enumerate(loader):
                if set(raw_batch["split"]) != {"train"}:
                    raise AssertionError("A non-train condition entered formal ST fitting")
                batch = move_batch(raw_batch, device)
                accumulation_boundary = (
                    microbatch_index + 1
                ) % GRADIENT_ACCUMULATION_STEPS == 0
                synchronization = nullcontext() if accumulation_boundary else model.no_sync()
                with synchronization:
                    prediction, loss, _values = forward_energy(model, batch)
                    if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
                        raise AssertionError("Formal ST prediction/Energy became non-finite")
                    (loss / GRADIENT_ACCUMULATION_STEPS).backward()
                report_loss = loss.detach().clone()
                dist.all_reduce(report_loss, op=dist.ReduceOp.SUM)
                report_loss /= WORLD_SIZE
                raw_energy = float(report_loss)
                global_conditions = int(batch["ctrl_cell_emb"].shape[0]) * WORLD_SIZE
                epoch_energy_weighted_sum += raw_energy * global_conditions
                epoch_condition_count += global_conditions
                epoch_microbatches += 1
                window_losses.append(raw_energy)
                if accumulation_boundary:
                    if any(
                        value.grad is not None and not torch.isfinite(value.grad).all()
                        for value in model.parameters()
                    ):
                        raise AssertionError("Formal ST gradient became non-finite")
                    before_clip = gradient_norm(model)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), GRADIENT_CLIP_NORM, error_if_nonfinite=True
                    )
                    after_clip = gradient_norm(model)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    epoch_updates += 1
                    if rank == 0:
                        writer.add_scalar(
                            "train/energy_step", float(np.mean(window_losses)), global_step
                        )
                        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                        writer.add_scalar(
                            "train/grad_norm_before_clip", before_clip, global_step
                        )
                        writer.add_scalar("train/grad_norm_after_clip", after_clip, global_step)
                    window_losses = []
            expected_shape = training_shape()
            if (
                epoch_microbatches != expected_shape["microbatches_per_epoch"]
                or epoch_updates != expected_shape["optimizer_updates_per_epoch"]
                or window_losses
            ):
                raise AssertionError("Formal epoch accumulation counts changed")
            train_energy = epoch_energy_weighted_sum / epoch_condition_count
            epoch_elapsed = time.perf_counter() - epoch_started

            dist.barrier()
            validation = validate_exact(raw_model, val_dataset, device) if rank == 0 else None
            metrics = torch.tensor(
                [
                    validation["condition_energy_mean"] if rank == 0 else 0.0,
                    validation["edge_level_mean_energy"] if rank == 0 else 0.0,
                ],
                device=device,
                dtype=torch.float64,
            )
            dist.broadcast(metrics, src=0)
            val_condition, val_edge = map(float, metrics)
            improved = val_edge < best_metric
            if improved:
                best_metric = val_edge
                best_epoch = epoch
                no_improvement = 0
            else:
                no_improvement += 1
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "train_energy_mean": train_energy,
                "train_energy_definition": "condition-weighted mean over raw unscaled microbatch Energy",
                "microbatches": epoch_microbatches,
                "optimizer_updates": epoch_updates,
                "val_condition_energy_mean": val_condition,
                "val_edge_energy_mean": val_edge,
                "best_val_edge_energy": best_metric,
                "best_epoch": best_epoch,
                "strict_improvement": improved,
                "no_improvement_epochs": no_improvement,
            }
            history.append(row)
            peaks = torch.tensor(
                [
                    torch.cuda.max_memory_allocated(device) / 2**30,
                    torch.cuda.max_memory_reserved(device) / 2**30,
                ],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
            if rank == 0:
                writer.add_scalar("train/energy_epoch", train_energy, epoch)
                writer.add_scalar("val/condition_energy", val_condition, epoch)
                writer.add_scalar("val/edge_energy", val_edge, epoch)
                writer.add_scalar(
                    "system/conditions_per_second", epoch_condition_count / epoch_elapsed, epoch
                )
                writer.add_scalar("system/peak_allocated_gib", float(peaks[0]), epoch)
                writer.add_scalar("system/peak_reserved_gib", float(peaks[1]), epoch)
                writer.add_scalar("best/val_edge_energy", best_metric, epoch)
                writer.add_scalar("best/epoch", best_epoch, epoch)
                writer.flush()
            rng_states = gather_rng_states(WORLD_SIZE)
            if rank == 0:
                checkpoint_args = {
                    "model": raw_model,
                    "optimizer": optimizer,
                    "variant": variant,
                    "epoch": epoch,
                    "global_step": global_step,
                    "current_val_edge_energy": val_edge,
                    "best_val_edge_energy": best_metric,
                    "best_epoch": best_epoch,
                    "no_improvement_epochs": no_improvement,
                    "protocol_sha256": protocol_sha,
                    "model_kwargs": model_kwargs,
                    "history": history,
                    "rng_states_by_rank": rng_states,
                }
                if improved:
                    save_checkpoint(best_path, kind="best", **checkpoint_args)
                save_checkpoint(last_path, kind="last", **checkpoint_args)
                print(
                    f"{variant} epoch={epoch} step={global_step} "
                    f"train_energy={row['train_energy_mean']:.8f} "
                    f"val_edge_energy={val_edge:.8f} best_epoch={best_epoch} "
                    f"no_improvement={no_improvement}",
                    flush=True,
                )
            dist.barrier()
            if no_improvement >= EARLY_STOPPING_PATIENCE:
                stop_reason = "early_stopping"
                break

        sync_error = ddp_parameter_sync_error(raw_model, WORLD_SIZE)
        if sync_error != 0.0:
            raise AssertionError("Formal DDP parameters are not synchronized")
        if rank != 0:
            dist.barrier()
            return None
        result = {
            "created_at_utc": utc_now(),
            "status": "pass",
            "variant": variant,
            "scope": "formal train+val only; test not constructed",
            "stop_reason": stop_reason,
            "epochs_completed": len(history),
            "global_step": global_step,
            "best_epoch": best_epoch,
            "best_val_edge_energy": best_metric,
            "protocol": {"path": relative(PROTOCOL_PATH), "sha256": protocol_sha},
            "checkpoints": {"best": artifact_record(best_path), "last": artifact_record(last_path)},
            "history": history,
            "ddp_parameter_moment_max_abs_error": sync_error,
            "elapsed_seconds": time.perf_counter() - started,
            "tensorboard_log_dir": relative(log_dir),
            "protocol_version": "v2",
            "precision": "bfloat16_autocast",
            "energy_dtype": "float32",
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_global_batch_size": EFFECTIVE_GLOBAL_BATCH_SIZE,
            "deepspeed": False,
            "test_dataset_constructed": False,
        }
        output = RESULTS / f"tahoe_experiment1_st_{variant}_formal_training_result.json"
        atomic_write_json(output, result)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        dist.barrier()
        return result
    finally:
        if writer is not None:
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--rewrite", action="store_true")
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--variant", choices=("st-a", "st-r"), required=True)
    smoke_parser.add_argument("--overwrite", action="store_true")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--variant", choices=("st-a", "st-r"), required=True)
    run_parser.add_argument("--resume-last", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(rewrite=args.rewrite)
    elif args.command == "smoke":
        run_smoke(args.variant, overwrite=args.overwrite)
    elif args.command == "run":
        run_formal(args.variant, resume_last=args.resume_last)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
