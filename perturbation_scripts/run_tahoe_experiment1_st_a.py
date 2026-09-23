#!/usr/bin/env python3
"""Prepare or run the formal Tahoe Experiment 1 ST-A training path."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import io
import json
import math
import os
import random
import time
from contextlib import nullcontext, redirect_stdout
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

from tahoe_experiment1_latent_data import (
    LATENT_DIM,
    PERT_DIM,
    PROJECT_ROOT,
    SET_SIZE,
    TahoeExperiment1LatentSetDataset,
    make_dataloader,
)


RESULTS = PROJECT_ROOT / "results"
SCRIPT_PATH = Path(__file__).resolve()
STATE_MODEL_PATH = (
    PROJECT_ROOT.parent / "state-main" / "src" / "state" / "tx" / "models" / "state_transition.py"
)
DATASET_PATH = PROJECT_ROOT / "perturbation_scripts" / "tahoe_experiment1_latent_data.py"
PATCH_PATH = PROJECT_ROOT / "patches" / "state_genejepa_st_a_compat.patch"
PROTOCOL_PATH = RESULTS / "tahoe_experiment1_st_a_training_protocol.json"
TENSORBOARD_ROOT = RESULTS / "tensorboard" / "tahoe_experiment1_st_a"
FORMAL_CHECKPOINT_ROOT = RESULTS / "tahoe_experiment1_st_formal_checkpoints_v2"
ST_A_VARIANT = "st-a"

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
EXPECTED_PARAMETERS = 101_560_320
EXPECTED_TRAINABLE_PARAMETERS = 76_984_320


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def assert_unique_set_indices(indices: torch.Tensor, side: str) -> None:
    if indices.ndim != 2 or indices.shape[1] != SET_SIZE:
        raise AssertionError(f"Unexpected {side} index shape: {tuple(indices.shape)}")
    ordered = torch.sort(indices, dim=1).values
    if (ordered[:, 1:] == ordered[:, :-1]).any():
        raise AssertionError(f"Duplicate cell within a {side} set")


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


def verify_source_provenance(*, verify_task: bool) -> None:
    del verify_task
    missing = [
        path
        for path in (STATE_MODEL_PATH, DATASET_PATH, PATCH_PATH)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing ST-A implementation inputs: {missing}")


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
        "schema": "tahoe_experiment1_st_a_training_protocol_v1",
        "status": "frozen",
        "model": {
            "class": "StateTransitionPerturbationModel",
            "prediction": "absolute treated latent",
            "formula": "Zpred = project_out(transformer_hidden)",
            "input_dim": LATENT_DIM,
            "output_dim": LATENT_DIM,
            "cell_set_len": SET_SIZE,
            "pert_dim": PERT_DIM,
            "predict_residual": False,
            "final_activation": "identity",
            "output_space": "embedding",
            "basal_encoder": "Linear(768, 768)",
            "perturbation_encoder": "Linear(380, 768)",
            "project_out": "Linear(768, 768)",
            "transformer": {
                "class": "LlamaBidirectionalModel",
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_hidden_layers": 8,
                "num_attention_heads": 12,
                "num_key_value_heads": 12,
                "head_dim": 64,
                "max_position_embeddings": SET_SIZE,
                "use_rotary_embeddings": False,
            },
            "parameter_count": EXPECTED_PARAMETERS,
            "trainable_parameter_count": EXPECTED_TRAINABLE_PARAMETERS,
        },
        "data": {
            "dataset_class": "TahoeExperiment1LatentSetDataset",
            "train_conditions": TRAIN_CONDITIONS,
            "val_conditions": VAL_CONDITIONS,
            "test_conditions": TEST_CONDITIONS,
            "input": {
                "ctrl_cell_emb": ["B", SET_SIZE, LATENT_DIM],
                "pert_emb": ["B", SET_SIZE, PERT_DIM],
            },
            "target": {"pert_cell_emb": ["B", SET_SIZE, LATENT_DIM]},
            "latent_preprocessing": "none; raw signed frozen Epoch25 embeddings",
        },
        "loss": {
            "class": "geomloss.SamplesLoss",
            "loss": "energy",
            "blur": ENERGY_BLUR,
            "reduction": "one Energy value per set, then batch mean",
        },
        "optimizer": optimizer_contract(),
        "training": {
            "seed": TRAINING_SEED,
            "precision": "bfloat16 model forward; float32 Energy",
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
            "world_size": WORLD_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_global_batch_size": EFFECTIVE_GLOBAL_BATCH_SIZE,
            "validation_metric": "edge-level mean Energy",
        },
        "implementation": {
            "runner": "perturbation_scripts/run_tahoe_experiment1_st_a.py",
            "dataset": "perturbation_scripts/tahoe_experiment1_latent_data.py",
            "state_upstream_commit": "f182478607c75f6b8f6256409cc0b4b902f991e9",
            "compatibility_patch": "patches/state_genejepa_st_a_compat.patch",
            "compatibility_patch_sha256": sha256_file(PATCH_PATH),
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
        raise AssertionError("Frozen ST-A protocol or implementation contract changed")
    return observed, sha256_file(PROTOCOL_PATH)


def build_model() -> tuple[nn.Module, dict[str, Any]]:
    from state.tx.models.state_transition import StateTransitionPerturbationModel

    kwargs = {
        "input_dim": LATENT_DIM,
        "hidden_dim": 768,
        "output_dim": LATENT_DIM,
        "pert_dim": PERT_DIM,
        "predict_residual": False,
        "final_activation": "identity",
        "distributional_loss": "energy",
        "transformer_backbone_key": "llama",
        "transformer_backbone_kwargs": {
            "bidirectional_attention": True,
            "max_position_embeddings": SET_SIZE,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 8,
            "num_attention_heads": 12,
            "num_key_value_heads": 12,
            "head_dim": 64,
            "use_cache": False,
            "attention_dropout": 0.0,
            "hidden_dropout": 0.0,
            "layer_norm_eps": 1e-6,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "tie_word_embeddings": False,
            "rotary_dim": 0,
            "use_rotary_embeddings": False,
        },
        "output_space": "embedding",
        "embed_key": "X_genejepa_epoch25",
        "gene_decoder_bool": False,
        "cell_set_len": SET_SIZE,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "activation": "gelu",
        "dropout": 0.0,
        "loss": "energy",
        "blur": ENERGY_BLUR,
        "mmd_num_chunks": 1,
        "randomize_mmd_chunks": False,
        "extra_tokens": 0,
        "batch_encoder": False,
        "batch_predictor": False,
        "use_batch_token": False,
        "confidence_token": False,
        "finetune_vci_decoder": False,
        "log1p_from_raw_counts": False,
        "lora": {"enable": False},
    }
    with redirect_stdout(io.StringIO()):
        model = StateTransitionPerturbationModel(**kwargs)
    return model, kwargs


def _assert_single_linear(
    module: nn.Module, in_features: int, out_features: int, name: str
) -> None:
    if not isinstance(module, nn.Sequential) or len(module) != 1:
        raise AssertionError(f"{name} is not a one-layer Sequential")
    layer = module[0]
    if not isinstance(layer, nn.Linear) or (layer.in_features, layer.out_features) != (
        in_features,
        out_features,
    ):
        raise AssertionError(f"{name} Linear dimensions changed")


def build_frozen_model(variant: str) -> tuple[nn.Module, dict[str, Any]]:
    if variant != ST_A_VARIANT:
        raise ValueError(f"Only {ST_A_VARIANT} is supported")
    seed_everything(TRAINING_SEED)
    model, kwargs = build_model()
    if (model.input_dim, model.output_dim, model.cell_sentence_len, model.pert_dim) != (
        LATENT_DIM,
        LATENT_DIM,
        SET_SIZE,
        PERT_DIM,
    ):
        raise AssertionError("ST-A model dimensions changed")
    _assert_single_linear(model.basal_encoder, 768, 768, "basal_encoder")
    _assert_single_linear(model.pert_encoder, 380, 768, "pert_encoder")
    _assert_single_linear(model.project_out, 768, 768, "project_out")
    if type(model.transformer_backbone).__name__ != "LlamaBidirectionalModel":
        raise AssertionError("ST-A requires LlamaBidirectionalModel")
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
        model.predict_residual
        or model.final_activation_name != "identity"
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
        raise AssertionError("ST-A absolute signed-output contract changed")
    if not isinstance(model.loss_fn, SamplesLoss) or model.distributional_loss != "energy":
        raise AssertionError("ST-A requires geomloss Energy")
    if model.loss_fn.loss != "energy" or model.loss_fn.blur != ENERGY_BLUR:
        raise AssertionError("Energy configuration changed")
    parameters = sum(value.numel() for value in model.parameters())
    trainable = sum(value.numel() for value in model.parameters() if value.requires_grad)
    if (parameters, trainable) != (EXPECTED_PARAMETERS, EXPECTED_TRAINABLE_PARAMETERS):
        raise AssertionError(f"ST-A parameter count changed: {(parameters, trainable)}")
    effective_kwargs = copy.deepcopy(kwargs)
    effective_kwargs["effective_hidden_act"] = config.hidden_act
    return model, effective_kwargs


def prepare(*, rewrite: bool) -> dict[str, Any]:
    verify_source_provenance(verify_task=False)
    model, kwargs = build_frozen_model(ST_A_VARIANT)
    protocol, protocol_sha = ensure_protocol(create=True, rewrite=rewrite)
    result = {
        "created_at_utc": utc_now(),
        "status": "pass",
        "scope": "ST-A contract preparation only; no data, optimizer step, or test",
        "protocol": {"path": relative(PROTOCOL_PATH), "sha256": protocol_sha},
        "parameters": sum(value.numel() for value in model.parameters()),
        "trainable_parameters": sum(
            value.numel() for value in model.parameters() if value.requires_grad
        ),
        "model_kwargs": kwargs,
        "training_shape": training_shape(),
        "tensorboard_environment": tensorboard_environment_audit(),
        "prediction": protocol["model"]["formula"],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


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


def gradient_norm(model: nn.Module) -> float:
    squares = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is not None:
            squares += parameter.grad.detach().double().square().sum()
    return float(squares.sqrt())


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
        "state_implementation_sha256": sha256_file(STATE_MODEL_PATH),
        "dataset_sha256": sha256_file(DATASET_PATH),
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
        "state_implementation_sha256": sha256_file(STATE_MODEL_PATH),
        "dataset_sha256": sha256_file(DATASET_PATH),
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
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--resume-last", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(rewrite=args.rewrite)
    elif args.command == "run":
        run_formal(ST_A_VARIANT, resume_last=args.resume_last)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
