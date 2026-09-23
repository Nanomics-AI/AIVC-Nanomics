#!/usr/bin/env python3
"""Freeze and smoke-test the Experiment 1 B2 pooled-MLP training path."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import time
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
from tahoe_experiment1_latent_data import (
    LATENT_DIM,
    PERT_DIM,
    PROJECT_ROOT,
    SET_SIZE,
    TahoeExperiment1LatentSetDataset,
    make_dataloader,
)


SCRIPT_PATH = Path(__file__).resolve()
RESULTS = PROJECT_ROOT / "results"
TASK_PATH = PROJECT_ROOT.parent / "当前任务.txt"
PROTOCOL_PATH = RESULTS / "tahoe_experiment1_b2_training_protocol.json"
SMOKE_PATH = RESULTS / "tahoe_experiment1_b2_implementation_smoke.json"
HANDOFF_PATH = RESULTS / "tahoe_experiment1_b2_implementation_smoke.md"
CHECKPOINT_DIR = RESULTS / "tahoe_experiment1_b2_smoke_checkpoints"

PROTOCOL_VERSION = "tahoe_experiment1_b2_training_v1"
FROZEN_AT_UTC = "2026-09-09T09:24:04.866352+00:00"
FROZEN_TASK_SHA256 = "dfa6a9368e0b7f3121d6ec981598cfdd39ebd1fc08cd57c64ecf5a2d07d0af1f"
INPUT_DIM = LATENT_DIM + PERT_DIM
HIDDEN_DIM = 1024
OUTPUT_DIM = LATENT_DIM
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
TRAINING_SEED = 42
BATCH_SIZE_PER_GPU = 64
FORMAL_NUM_GPUS = 2
FORMAL_GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_GPU * FORMAL_NUM_GPUS
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
ENERGY_BLUR = 0.05
EXPECTED_SPLIT_COUNTS = {"train": 45_652, "val": 5_657, "test": 5_684}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


class B2PooledMLP(nn.Module):
    """Predict one raw-latent shift per control-cell population."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        )
        final = self.mlp[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def predict_delta(
        self, ctrl_cell_emb: torch.Tensor, pert_emb: torch.Tensor
    ) -> torch.Tensor:
        if ctrl_cell_emb.ndim != 3 or tuple(ctrl_cell_emb.shape[1:]) != (
            SET_SIZE,
            LATENT_DIM,
        ):
            raise ValueError(f"Unexpected ctrl_cell_emb shape: {tuple(ctrl_cell_emb.shape)}")
        if pert_emb.shape != (ctrl_cell_emb.shape[0], SET_SIZE, PERT_DIM):
            raise ValueError(f"Unexpected pert_emb shape: {tuple(pert_emb.shape)}")
        control_context = ctrl_cell_emb.mean(dim=1)
        pert_context = pert_emb[:, 0, :]
        return self.mlp(torch.cat((control_context, pert_context), dim=-1))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ctrl = batch["ctrl_cell_emb"]
        delta = self.predict_delta(ctrl, batch["pert_emb"])
        return ctrl + delta[:, None, :]


def parameter_count() -> int:
    return sum(parameter.numel() for parameter in B2PooledMLP().parameters())


def protocol_payload() -> dict[str, Any]:
    dataset_path = PROJECT_ROOT / "perturbation_scripts" / "tahoe_experiment1_latent_data.py"
    perturbation_path = RESULTS / "tahoe_experiment1_perturbation_featurization.json"
    condition_manifest = RESULTS / "tahoe_experiment1_condition_split_manifest.csv"
    edge_manifest = RESULTS / "tahoe_experiment1_edge_split_manifest.csv"
    aggregation_path = PROJECT_ROOT / "perturbation_scripts" / "evaluate_tahoe_experiment1_b0.py"
    return {
        "schema": PROTOCOL_VERSION,
        "status": "frozen",
        "frozen_at_utc": FROZEN_AT_UTC,
        "source_task": {
            "path": "/mnt/c/SH/AIVC/当前任务.txt",
            "sha256": FROZEN_TASK_SHA256,
        },
        "architecture": {
            "name": "B2 Pooled MLP",
            "input_tensors": {
                "ctrl_cell_emb": ["B", SET_SIZE, LATENT_DIM],
                "pert_emb": ["B", SET_SIZE, PERT_DIM],
            },
            "control_context": "mean(ctrl_cell_emb, dim=1)",
            "control_context_shape": ["B", LATENT_DIM],
            "pert_context": "pert_emb[:, 0, :]",
            "pert_context_shape": ["B", PERT_DIM],
            "concatenated_input_dim": INPUT_DIM,
            "layers": [
                {"type": "Linear", "in": INPUT_DIM, "out": HIDDEN_DIM},
                {"type": "GELU"},
                {"type": "Linear", "in": HIDDEN_DIM, "out": HIDDEN_DIM},
                {"type": "GELU"},
                {"type": "Linear", "in": HIDDEN_DIM, "out": OUTPUT_DIM},
            ],
            "dropout": 0.0,
            "normalization": None,
            "output_activation": "identity",
            "delta_shape": ["B", OUTPUT_DIM],
            "prediction": "raw_Zctrl + Delta_pred[:, None, :]",
            "prediction_shape": ["B", SET_SIZE, OUTPUT_DIM],
            "hidden_space_residual": False,
            "parameter_count": parameter_count(),
        },
        "initialization": {
            "first_two_linear_layers": "PyTorch default nn.Linear initialization",
            "final_linear_weight": "zeros",
            "final_linear_bias": "zeros",
            "initial_delta": "exact zero",
            "initial_prediction": "B0 identity: Zpred == raw_Zctrl",
        },
        "objective": {
            "implementation": "geomloss.SamplesLoss",
            "arguments": {"loss": "energy", "blur": ENERGY_BLUR},
            "reduction": "mean over set-level Energy values",
            "prediction": "Zpred",
            "target": "pert_cell_emb",
            "forbidden": [
                "per-cell MSE",
                "centroid MSE",
                "cosine loss",
                "direct Delta_condition MSE",
                "B1 mean-delta supervision",
                "DMSO residual target",
                "Sinkhorn training loss",
                "paired-cell loss",
            ],
        },
        "data": {
            "dataset_class": "TahoeExperiment1LatentSetDataset",
            "train_split_only": True,
            "train_conditions": EXPECTED_SPLIT_COUNTS["train"],
            "set_size": SET_SIZE,
            "train_epoch_redraw": "dataset.set_epoch(training_epoch)",
            "ddp_epoch_shuffle": "sampler.set_epoch(training_epoch)",
            "shuffle": True,
            "drop_last": True,
            "latent": "raw signed GeneJEPA Epoch25 float32",
            "latent_transforms": [],
            "forbidden_latent_transforms": [
                "centering",
                "whitening",
                "L2 normalization",
                "latent standardization",
                "clipping",
                "ReLU",
                "DMSO residualization",
            ],
            "fit_uses_val_or_test_treated_cells": False,
            "capacity_caveat": {
                "train_treated_conditions_with_exactly_256_cached_cells": 12,
                "interpretation": "set_epoch changes deterministic draw order, but membership cannot change for these 12 exhausted treated populations",
                "fallback_applied": False,
            },
        },
        "perturbation": {
            "dimensions": PERT_DIM,
            "drug": "379-d exact-drug one-hot",
            "dose": "1-d train-only standardized log10(dose_uM)",
            "excluded": [
                "cell_line_id",
                "plate",
                "sample",
                "MOA",
                "SMILES",
                "PubChem",
                "ESM",
            ],
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "training_seed": TRAINING_SEED,
            "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
            "num_gpus": FORMAL_NUM_GPUS,
            "global_batch_size": FORMAL_GLOBAL_BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "scheduler": None,
            "linear_lr_scaling": False,
        },
        "validation_and_checkpoint": {
            "split": "val",
            "test_access_during_training": False,
            "set_size": SET_SIZE,
            "dataset_epoch": 0,
            "shuffle": False,
            "drop_last": False,
            "formal_five_repeat_evaluation_during_training": False,
            "metric": "val edge-level Energy distance",
            "aggregation": [
                "repeat",
                "plate-level condition",
                "biological replicate",
                "(cell_line_id, drug, dose_uM)",
                "equal dose average",
                "(cell_line_id, drug) edge",
            ],
            "plate6_plate14_rule": "only dose_uM == 5.0 high-dose groups are paired when both plate6 and plate14 exist",
            "best_checkpoint": "minimum val edge-level mean Energy",
            "tie_break": "retain earlier epoch",
            "early_stopping": "stop after 5 consecutive epochs without strict improvement",
        },
        "smoke": {
            "scientific_result": False,
            "suggested_train_conditions": 256,
            "suggested_val_conditions": 64,
            "optimizer_steps_max": 200,
            "formal_test_evaluation": False,
        },
        "implementation": {
            "path": "perturbation_scripts/train_tahoe_experiment1_b2.py",
            "sha256": sha256_file(SCRIPT_PATH),
            "dataset_path": "perturbation_scripts/tahoe_experiment1_latent_data.py",
            "dataset_sha256": sha256_file(dataset_path),
            "aggregation_path": "perturbation_scripts/evaluate_tahoe_experiment1_b0.py",
            "aggregation_sha256": sha256_file(aggregation_path),
        },
        "frozen_inputs": {
            "perturbation_specification": {
                "path": "results/tahoe_experiment1_perturbation_featurization.json",
                "sha256": sha256_file(perturbation_path),
            },
            "condition_split_manifest": {
                "path": "results/tahoe_experiment1_condition_split_manifest.csv",
                "sha256": sha256_file(condition_manifest),
            },
            "edge_split_manifest": {
                "path": "results/tahoe_experiment1_edge_split_manifest.csv",
                "sha256": sha256_file(edge_manifest),
            },
        },
    }


def ensure_protocol(*, rewrite: bool) -> tuple[dict[str, Any], str]:
    expected = protocol_payload()
    if rewrite or not PROTOCOL_PATH.exists():
        atomic_write_json(PROTOCOL_PATH, expected)
    observed = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if observed != expected:
        raise AssertionError(
            "B2 protocol differs from the frozen contract; inspect it or use "
            "--rewrite-protocol only after an authorized implementation update"
        )
    return observed, sha256_file(PROTOCOL_PATH)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def core_model(model: nn.Module) -> B2PooledMLP:
    return model.module if isinstance(model, DistributedDataParallel) else model


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in ("ctrl_cell_emb", "pert_cell_emb", "pert_emb")
    }


def make_split_dataset(split: str, limit: int) -> TahoeExperiment1LatentSetDataset:
    dataset = TahoeExperiment1LatentSetDataset(
        split=split,
        seed=TRAINING_SEED,
        epoch=0,
    )
    if dataset.split_counts != EXPECTED_SPLIT_COUNTS:
        raise AssertionError(f"Frozen split counts changed: {dataset.split_counts}")
    if limit > 0:
        if limit > len(dataset):
            raise ValueError(f"Requested {limit} {split} conditions, only {len(dataset)} exist")
        dataset.conditions = dataset.conditions.iloc[:limit].copy().reset_index(drop=True)
    if not dataset.conditions["split"].eq(split).all():
        raise AssertionError(f"A non-{split} condition entered the Dataset")
    return dataset


def indices_sha256(*arrays: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.asarray(array.detach().cpu().numpy(), dtype="<i8")
        digest.update(values.tobytes())
    return digest.hexdigest()


def train_redraw_audit(dataset: TahoeExperiment1LatentSetDataset) -> dict[str, Any]:
    row = dataset.conditions.iloc[0]
    snapshots: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for epoch in (0, 1):
        dataset.set_epoch(epoch)
        source = dataset._sample_range(
            int(row["control_embedding_start"]),
            int(row["control_embedding_stop_exclusive"]),
            str(row["pair_id"]),
            "control",
        )
        target = dataset._sample_range(
            int(row["treated_embedding_start"]),
            int(row["treated_embedding_stop_exclusive"]),
            str(row["pair_id"]),
            "treated",
        )
        snapshots[epoch] = (source, target)
    dataset.set_epoch(0)
    source_changed = not np.array_equal(
        np.sort(snapshots[0][0]), np.sort(snapshots[1][0])
    )
    target_changed = not np.array_equal(
        np.sort(snapshots[0][1]), np.sort(snapshots[1][1])
    )
    if not source_changed or not target_changed:
        raise AssertionError("Smoke condition did not redraw control and treated memberships")
    return {
        "status": "pass",
        "condition_id": str(row["pair_id"]),
        "control_population_size": int(row["control_cached_cell_count"]),
        "treated_population_size": int(row["treated_cached_cell_count"]),
        "epoch_0_index_sha256": indices_sha256(
            torch.from_numpy(snapshots[0][0]), torch.from_numpy(snapshots[0][1])
        ),
        "epoch_1_index_sha256": indices_sha256(
            torch.from_numpy(snapshots[1][0]), torch.from_numpy(snapshots[1][1])
        ),
        "control_membership_changed": source_changed,
        "treated_membership_changed": target_changed,
        "dataset_set_epoch_called_each_training_epoch": True,
    }


def initial_identity_audit(
    model: B2PooledMLP,
    dataset: TahoeExperiment1LatentSetDataset,
    device: torch.device,
    metric: SamplesLoss,
) -> dict[str, Any]:
    dataset.set_epoch(0)
    loader = make_dataloader(
        dataset,
        batch_size=min(2, len(dataset)),
        shuffle=False,
        seed=TRAINING_SEED,
        drop_last=False,
    )
    raw = next(iter(loader))
    batch = move_batch(raw, device)
    if not torch.equal(
        batch["pert_emb"], batch["pert_emb"][:, :1, :].expand_as(batch["pert_emb"])
    ):
        raise AssertionError("Perturbation vector differs within an S=256 set")
    final = model.mlp[-1]
    assert isinstance(final, nn.Linear)
    with torch.no_grad():
        delta = model.predict_delta(batch["ctrl_cell_emb"], batch["pert_emb"])
        prediction = model(batch)
        energy = metric(prediction, batch["pert_cell_emb"]).mean()
    delta_max = float(delta.abs().max())
    identity_max = float((prediction - batch["ctrl_cell_emb"]).abs().max())
    checks = {
        "final_weight_exact_zero": bool(torch.count_nonzero(final.weight) == 0),
        "final_bias_exact_zero": bool(torch.count_nonzero(final.bias) == 0),
        "delta_exact_zero": bool(torch.count_nonzero(delta) == 0),
        "prediction_exactly_equals_control": bool(
            torch.equal(prediction, batch["ctrl_cell_emb"])
        ),
        "finite_energy": bool(torch.isfinite(energy)),
        "signed_control": bool((batch["ctrl_cell_emb"] < 0).any()),
        "signed_prediction": bool((prediction < 0).any()),
    }
    if not all(checks.values()):
        raise AssertionError(f"B2 initialization contract failed: {checks}")
    return {
        "status": "pass",
        "batch_size": int(prediction.shape[0]),
        "input_shapes": {
            "ctrl_cell_emb": list(batch["ctrl_cell_emb"].shape),
            "pert_cell_emb": list(batch["pert_cell_emb"].shape),
            "pert_emb": list(batch["pert_emb"].shape),
        },
        "delta_shape": list(delta.shape),
        "prediction_shape": list(prediction.shape),
        "delta_max_abs": delta_max,
        "prediction_minus_control_max_abs": identity_max,
        "initial_energy": float(energy),
        "control_negative_ratio": float((batch["ctrl_cell_emb"] < 0).float().mean()),
        "prediction_negative_ratio": float((prediction < 0).float().mean()),
        "checks": checks,
    }


def assert_unique_set_indices(indices: torch.Tensor, side: str) -> None:
    if indices.ndim != 2 or indices.shape[1] != SET_SIZE:
        raise AssertionError(f"Unexpected {side} index shape: {tuple(indices.shape)}")
    ordered = torch.sort(indices, dim=1).values
    if (ordered[:, 1:] == ordered[:, :-1]).any():
        raise AssertionError(f"Duplicate cell within a {side} set")


@torch.no_grad()
def validate_model(
    model: B2PooledMLP,
    dataset: TahoeExperiment1LatentSetDataset,
    device: torch.device,
    metric: SamplesLoss,
) -> dict[str, Any]:
    dataset.set_epoch(0)
    loader = make_dataloader(
        dataset,
        batch_size=BATCH_SIZE_PER_GPU,
        shuffle=False,
        seed=TRAINING_SEED,
        drop_last=False,
    )
    if loader.drop_last or type(loader.sampler).__name__ != "SequentialSampler":
        raise AssertionError("Validation must use shuffle=False and drop_last=False")
    metadata = dataset.conditions.set_index("pair_id")
    records: list[dict[str, Any]] = []
    energies: list[float] = []
    index_digest = hashlib.sha256()
    model.eval()
    for raw_batch in loader:
        if set(raw_batch["split"]) != {"val"}:
            raise AssertionError("Non-val condition entered B2 validation")
        assert_unique_set_indices(raw_batch["source_embedding_index"], "control")
        assert_unique_set_indices(raw_batch["target_embedding_index"], "treated")
        index_digest.update(
            np.asarray(raw_batch["source_embedding_index"].numpy(), dtype="<i8").tobytes()
        )
        index_digest.update(
            np.asarray(raw_batch["target_embedding_index"].numpy(), dtype="<i8").tobytes()
        )
        batch = move_batch(raw_batch, device)
        prediction = model(batch)
        values = metric(prediction, batch["pert_cell_emb"]).reshape(-1)
        if values.shape != (len(raw_batch["condition_id"]),) or not torch.isfinite(
            values
        ).all():
            raise AssertionError("Validation Energy is non-finite or has the wrong shape")
        values_cpu = values.detach().cpu().numpy().astype(np.float64, copy=False)
        energies.extend(values_cpu.tolist())
        for condition_id, value in zip(
            raw_batch["condition_id"], values_cpu, strict=True
        ):
            row = metadata.loc[str(condition_id)]
            records.append(
                {
                    "condition_id": str(condition_id),
                    "edge_id": str(row["edge_id"]),
                    "cell_line_id": str(row["cell_line_id"]),
                    "drug": str(row["drug"]),
                    "dose_uM": float(row["dose_uM"]),
                    "plate": str(row["plate"]),
                    "repeat_epoch": 0,
                    "energy_distance": float(value),
                }
            )
    raw = pd.DataFrame.from_records(records)
    if len(raw) != len(dataset) or raw["condition_id"].nunique() != len(dataset):
        raise AssertionError("Validation did not evaluate each selected condition exactly once")
    edge, details = aggregate(raw)
    edge_mean = float(edge["energy_distance"].mean())
    energy_array = np.asarray(energies, dtype="<f8")
    return {
        "condition_count": int(len(raw)),
        "edge_count": int(len(edge)),
        "edge_level_mean_energy": edge_mean,
        "condition_energy_mean": float(energy_array.mean()),
        "condition_energy_values": energies,
        "condition_energy_sha256": hashlib.sha256(energy_array.tobytes()).hexdigest(),
        "cell_index_sha256": index_digest.hexdigest(),
        "plate6_plate14_replicate_groups": int(
            details["replicate_audit"].iloc[0]["plate6_plate14_groups"]
        ),
        "dataset_epoch": dataset.epoch,
        "shuffle": False,
        "drop_last": False,
        "finite": bool(np.isfinite(energy_array).all() and math.isfinite(edge_mean)),
    }


def save_checkpoint(
    path: Path,
    model: B2PooledMLP,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    global_step: int,
    val_edge_mean_energy: float,
    protocol_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "format": "tahoe_experiment1_b2_v1",
            "epoch": epoch,
            "global_step": global_step,
            "val_edge_level_mean_energy": val_edge_mean_energy,
            "protocol_sha256": protocol_sha256,
            "model_state": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "optimizer_state": optimizer.state_dict(),
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint_model(
    path: Path, protocol_sha256: str, device: torch.device
) -> tuple[B2PooledMLP, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("format") != "tahoe_experiment1_b2_v1"
        or payload.get("protocol_sha256") != protocol_sha256
        or "optimizer_state" not in payload
    ):
        raise AssertionError("B2 checkpoint provenance or optimizer state is invalid")
    model = B2PooledMLP()
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    return model, payload


def parameter_update_audit(
    model: B2PooledMLP, initial: dict[str, torch.Tensor]
) -> dict[str, Any]:
    changed = 0
    max_abs = 0.0
    for name, parameter in model.named_parameters():
        difference = (parameter.detach().cpu() - initial[name]).abs()
        if torch.count_nonzero(difference):
            changed += 1
            max_abs = max(max_abs, float(difference.max()))
    if changed == 0 or not math.isfinite(max_abs) or max_abs <= 0:
        raise AssertionError("B2 parameters did not update")
    return {
        "status": "pass",
        "changed_parameter_tensors": changed,
        "total_parameter_tensors": len(initial),
        "maximum_absolute_change": max_abs,
    }


def ddp_parameter_sync_error(model: B2PooledMLP, world_size: int) -> float:
    moments = torch.zeros(3, device=next(model.parameters()).device, dtype=torch.float64)
    for parameter in model.parameters():
        values = parameter.detach().double()
        moments[0] += values.sum()
        moments[1] += values.square().sum()
        moments[2] += values.abs().sum()
    gathered = [torch.empty_like(moments) for _ in range(world_size)]
    dist.all_gather(gathered, moments)
    return max(float((value - gathered[0]).abs().max()) for value in gathered)


def environment_payload(device: torch.device, world_size: int) -> dict[str, Any]:
    return {
        "python": ".".join(map(str, os.sys.version_info[:3])),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "geomloss": importlib.metadata.version("geomloss"),
        "world_size": world_size,
        "visible_cuda_devices": torch.cuda.device_count(),
        "gpu": torch.cuda.get_device_name(device),
        "device": str(device),
        "nccl_cumem_enable": os.environ.get("NCCL_CUMEM_ENABLE"),
        "nccl_cumem_host_enable": os.environ.get("NCCL_CUMEM_HOST_ENABLE"),
    }


def run_training_smoke(
    *,
    run_kind: str,
    device: torch.device,
    rank: int,
    world_size: int,
    train_condition_limit: int,
    val_condition_limit: int,
    max_steps: int,
    max_epochs: int,
    protocol_sha256: str,
) -> dict[str, Any] | None:
    distributed = world_size > 1
    if distributed != (run_kind == "ddp"):
        raise AssertionError("run-kind and torch distributed world size disagree")
    seed_everything(TRAINING_SEED)
    train_dataset = make_split_dataset("train", train_condition_limit)
    val_dataset = make_split_dataset("val", val_condition_limit)
    if set(train_dataset.conditions["edge_id"]) & set(val_dataset.conditions["edge_id"]):
        raise AssertionError("Smoke train/val treated edges overlap")
    redraw = train_redraw_audit(train_dataset)

    sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=TRAINING_SEED,
            drop_last=True,
        )
        if distributed
        else None
    )
    train_loader = make_dataloader(
        train_dataset,
        batch_size=BATCH_SIZE_PER_GPU,
        shuffle=not distributed,
        seed=TRAINING_SEED,
        sampler=sampler,
        drop_last=True,
    )
    if not train_loader.drop_last:
        raise AssertionError("Training must use drop_last=True")
    if distributed and not isinstance(train_loader.sampler, DistributedSampler):
        raise AssertionError("DDP training lacks DistributedSampler")
    if not distributed and type(train_loader.sampler).__name__ != "RandomSampler":
        raise AssertionError("Single-GPU training lacks shuffle=True")

    raw_model = B2PooledMLP().to(device)
    metric = SamplesLoss(loss="energy", blur=ENERGY_BLUR)
    identity = initial_identity_audit(raw_model, train_dataset, device, metric)
    initial_parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in raw_model.named_parameters()
    }
    model: nn.Module = (
        DistributedDataParallel(raw_model, device_ids=[device.index])
        if distributed
        else raw_model
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    checkpoint_path = CHECKPOINT_DIR / f"b2_{run_kind}_best.pt"

    epoch_history: list[dict[str, Any]] = []
    step_losses: list[float] = []
    gradient_norms: list[float] = []
    global_step = 0
    best_metric = float("inf")
    best_epoch = -1
    no_improvement_epochs = 0
    early_stopped = False
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()

    for epoch in range(max_epochs):
        if global_step >= max_steps:
            break
        train_dataset.set_epoch(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        epoch_losses: list[float] = []
        for raw_batch in train_loader:
            if global_step >= max_steps:
                break
            if set(raw_batch["split"]) != {"train"}:
                raise AssertionError("Non-train condition entered B2 fitting")
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = metric(prediction, batch["pert_cell_emb"]).mean()
            if not torch.isfinite(loss):
                raise AssertionError("B2 training Energy became non-finite")
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise AssertionError("B2 gradient became non-finite")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP_NORM,
                error_if_nonfinite=True,
            )
            optimizer.step()
            report_loss = loss.detach().clone()
            if distributed:
                dist.all_reduce(report_loss, op=dist.ReduceOp.SUM)
                report_loss /= world_size
            loss_value = float(report_loss)
            step_losses.append(loss_value)
            epoch_losses.append(loss_value)
            gradient_norms.append(float(gradient_norm.detach()))
            global_step += 1
        if not epoch_losses:
            break

        if distributed:
            dist.barrier()
        validation = validate_model(raw_model, val_dataset, device, metric) if rank == 0 else None
        metric_tensor = torch.tensor(
            [validation["edge_level_mean_energy"] if rank == 0 else 0.0],
            device=device,
            dtype=torch.float64,
        )
        if distributed:
            dist.broadcast(metric_tensor, src=0)
        val_metric = float(metric_tensor.item())
        improved = val_metric < best_metric
        if improved:
            best_metric = val_metric
            best_epoch = epoch
            no_improvement_epochs = 0
            if rank == 0:
                save_checkpoint(
                    checkpoint_path,
                    raw_model,
                    optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    val_edge_mean_energy=val_metric,
                    protocol_sha256=protocol_sha256,
                )
        else:
            no_improvement_epochs += 1
        if rank == 0:
            epoch_history.append(
                {
                    "epoch": epoch,
                    "optimizer_steps": len(epoch_losses),
                    "global_step": global_step,
                    "train_energy_mean": float(np.mean(epoch_losses)),
                    "train_energy_first": epoch_losses[0],
                    "train_energy_last": epoch_losses[-1],
                    "val_edge_level_mean_energy": val_metric,
                    "strict_improvement": improved,
                    "no_improvement_epochs": no_improvement_epochs,
                }
            )
            print(
                f"B2 {run_kind} epoch={epoch} step={global_step} "
                f"train_energy={np.mean(epoch_losses):.8f} "
                f"val_edge_energy={val_metric:.8f} best_epoch={best_epoch}",
                flush=True,
            )
        if distributed:
            dist.barrier()
        if no_improvement_epochs >= EARLY_STOPPING_PATIENCE:
            early_stopped = True
            break

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if not step_losses or best_epoch < 0:
        raise AssertionError("B2 smoke performed no optimizer step or saved no checkpoint")
    update = parameter_update_audit(raw_model, initial_parameters)
    sync_error = ddp_parameter_sync_error(raw_model, world_size) if distributed else 0.0
    if sync_error != 0:
        raise AssertionError(f"DDP parameters are not synchronized: {sync_error}")

    local_runtime = {
        "rank": rank,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }
    runtimes: list[dict[str, Any] | None]
    if distributed:
        gathered: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(gathered, local_runtime)
        runtimes = gathered
        dist.barrier()
    else:
        runtimes = [local_runtime]

    if rank != 0:
        if distributed:
            dist.barrier()
        return None

    best_model, checkpoint = load_checkpoint_model(
        checkpoint_path, protocol_sha256, device
    )
    fixed_val_first = validate_model(best_model, val_dataset, device, metric)
    fixed_val_second = validate_model(best_model, val_dataset, device, metric)
    energy_first = np.asarray(fixed_val_first.pop("condition_energy_values"), dtype=np.float64)
    energy_second = np.asarray(fixed_val_second.pop("condition_energy_values"), dtype=np.float64)
    fixed_indices_equal = (
        fixed_val_first["cell_index_sha256"] == fixed_val_second["cell_index_sha256"]
    )
    fixed_energy_equal = np.array_equal(energy_first, energy_second)
    fixed_edge_equal = (
        fixed_val_first["edge_level_mean_energy"]
        == fixed_val_second["edge_level_mean_energy"]
    )
    if not fixed_indices_equal or not fixed_energy_equal or not fixed_edge_equal:
        raise AssertionError("Fixed validation epoch=0 is not exactly reproducible")
    checkpoint_audit = {
        "path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
        "round_trip_load": True,
        "contains_optimizer_state": True,
        "selected_epoch": int(checkpoint["epoch"]),
        "selected_global_step": int(checkpoint["global_step"]),
        "selected_val_edge_level_mean_energy": float(
            checkpoint["val_edge_level_mean_energy"]
        ),
        "selection_rule": "strict minimum; equal value retains earlier epoch",
    }
    first_epoch_energy = float(epoch_history[0]["train_energy_mean"])
    final_epoch_energy = float(epoch_history[-1]["train_energy_mean"])
    train_decreased = final_epoch_energy < first_epoch_energy
    if run_kind == "single" and not train_decreased:
        raise AssertionError("Single-GPU smoke train Energy did not decrease")
    checks = {
        "forward_shape": identity["prediction_shape"][-2:] == [SET_SIZE, LATENT_DIM],
        "initial_delta_zero": identity["checks"]["delta_exact_zero"],
        "initial_prediction_equals_b0": identity["checks"][
            "prediction_exactly_equals_control"
        ],
        "energy_finite": all(math.isfinite(value) for value in step_losses),
        "backward_and_gradients_finite": all(
            math.isfinite(value) for value in gradient_norms
        ),
        "parameters_updated": update["status"] == "pass",
        "single_gpu_train_energy_decreased": train_decreased
        if run_kind == "single"
        else "covered_by_single_gpu_smoke",
        "raw_signed_latent_preserved": identity["checks"]["signed_prediction"],
        "no_val_test_fitting_leakage": True,
        "train_epoch_redraw": redraw["status"] == "pass",
        "fixed_val_exact_reproducibility": True,
        "checkpoint_round_trip": checkpoint_audit["round_trip_load"],
        "ddp_parameters_synchronized": sync_error == 0 if distributed else "not_applicable",
    }
    if any(value is False for value in checks.values()):
        raise AssertionError(f"B2 smoke checks failed: {checks}")
    result = {
        "status": "pass",
        "run_kind": run_kind,
        "scientific_result": False,
        "world_size": world_size,
        "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
        "global_batch_size": BATCH_SIZE_PER_GPU * world_size,
        "train_conditions": len(train_dataset),
        "val_conditions": len(val_dataset),
        "max_epochs_cap": max_epochs,
        "max_optimizer_steps_cap": max_steps,
        "epochs_completed": len(epoch_history),
        "optimizer_steps": global_step,
        "early_stopped": early_stopped,
        "best_epoch": best_epoch,
        "parameter_count": sum(parameter.numel() for parameter in raw_model.parameters()),
        "initialization": identity,
        "training_energy": {
            "first_step": step_losses[0],
            "last_step": step_losses[-1],
            "first_epoch_mean": first_epoch_energy,
            "final_epoch_mean": final_epoch_energy,
            "absolute_epoch_mean_decrease": first_epoch_energy - final_epoch_energy,
            "relative_epoch_mean_decrease": (
                first_epoch_energy - final_epoch_energy
            )
            / first_epoch_energy,
            "minimum_step": min(step_losses),
            "all_finite": True,
            "decreased": train_decreased,
        },
        "gradient_norm": {
            "minimum": min(gradient_norms),
            "maximum": max(gradient_norms),
            "all_finite": True,
            "clip_norm": GRADIENT_CLIP_NORM,
        },
        "parameter_update": update,
        "train_sampling": redraw,
        "fixed_validation_reproducibility": {
            "status": "pass",
            "dataset_epoch": 0,
            "cell_indices_exactly_equal": fixed_indices_equal,
            "condition_energies_bitwise_equal": fixed_energy_equal,
            "edge_mean_bitwise_equal": fixed_edge_equal,
            "maximum_condition_energy_absolute_difference": float(
                np.max(np.abs(energy_first - energy_second))
            ),
            "first": fixed_val_first,
            "second": fixed_val_second,
        },
        "checkpoint": checkpoint_audit,
        "epoch_history": epoch_history,
        "optimizer": {
            "name": type(optimizer).__name__,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": None,
        },
        "runtime": {
            "wall_seconds": elapsed,
            "mean_optimizer_step_seconds_including_epoch_validation": elapsed
            / global_step,
            "rank": runtimes,
        },
        "ddp_parameter_moment_max_abs_error": sync_error,
        "data_contract": {
            "train_split_only_for_fit": True,
            "val_split_only_for_checkpoint": True,
            "test_dataset_constructed": False,
            "latent_transforms": list(train_dataset.embedding_transforms),
            "dose_fit_scope": train_dataset.dose_statistics["fit_scope"],
            "edge_overlap": train_dataset.edge_overlap,
            "train_shuffle": True,
            "train_drop_last": True,
            "val_shuffle": False,
            "val_drop_last": False,
        },
        "checks": checks,
        "environment": environment_payload(device, world_size),
    }
    if distributed:
        dist.barrier()
    return result


def write_handoff(payload: dict[str, Any]) -> None:
    single = payload.get("single_gpu")
    ddp = payload.get("two_gpu_ddp")

    def run_line(name: str, result: dict[str, Any] | None) -> str:
        if result is None:
            return f"- {name}: **NOT RUN**"
        energy = result["training_energy"]
        return (
            f"- {name}: **{result['status'].upper()}**, "
            f"{result['optimizer_steps']} steps, train epoch-mean Energy "
            f"{energy['first_epoch_mean']:.8f} -> {energy['final_epoch_mean']:.8f}."
        )

    text = f"""# Tahoe Experiment 1 B2 implementation smoke

- Status: **{payload['status'].upper()}**
- Scope: engineering smoke only; no formal full training, formal test evaluation, STATE, or ST.
- Model parameters: **{payload['parameter_count']:,}**
- Frozen protocol: `results/tahoe_experiment1_b2_training_protocol.json`

## Runtime smoke

{run_line('Single GPU', single)}
{run_line('Two-GPU DDP', ddp)}

## Contract checks

- Initial final-layer weight/bias are zero; `Delta_pred == 0` and `Zpred == Zctrl` exactly.
- Output shape is `[B,256,768]`; raw signed latent coordinates remain signed.
- Training uses real `geomloss.SamplesLoss(loss="energy", blur=0.05)`, AdamW, and finite clipped gradients.
- Train uses only the frozen train split with `shuffle=True`, `drop_last=True`, and epoch redraw.
- Validation uses only val, fixed Dataset epoch 0, `shuffle=False`, `drop_last=False`, and the frozen edge-level aggregation hierarchy.
- Fixed validation indices and Energy values repeat exactly.
- Checkpoints select the strict minimum val edge-level mean Energy; equal values retain the earlier epoch.

## Data feasibility note

Twelve of 45,652 train treated conditions have exactly 256 cached cells. For those exhausted populations, changing Dataset epoch can only reorder the same 256 members; no fallback or population change was applied. The smoke subset contains populations larger than 256 and verifies actual membership redraw.

Machine-readable result: `results/tahoe_experiment1_b2_implementation_smoke.json`
"""
    atomic_write_text(HANDOFF_PATH, text)


def update_combined_result(
    run_kind: str,
    result: dict[str, Any],
    protocol_sha256: str,
    *,
    reset: bool,
) -> dict[str, Any]:
    existing: dict[str, Any] = {}
    if SMOKE_PATH.exists() and not reset:
        existing = json.loads(SMOKE_PATH.read_text(encoding="utf-8"))
    run_key = "single_gpu" if run_kind == "single" else "two_gpu_ddp"
    payload = {
        "schema": "tahoe_experiment1_b2_implementation_smoke_v1",
        "created_at_utc": existing.get("created_at_utc", utc_now()),
        "updated_at_utc": utc_now(),
        "status": "partial_pass",
        "scope": {
            "engineering_smoke_only": True,
            "formal_full_training_started": False,
            "formal_test_evaluation_run": False,
            "state_or_st_run": False,
            "scientific_result": False,
        },
        "protocol": {
            "path": "results/tahoe_experiment1_b2_training_protocol.json",
            "sha256": protocol_sha256,
            "schema": PROTOCOL_VERSION,
        },
        "implementation": {
            "path": "perturbation_scripts/train_tahoe_experiment1_b2.py",
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "parameter_count": parameter_count(),
        "single_gpu": existing.get("single_gpu"),
        "two_gpu_ddp": existing.get("two_gpu_ddp"),
        "data_feasibility_note": {
            "train_treated_conditions_with_exactly_256_cached_cells": 12,
            "effect": "membership redraw is impossible only for these exhausted treated populations",
            "smoke_redraw_uses_population_gt_256": True,
            "fallback_applied": False,
        },
        "outputs": {
            "protocol_json": "results/tahoe_experiment1_b2_training_protocol.json",
            "smoke_json": "results/tahoe_experiment1_b2_implementation_smoke.json",
            "handoff_md": "results/tahoe_experiment1_b2_implementation_smoke.md",
        },
    }
    payload[run_key] = result
    if (
        payload["single_gpu"] is not None
        and payload["two_gpu_ddp"] is not None
        and payload["single_gpu"].get("status") == "pass"
        and payload["two_gpu_ddp"].get("status") == "pass"
    ):
        payload["status"] = "pass"
    atomic_write_json(SMOKE_PATH, payload)
    write_handoff(payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-kind", choices=("single", "ddp"))
    parser.add_argument("--write-protocol-only", action="store_true")
    parser.add_argument("--rewrite-protocol", action="store_true")
    parser.add_argument("--reset-output", action="store_true")
    parser.add_argument("--train-conditions", type=int, default=256)
    parser.add_argument("--val-conditions", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()
    if not args.write_protocol_only and args.run_kind is None:
        parser.error("--run-kind is required unless --write-protocol-only is used")
    if min(args.train_conditions, args.val_conditions, args.max_steps, args.max_epochs) < 1:
        parser.error("condition limits, max-steps, and max-epochs must be positive")
    return args


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if args.write_protocol_only:
        if distributed:
            raise RuntimeError("Write the protocol with one ordinary Python process")
        protocol, protocol_sha = ensure_protocol(rewrite=args.rewrite_protocol)
        print(
            json.dumps(
                {
                    "status": "pass",
                    "path": str(PROTOCOL_PATH),
                    "sha256": protocol_sha,
                    "parameter_count": protocol["architecture"]["parameter_count"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if not torch.cuda.is_available():
        raise RuntimeError("B2 smoke requires CUDA")
    if distributed:
        if args.run_kind != "ddp" or world_size != FORMAL_NUM_GPUS:
            raise RuntimeError("B2 DDP smoke requires exactly two processes/GPUs")
        if os.environ.get("NCCL_CUMEM_ENABLE") != "0" or os.environ.get(
            "NCCL_CUMEM_HOST_ENABLE"
        ) != "0":
            raise RuntimeError("This host requires both NCCL CUMEM workaround variables set to 0")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        if args.run_kind != "single" or torch.cuda.device_count() != 1:
            raise RuntimeError("Single smoke requires CUDA_VISIBLE_DEVICES to expose one GPU")
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
    try:
        _protocol, protocol_sha = ensure_protocol(rewrite=args.rewrite_protocol)
        result = run_training_smoke(
            run_kind=args.run_kind,
            device=device,
            rank=rank,
            world_size=world_size,
            train_condition_limit=args.train_conditions,
            val_condition_limit=args.val_conditions,
            max_steps=args.max_steps,
            max_epochs=args.max_epochs,
            protocol_sha256=protocol_sha,
        )
        if rank == 0:
            assert result is not None
            payload = update_combined_result(
                args.run_kind,
                result,
                protocol_sha,
                reset=args.reset_output,
            )
            print(json.dumps(payload, indent=2, ensure_ascii=False))
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
