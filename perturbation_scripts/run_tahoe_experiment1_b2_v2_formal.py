#!/usr/bin/env python3
"""Run formal B2-v2 MSE training and select checkpoints by val Energy."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from geomloss import SamplesLoss
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from train_tahoe_experiment1_b2 import (
    ENERGY_BLUR,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    RESULTS,
    SCRIPT_PATH as B2_V1_IMPLEMENTATION_PATH,
    TRAINING_SEED,
    WEIGHT_DECAY,
    atomic_write_json,
    make_dataloader,
    make_split_dataset,
    seed_everything,
    sha256_file,
    utc_now,
    validate_model,
)
from train_tahoe_experiment1_b2_v2 import (
    B1_BUILD_AUDIT_PATH,
    DeltaPredictor,
    SCRIPT_PATH as B2_V2_IMPLEMENTATION_PATH,
    SHIFT_METADATA_PATH,
    SHIFT_PATH,
    load_targets,
    targets_for_batch,
)


PROJECT_ROOT = RESULTS.parent
SCRIPT_PATH = Path(__file__).resolve()
TASK_PATH = PROJECT_ROOT.parent / "当前任务.txt"
PROTOCOL_PATH = RESULTS / "tahoe_experiment1_b2_v2_training_protocol.json"
CHECKPOINT_DIR = RESULTS / "tahoe_experiment1_b2_v2_formal_checkpoints"
BEST_CHECKPOINT = CHECKPOINT_DIR / "b2_v2_best.pt"
LAST_CHECKPOINT = CHECKPOINT_DIR / "b2_v2_last.pt"
HISTORY_PATH = RESULTS / "tahoe_experiment1_b2_v2_training_history.csv"
RESULT_PATH = RESULTS / "tahoe_experiment1_b2_v2_formal_training_result.json"

FORMAL_TASK_SHA256 = "cb5d8e033ab6e216f5f2deee046b6cf104a566748bb672fd475a1470c62dedb7"
FROZEN_AT_UTC = "2026-09-10T03:30:00+00:00"
BATCH_SIZE_PER_GPU = 128
WORLD_SIZE = 2
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_GPU * WORLD_SIZE
MAX_EPOCHS = 30
PATIENCE = 5

V1_ARTIFACT_PATHS = (
    RESULTS / "tahoe_experiment1_b2_training_protocol.json",
    RESULTS / "tahoe_experiment1_b2_formal_checkpoints" / "b2_best.pt",
    RESULTS / "tahoe_experiment1_b2_formal_checkpoints" / "b2_last.pt",
    RESULTS / "tahoe_experiment1_b2_training_history.csv",
    RESULTS / "tahoe_experiment1_b2_formal_training_console.log",
)


def relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def artifact_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required preserved artifact is missing: {path}")
    return {
        "path": relative(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def v1_artifacts() -> list[dict[str, Any]]:
    return [artifact_record(path) for path in V1_ARTIFACT_PATHS]


def protocol_payload() -> dict[str, Any]:
    return {
        "schema": "tahoe_experiment1_b2_v2_training_v1",
        "status": "frozen",
        "frozen_at_utc": FROZEN_AT_UTC,
        "source_task": {"path": str(TASK_PATH), "sha256": FORMAL_TASK_SHA256},
        "model": {
            "architecture": "B2PooledMLP 1148 -> 1024 -> GELU -> 1024 -> GELU -> 768",
            "control_context": "mean(ctrl_cell_emb, dim=1)",
            "input": "concat(control_context, perturbation_380d)",
            "output": "Delta_pred [B,768]",
            "future_prediction": "raw_Zctrl + Delta_pred[:,None,:]",
            "final_linear_initialization": "zero weight and zero bias",
        },
        "target": {
            "definition": "centroid(all cached treated cells) - centroid(all matched cached control cells)",
            "recomputed": False,
            "array": artifact_record(SHIFT_PATH),
            "metadata": artifact_record(SHIFT_METADATA_PATH),
            "build_audit": artifact_record(B1_BUILD_AUDIT_PATH),
            "shape": [45_652, 768],
            "dtype": "float32",
            "fit_split": "train only",
        },
        "training": {
            "conditions": 45_652,
            "set_size": 256,
            "loss": "torch.nn.functional.mse_loss(Delta_pred, Delta_condition)",
            "other_training_losses": [],
            "dataset_set_epoch": True,
            "sampler_set_epoch": True,
            "shuffle": True,
            "drop_last": True,
            "world_size": WORLD_SIZE,
            "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "scheduler": None,
            "training_seed": TRAINING_SEED,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
        },
        "validation": {
            "conditions": 5_657,
            "dataset_epoch": 0,
            "shuffle": False,
            "drop_last": False,
            "prediction": "raw_Zctrl + Delta_pred[:,None,:]",
            "metric": "geomloss.SamplesLoss(loss='energy', blur=0.05)",
            "aggregation": [
                "condition",
                "biological replicate",
                "(cell_line,drug,dose)",
                "equal dose average",
                "(cell_line,drug) edge",
            ],
            "plate6_plate14_rule": "pair only dose_uM == 5.0 when both plates exist",
            "checkpoint_selection": "strict minimum val edge-level mean Energy",
            "test_dataset_used": False,
        },
        "implementation": {
            "architecture": artifact_record(B2_V1_IMPLEMENTATION_PATH),
            "b2_v2_mse": artifact_record(B2_V2_IMPLEMENTATION_PATH),
            "formal_runner": artifact_record(SCRIPT_PATH),
        },
        "b2_v1_preserved_artifacts": v1_artifacts(),
    }


def ensure_protocol(*, create: bool) -> tuple[dict[str, Any], str]:
    expected = protocol_payload()
    if not PROTOCOL_PATH.exists():
        if not create:
            raise FileNotFoundError(f"Run prepare first to freeze {PROTOCOL_PATH}")
        atomic_write_json(PROTOCOL_PATH, expected)
    observed = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if observed != expected:
        raise AssertionError("B2-v2 frozen training protocol or an input artifact changed")
    return observed, sha256_file(PROTOCOL_PATH)


def make_datasets() -> tuple[Any, Any]:
    train_dataset = make_split_dataset("train", 0)
    val_dataset = make_split_dataset("val", 0)
    if len(train_dataset) != 45_652 or len(val_dataset) != 5_657:
        raise AssertionError("Frozen formal train/val condition counts changed")
    if set(train_dataset.conditions["edge_id"]) & set(val_dataset.conditions["edge_id"]):
        raise AssertionError("Frozen train and val edges overlap")
    if train_dataset.embedding_transforms or val_dataset.embedding_transforms:
        raise AssertionError("B2-v2 requires raw latent with no transforms")
    return train_dataset, val_dataset


def training_shape() -> dict[str, int]:
    samples_per_rank = 45_652 // WORLD_SIZE
    steps_per_epoch = samples_per_rank // BATCH_SIZE_PER_GPU
    consumed = steps_per_epoch * GLOBAL_BATCH_SIZE
    return {
        "samples_per_rank": samples_per_rank,
        "optimizer_steps_per_epoch": steps_per_epoch,
        "conditions_consumed_per_epoch": consumed,
        "conditions_dropped_per_epoch": 45_652 - consumed,
        "validation_batches_per_epoch_on_rank0": math.ceil(5_657 / 64),
        "maximum_optimizer_steps": steps_per_epoch * MAX_EPOCHS,
    }


def formal_outputs() -> list[Path]:
    return [BEST_CHECKPOINT, LAST_CHECKPOINT, HISTORY_PATH, RESULT_PATH]


def prepare() -> dict[str, Any]:
    if sha256_file(TASK_PATH) != FORMAL_TASK_SHA256:
        raise AssertionError("当前任务.txt changed before B2-v2 protocol freeze")
    train_dataset, val_dataset = make_datasets()
    shifts, metadata, _row_by_condition, alignment = load_targets(train_dataset)
    protocol, protocol_sha = ensure_protocol(create=True)
    existing = [relative(path) for path in formal_outputs() if path.exists()]
    return {
        "status": "pass" if not existing else "fail",
        "scope": "read-only formal B2-v2 readiness; no optimizer, validation, or test Dataset",
        "protocol": {"path": relative(PROTOCOL_PATH), "sha256": protocol_sha},
        "train_conditions": len(train_dataset),
        "val_conditions": len(val_dataset),
        "test_dataset_constructed": False,
        "condition_target_alignment": alignment,
        "target_shape": list(shifts.shape),
        "target_metadata_rows": len(metadata),
        "schedule": training_shape(),
        "b2_v1_artifacts_preserved": protocol["b2_v1_preserved_artifacts"],
        "formal_outputs_already_present": existing,
        "fresh_run_ready": not existing,
    }


def artifact_guard(*, resume_last: bool) -> None:
    if resume_last:
        if not BEST_CHECKPOINT.is_file() or not LAST_CHECKPOINT.is_file():
            raise FileNotFoundError("--resume-last requires B2-v2 best and last checkpoints")
        if RESULT_PATH.exists() and json.loads(
            RESULT_PATH.read_text(encoding="utf-8")
        ).get("status") == "pass":
            raise RuntimeError("B2-v2 formal training is already complete")
        return
    existing = [relative(path) for path in formal_outputs() if path.exists()]
    if existing or (CHECKPOINT_DIR.exists() and any(CHECKPOINT_DIR.iterdir())):
        raise FileExistsError("Fresh B2-v2 run refused; artifacts exist: " + ", ".join(existing))


def atomic_write_history(history: list[dict[str, Any]]) -> None:
    temporary = HISTORY_PATH.with_name(HISTORY_PATH.name + ".tmp")
    pd.DataFrame.from_records(history).to_csv(
        temporary, index=False, encoding="utf-8-sig", lineterminator="\n"
    )
    temporary.replace(HISTORY_PATH)


def save_checkpoint(
    path: Path,
    model: DeltaPredictor,
    optimizer: torch.optim.Optimizer,
    *,
    kind: str,
    epoch: int,
    global_step: int,
    train_mse: float,
    current_val_edge_energy: float,
    best_val_edge_energy: float,
    best_epoch: int,
    no_improvement_epochs: int,
    history: list[dict[str, Any]],
    protocol_sha256: str,
    elapsed_seconds_so_far: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "format": "tahoe_experiment1_b2_v2_formal_v1",
            "checkpoint_kind": kind,
            "fresh_initialization": True,
            "b2_v1_or_smoke_checkpoint_loaded": False,
            "model_state": {
                key: value.detach().cpu() for key, value in model.b2.state_dict().items()
            },
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "train_mse": train_mse,
            "current_val_edge_energy": current_val_edge_energy,
            "best_val_edge_energy": best_val_edge_energy,
            "best_epoch": best_epoch,
            "no_improvement_epochs": no_improvement_epochs,
            "history": history,
            "b2_v2_protocol_sha256": protocol_sha256,
            "implementation_sha256": sha256_file(B2_V2_IMPLEMENTATION_PATH),
            "formal_runner_sha256": sha256_file(SCRIPT_PATH),
            "training_seed": TRAINING_SEED,
            "condition_shift_target_sha256": sha256_file(SHIFT_PATH),
            "target_metadata_sha256": sha256_file(SHIFT_METADATA_PATH),
            "elapsed_seconds_so_far": elapsed_seconds_so_far,
        },
        temporary,
    )
    temporary.replace(path)


def load_last(
    model: DeltaPredictor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    protocol_sha256: str,
) -> dict[str, Any]:
    payload = torch.load(LAST_CHECKPOINT, map_location=device, weights_only=False)
    required = {
        "format": "tahoe_experiment1_b2_v2_formal_v1",
        "checkpoint_kind": "last",
        "fresh_initialization": True,
        "b2_v1_or_smoke_checkpoint_loaded": False,
        "b2_v2_protocol_sha256": protocol_sha256,
        "implementation_sha256": sha256_file(B2_V2_IMPLEMENTATION_PATH),
        "formal_runner_sha256": sha256_file(SCRIPT_PATH),
        "training_seed": TRAINING_SEED,
        "condition_shift_target_sha256": sha256_file(SHIFT_PATH),
        "target_metadata_sha256": sha256_file(SHIFT_METADATA_PATH),
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise AssertionError(f"B2-v2 last checkpoint mismatch for {key}")
    model.b2.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def validate_checkpoint(
    path: Path,
    *,
    kind: str,
    epoch: int,
    global_step: int,
    protocol_sha256: str,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "checkpoint_kind": kind,
        "epoch": epoch,
        "global_step": global_step,
        "b2_v2_protocol_sha256": protocol_sha256,
        "implementation_sha256": sha256_file(B2_V2_IMPLEMENTATION_PATH),
        "formal_runner_sha256": sha256_file(SCRIPT_PATH),
        "training_seed": TRAINING_SEED,
        "condition_shift_target_sha256": sha256_file(SHIFT_PATH),
        "target_metadata_sha256": sha256_file(SHIFT_METADATA_PATH),
    }
    if "model_state" not in checkpoint or "optimizer_state" not in checkpoint:
        raise AssertionError(f"B2-v2 {kind} checkpoint lacks model/optimizer state")
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise AssertionError(f"B2-v2 {kind} checkpoint mismatch for {key}")
    return checkpoint


def run(*, resume_last: bool) -> dict[str, Any] | None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != WORLD_SIZE:
        raise RuntimeError("Formal B2-v2 requires exactly two torchrun processes")
    if os.environ.get("NCCL_CUMEM_ENABLE") != "0" or os.environ.get(
        "NCCL_CUMEM_HOST_ENABLE"
    ) != "0":
        raise RuntimeError("Set both NCCL CUMEM workaround variables to 0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("Formal B2-v2 requires exactly two visible CUDA GPUs")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    session_started = time.perf_counter()
    try:
        protocol, protocol_sha = ensure_protocol(create=False)
        if rank == 0:
            artifact_guard(resume_last=resume_last)
        dist.barrier()

        seed_everything(TRAINING_SEED)
        train_dataset, val_dataset = make_datasets()
        shifts, metadata, row_by_condition, alignment = load_targets(train_dataset)
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=TRAINING_SEED,
            drop_last=True,
        )
        train_loader = make_dataloader(
            train_dataset,
            batch_size=BATCH_SIZE_PER_GPU,
            shuffle=False,
            seed=TRAINING_SEED,
            sampler=sampler,
            drop_last=True,
        )
        expected_steps = training_shape()["optimizer_steps_per_epoch"]
        if len(train_loader) != expected_steps or not train_loader.drop_last:
            raise AssertionError("B2-v2 formal train DataLoader contract changed")

        raw_model = DeltaPredictor().to(device)
        final = raw_model.b2.mlp[-1]
        if torch.count_nonzero(final.weight) or torch.count_nonzero(final.bias):
            raise AssertionError("B2-v2 final Linear is not zero-initialized")
        optimizer = torch.optim.AdamW(
            raw_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        if resume_last:
            checkpoint = load_last(raw_model, optimizer, device, protocol_sha)
            history = list(checkpoint["history"])
            start_epoch = int(checkpoint["epoch"]) + 1
            global_step = int(checkpoint["global_step"])
            best_metric = float(checkpoint["best_val_edge_energy"])
            best_epoch = int(checkpoint["best_epoch"])
            no_improvement_epochs = int(checkpoint["no_improvement_epochs"])
            prior_elapsed = float(checkpoint.get("elapsed_seconds_so_far", 0.0))
            if start_epoch >= MAX_EPOCHS or no_improvement_epochs >= PATIENCE:
                raise RuntimeError("B2-v2 last checkpoint is already terminal")
        else:
            history: list[dict[str, Any]] = []
            start_epoch = 0
            global_step = 0
            best_metric = float("inf")
            best_epoch = -1
            no_improvement_epochs = 0
            prior_elapsed = 0.0

        model = DistributedDataParallel(raw_model, device_ids=[local_rank])
        validation_metric = SamplesLoss(loss="energy", blur=ENERGY_BLUR)
        early_stopped = False
        all_training_losses_finite = True
        all_gradients_finite = True
        torch.cuda.reset_peak_memory_stats(device)

        for epoch in range(start_epoch, MAX_EPOCHS):
            train_dataset.set_epoch(epoch)
            sampler.set_epoch(epoch)
            if train_dataset.epoch != epoch or sampler.epoch != epoch:
                raise AssertionError("Dataset or DistributedSampler epoch was not updated")
            model.train()
            epoch_losses: list[float] = []
            for raw_batch in train_loader:
                if set(raw_batch["split"]) != {"train"}:
                    raise AssertionError("Non-train condition entered B2-v2 fitting")
                condition_ids = list(raw_batch["condition_id"])
                target, _rows = targets_for_batch(
                    condition_ids, shifts, metadata, row_by_condition, device
                )
                ctrl = raw_batch["ctrl_cell_emb"].to(device, non_blocking=True)
                pert = raw_batch["pert_emb"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                delta = model(ctrl, pert)
                if delta.shape != target.shape or tuple(delta.shape[1:]) != (768,):
                    raise AssertionError("Delta_pred/Delta_condition is not [B,768]")
                loss = F.mse_loss(delta, target)
                if not torch.isfinite(loss):
                    all_training_losses_finite = False
                    raise AssertionError("B2-v2 training MSE became non-finite")
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        all_gradients_finite = False
                        raise AssertionError("B2-v2 gradient became non-finite")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), GRADIENT_CLIP_NORM, error_if_nonfinite=True
                )
                optimizer.step()
                report_loss = loss.detach().clone()
                dist.all_reduce(report_loss, op=dist.ReduceOp.SUM)
                report_loss /= world_size
                epoch_losses.append(float(report_loss))
                global_step += 1
            if len(epoch_losses) != expected_steps:
                raise AssertionError(f"Epoch {epoch} did not contain {expected_steps} steps")
            train_mse = float(np.mean(epoch_losses))

            dist.barrier()
            validation = (
                validate_model(raw_model.b2, val_dataset, device, validation_metric)
                if rank == 0
                else None
            )
            if rank == 0 and (
                not validation["finite"]
                or validation["condition_count"] != 5_657
                or validation["dataset_epoch"] != 0
                or validation["shuffle"]
                or validation["drop_last"]
            ):
                raise AssertionError(f"B2-v2 fixed validation failed: {validation}")
            val_values = torch.tensor(
                [
                    validation["condition_energy_mean"] if rank == 0 else 0.0,
                    validation["edge_level_mean_energy"] if rank == 0 else 0.0,
                ],
                device=device,
                dtype=torch.float64,
            )
            dist.broadcast(val_values, src=0)
            val_condition = float(val_values[0])
            val_edge = float(val_values[1])
            if not math.isfinite(val_condition) or not math.isfinite(val_edge):
                raise AssertionError("B2-v2 validation Energy became non-finite")
            improved = val_edge < best_metric
            if improved:
                best_metric = val_edge
                best_epoch = epoch
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1
            early_stopped = no_improvement_epochs >= PATIENCE

            if rank == 0:
                assert validation is not None
                validation.pop("condition_energy_values", None)
                history.append(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_mse_mean": train_mse,
                        "val_condition_energy_mean": val_condition,
                        "val_edge_energy_mean": val_edge,
                        "best_so_far": best_metric,
                        "strict_improvement": improved,
                        "no_improvement_epochs": no_improvement_epochs,
                    }
                )
                elapsed_so_far = prior_elapsed + time.perf_counter() - session_started
                checkpoint_args = {
                    "model": raw_model,
                    "optimizer": optimizer,
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_mse": train_mse,
                    "current_val_edge_energy": val_edge,
                    "best_val_edge_energy": best_metric,
                    "best_epoch": best_epoch,
                    "no_improvement_epochs": no_improvement_epochs,
                    "history": history,
                    "protocol_sha256": protocol_sha,
                    "elapsed_seconds_so_far": elapsed_so_far,
                }
                if improved:
                    save_checkpoint(BEST_CHECKPOINT, kind="best", **checkpoint_args)
                save_checkpoint(LAST_CHECKPOINT, kind="last", **checkpoint_args)
                atomic_write_history(history)
                print(
                    f"B2-v2 epoch={epoch} step={global_step} train_mse={train_mse:.10f} "
                    f"val_condition_energy={val_condition:.8f} val_edge_energy={val_edge:.8f} "
                    f"best_epoch={best_epoch} no_improvement={no_improvement_epochs}",
                    flush=True,
                )
            dist.barrier()
            if early_stopped:
                break

        sync_tensor = torch.zeros(3, device=device, dtype=torch.float64)
        for parameter in raw_model.parameters():
            values = parameter.detach().double()
            sync_tensor[0] += values.sum()
            sync_tensor[1] += values.square().sum()
            sync_tensor[2] += values.abs().sum()
        gathered_sync = [torch.empty_like(sync_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_sync, sync_tensor)
        sync_error = max(
            float((value - gathered_sync[0]).abs().max()) for value in gathered_sync
        )
        if sync_error != 0.0:
            raise AssertionError(f"B2-v2 DDP parameters are not synchronized: {sync_error}")
        torch.cuda.synchronize(device)
        session_elapsed = time.perf_counter() - session_started
        local_runtime = {
            "rank": rank,
            "local_rank": local_rank,
            "gpu_name": torch.cuda.get_device_name(device),
            "session_elapsed_seconds": session_elapsed,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        }
        gathered_runtime: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(gathered_runtime, local_runtime)
        runtimes = [item for item in gathered_runtime if item is not None]
        if len(runtimes) != 2 or {item["local_rank"] for item in runtimes} != {0, 1}:
            raise AssertionError("Both B2-v2 DDP ranks did not complete")

        if rank != 0:
            dist.barrier()
            return None

        historical_best = min(row["val_edge_energy_mean"] for row in history)
        historical_best_epoch = min(
            row["epoch"] for row in history if row["val_edge_energy_mean"] == historical_best
        )
        if best_metric != historical_best or best_epoch != historical_best_epoch:
            raise AssertionError("Best B2-v2 checkpoint is not the strict historical minimum")
        best_payload = validate_checkpoint(
            BEST_CHECKPOINT,
            kind="best",
            epoch=best_epoch,
            global_step=int(history[best_epoch]["global_step"]),
            protocol_sha256=protocol_sha,
        )
        validate_checkpoint(
            LAST_CHECKPOINT,
            kind="last",
            epoch=int(history[-1]["epoch"]),
            global_step=global_step,
            protocol_sha256=protocol_sha,
        )
        if best_payload["best_val_edge_energy"] != best_metric:
            raise AssertionError("Best checkpoint Energy value is stale")
        if v1_artifacts() != protocol["b2_v1_preserved_artifacts"]:
            raise AssertionError("A B2-v1 artifact changed during B2-v2 training")

        epoch0_energy = float(history[0]["val_edge_energy_mean"])
        total_elapsed = prior_elapsed + max(
            float(item["session_elapsed_seconds"]) for item in runtimes
        )
        result = {
            "created_at_utc": utc_now(),
            "status": "pass",
            "scope": "formal B2-v2 train and val checkpoint selection; no test or STATE/ST",
            "epochs_completed": len(history),
            "optimizer_steps": global_step,
            "early_stopped": early_stopped,
            "stop_reason": "early_stopping" if early_stopped else "max_epochs",
            "best_epoch": best_epoch,
            "best_val_edge_energy": best_metric,
            "epoch_0_val_edge_energy": epoch0_energy,
            "best_relative_energy_improvement_from_epoch_0": (
                epoch0_energy - best_metric
            )
            / epoch0_energy,
            "first_train_mse": float(history[0]["train_mse_mean"]),
            "final_train_mse": float(history[-1]["train_mse_mean"]),
            "protocol": {"path": relative(PROTOCOL_PATH), "sha256": protocol_sha},
            "checkpoint": {
                "best": artifact_record(BEST_CHECKPOINT),
                "last": artifact_record(LAST_CHECKPOINT),
            },
            "history": artifact_record(HISTORY_PATH),
            "runtime": {"total_elapsed_seconds": total_elapsed, "rank": runtimes},
            "training": {
                **training_shape(),
                "world_size": 2,
                "batch_size_per_gpu": 128,
                "global_batch_size": 256,
                "loss": "MSE only",
                "all_losses_finite": all_training_losses_finite,
                "all_gradients_finite": all_gradients_finite,
                "condition_target_alignment": alignment,
                "target_recomputed": False,
                "raw_latent_transforms": list(train_dataset.embedding_transforms),
                "dataset_set_epoch_each_epoch": True,
                "sampler_set_epoch_each_epoch": True,
            },
            "validation": {
                "conditions": len(val_dataset),
                "dataset_epoch": 0,
                "shuffle": False,
                "drop_last": False,
                "checkpoint_selection": "strict minimum val edge-level mean Energy",
            },
            "test_dataset_constructed": False,
            "ddp_parameter_moment_max_abs_error": sync_error,
            "b2_v1_artifacts_preserved": True,
            "resumed_from_b2_v2_last_checkpoint": resume_last,
            "blockers": [],
            "warnings": [
                "drop_last=True consumes 45,568 of 45,652 train conditions per epoch; sampler epoch changes which 84 are dropped."
            ],
        }
        atomic_write_json(RESULT_PATH, result)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        dist.barrier()
        return result
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="Freeze and audit the full-run contract without training")
    run_parser = subparsers.add_parser("run", help="Run formal two-GPU B2-v2 training")
    run_parser.add_argument(
        "--resume-last",
        action="store_true",
        help="Resume only from B2-v2's last complete epoch checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError("Run prepare with one ordinary Python process")
        print(json.dumps(prepare(), indent=2, ensure_ascii=False))
        return
    run(resume_last=args.resume_last)


if __name__ == "__main__":
    main()
