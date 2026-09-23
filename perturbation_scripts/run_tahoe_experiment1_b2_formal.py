#!/usr/bin/env python3
"""Run the frozen Experiment 1 B2 full training with two-GPU DDP."""

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
from geomloss import SamplesLoss
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from train_tahoe_experiment1_b2 import (
    B2PooledMLP,
    BATCH_SIZE_PER_GPU,
    EARLY_STOPPING_PATIENCE,
    ENERGY_BLUR,
    EXPECTED_SPLIT_COUNTS,
    FORMAL_GLOBAL_BATCH_SIZE,
    FORMAL_NUM_GPUS,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    MAX_EPOCHS,
    PROTOCOL_PATH,
    RESULTS,
    SCRIPT_PATH as CORE_SCRIPT_PATH,
    TRAINING_SEED,
    WEIGHT_DECAY,
    atomic_write_json,
    atomic_write_text,
    ddp_parameter_sync_error,
    ensure_protocol,
    initial_identity_audit,
    make_dataloader,
    make_split_dataset,
    move_batch,
    seed_everything,
    sha256_file,
    train_redraw_audit,
    utc_now,
    validate_model,
)


RUNNER_PATH = Path(__file__).resolve()
CHECKPOINT_DIR = RESULTS / "tahoe_experiment1_b2_formal_checkpoints"
BEST_CHECKPOINT = CHECKPOINT_DIR / "b2_best.pt"
LAST_CHECKPOINT = CHECKPOINT_DIR / "b2_last.pt"
HISTORY_PATH = RESULTS / "tahoe_experiment1_b2_training_history.csv"
RESULT_PATH = RESULTS / "tahoe_experiment1_b2_formal_training_result.json"
HANDOFF_PATH = RESULTS / "tahoe_experiment1_b2_formal_training_handoff.md"
READINESS_PATH = RESULTS / "tahoe_experiment1_b2_formal_training_readiness.json"
CONSOLE_LOG_PATH = RESULTS / "tahoe_experiment1_b2_formal_training_console.log"
FORMAL_TASK_SHA256 = "10024db5cc04aae3d54fcb281fec3d589f8808cd720cebec89037d925ba8d87e"


def relative(path: Path) -> str:
    return str(path.relative_to(RESULTS.parent)).replace("\\", "/")


def atomic_write_history(history: list[dict[str, Any]]) -> None:
    temporary = HISTORY_PATH.with_name(HISTORY_PATH.name + ".tmp")
    pd.DataFrame.from_records(history).to_csv(
        temporary, index=False, encoding="utf-8-sig", lineterminator="\n"
    )
    temporary.replace(HISTORY_PATH)


def assert_protocol(protocol: dict[str, Any]) -> None:
    expected = {
        "train_conditions": EXPECTED_SPLIT_COUNTS["train"],
        "val_conditions": EXPECTED_SPLIT_COUNTS["val"],
        "set_size": 256,
        "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
        "num_gpus": FORMAL_NUM_GPUS,
        "global_batch_size": FORMAL_GLOBAL_BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": EARLY_STOPPING_PATIENCE,
        "training_seed": TRAINING_SEED,
    }
    observed = {
        "train_conditions": protocol["data"]["train_conditions"],
        "val_conditions": EXPECTED_SPLIT_COUNTS["val"],
        "set_size": protocol["data"]["set_size"],
        "batch_size_per_gpu": protocol["optimizer"]["batch_size_per_gpu"],
        "num_gpus": protocol["optimizer"]["num_gpus"],
        "global_batch_size": protocol["optimizer"]["global_batch_size"],
        "max_epochs": protocol["optimizer"]["max_epochs"],
        "patience": protocol["optimizer"]["early_stopping_patience"],
        "training_seed": protocol["optimizer"]["training_seed"],
    }
    if observed != expected:
        raise AssertionError(f"Frozen B2 protocol values changed: {observed}")
    if protocol["objective"]["arguments"] != {"loss": "energy", "blur": 0.05}:
        raise AssertionError("Frozen Energy objective changed")
    if protocol["optimizer"]["scheduler"] is not None:
        raise AssertionError("Formal B2 must not use a scheduler")
    if protocol["implementation"]["sha256"] != sha256_file(CORE_SCRIPT_PATH):
        raise AssertionError("Frozen B2 implementation SHA256 changed")


def make_datasets() -> tuple[Any, Any]:
    train_dataset = make_split_dataset("train", 0)
    val_dataset = make_split_dataset("val", 0)
    if len(train_dataset) != 45_652 or len(val_dataset) != 5_657:
        raise AssertionError("Formal B2 train/val condition counts changed")
    if set(train_dataset.conditions["edge_id"]) & set(val_dataset.conditions["edge_id"]):
        raise AssertionError("Frozen train and val edges overlap")
    if train_dataset.embedding_transforms or val_dataset.embedding_transforms:
        raise AssertionError("Formal B2 must consume untransformed raw cached latent")
    exact_capacity = int(
        train_dataset.conditions["treated_cached_cell_count"].eq(256).sum()
    )
    if exact_capacity != 12:
        raise AssertionError(f"Expected 12 exact-capacity train conditions, found {exact_capacity}")
    return train_dataset, val_dataset


def training_shape(train_conditions: int, val_conditions: int) -> dict[str, int]:
    samples_per_rank = train_conditions // FORMAL_NUM_GPUS
    steps_per_epoch = samples_per_rank // BATCH_SIZE_PER_GPU
    consumed = steps_per_epoch * FORMAL_GLOBAL_BATCH_SIZE
    return {
        "samples_per_rank_before_dataloader_drop_last": samples_per_rank,
        "optimizer_steps_per_epoch": steps_per_epoch,
        "global_train_sets_consumed_per_epoch": consumed,
        "global_train_sets_dropped_per_epoch": train_conditions - consumed,
        "validation_batches_per_epoch_on_rank0": math.ceil(
            val_conditions / BATCH_SIZE_PER_GPU
        ),
        "maximum_optimizer_steps": steps_per_epoch * MAX_EPOCHS,
    }


def prepare() -> dict[str, Any]:
    protocol, protocol_sha = ensure_protocol(rewrite=False)
    assert_protocol(protocol)
    train_dataset, val_dataset = make_datasets()
    formal_outputs = [BEST_CHECKPOINT, LAST_CHECKPOINT, HISTORY_PATH, RESULT_PATH, HANDOFF_PATH]
    existing = [relative(path) for path in formal_outputs if path.exists()]
    payload = {
        "created_at_utc": utc_now(),
        "status": "pass" if not existing else "fail",
        "scope": "formal B2 training readiness only; no optimizer step or test Dataset",
        "protocol": {"path": relative(PROTOCOL_PATH), "sha256": protocol_sha},
        "frozen_implementation": {
            "path": relative(CORE_SCRIPT_PATH),
            "sha256": sha256_file(CORE_SCRIPT_PATH),
        },
        "formal_runner": {
            "path": relative(RUNNER_PATH),
            "sha256": sha256_file(RUNNER_PATH),
        },
        "formal_request_sha256": FORMAL_TASK_SHA256,
        "data": {
            "train_conditions": len(train_dataset),
            "val_conditions": len(val_dataset),
            "test_dataset_constructed": False,
            "train_treated_conditions_with_exactly_256_cached_cells": 12,
            "latent_transforms": list(train_dataset.embedding_transforms),
            "dose_fit_scope": train_dataset.dose_statistics["fit_scope"],
        },
        "schedule": training_shape(len(train_dataset), len(val_dataset)),
        "formal_outputs_already_present": existing,
        "fresh_run_ready": not existing,
    }
    atomic_write_json(READINESS_PATH, payload)
    return payload


def formal_artifact_guard(*, resume_last: bool) -> None:
    outputs = [BEST_CHECKPOINT, LAST_CHECKPOINT, HISTORY_PATH, RESULT_PATH, HANDOFF_PATH]
    if resume_last:
        if not LAST_CHECKPOINT.exists() or not BEST_CHECKPOINT.exists():
            raise FileNotFoundError("--resume-last requires both formal last and best checkpoints")
        if RESULT_PATH.exists():
            result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
            if result.get("status") == "pass":
                raise RuntimeError("Formal B2 training is already complete")
        return
    existing = [relative(path) for path in outputs if path.exists()]
    if existing or (CHECKPOINT_DIR.exists() and any(CHECKPOINT_DIR.iterdir())):
        raise FileExistsError(
            "Fresh formal run refused because formal artifacts already exist: "
            + ", ".join(existing or [relative(CHECKPOINT_DIR)])
        )


def save_checkpoint(
    path: Path,
    model: B2PooledMLP,
    optimizer: torch.optim.Optimizer,
    *,
    checkpoint_kind: str,
    epoch: int,
    global_step: int,
    current_val_edge_energy: float,
    best_val_edge_energy: float,
    best_epoch: int,
    no_improvement_epochs: int,
    history: list[dict[str, Any]],
    initialization_audit: dict[str, Any],
    protocol_sha256: str,
    core_implementation_sha256: str,
    runner_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "format": "tahoe_experiment1_b2_formal_v1",
            "checkpoint_kind": checkpoint_kind,
            "fresh_initialization": True,
            "smoke_checkpoint_loaded": False,
            "epoch": epoch,
            "global_step": global_step,
            "current_val_edge_level_energy": current_val_edge_energy,
            "best_val_edge_level_energy": best_val_edge_energy,
            "best_epoch": best_epoch,
            "no_improvement_epochs": no_improvement_epochs,
            "training_protocol_sha256": protocol_sha256,
            "implementation_sha256": core_implementation_sha256,
            "formal_runner_sha256": runner_sha256,
            "training_seed": TRAINING_SEED,
            "model_state": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "initialization_audit": initialization_audit,
        },
        temporary,
    )
    temporary.replace(path)


def load_last_checkpoint(
    model: B2PooledMLP,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    protocol_sha256: str,
    core_implementation_sha256: str,
    runner_sha256: str,
) -> dict[str, Any]:
    payload = torch.load(LAST_CHECKPOINT, map_location=device, weights_only=False)
    required = {
        "format": "tahoe_experiment1_b2_formal_v1",
        "checkpoint_kind": "last",
        "fresh_initialization": True,
        "smoke_checkpoint_loaded": False,
        "training_protocol_sha256": protocol_sha256,
        "implementation_sha256": core_implementation_sha256,
        "formal_runner_sha256": runner_sha256,
        "training_seed": TRAINING_SEED,
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise AssertionError(f"Formal last checkpoint mismatch for {key}")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def result_payload(
    *,
    history: list[dict[str, Any]],
    early_stopped: bool,
    global_step: int,
    best_epoch: int,
    best_metric: float,
    protocol_sha256: str,
    core_sha256: str,
    runner_sha256: str,
    sync_error: float,
    runtimes: list[dict[str, Any]],
    elapsed_seconds: float,
    resumed_from_last: bool,
) -> dict[str, Any]:
    historical_best = min(row["val_edge_energy_mean"] for row in history)
    historical_best_epoch = min(
        row["epoch"]
        for row in history
        if row["val_edge_energy_mean"] == historical_best
    )
    if best_metric != historical_best or best_epoch != historical_best_epoch:
        raise AssertionError("Best checkpoint selection differs from strict historical minimum")
    if not BEST_CHECKPOINT.exists() or not LAST_CHECKPOINT.exists():
        raise AssertionError("Formal best/last checkpoint missing")
    best_checkpoint = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)
    last_checkpoint = torch.load(LAST_CHECKPOINT, map_location="cpu", weights_only=False)
    required_best = {
        "checkpoint_kind": "best",
        "epoch": best_epoch,
        "global_step": int(history[best_epoch]["global_step"]),
        "best_val_edge_level_energy": best_metric,
        "training_protocol_sha256": protocol_sha256,
        "implementation_sha256": core_sha256,
        "formal_runner_sha256": runner_sha256,
        "training_seed": TRAINING_SEED,
        "fresh_initialization": True,
        "smoke_checkpoint_loaded": False,
    }
    required_last = {
        "checkpoint_kind": "last",
        "epoch": int(history[-1]["epoch"]),
        "global_step": global_step,
        "training_protocol_sha256": protocol_sha256,
        "implementation_sha256": core_sha256,
        "formal_runner_sha256": runner_sha256,
        "training_seed": TRAINING_SEED,
    }
    for name, checkpoint, required in (
        ("best", best_checkpoint, required_best),
        ("last", last_checkpoint, required_last),
    ):
        if "model_state" not in checkpoint or "optimizer_state" not in checkpoint:
            raise AssertionError(f"Formal {name} checkpoint lacks model/optimizer state")
        for key, value in required.items():
            if checkpoint.get(key) != value:
                raise AssertionError(f"Formal {name} checkpoint mismatch for {key}")
    epoch0_val = float(history[0]["val_edge_energy_mean"])
    checks = {
        "train_split_exactly_45652": True,
        "val_split_exactly_5657": True,
        "test_dataset_constructed": False,
        "world_size_2": True,
        "batch_64_per_gpu_global_128": True,
        "raw_signed_latent_no_transforms": True,
        "perturbation_frozen_380d": True,
        "extra_context_inputs": False,
        "energy_finite": True,
        "gradients_finite": True,
        "ddp_parameters_synchronized": sync_error == 0.0,
        "dataset_set_epoch_each_epoch": True,
        "sampler_set_epoch_each_epoch": True,
        "fixed_val_dataset_epoch_0": True,
        "val_shuffle_false": True,
        "val_drop_last_false": True,
        "best_is_historical_strict_minimum": True,
        "smoke_checkpoint_loaded": False,
    }
    if any(value is not expected for value, expected in [
        (checks["test_dataset_constructed"], False),
        (checks["extra_context_inputs"], False),
    ]) or any(
        not value
        for key, value in checks.items()
        if key not in {"test_dataset_constructed", "extra_context_inputs"}
    ):
        raise AssertionError(f"Formal B2 final checks failed: {checks}")
    return {
        "created_at_utc": utc_now(),
        "status": "pass",
        "scope": "B2 formal full train plus fixed val checkpoint selection; no test evaluation",
        "fresh_initialization": True,
        "smoke_checkpoint_loaded": False,
        "resumed_from_formal_last_checkpoint": resumed_from_last,
        "epochs_completed": len(history),
        "optimizer_steps": global_step,
        "early_stopped": early_stopped,
        "stop_reason": "early_stopping" if early_stopped else "max_epochs",
        "best_epoch": best_epoch,
        "best_val_edge_level_energy": best_metric,
        "epoch_0_val_edge_level_energy": epoch0_val,
        "best_relative_improvement_from_epoch_0": (epoch0_val - best_metric) / epoch0_val,
        "first_train_energy_mean": float(history[0]["train_energy_mean"]),
        "final_train_energy_mean": float(history[-1]["train_energy_mean"]),
        "protocol": {"path": relative(PROTOCOL_PATH), "sha256": protocol_sha256},
        "implementation": {"path": relative(CORE_SCRIPT_PATH), "sha256": core_sha256},
        "formal_runner": {"path": relative(RUNNER_PATH), "sha256": runner_sha256},
        "training": {
            "train_conditions": 45_652,
            "val_conditions": 5_657,
            "test_dataset_constructed": False,
            "world_size": 2,
            "batch_size_per_gpu": 64,
            "global_batch_size": 128,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "scheduler": None,
            "seed": TRAINING_SEED,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "steps_per_epoch": training_shape(45_652, 5_657)["optimizer_steps_per_epoch"],
        },
        "validation": {
            "dataset_epoch": 0,
            "shuffle": False,
            "drop_last": False,
            "checkpoint_metric": "val edge-level mean Energy",
            "plate6_plate14_rule": "only dose_uM == 5.0 high-dose groups are paired when both plate6 and plate14 exist",
        },
        "capacity_caveat": {
            "train_treated_conditions_with_exactly_256_cached_cells": 12,
            "fallback_applied": False,
        },
        "checkpoints": {
            "best": {
                "path": relative(BEST_CHECKPOINT),
                "sha256": sha256_file(BEST_CHECKPOINT),
                "size_bytes": BEST_CHECKPOINT.stat().st_size,
            },
            "last": {
                "path": relative(LAST_CHECKPOINT),
                "sha256": sha256_file(LAST_CHECKPOINT),
                "size_bytes": LAST_CHECKPOINT.stat().st_size,
            },
        },
        "history": {"path": relative(HISTORY_PATH), "rows": len(history)},
        "ddp_parameter_moment_max_abs_error": sync_error,
        "gpu_participation": runtimes,
        "elapsed_seconds": elapsed_seconds,
        "checks": checks,
        "warnings": [
            "12 train treated conditions have exactly 256 cached cells; their membership cannot redraw.",
            "drop_last=True consumes 45,568 of 45,652 train conditions per epoch; the dropped 84 vary with sampler epoch.",
        ],
    }


def write_handoff(payload: dict[str, Any]) -> None:
    text = f"""# Tahoe Experiment 1 B2 formal training handoff

- Status: **{payload['status'].upper()}**
- Scope: formal B2 train + fixed val only; no formal test evaluation and no STATE/ST.
- Epochs / optimizer steps: **{payload['epochs_completed']} / {payload['optimizer_steps']}**
- Stop: **{payload['stop_reason']}**
- Best epoch / val edge Energy: **{payload['best_epoch']} / {payload['best_val_edge_level_energy']:.10f}**
- Epoch 0 val edge Energy: **{payload['epoch_0_val_edge_level_energy']:.10f}**
- Best checkpoint: `{payload['checkpoints']['best']['path']}`
- Best checkpoint SHA-256: `{payload['checkpoints']['best']['sha256']}`
- Training history: `{payload['history']['path']}`

Both DDP ranks completed the same optimizer-step count and the final parameter-moment synchronization error was `{payload['ddp_parameter_moment_max_abs_error']}`.
"""
    atomic_write_text(HANDOFF_PATH, text)


def run(*, resume_last: bool) -> dict[str, Any] | None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != FORMAL_NUM_GPUS:
        raise RuntimeError("Formal B2 requires exactly two torchrun processes")
    if os.environ.get("NCCL_CUMEM_ENABLE") != "0" or os.environ.get(
        "NCCL_CUMEM_HOST_ENABLE"
    ) != "0":
        raise RuntimeError("Set both NCCL CUMEM workaround variables to 0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != FORMAL_NUM_GPUS:
        raise RuntimeError("Formal B2 requires exactly two visible CUDA GPUs")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    try:
        protocol, protocol_sha = ensure_protocol(rewrite=False)
        assert_protocol(protocol)
        core_sha = sha256_file(CORE_SCRIPT_PATH)
        runner_sha = sha256_file(RUNNER_PATH)
        if rank == 0:
            formal_artifact_guard(resume_last=resume_last)
        guard = torch.tensor([1], device=device)
        dist.broadcast(guard, src=0)

        seed_everything(TRAINING_SEED)
        train_dataset, val_dataset = make_datasets()
        redraw = train_redraw_audit(train_dataset)
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
        expected_steps = training_shape(len(train_dataset), len(val_dataset))[
            "optimizer_steps_per_epoch"
        ]
        if len(train_loader) != expected_steps or not train_loader.drop_last:
            raise AssertionError("Formal train DataLoader shape/drop_last changed")

        raw_model = B2PooledMLP().to(device)
        metric = SamplesLoss(loss="energy", blur=ENERGY_BLUR)
        optimizer = torch.optim.AdamW(
            raw_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        if resume_last:
            checkpoint = load_last_checkpoint(
                raw_model,
                optimizer,
                device,
                protocol_sha256=protocol_sha,
                core_implementation_sha256=core_sha,
                runner_sha256=runner_sha,
            )
            history = list(checkpoint["history"])
            initialization = dict(checkpoint["initialization_audit"])
            start_epoch = int(checkpoint["epoch"]) + 1
            global_step = int(checkpoint["global_step"])
            best_metric = float(checkpoint["best_val_edge_level_energy"])
            best_epoch = int(checkpoint["best_epoch"])
            no_improvement_epochs = int(checkpoint["no_improvement_epochs"])
            if start_epoch >= MAX_EPOCHS or no_improvement_epochs >= EARLY_STOPPING_PATIENCE:
                raise RuntimeError("Formal last checkpoint is already at a terminal epoch")
        else:
            initialization = initial_identity_audit(raw_model, train_dataset, device, metric)
            history: list[dict[str, Any]] = []
            start_epoch = 0
            global_step = 0
            best_metric = float("inf")
            best_epoch = -1
            no_improvement_epochs = 0

        model = DistributedDataParallel(raw_model, device_ids=[local_rank])
        if resume_last:
            # Optimizer parameter identities remain those of raw_model after DDP wrapping.
            pass
        early_stopped = False
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        local_gradient_min = float("inf")
        local_gradient_max = 0.0

        for epoch in range(start_epoch, MAX_EPOCHS):
            train_dataset.set_epoch(epoch)
            sampler.set_epoch(epoch)
            if train_dataset.epoch != epoch or sampler.epoch != epoch:
                raise AssertionError("Dataset or DistributedSampler epoch was not updated")
            model.train()
            epoch_losses: list[float] = []
            for raw_batch in train_loader:
                if set(raw_batch["split"]) != {"train"}:
                    raise AssertionError("Non-train condition entered formal B2 fitting")
                batch = move_batch(raw_batch, device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch)
                loss = metric(prediction, batch["pert_cell_emb"]).mean()
                if not torch.isfinite(loss):
                    raise AssertionError("Formal B2 Energy became non-finite")
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise AssertionError("Formal B2 gradient became non-finite")
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), GRADIENT_CLIP_NORM, error_if_nonfinite=True
                )
                local_gradient_min = min(local_gradient_min, float(gradient_norm))
                local_gradient_max = max(local_gradient_max, float(gradient_norm))
                optimizer.step()
                report_loss = loss.detach().clone()
                dist.all_reduce(report_loss, op=dist.ReduceOp.SUM)
                report_loss /= world_size
                epoch_losses.append(float(report_loss))
                global_step += 1
            if len(epoch_losses) != expected_steps:
                raise AssertionError(
                    f"Expected {expected_steps} optimizer steps in epoch {epoch}, got {len(epoch_losses)}"
                )

            dist.barrier()
            validation = validate_model(raw_model, val_dataset, device, metric) if rank == 0 else None
            if rank == 0 and (
                not validation["finite"]
                or validation["condition_count"] != 5_657
                or validation["dataset_epoch"] != 0
                or validation["shuffle"]
                or validation["drop_last"]
            ):
                raise AssertionError(f"Formal fixed-validation contract failed: {validation}")
            validation_metrics = torch.tensor(
                [
                    validation["edge_level_mean_energy"] if rank == 0 else 0.0,
                    validation["condition_energy_mean"] if rank == 0 else 0.0,
                ],
                device=device,
                dtype=torch.float64,
            )
            dist.broadcast(validation_metrics, src=0)
            val_edge = float(validation_metrics[0])
            val_condition = float(validation_metrics[1])
            if not math.isfinite(val_edge) or not math.isfinite(val_condition):
                raise AssertionError("Formal validation Energy became non-finite")
            improved = val_edge < best_metric
            if improved:
                best_metric = val_edge
                best_epoch = epoch
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1
            early_stopped = no_improvement_epochs >= EARLY_STOPPING_PATIENCE

            if rank == 0:
                assert validation is not None
                validation.pop("condition_energy_values", None)
                history.append(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "optimizer_steps": len(epoch_losses),
                        "train_energy_mean": float(np.mean(epoch_losses)),
                        "val_condition_energy_mean": val_condition,
                        "val_edge_energy_mean": val_edge,
                        "best_so_far": best_metric,
                        "strict_improvement": improved,
                        "no_improvement_epochs": no_improvement_epochs,
                    }
                )
                checkpoint_kwargs = {
                    "model": raw_model,
                    "optimizer": optimizer,
                    "epoch": epoch,
                    "global_step": global_step,
                    "current_val_edge_energy": val_edge,
                    "best_val_edge_energy": best_metric,
                    "best_epoch": best_epoch,
                    "no_improvement_epochs": no_improvement_epochs,
                    "history": history,
                    "initialization_audit": initialization,
                    "protocol_sha256": protocol_sha,
                    "core_implementation_sha256": core_sha,
                    "runner_sha256": runner_sha,
                }
                if improved:
                    save_checkpoint(BEST_CHECKPOINT, checkpoint_kind="best", **checkpoint_kwargs)
                save_checkpoint(LAST_CHECKPOINT, checkpoint_kind="last", **checkpoint_kwargs)
                atomic_write_history(history)
                print(
                    f"B2 formal epoch={epoch} step={global_step} "
                    f"train_energy={np.mean(epoch_losses):.8f} "
                    f"val_condition_energy={val_condition:.8f} "
                    f"val_edge_energy={val_edge:.8f} best_epoch={best_epoch} "
                    f"no_improvement={no_improvement_epochs}",
                    flush=True,
                )
            dist.barrier()
            if early_stopped:
                break

        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        sync_error = ddp_parameter_sync_error(raw_model, world_size)
        if sync_error != 0.0:
            raise AssertionError(f"DDP parameters are not synchronized: {sync_error}")
        local_runtime = {
            "rank": rank,
            "local_rank": local_rank,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "optimizer_steps": global_step,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            "gradient_norm_min": local_gradient_min,
            "gradient_norm_max": local_gradient_max,
        }
        gathered: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(gathered, local_runtime)
        runtimes = [item for item in gathered if item is not None]
        if len(runtimes) != world_size or {item["local_rank"] for item in runtimes} != {0, 1}:
            raise AssertionError("Both formal DDP ranks did not report completion")

        if rank != 0:
            dist.barrier()
            return None
        payload = result_payload(
            history=history,
            early_stopped=early_stopped,
            global_step=global_step,
            best_epoch=best_epoch,
            best_metric=best_metric,
            protocol_sha256=protocol_sha,
            core_sha256=core_sha,
            runner_sha256=runner_sha,
            sync_error=sync_error,
            runtimes=runtimes,
            elapsed_seconds=elapsed,
            resumed_from_last=resume_last,
        )
        payload["train_sampling"] = redraw
        atomic_write_json(RESULT_PATH, payload)
        write_handoff(payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
        dist.barrier()
        return payload
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="Validate the frozen full-run contract without CUDA training")
    run_parser = subparsers.add_parser("run", help="Run the exact frozen two-GPU formal training")
    run_parser.add_argument(
        "--resume-last",
        action="store_true",
        help="Resume only from this formal run's last complete epoch checkpoint.",
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
