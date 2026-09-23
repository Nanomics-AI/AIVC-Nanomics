#!/usr/bin/env python3
"""Smoke-test B2 with the frozen train condition-shift MSE target."""

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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from train_tahoe_experiment1_b2 import (
    B2PooledMLP,
    GRADIENT_CLIP_NORM,
    LATENT_DIM,
    LEARNING_RATE,
    PERT_DIM,
    RESULTS,
    SCRIPT_PATH as B2_V1_SCRIPT_PATH,
    TRAINING_SEED,
    WEIGHT_DECAY,
    atomic_write_json,
    ddp_parameter_sync_error,
    make_dataloader,
    make_split_dataset,
    parameter_update_audit,
    seed_everything,
    sha256_file,
    utc_now,
)


SCRIPT_PATH = Path(__file__).resolve()
SHIFT_PATH = RESULTS / "tahoe_experiment1_b1_train_condition_shifts.npy"
SHIFT_METADATA_PATH = RESULTS / "tahoe_experiment1_b1_train_condition_shifts_metadata.csv"
B1_BUILD_AUDIT_PATH = RESULTS / "tahoe_experiment1_b1_mean_delta_build_audit.json"
V1_HISTORY_PATH = RESULTS / "tahoe_experiment1_b2_training_history.csv"
V1_BEST_PATH = RESULTS / "tahoe_experiment1_b2_formal_checkpoints" / "b2_best.pt"
V1_LAST_PATH = RESULTS / "tahoe_experiment1_b2_formal_checkpoints" / "b2_last.pt"
OUTPUT_PATH = RESULTS / "tahoe_experiment1_b2_v2_smoke.json"
EXPECTED_TRAIN_CONDITIONS = 45_652
DEFAULT_SMOKE_CONDITIONS = 512
DEFAULT_BATCH_SIZE_PER_GPU = 128
DEFAULT_STEPS = 20
WORLD_SIZE = 2


class DeltaPredictor(nn.Module):
    """Expose the existing B2 MLP delta as the DDP forward output."""

    def __init__(self) -> None:
        super().__init__()
        self.b2 = B2PooledMLP()

    def forward(
        self, ctrl_cell_emb: torch.Tensor, pert_emb: torch.Tensor
    ) -> torch.Tensor:
        return self.b2.predict_delta(ctrl_cell_emb, pert_emb)


def load_targets(dataset: Any) -> tuple[np.memmap, pd.DataFrame, dict[str, int], dict[str, Any]]:
    audit = json.loads(B1_BUILD_AUDIT_PATH.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "pass"
        or audit.get("counts", {}).get("train_conditions") != EXPECTED_TRAIN_CONDITIONS
        or audit.get("checks", {}).get("train_conditions_only") != "pass"
    ):
        raise AssertionError("B1 train condition-shift audit is not a valid PASS artifact")
    if sha256_file(SHIFT_PATH) != audit["outputs"]["condition_shift"]["sha256"]:
        raise AssertionError("B1 train condition-shift NPY SHA-256 changed")
    if sha256_file(SHIFT_METADATA_PATH) != audit["outputs"]["condition_metadata"]["sha256"]:
        raise AssertionError("B1 train condition-shift metadata SHA-256 changed")

    shifts = np.load(SHIFT_PATH, mmap_mode="r")
    metadata = pd.read_csv(
        SHIFT_METADATA_PATH, keep_default_na=False, encoding="utf-8-sig"
    )
    if (
        not isinstance(shifts, np.memmap)
        or shifts.shape != (EXPECTED_TRAIN_CONDITIONS, LATENT_DIM)
        or shifts.dtype != np.float32
        or not np.isfinite(shifts).all()
    ):
        raise AssertionError("B1 condition-shift target array is invalid")
    if (
        len(metadata) != EXPECTED_TRAIN_CONDITIONS
        or metadata["condition_id"].duplicated().any()
        or not np.array_equal(
            metadata["condition_shift_row"].to_numpy(np.int64),
            np.arange(EXPECTED_TRAIN_CONDITIONS, dtype=np.int64),
        )
    ):
        raise AssertionError("B1 condition-shift metadata row mapping is invalid")

    train = (
        dataset.all_conditions.loc[dataset.all_conditions["split"].eq("train")]
        .sort_values("cache_condition_index", kind="stable")
        .reset_index(drop=True)
    )
    comparisons = {
        "condition_id": train["pair_id"].astype(str),
        "cell_line_id": train["cell_line_id"].astype(str),
        "drug": train["drug"].astype(str),
        "control_pool_id": train["control_pool_id"].astype(str),
        "dose_uM": train["dose_uM"].astype(np.float64),
        "treated_cached_cell_count": train["treated_cached_cell_count"].astype(np.int64),
        "control_cached_cell_count": train["control_cached_cell_count"].astype(np.int64),
    }
    for column, expected in comparisons.items():
        observed = metadata[column]
        if column == "dose_uM":
            equal = np.array_equal(observed.to_numpy(np.float64), expected.to_numpy())
        elif column.endswith("_count"):
            equal = np.array_equal(observed.to_numpy(np.int64), expected.to_numpy())
        else:
            equal = observed.astype(str).equals(expected.reset_index(drop=True))
        if not equal:
            raise AssertionError(f"Condition-shift row alignment failed for {column}")

    row_by_condition = dict(
        zip(
            metadata["condition_id"].astype(str),
            metadata["condition_shift_row"].astype(np.int64),
            strict=True,
        )
    )
    alignment = {
        "status": "pass",
        "all_train_conditions_aligned": True,
        "aligned_rows": len(metadata),
        "row_is_contiguous": True,
        "metadata_columns_checked": list(comparisons),
        "sample": [
            {
                "condition_shift_row": int(position),
                "condition_id": str(metadata.iloc[position]["condition_id"]),
            }
            for position in (0, len(metadata) // 2, len(metadata) - 1)
        ],
    }
    return shifts, metadata, row_by_condition, alignment


def targets_for_batch(
    condition_ids: list[str],
    shifts: np.memmap,
    metadata: pd.DataFrame,
    row_by_condition: dict[str, int],
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    rows = np.fromiter(
        (row_by_condition[condition_id] for condition_id in condition_ids),
        dtype=np.int64,
        count=len(condition_ids),
    )
    if metadata.iloc[rows]["condition_id"].astype(str).tolist() != condition_ids:
        raise AssertionError("Batch condition_id does not match Delta_condition row")
    target = torch.from_numpy(np.asarray(shifts[rows], dtype=np.float32)).to(
        device, non_blocking=True
    )
    if target.shape != (len(condition_ids), LATENT_DIM) or not torch.isfinite(target).all():
        raise AssertionError(f"Invalid Delta_condition target shape: {tuple(target.shape)}")
    return target, rows


@torch.inference_mode()
def fixed_train_mse(
    model: DeltaPredictor,
    dataset: Any,
    shifts: np.memmap,
    metadata: pd.DataFrame,
    row_by_condition: dict[str, int],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    dataset.set_epoch(0)
    loader = make_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=TRAINING_SEED,
        drop_last=False,
    )
    raw = next(iter(loader))
    condition_ids = list(raw["condition_id"])
    target, rows = targets_for_batch(
        condition_ids, shifts, metadata, row_by_condition, device
    )
    ctrl = raw["ctrl_cell_emb"].to(device, non_blocking=True)
    pert = raw["pert_emb"].to(device, non_blocking=True)
    model.eval()
    delta = model(ctrl, pert)
    loss = F.mse_loss(delta, target)
    if delta.shape != target.shape or not torch.isfinite(loss):
        raise AssertionError("Fixed train MSE audit failed")
    prediction = ctrl + delta[:, None, :]
    return {
        "conditions": len(condition_ids),
        "condition_id_first": condition_ids[0],
        "condition_id_last": condition_ids[-1],
        "target_row_first": int(rows[0]),
        "target_row_last": int(rows[-1]),
        "delta_pred_shape": list(delta.shape),
        "target_shape": list(target.shape),
        "prediction_shape": list(prediction.shape),
        "mse": float(loss),
        "finite": True,
    }


def v1_status() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "stopped",
        "scientific_result": False,
        "files_preserved": True,
        "best_checkpoint_exists": V1_BEST_PATH.exists(),
        "last_checkpoint_exists": V1_LAST_PATH.exists(),
    }
    if V1_HISTORY_PATH.exists():
        history = pd.read_csv(V1_HISTORY_PATH, encoding="utf-8-sig")
        payload.update(
            {
                "completed_epochs": len(history),
                "last_epoch": int(history.iloc[-1]["epoch"]),
                "optimizer_steps": int(history.iloc[-1]["global_step"]),
            }
        )
    return payload


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != WORLD_SIZE:
        raise RuntimeError("B2 v2 smoke requires exactly two torchrun processes")
    if os.environ.get("NCCL_CUMEM_ENABLE") != "0" or os.environ.get(
        "NCCL_CUMEM_HOST_ENABLE"
    ) != "0":
        raise RuntimeError("Set both NCCL CUMEM workaround variables to 0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("B2 v2 smoke requires exactly two visible CUDA GPUs")
    if args.batch_size_per_gpu not in (64, 128):
        raise ValueError("Only the requested 128/GPU or OOM fallback 64/GPU is allowed")
    if args.conditions != DEFAULT_SMOKE_CONDITIONS or not 10 <= args.steps <= 20:
        raise ValueError("This smoke is frozen to 512 conditions and 10-20 steps")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    try:
        seed_everything(TRAINING_SEED)
        dataset = make_split_dataset("train", args.conditions)
        if len(dataset) != DEFAULT_SMOKE_CONDITIONS or not dataset.conditions["split"].eq("train").all():
            raise AssertionError("B2 v2 smoke Dataset is not exactly 512 train conditions")
        shifts, metadata, row_by_condition, alignment = load_targets(dataset)

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=TRAINING_SEED,
            drop_last=True,
        )
        loader = make_dataloader(
            dataset,
            batch_size=args.batch_size_per_gpu,
            shuffle=False,
            seed=TRAINING_SEED,
            sampler=sampler,
            drop_last=True,
        )
        if len(loader) < 1:
            raise AssertionError("B2 v2 smoke DataLoader has no full batch")

        raw_model = DeltaPredictor().to(device)
        initial_parameters = {
            name: parameter.detach().cpu().clone()
            for name, parameter in raw_model.named_parameters()
        }
        initial_fixed = fixed_train_mse(
            raw_model,
            dataset,
            shifts,
            metadata,
            row_by_condition,
            device,
            args.batch_size_per_gpu,
        )
        model = DistributedDataParallel(raw_model, device_ids=[local_rank])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        step_losses: list[float] = []
        first_shapes: dict[str, list[int]] | None = None
        torch.cuda.reset_peak_memory_stats(device)
        dist.barrier()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        step = 0
        epoch = 0
        while step < args.steps:
            dataset.set_epoch(epoch)
            sampler.set_epoch(epoch)
            model.train()
            iterator = iter(loader)
            while step < args.steps:
                try:
                    raw = next(iterator)
                except StopIteration:
                    break
                condition_ids = list(raw["condition_id"])
                target, _rows = targets_for_batch(
                    condition_ids, shifts, metadata, row_by_condition, device
                )
                ctrl = raw["ctrl_cell_emb"].to(device, non_blocking=True)
                pert = raw["pert_emb"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                delta = model(ctrl, pert)
                if delta.shape != target.shape or delta.shape[1] != LATENT_DIM:
                    raise AssertionError("Delta_pred and Delta_condition are not [B,768]")
                loss = F.mse_loss(delta, target)
                if not torch.isfinite(loss):
                    raise AssertionError("B2 v2 MSE became non-finite")
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise AssertionError("B2 v2 gradient became non-finite")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), GRADIENT_CLIP_NORM, error_if_nonfinite=True
                )
                optimizer.step()
                report_loss = loss.detach().clone()
                dist.all_reduce(report_loss, op=dist.ReduceOp.SUM)
                report_loss /= world_size
                step_losses.append(float(report_loss))
                if first_shapes is None:
                    first_shapes = {
                        "ctrl_cell_emb": list(ctrl.shape),
                        "pert_emb": list(pert.shape),
                        "delta_pred": list(delta.shape),
                        "delta_condition": list(target.shape),
                    }
                step += 1
            epoch += 1
        torch.cuda.synchronize(device)
        dist.barrier()
        training_elapsed = time.perf_counter() - started

        final_fixed = fixed_train_mse(
            raw_model,
            dataset,
            shifts,
            metadata,
            row_by_condition,
            device,
            args.batch_size_per_gpu,
        )
        update = parameter_update_audit(raw_model, initial_parameters)
        sync_error = ddp_parameter_sync_error(raw_model, world_size)
        if sync_error != 0.0:
            raise AssertionError(f"B2 v2 DDP parameters are not synchronized: {sync_error}")

        local_runtime = {
            "rank": rank,
            "local_rank": local_rank,
            "gpu_name": torch.cuda.get_device_name(device),
            "training_elapsed_seconds": training_elapsed,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        }
        gathered: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(gathered, local_runtime)
        runtimes = [item for item in gathered if item is not None]
        if len(runtimes) != WORLD_SIZE or {item["local_rank"] for item in runtimes} != {0, 1}:
            raise AssertionError("Both A6000 ranks did not complete B2 v2 smoke")
        wall = max(float(item["training_elapsed_seconds"]) for item in runtimes)
        conditions_per_second = args.steps * args.batch_size_per_gpu * world_size / wall
        eta_seconds = EXPECTED_TRAIN_CONDITIONS * 30 / conditions_per_second
        fixed_decreased = final_fixed["mse"] < initial_fixed["mse"]
        if not fixed_decreased or not all(math.isfinite(value) for value in step_losses):
            raise AssertionError("B2 v2 fixed-train MSE did not decrease or was non-finite")

        if rank != 0:
            dist.barrier()
            return None
        payload = {
            "created_at_utc": utc_now(),
            "status": "pass",
            "scope": "512 real train conditions; no formal full training, val, test, or STATE/ST",
            "b2_v1_energy_formal_training": v1_status(),
            "b2_v2": {
                "objective": "MSE(Delta_pred, Delta_condition)",
                "energy_loss_used": False,
                "auxiliary_loss_used": False,
                "future_prediction": "raw_Zctrl + Delta_pred[:, None, :]",
                "architecture_reused": "B2PooledMLP 1148 -> 1024 -> 1024 -> 768",
                "architecture_source": {
                    "path": "perturbation_scripts/train_tahoe_experiment1_b2.py",
                    "sha256": sha256_file(B2_V1_SCRIPT_PATH),
                },
            },
            "target": {
                "definition": "precomputed treated centroid - matched control centroid",
                "recomputed": False,
                "npy": {
                    "path": "results/tahoe_experiment1_b1_train_condition_shifts.npy",
                    "sha256": sha256_file(SHIFT_PATH),
                    "shape": list(shifts.shape),
                    "dtype": str(shifts.dtype),
                },
                "metadata": {
                    "path": "results/tahoe_experiment1_b1_train_condition_shifts_metadata.csv",
                    "sha256": sha256_file(SHIFT_METADATA_PATH),
                },
                "alignment": alignment,
            },
            "training": {
                "conditions": len(dataset),
                "split": "train",
                "val_dataset_constructed": False,
                "test_dataset_constructed": False,
                "steps": args.steps,
                "epochs_used_to_reach_steps": epoch,
                "world_size": world_size,
                "batch_size_per_gpu": args.batch_size_per_gpu,
                "global_batch_size": args.batch_size_per_gpu * world_size,
                "optimizer": "AdamW",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "gradient_clip_norm": GRADIENT_CLIP_NORM,
                "seed": TRAINING_SEED,
                "first_batch_shapes": first_shapes,
                "initial_fixed_train_mse": initial_fixed["mse"],
                "final_fixed_train_mse": final_fixed["mse"],
                "fixed_train_mse_decreased": fixed_decreased,
                "first_step_mse": step_losses[0],
                "final_step_mse": step_losses[-1],
                "minimum_step_mse": min(step_losses),
                "all_losses_finite": True,
                "parameters_updated": update,
                "ddp_parameter_moment_max_abs_error": sync_error,
            },
            "fixed_train_audit": {"initial": initial_fixed, "final": final_fixed},
            "runtime": {
                "rank": runtimes,
                "average_step_seconds_end_to_end": wall / args.steps,
                "global_conditions_per_second": conditions_per_second,
                "eta_45652_conditions_x_30_epochs_seconds": eta_seconds,
                "eta_45652_conditions_x_30_epochs_hours": eta_seconds / 3600,
                "peak_allocated_gib_max": max(item["peak_allocated_gib"] for item in runtimes),
                "peak_reserved_gib_max": max(item["peak_reserved_gib"] for item in runtimes),
            },
            "checks": {
                "condition_target_alignment": "pass",
                "delta_and_target_shape_Bx768": "pass",
                "loss_finite": "pass",
                "backward_finite": "pass",
                "parameters_updated": "pass",
                "fixed_train_mse_decreased": "pass",
                "two_gpu_ddp": "pass",
                "val_or_test_run": "pass: none",
            },
            "implementation": {
                "path": "perturbation_scripts/train_tahoe_experiment1_b2_v2.py",
                "sha256": sha256_file(SCRIPT_PATH),
            },
            "blockers": [],
        }
        atomic_write_json(OUTPUT_PATH, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
        dist.barrier()
        return payload
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=int, default=DEFAULT_SMOKE_CONDITIONS)
    parser.add_argument("--batch-size-per-gpu", type=int, default=DEFAULT_BATCH_SIZE_PER_GPU)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
