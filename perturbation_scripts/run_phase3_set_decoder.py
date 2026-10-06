#!/usr/bin/env python3
"""Prepare, smoke, train, and validation-select frozen Phase-III D1/D2."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from phase3_set_decoder_data import (
    CACHE_PLAN,
    EMBEDDING_MANIFEST,
    EXPRESSION_MANIFEST,
    GENE_DIM,
    Phase3SetDecoderDataset,
    SAMPLING_AUDIT,
    SET_SIZE,
    SUBSET_MANIFEST,
    TASK_PATH,
    TASK_SHA256,
    artifact,
    atomic_json,
    make_dataloader,
    sha256_file,
    write_sampling_fairness_audit,
)
from phase3_set_decoder_model import (
    SetDeltaDecoder,
    architecture_audit,
    build_set_decoder,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
SCRIPT_PATH = Path(__file__).resolve()
DATASET_PATH = PROJECT_ROOT / "perturbation_scripts/phase3_set_decoder_data.py"
MODEL_PATH = PROJECT_ROOT / "perturbation_scripts/phase3_set_decoder_model.py"
PROTOCOL_PATH = RESULTS / "phase3_set_decoder_training_protocol.json"
ARCHITECTURE_AUDIT = RESULTS / "phase3_set_decoder_architecture_audit.json"
SMOKE_PATH = RESULTS / "phase3_set_decoder_exact_config_smoke.json"
SELECTION_PATH = RESULTS / "phase3_decoder_selection.json"

SEED = 42
WORLD_SIZE = 2
BATCH_SIZE_PER_GPU = 20
GRADIENT_ACCUMULATION_STEPS = 2
EFFECTIVE_GLOBAL_BATCH = 80
NUM_WORKERS_PER_RANK = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
MAX_EPOCHS = 30
TRAIN_CONDITIONS = 4000
VAL_CONDITIONS = 500
MICROBATCHES_PER_RANK = 100
OPTIMIZER_STEPS_PER_EPOCH = 50
EXPECTED_PARAMETERS = {"d1": 1_699_012, "d2": 1_877_909}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_core(model: nn.Module) -> SetDeltaDecoder:
    value = model.module if isinstance(model, DistributedDataParallel) else model
    if not isinstance(value, SetDeltaDecoder):
        raise TypeError(type(value))
    return value


def variant_paths(variant: str) -> dict[str, Path]:
    root = RESULTS / f"phase3_{variant}_checkpoints"
    return {
        "dir": root,
        "best": root / "best.pt",
        "last": root / "last.pt",
        "result": RESULTS / f"phase3_{variant}_training_result.json",
    }


def _architecture_payload() -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for variant in ("d1", "d2"):
        model = build_set_decoder(variant, SEED)
        audit = architecture_audit(model)
        audit["expected_parameter_count"] = EXPECTED_PARAMETERS[variant]
        audit["parameter_count_exact"] = audit["parameter_count"] == EXPECTED_PARAMETERS[variant]
        audit["trainable_parameter_count_exact"] = (
            audit["trainable_parameter_count"] == EXPECTED_PARAMETERS[variant]
        )
        if audit["status"] != "pass" or not audit["parameter_count_exact"] or not audit[
            "trainable_parameter_count_exact"
        ]:
            raise AssertionError(audit)
        variants[variant] = audit
    return {
        "schema": "phase3_set_decoder_architecture_audit_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "variants": variants,
        "shared_formulation": "shared_encoder(treated) - shared_encoder(control) -> shared signed readout",
        "model_source": artifact(MODEL_PATH),
        "phase3_complete": False,
        "phase4_started": False,
    }


def _protocol_payload(architecture: dict[str, Any]) -> dict[str, Any]:
    expression = json.loads(EXPRESSION_MANIFEST.read_text(encoding="utf-8"))
    sampling = json.loads(SAMPLING_AUDIT.read_text(encoding="utf-8"))
    if expression.get("status") != "pass" or sampling.get("status") != "pass":
        raise AssertionError("Expression/sampling audits must PASS before freezing protocol")
    return {
        "schema": "phase3_set_decoder_training_protocol_v1",
        "created_at_utc": utc_now(),
        "status": "frozen",
        "phase": "Task 3 / Phase III",
        "task": {**artifact(TASK_PATH), "expected_sha256": TASK_SHA256},
        "data": {
            "phase2_subset_manifest": artifact(SUBSET_MANIFEST),
            "phase2_cache_plan": artifact(CACHE_PLAN),
            "author_embedding_manifest": artifact(EMBEDDING_MANIFEST),
            "expression_cache_manifest": artifact(EXPRESSION_MANIFEST),
            "sampling_fairness_audit": artifact(SAMPLING_AUDIT),
            "conditions": {"train": 4000, "val": 500, "test": 500},
            "pool_cap": 512,
            "set_size": SET_SIZE,
            "sampling": "Phase-II deterministic seed+epoch+pair_id+side, without replacement",
            "canonical_order": "stable phase2_cell_index ascending after membership selection",
            "train_sampling_epoch": "current training epoch",
            "validation_sampling_epoch": 0,
            "test_repeat_epochs": [0, 1, 2, 3, 4],
        },
        "models": architecture["variants"],
        "target": "mean(real treated Top20 expression)-mean(matched real control Top20 expression)",
        "loss": {"class": "MSELoss", "space": "signed condition-level Top20 delta", "dtype": "float32"},
        "training": {
            "seed": SEED,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": None,
            "max_epochs": MAX_EPOCHS,
            "precision": "BF16 autocast forward; FP32 MSE",
            "world_size": WORLD_SIZE,
            "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_global_batch": EFFECTIVE_GLOBAL_BATCH,
            "distributed_sampler_shuffle": True,
            "distributed_sampler_seed": SEED,
            "distributed_sampler_drop_last": False,
            "dataloader_drop_last": False,
            "conditions_per_rank": 2000,
            "microbatches_per_rank": MICROBATCHES_PER_RANK,
            "optimizer_steps_per_epoch": OPTIMIZER_STEPS_PER_EPOCH,
            "conditions_consumed_per_epoch": TRAIN_CONDITIONS,
            "checkpoint_metric": "lowest validation macro condition MSE",
            "test_used_for_checkpoint_or_architecture_selection": False,
        },
        "architecture_selection": {
            "data": "500 validation conditions, epoch 0 only",
            "primary": "higher shared-valid macro Pearson delta",
            "tie_break": "lower macro MAE",
            "fallback": "lower macro MAE across all 500 if either model has finite constant predictions",
            "test_used": False,
        },
        "implementation": {
            "runner": artifact(SCRIPT_PATH),
            "dataset": artifact(DATASET_PATH),
            "model": artifact(MODEL_PATH),
        },
        "phase3_complete": False,
        "phase4_started": False,
    }


def prepare(rewrite: bool) -> dict[str, Any]:
    if sha256_file(TASK_PATH) != TASK_SHA256:
        raise AssertionError("Frozen Phase-III task changed")
    expression = json.loads(EXPRESSION_MANIFEST.read_text(encoding="utf-8"))
    if expression.get("status") != "pass":
        raise AssertionError("Formal Phase-III expression cache is not PASS")
    if any(path.exists() for path in (ARCHITECTURE_AUDIT, PROTOCOL_PATH, SAMPLING_AUDIT)) and not rewrite:
        raise FileExistsError("Phase-III prepare outputs exist; use --rewrite to regenerate audits")
    architecture = _architecture_payload()
    atomic_json(ARCHITECTURE_AUDIT, architecture)
    write_sampling_fairness_audit()
    protocol = _protocol_payload(architecture)
    atomic_json(PROTOCOL_PATH, protocol)
    result = {
        "status": "pass",
        "architecture_audit": artifact(ARCHITECTURE_AUDIT),
        "sampling_fairness_audit": artifact(SAMPLING_AUDIT),
        "training_protocol": artifact(PROTOCOL_PATH),
        "formal_training_started": False,
        "phase3_complete": False,
        "phase4_started": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def verify_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    architecture = json.loads(ARCHITECTURE_AUDIT.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen" or architecture.get("status") != "pass":
        raise AssertionError("Phase-III protocol/architecture audit is not ready")
    for name, path in (("runner", SCRIPT_PATH), ("dataset", DATASET_PATH), ("model", MODEL_PATH)):
        if protocol["implementation"][name]["sha256"] != sha256_file(path):
            raise AssertionError(f"Phase-III {name} changed after protocol freeze; rerun prepare --rewrite")
    for variant in ("d1", "d2"):
        recorded = architecture["variants"][variant]
        if recorded["parameter_count"] != EXPECTED_PARAMETERS[variant]:
            raise AssertionError(f"Frozen {variant} parameter count changed")
    return protocol


def setup_distributed() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != WORLD_SIZE:
        raise RuntimeError("Phase-III smoke/training requires torchrun with exactly two ranks")
    if os.environ.get("NCCL_CUMEM_ENABLE") != "0" or os.environ.get("NCCL_CUMEM_HOST_ENABLE") != "0":
        raise RuntimeError("This host requires NCCL_CUMEM_ENABLE=0 and NCCL_CUMEM_HOST_ENABLE=0")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    return rank, local_rank, device


def move_batch(raw: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: raw[key].to(device, non_blocking=True)
        for key in ("ctrl_cell_emb", "pert_cell_emb", "gt_delta")
    }


def forward_mse(
    model: nn.Module, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction = model(batch["ctrl_cell_emb"], batch["pert_cell_emb"])
    prediction32 = prediction.float()
    target32 = batch["gt_delta"].float()
    if prediction32.shape != target32.shape or prediction32.shape[1:] != (GENE_DIM,):
        raise AssertionError("Phase-III prediction/target shape changed")
    per_condition = (prediction32 - target32).square().mean(dim=1)
    if not torch.isfinite(prediction32).all() or not torch.isfinite(per_condition).all():
        raise AssertionError("Phase-III forward/MSE is non-finite")
    return prediction, per_condition.mean(), per_condition


def parameter_sync_error(model: nn.Module) -> float:
    moments = torch.stack(
        [
            torch.stack(
                [parameter.detach().double().sum(), parameter.detach().double().square().sum()]
            )
            for parameter in model_core(model).parameters()
        ]
    ).sum(dim=0)
    gathered = [torch.zeros_like(moments) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered, moments)
    return max(float((value - gathered[0]).abs().max()) for value in gathered)


def _gradient_audit(model: SetDeltaDecoder) -> dict[str, Any]:
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    finite = bool(gradients) and all(torch.isfinite(value).all().item() for value in gradients)
    nonzero = sum(int(torch.count_nonzero(value)) for value in gradients)
    return {
        "parameter_tensors_with_grad": len(gradients),
        "all_finite": finite,
        "nonzero_coordinates": nonzero,
        "global_l2_norm": math.sqrt(
            sum(float(value.detach().float().square().sum()) for value in gradients)
        ) if gradients else 0.0,
    }


def _smoke_variant(
    variant: str,
    raw_batches: list[dict[str, Any]],
    rank: int,
    local_rank: int,
    device: torch.device,
) -> dict[str, Any] | None:
    raw_model = build_set_decoder(variant, SEED).to(device)
    architecture = architecture_audit(raw_model)
    if architecture["status"] != "pass" or architecture["parameter_count"] != EXPECTED_PARAMETERS[variant]:
        raise AssertionError(architecture)
    model = DistributedDataParallel(raw_model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    probe = next(raw_model.parameters())
    before = probe.detach().clone()
    permutation: dict[str, Any] | None = None
    first = move_batch(raw_batches[0], device)
    if variant == "d2":
        raw_model.eval()
        generator = torch.Generator(device=device).manual_seed(SEED)
        control_order = torch.randperm(SET_SIZE, generator=generator, device=device)
        treated_order = torch.randperm(SET_SIZE, generator=generator, device=device)
        with torch.no_grad():
            original = raw_model(first["ctrl_cell_emb"][:1], first["pert_cell_emb"][:1])
            permuted = raw_model(
                first["ctrl_cell_emb"][:1, control_order],
                first["pert_cell_emb"][:1, treated_order],
            )
        difference = float((original - permuted).abs().max())
        permutation = {
            "max_abs_difference": difference,
            "atol": 2e-5,
            "rtol": 2e-5,
            "allclose": bool(torch.allclose(original, permuted, atol=2e-5, rtol=2e-5)),
        }
    raw_model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    losses: list[float] = []
    output_shape: list[int] = []
    started = time.perf_counter()
    for microbatch, raw in enumerate(raw_batches):
        batch = move_batch(raw, device)
        synchronize = microbatch + 1 == GRADIENT_ACCUMULATION_STEPS
        with contextlib.nullcontext() if synchronize else model.no_sync():
            prediction, loss, _ = forward_mse(model, batch)
            (loss / GRADIENT_ACCUMULATION_STEPS).backward()
        losses.append(float(loss.detach()))
        output_shape = list(prediction.shape)
    gradients = _gradient_audit(raw_model)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    updated = not torch.equal(before, probe.detach())
    sync_error = parameter_sync_error(model)
    local = {
        "rank": rank,
        "gpu": torch.cuda.get_device_name(device),
        "losses": losses,
        "peak_allocated_GiB": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_GiB": torch.cuda.max_memory_reserved(device) / 2**30,
        "elapsed_seconds": time.perf_counter() - started,
    }
    gathered: list[dict[str, Any] | None] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, local)
    result = None
    if rank == 0:
        shapes = dict(raw_model.shared_encoder.last_shapes)
        result = {
            "variant": variant,
            "architecture": architecture,
            "input_shapes": {
                "control": list(first["ctrl_cell_emb"].shape),
                "treated": list(first["pert_cell_emb"].shape),
                "gt_delta": list(first["gt_delta"].shape),
            },
            "output_shape": output_shape,
            "encoder_intermediate_shapes": shapes,
            "single_shared_encoder": not hasattr(raw_model, "control_encoder")
            and not hasattr(raw_model, "treated_encoder"),
            "losses": [value for item in gathered if item for value in item["losses"]],
            "loss_finite": all(
                math.isfinite(value) for item in gathered if item for value in item["losses"]
            ),
            "gradients": gradients,
            "parameter_updated": updated,
            "ddp_parameter_moment_max_abs_error": sync_error,
            "permutation_invariance": permutation,
            "rank_runtime": gathered,
        }
    del model, raw_model, optimizer
    torch.cuda.empty_cache()
    dist.barrier()
    return result


def run_smoke(overwrite: bool) -> dict[str, Any] | None:
    rank, local_rank, device = setup_distributed()
    try:
        verify_protocol()
        if rank == 0 and SMOKE_PATH.exists() and not overwrite:
            raise FileExistsError(SMOKE_PATH)
        seed_everything(SEED)
        dataset = Phase3SetDecoderDataset(split="train", seed=SEED, epoch=0)
        sampler = DistributedSampler(
            dataset, num_replicas=WORLD_SIZE, rank=rank, shuffle=True, seed=SEED, drop_last=False
        )
        loader = make_dataloader(
            dataset,
            batch_size=BATCH_SIZE_PER_GPU,
            shuffle=False,
            seed=SEED,
            sampler=sampler,
            drop_last=False,
            num_workers=NUM_WORKERS_PER_RANK,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2,
        )
        raw_batches = []
        for batch in loader:
            raw_batches.append(batch)
            if len(raw_batches) == GRADIENT_ACCUMULATION_STEPS:
                break
        if len(raw_batches) != GRADIENT_ACCUMULATION_STEPS:
            raise AssertionError("Combined smoke did not obtain two exact-config microbatches")
        variants: dict[str, Any] = {}
        for variant in ("d1", "d2"):
            result = _smoke_variant(variant, raw_batches, rank, local_rank, device)
            if rank == 0 and result is not None:
                variants[variant] = result
        if rank != 0:
            return None
        d1_shapes = variants["d1"]["encoder_intermediate_shapes"]
        d2_shapes = variants["d2"]["encoder_intermediate_shapes"]
        checks = {
            "real_input_shapes": all(
                item["input_shapes"] == {
                    "control": [BATCH_SIZE_PER_GPU, SET_SIZE, 768],
                    "treated": [BATCH_SIZE_PER_GPU, SET_SIZE, 768],
                    "gt_delta": [BATCH_SIZE_PER_GPU, GENE_DIM],
                }
                for item in variants.values()
            ),
            "output_shape_b20x20": all(
                item["output_shape"] == [BATCH_SIZE_PER_GPU, GENE_DIM]
                for item in variants.values()
            ),
            "shared_encoder": all(item["single_shared_encoder"] for item in variants.values()),
            "d1_exact_shapes": d1_shapes.get("conv1") == [BATCH_SIZE_PER_GPU, 8, 128, 192]
            and d1_shapes.get("conv2") == [BATCH_SIZE_PER_GPU, 16, 64, 48]
            and d1_shapes.get("conv3") == [BATCH_SIZE_PER_GPU, 32, 32, 12]
            and d1_shapes.get("tokens32") == [BATCH_SIZE_PER_GPU, 384, 32]
            and d1_shapes.get("tokens256") == [BATCH_SIZE_PER_GPU, 384, 256],
            "d2_exact_shapes": d2_shapes.get("projected") == [BATCH_SIZE_PER_GPU, 256, 256]
            and d2_shapes.get("transformer") == [BATCH_SIZE_PER_GPU, 256, 256]
            and d2_shapes.get("attention_weights") == [BATCH_SIZE_PER_GPU, 256, 1],
            "d2_permutation_invariant": variants["d2"]["permutation_invariance"]["allclose"],
            "finite_fp32_mse": all(item["loss_finite"] for item in variants.values()),
            "finite_nonzero_gradients": all(
                item["gradients"]["all_finite"]
                and item["gradients"]["nonzero_coordinates"] > 0
                and item["gradients"]["global_l2_norm"] > 0
                for item in variants.values()
            ),
            "optimizer_updates": all(item["parameter_updated"] for item in variants.values()),
            "ddp_synchronized": all(
                item["ddp_parameter_moment_max_abs_error"] == 0.0 for item in variants.values()
            ),
        }
        result = {
            "schema": "phase3_set_decoder_exact_config_smoke_v1",
            "created_at_utc": utc_now(),
            "status": "pass" if all(checks.values()) else "fail",
            "scientific_result": False,
            "combined_single_smoke": True,
            "world_size": WORLD_SIZE,
            "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "variants": variants,
            "checks": checks,
            "protocol": artifact(PROTOCOL_PATH),
            "phase3_complete": False,
            "phase4_started": False,
        }
        if result["status"] != "pass":
            raise AssertionError(result)
        atomic_json(SMOKE_PATH, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


@torch.no_grad()
def validate_mse(
    model: nn.Module, dataset: Phase3SetDecoderDataset, device: torch.device
) -> dict[str, Any]:
    dataset.set_epoch(0)
    loader = make_dataloader(
        dataset,
        batch_size=BATCH_SIZE_PER_GPU,
        shuffle=False,
        seed=SEED,
        drop_last=False,
        num_workers=NUM_WORKERS_PER_RANK,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
    )
    model.eval()
    ids: list[str] = []
    losses: list[float] = []
    for raw in loader:
        ids.extend(str(value) for value in raw["condition_id"])
        batch = move_batch(raw, device)
        _, _, per_condition = forward_mse(model, batch)
        losses.extend(per_condition.cpu().numpy().astype(float).tolist())
    expected = dataset.conditions["pair_id"].astype(str).tolist()
    if ids != expected or len(losses) != VAL_CONDITIONS or not np.isfinite(losses).all():
        raise AssertionError("Validation was shuffled, dropped, duplicated, or non-finite")
    return {"conditions": len(losses), "macro_condition_mse": float(np.mean(losses))}


def capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"])


def checkpoint_payload(
    *,
    variant: str,
    kind: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_metric: float,
    best_epoch: int,
    history: list[dict[str, Any]],
    rng_states: list[dict[str, Any]],
    terminal: bool,
) -> dict[str, Any]:
    return {
        "schema": "phase3_set_decoder_checkpoint_v1",
        "variant": variant,
        "kind": kind,
        "created_at_utc": utc_now(),
        "model_state_dict": model_core(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_mse": best_metric,
        "best_epoch": best_epoch,
        "history": history,
        "rng_states": rng_states,
        "terminal": terminal,
        "seed": SEED,
        "protocol": artifact(PROTOCOL_PATH),
        "expression_cache_manifest": artifact(EXPRESSION_MANIFEST),
        "runner": artifact(SCRIPT_PATH),
        "dataset": artifact(DATASET_PATH),
        "model": artifact(MODEL_PATH),
        "phase3_complete": False,
        "phase4_started": False,
    }


def run_train(variant: str, resume: bool) -> dict[str, Any] | None:
    rank, local_rank, device = setup_distributed()
    paths = variant_paths(variant)
    try:
        verify_protocol()
        smoke = json.loads(SMOKE_PATH.read_text(encoding="utf-8"))
        if smoke.get("status") != "pass":
            raise AssertionError("Combined exact-config smoke is not PASS")
        if rank == 0:
            paths["dir"].mkdir(parents=True, exist_ok=True)
            if not resume and any(paths[name].exists() for name in ("best", "last", "result")):
                raise FileExistsError(f"Phase-III {variant} formal outputs exist; use --resume-last")
        seed_everything(SEED)
        train_dataset = Phase3SetDecoderDataset(split="train", seed=SEED, epoch=0)
        val_dataset = (
            Phase3SetDecoderDataset(split="val", seed=SEED, epoch=0) if rank == 0 else None
        )
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=WORLD_SIZE,
            rank=rank,
            shuffle=True,
            seed=SEED,
            drop_last=False,
        )
        loader = make_dataloader(
            train_dataset,
            batch_size=BATCH_SIZE_PER_GPU,
            shuffle=False,
            seed=SEED,
            sampler=sampler,
            drop_last=False,
            num_workers=NUM_WORKERS_PER_RANK,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2,
        )
        if len(loader) != MICROBATCHES_PER_RANK or len(loader) % GRADIENT_ACCUMULATION_STEPS:
            raise AssertionError("Formal loader no longer yields exactly 100 divisible microbatches/rank")
        raw_model = build_set_decoder(variant, SEED).to(device)
        audit = architecture_audit(raw_model)
        if audit["status"] != "pass" or audit["parameter_count"] != EXPECTED_PARAMETERS[variant]:
            raise AssertionError(audit)
        model = DistributedDataParallel(raw_model, device_ids=[local_rank], output_device=local_rank)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        start_epoch = 0
        global_step = 0
        best_metric = float("inf")
        best_epoch = -1
        history: list[dict[str, Any]] = []
        if resume:
            checkpoint = torch.load(paths["last"], map_location="cpu", weights_only=False)
            if (
                checkpoint.get("schema") != "phase3_set_decoder_checkpoint_v1"
                or checkpoint.get("variant") != variant
                or checkpoint.get("kind") != "last"
            ):
                raise AssertionError(f"Not a Phase-III {variant} last checkpoint")
            if checkpoint.get("terminal"):
                raise RuntimeError(f"Phase-III {variant} last checkpoint is already terminal")
            raw_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = int(checkpoint["epoch"]) + 1
            global_step = int(checkpoint["global_step"])
            best_metric = float(checkpoint["best_val_mse"])
            best_epoch = int(checkpoint["best_epoch"])
            history = list(checkpoint["history"])
            restore_rng(checkpoint["rng_states"][rank])
        dist.barrier()
        started = time.perf_counter()
        peak_allocated = 0.0
        peak_reserved = 0.0
        expected_ids = set(train_dataset.conditions["pair_id"].astype(str))
        for epoch in range(start_epoch, MAX_EPOCHS):
            train_dataset.set_epoch(epoch)
            sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            loss_sum = 0.0
            condition_count = 0
            optimizer_updates = 0
            consumed_ids: list[str] = []
            epoch_started = time.perf_counter()
            for microbatch, raw in enumerate(loader):
                consumed_ids.extend(str(value) for value in raw["condition_id"])
                batch = move_batch(raw, device)
                synchronize = (microbatch + 1) % GRADIENT_ACCUMULATION_STEPS == 0
                with contextlib.nullcontext() if synchronize else model.no_sync():
                    _, loss, per_condition = forward_mse(model, batch)
                    (loss / GRADIENT_ACCUMULATION_STEPS).backward()
                loss_sum += float(per_condition.detach().sum())
                condition_count += len(per_condition)
                if synchronize:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    optimizer_updates += 1
            if optimizer_updates != OPTIMIZER_STEPS_PER_EPOCH:
                raise AssertionError("Formal epoch did not execute exactly 50 optimizer updates")
            torch.cuda.synchronize(device)
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device) / 2**30)
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device) / 2**30)
            totals = torch.tensor([loss_sum, condition_count], dtype=torch.float64, device=device)
            dist.all_reduce(totals)
            train_mse = float(totals[0] / totals[1])
            gathered_ids: list[list[str] | None] = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_ids, consumed_ids)
            flat_ids = [value for values in gathered_ids if values for value in values]
            counts = Counter(flat_ids)
            coverage = {
                "global_conditions_consumed": len(flat_ids),
                "global_unique_pair_ids": len(counts),
                "missing_pair_ids": len(expected_ids - set(counts)),
                "duplicate_pair_ids": sum(value - 1 for value in counts.values() if value > 1),
                "pair_id_set_sha256": hashlib.sha256(
                    "\n".join(sorted(counts)).encode("utf-8")
                ).hexdigest(),
            }
            if not (
                coverage["global_conditions_consumed"] == TRAIN_CONDITIONS
                and coverage["global_unique_pair_ids"] == TRAIN_CONDITIONS
                and coverage["missing_pair_ids"] == 0
                and coverage["duplicate_pair_ids"] == 0
            ):
                raise AssertionError(f"Formal epoch coverage failed: {coverage}")
            dist.barrier()
            validation: dict[str, Any] | None = None
            if rank == 0:
                assert val_dataset is not None
                validation = validate_mse(raw_model, val_dataset, device)
            holder: list[Any] = [validation]
            dist.broadcast_object_list(holder, src=0)
            validation = holder[0]
            val_mse = float(validation["macro_condition_mse"])
            improved = val_mse < best_metric
            if improved:
                best_metric = val_mse
                best_epoch = epoch
            row = {
                "epoch": epoch,
                "train_mse": train_mse,
                "val_mse": val_mse,
                "improved": improved,
                "best_epoch": best_epoch,
                "best_val_mse": best_metric,
                "optimizer_updates": optimizer_updates,
                "global_step": global_step,
                "coverage": coverage,
                "elapsed_seconds": time.perf_counter() - epoch_started,
            }
            history.append(row)
            rng = capture_rng()
            rng_states: list[dict[str, Any] | None] = [None] * WORLD_SIZE
            dist.all_gather_object(rng_states, rng)
            terminal = epoch + 1 == MAX_EPOCHS
            if rank == 0:
                common = dict(
                    variant=variant,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    best_metric=best_metric,
                    best_epoch=best_epoch,
                    history=history,
                    rng_states=[value for value in rng_states if value is not None],
                    terminal=terminal,
                )
                if improved:
                    atomic_torch(paths["best"], checkpoint_payload(kind="best", **common))
                atomic_torch(paths["last"], checkpoint_payload(kind="last", **common))
                print(
                    f"phase3 {variant} epoch={epoch} step={global_step} "
                    f"train_mse={train_mse:.8f} val_mse={val_mse:.8f} best_epoch={best_epoch} "
                    f"coverage={coverage['global_unique_pair_ids']}/4000",
                    flush=True,
                )
            dist.barrier()
        sync_error = parameter_sync_error(model)
        local_runtime = {
            "rank": rank,
            "gpu": torch.cuda.get_device_name(device),
            "peak_allocated_GiB": peak_allocated,
            "peak_reserved_GiB": peak_reserved,
        }
        runtimes: list[dict[str, Any] | None] = [None] * WORLD_SIZE
        dist.all_gather_object(runtimes, local_runtime)
        if rank != 0:
            return None
        coverage_pass = all(
            item["coverage"]["global_unique_pair_ids"] == TRAIN_CONDITIONS
            and item["coverage"]["missing_pair_ids"] == 0
            and item["coverage"]["duplicate_pair_ids"] == 0
            and item["coverage"]["global_conditions_consumed"] == TRAIN_CONDITIONS
            for item in history
        )
        result = {
            "schema": "phase3_set_decoder_training_result_v1",
            "created_at_utc": utc_now(),
            "status": "pass" if sync_error == 0.0 and coverage_pass else "fail",
            "variant": variant,
            "architecture": architecture_audit(raw_model),
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_val_mse": best_metric,
            "stop_reason": "max_epochs",
            "history": history,
            "optimizer": {"class": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
            "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_global_batch": EFFECTIVE_GLOBAL_BATCH,
            "distributed_sampler_drop_last": False,
            "dataloader_drop_last": False,
            "conditions_used_per_epoch": TRAIN_CONDITIONS,
            "optimizer_updates_per_epoch": OPTIMIZER_STEPS_PER_EPOCH,
            "coverage_all_epochs_pass": coverage_pass,
            "ddp_parameter_moment_max_abs_error": sync_error,
            "rank_runtime": runtimes,
            "elapsed_seconds": time.perf_counter() - started,
            "checkpoints": {"best": artifact(paths["best"]), "last": artifact(paths["last"])},
            "protocol": artifact(PROTOCOL_PATH),
            "smoke": artifact(SMOKE_PATH),
            "test_dataset_constructed_during_training": False,
            "phase3_complete": False,
            "phase4_started": False,
        }
        if result["status"] != "pass":
            raise AssertionError(result)
        atomic_json(paths["result"], result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Pearson received a non-finite vector")
    if float(np.var(left)) == 0.0 or float(np.var(right)) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


@torch.inference_mode()
def _validation_predictions(variant: str, device: torch.device) -> dict[str, Any]:
    paths = variant_paths(variant)
    checkpoint = torch.load(paths["best"], map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "phase3_set_decoder_checkpoint_v1" or checkpoint.get("variant") != variant:
        raise AssertionError(f"Invalid Phase-III {variant} best checkpoint")
    model = build_set_decoder(variant, SEED)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().requires_grad_(False).to(device)
    dataset = Phase3SetDecoderDataset(split="val", seed=SEED, epoch=0)
    loader = make_dataloader(
        dataset, batch_size=BATCH_SIZE_PER_GPU, shuffle=False, seed=SEED, drop_last=False,
        num_workers=NUM_WORKERS_PER_RANK, pin_memory=True, persistent_workers=False,
        prefetch_factor=2,
    )
    ids: list[str] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    membership = hashlib.sha256()
    for raw in loader:
        ids.extend(str(value) for value in raw["condition_id"])
        membership.update(raw["source_embedding_index"].numpy().astype("<i8", copy=False).tobytes())
        membership.update(raw["target_embedding_index"].numpy().astype("<i8", copy=False).tobytes())
        batch = move_batch(raw, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(batch["ctrl_cell_emb"], batch["pert_cell_emb"])
        predictions.append(prediction.float().cpu().numpy())
        targets.append(batch["gt_delta"].float().cpu().numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    if len(ids) != VAL_CONDITIONS or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise AssertionError("Validation predictions are incomplete or non-finite")
    del model
    torch.cuda.empty_cache()
    return {
        "condition_ids": ids,
        "prediction": prediction,
        "target": target,
        "membership_sha256": membership.hexdigest(),
    }


def select_architecture() -> dict[str, Any]:
    verify_protocol()
    for variant in ("d1", "d2"):
        training = json.loads(variant_paths(variant)["result"].read_text(encoding="utf-8"))
        if training.get("status") != "pass":
            raise AssertionError(f"Phase-III {variant} training is not PASS")
    if not torch.cuda.is_available():
        raise RuntimeError("Validation architecture selection requires one visible CUDA GPU")
    device = torch.device("cuda:0")
    values = {variant: _validation_predictions(variant, device) for variant in ("d1", "d2")}
    if values["d1"]["condition_ids"] != values["d2"]["condition_ids"]:
        raise AssertionError("D1/D2 validation condition order differs")
    if values["d1"]["membership_sha256"] != values["d2"]["membership_sha256"]:
        raise AssertionError("D1/D2 validation physical-cell membership differs")
    if not np.array_equal(values["d1"]["target"], values["d2"]["target"]):
        raise AssertionError("D1/D2 validation GT differs")
    target = values["d1"]["target"]
    true_valid = np.var(target, axis=1) > 0.0
    true_zero = int((~true_valid).sum())
    metrics: dict[str, Any] = {}
    model_induced: dict[str, int] = {}
    for variant in ("d1", "d2"):
        prediction = values[variant]["prediction"]
        if not np.isfinite(prediction).all():
            raise AssertionError(f"{variant} has non-finite validation predictions")
        pred_constant = np.var(prediction, axis=1) == 0.0
        model_induced[variant] = int(np.count_nonzero(true_valid & pred_constant))
        correlations = np.asarray(
            [_pearson(prediction[index], target[index]) for index in range(VAL_CONDITIONS)]
        )
        shared_values = correlations[true_valid & ~pred_constant]
        metrics[variant] = {
            "macro_mae_all_500": float(np.abs(prediction - target).mean(axis=1).mean()),
            "macro_mse_all_500": float(np.square(prediction - target).mean(axis=1).mean()),
            "macro_pearson_shared_mathematically_valid": float(shared_values.mean())
            if len(shared_values) else None,
            "finite_pearson_n": int(np.isfinite(correlations).sum()),
            "model_induced_undefined_pearson_n": model_induced[variant],
            "prediction_signed": bool(np.any(prediction < 0) and np.any(prediction > 0)),
            "membership_sha256": values[variant]["membership_sha256"],
        }
    fallback = any(model_induced.values())
    if fallback:
        selection_metric = "lower validation macro MAE across all 500 conditions"
        selected = min(("d1", "d2"), key=lambda name: metrics[name]["macro_mae_all_500"])
    else:
        selection_metric = "higher validation macro Pearson delta on shared valid conditions; lower MAE tie-break"
        left = metrics["d1"]["macro_pearson_shared_mathematically_valid"]
        right = metrics["d2"]["macro_pearson_shared_mathematically_valid"]
        if left is None or right is None:
            raise AssertionError("No mathematically valid validation Pearson conditions")
        if left == right:
            selected = min(("d1", "d2"), key=lambda name: metrics[name]["macro_mae_all_500"])
        else:
            selected = "d1" if left > right else "d2"
    result = {
        "schema": "phase3_decoder_selection_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "validation_conditions": VAL_CONDITIONS,
        "sampling_epoch": 0,
        "shared_mathematically_valid_pearson_n": int(true_valid.sum()),
        "true_delta_zero_variance_n": true_zero,
        "d1_model_induced_undefined_pearson_n": model_induced["d1"],
        "d2_model_induced_undefined_pearson_n": model_induced["d2"],
        "nonfinite_prediction_n": 0,
        "fallback_triggered": fallback,
        "selection_metric_actually_used": selection_metric,
        "metrics": metrics,
        "selected_architecture": selected,
        "test_used_for_selection": False,
        "protocol": artifact(PROTOCOL_PATH),
        "checkpoints": {
            variant: artifact(variant_paths(variant)["best"]) for variant in ("d1", "d2")
        },
        "phase3_complete": False,
        "phase4_started": False,
    }
    atomic_json(SELECTION_PATH, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--rewrite", action="store_true")
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--overwrite", action="store_true")
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--variant", choices=("d1", "d2"), required=True)
    train_parser.add_argument("--resume-last", action="store_true")
    commands.add_parser("select")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args.rewrite)
    elif args.command == "smoke":
        run_smoke(args.overwrite)
    elif args.command == "train":
        run_train(args.variant, args.resume_last)
    else:
        select_architecture()


if __name__ == "__main__":
    main()
