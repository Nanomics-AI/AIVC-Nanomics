#!/usr/bin/env python3
"""Prepare, smoke, train, and latent-evaluate the frozen Phase-II ST-A v2."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
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
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from evaluate_tahoe_experiment1_b0 import aggregate
from phase2_stav2_dataset import Phase2STAv2Dataset, make_dataloader
from phase2_stav2_model import (
    DRUG_COUNT,
    EXPECTED_TOTAL_PARAMETERS,
    EXPECTED_TRAINABLE_PARAMETERS,
    LATENT_DIM,
    SET_SIZE,
    architecture_audit,
    build_stav2,
    model_kwargs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
SCRIPT_PATH = Path(__file__).resolve()
TASK_PATH = PROJECT_ROOT.parent / "Phase II.md"
SUBSET_MANIFEST = RESULTS / "phase2_stav2_subset_manifest.json"
CACHE_PLAN = RESULTS / "phase2_stav2_cache_plan.json"
DOSE_NORMALIZATION = RESULTS / "phase2_stav2_dose_normalization.json"
SAMPLING_AUDIT = RESULTS / "phase2_stav2_dynamic_sampling_audit.json"
EMBEDDING_MANIFEST = RESULTS / "phase2_author_genejepa_embedding_manifest.json"
DATASET_PATH = PROJECT_ROOT / "perturbation_scripts/phase2_stav2_dataset.py"
MODEL_PATH = PROJECT_ROOT / "perturbation_scripts/phase2_stav2_model.py"
ARCHITECTURE_AUDIT = RESULTS / "phase2_stav2_architecture_audit.json"
PROTOCOL_PATH = RESULTS / "phase2_stav2_training_protocol.json"
SMOKE_PATH = RESULTS / "phase2_stav2_exact_config_smoke.json"
CHECKPOINT_DIR = RESULTS / "phase2_stav2_checkpoints"
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"
LAST_CHECKPOINT = CHECKPOINT_DIR / "last.pt"
TRAINING_RESULT = RESULTS / "phase2_stav2_training_result.json"
TENSORBOARD_DIR = RESULTS / "tensorboard/phase2_stav2"
LATENT_RESULT = RESULTS / "phase2_stav2_latent_evaluation.json"
LATENT_REPEATS = RESULTS / "phase2_stav2_latent_evaluation_repeats.csv"
LATENT_CONDITIONS = RESULTS / "phase2_stav2_latent_evaluation_conditions.csv"
TEST_SAMPLING_INDICES = RESULTS / "phase2_stav2_test_sampling_indices.npz"

SEED = 42
WORLD_SIZE = 2
BATCH_SIZE_PER_GPU = 64
GRADIENT_ACCUMULATION_STEPS = 2
EFFECTIVE_GLOBAL_BATCH = WORLD_SIZE * BATCH_SIZE_PER_GPU * GRADIENT_ACCUMULATION_STEPS
NUM_WORKERS_PER_RANK = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
ENERGY_BLUR = 0.05
FINAL_REPEAT_EPOCHS = [0, 1, 2, 3, 4]
SMOKE_MICROBATCHES = 4


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
        return str(resolved)


def artifact(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[str, Any] = {"path": relative(path), "size_bytes": path.stat().st_size}
    if hash_file:
        result["sha256"] = sha256_file(path)
    return result


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", lineterminator="\n")
    os.replace(temporary, path)


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_core(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def protocol_payload(architecture: dict[str, Any]) -> dict[str, Any]:
    subset = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))
    dose = json.loads(DOSE_NORMALIZATION.read_text(encoding="utf-8"))
    return {
        "schema": "phase2_stav2_training_protocol_v1",
        "created_at_utc": utc_now(),
        "status": "frozen",
        "phase": "Task 2 / Phase II",
        "task": artifact(TASK_PATH),
        "data": {
            "subset_manifest": artifact(SUBSET_MANIFEST),
            "cache_plan": artifact(CACHE_PLAN),
            "dynamic_sampling_audit": artifact(SAMPLING_AUDIT),
            "conditions": subset["counts"]["conditions_by_split"],
            "cell_lines": subset["selection"]["selected_cell_lines"],
            "treated_pool_cap": 512,
            "control_pool_cap": 512,
            "set_size": SET_SIZE,
            "training_sampling_epoch": "training epoch",
            "validation_sampling_epoch": 0,
            "final_test_sampling_epochs": FINAL_REPEAT_EPOCHS,
            "sampling_seed": SEED,
            "latent_transforms": [],
        },
        "conditioning": {
            "drug": "Embedding(379,768)",
            "dose": dose,
            "formula": "Zinput = raw_Zcontrol + drug_embedding(drug_id) * dose_scaled",
            "independent_380d_perturbation_encoder": "removed",
            "double_conditioning": False,
        },
        "model": {
            "class": "local minimal Phase2STAv2 subclass of official StateTransitionPerturbationModel",
            "kwargs": model_kwargs(),
            "architecture_audit": architecture,
            "absolute_prediction": True,
            "output_activation": "identity",
            "output_shape": ["B", SET_SIZE, LATENT_DIM],
        },
        "loss": {"class": "geomloss.SamplesLoss", "loss": "energy", "blur": ENERGY_BLUR},
        "optimizer": {
            "class": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "scheduler": None,
        },
        "training": {
            "seed": SEED,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "precision": "BF16 model forward + FP32 Energy",
            "world_size": WORLD_SIZE,
            "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_global_batch": EFFECTIVE_GLOBAL_BATCH,
            "num_workers_per_rank": NUM_WORKERS_PER_RANK,
            "validation_metric": "edge-level mean Energy",
            "test_used_for_checkpoint_selection": False,
        },
        "implementation": {
            "runner": artifact(SCRIPT_PATH),
            "dataset": artifact(DATASET_PATH),
            "model": artifact(MODEL_PATH),
            "reused_sampler": "TahoeExperiment1LatentSetDataset._sample_range",
            "reused_dataloader": "tahoe_experiment1_latent_data.make_dataloader",
            "reused_aggregation": "evaluate_tahoe_experiment1_b0.aggregate",
        },
        "phase3_started": False,
    }


def prepare(rewrite: bool) -> dict[str, Any]:
    if ARCHITECTURE_AUDIT.exists() and not rewrite:
        raise FileExistsError(f"Architecture audit exists: {ARCHITECTURE_AUDIT}")
    model, _ = build_stav2(SEED)
    architecture = architecture_audit(model)
    if architecture["status"] != "pass":
        raise AssertionError(architecture)
    result = {
        "schema": "phase2_stav2_architecture_audit_v1",
        "created_at_utc": utc_now(),
        **architecture,
        "conditioning": "Embedding(379,768) * dose_scaled + control",
        "independent_perturbation_encoder": "removed",
        "source": artifact(MODEL_PATH),
    }
    atomic_json(ARCHITECTURE_AUDIT, result)
    protocol = protocol_payload(result)
    atomic_json(PROTOCOL_PATH, protocol)
    del model
    payload = {
        "status": "pass",
        "architecture_audit": artifact(ARCHITECTURE_AUDIT),
        "training_protocol": artifact(PROTOCOL_PATH),
        "formal_embedding_ready": EMBEDDING_MANIFEST.is_file(),
        "formal_training_started": False,
        "phase3_started": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def verify_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    architecture = json.loads(ARCHITECTURE_AUDIT.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen" or architecture.get("status") != "pass":
        raise AssertionError("Phase-II protocol or architecture audit is not ready")
    for key, expected in (
        ("total_parameters", EXPECTED_TOTAL_PARAMETERS),
        ("trainable_parameters", EXPECTED_TRAINABLE_PARAMETERS),
    ):
        if architecture.get(key) != expected:
            raise AssertionError(f"Architecture parameter contract changed: {key}")
    implementation = protocol["implementation"]
    for name, path in (("runner", SCRIPT_PATH), ("dataset", DATASET_PATH), ("model", MODEL_PATH)):
        if implementation[name]["sha256"] != sha256_file(path):
            raise AssertionError(f"Frozen Phase-II {name} source changed; rerun prepare --rewrite")
    return protocol


def setup_distributed() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != WORLD_SIZE:
        raise RuntimeError("Phase-II smoke/training requires torchrun with exactly 2 ranks")
    if os.environ.get("NCCL_CUMEM_ENABLE") != "0" or os.environ.get("NCCL_CUMEM_HOST_ENABLE") != "0":
        raise RuntimeError("This host requires NCCL_CUMEM_ENABLE=0 and NCCL_CUMEM_HOST_ENABLE=0")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, local_rank, torch.device("cuda", local_rank)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in ("ctrl_cell_emb", "pert_cell_emb", "drug_id", "dose_scaled")
    }


def forward_energy(
    model: nn.Module, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = batch["ctrl_cell_emb"].shape[0]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction = model(batch).reshape(batch_size, SET_SIZE, LATENT_DIM)
    prediction_loss = prediction.float()
    target = batch["pert_cell_emb"].reshape(batch_size, SET_SIZE, LATENT_DIM).float()
    per_set = model_core(model)._compute_distribution_loss(prediction_loss, target).reshape(-1)
    if per_set.shape != (batch_size,) or not torch.isfinite(per_set).all():
        raise AssertionError("Phase-II Energy is not one finite value per condition")
    return prediction, per_set.mean(), per_set


def branch_gradient_audit(core: nn.Module) -> dict[str, Any]:
    groups = {
        "drug_embedding": [core.drug_embedding.weight],
        "transformer": list(core.transformer_backbone.parameters()),
        "project_out": list(core.project_out.parameters()),
    }
    result: dict[str, Any] = {}
    for name, parameters in groups.items():
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        finite = bool(gradients) and all(torch.isfinite(value).all().item() for value in gradients)
        norm = math.sqrt(sum(float(value.detach().float().square().sum()) for value in gradients)) if gradients else 0.0
        result[name] = {"has_gradient": bool(gradients), "finite": finite, "norm": norm}
    return result


def parameter_sync_error(model: nn.Module) -> float:
    core = model_core(model)
    moments = torch.stack(
        [
            torch.stack([parameter.detach().double().sum(), parameter.detach().double().square().sum()])
            for parameter in core.parameters()
        ]
    ).sum(dim=0)
    gathered = [torch.zeros_like(moments) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered, moments)
    return max(float((value - gathered[0]).abs().max()) for value in gathered)


def sampling_smoke(dataset: Phase2STAv2Dataset) -> dict[str, Any]:
    candidates = dataset.conditions.loc[
        dataset.conditions["treated_cached_cell_count"].gt(SET_SIZE)
        & dataset.conditions["control_cached_cell_count"].gt(SET_SIZE)
    ]
    if candidates.empty:
        raise AssertionError("No pool>256 train condition is available")
    local_index = int(candidates.index[0])
    dataset.set_epoch(0)
    epoch0_a = dataset[local_index]
    epoch0_b = dataset[local_index]
    dataset.set_epoch(1)
    epoch1 = dataset[local_index]
    dataset.set_epoch(0)
    return {
        "pair_id": epoch0_a["condition_id"],
        "epoch0_reproducible_control": bool(torch.equal(epoch0_a["source_embedding_index"], epoch0_b["source_embedding_index"])),
        "epoch0_reproducible_treated": bool(torch.equal(epoch0_a["target_embedding_index"], epoch0_b["target_embedding_index"])),
        "epoch0_vs_epoch1_control_changed": bool(not torch.equal(epoch0_a["source_embedding_index"], epoch1["source_embedding_index"])),
        "epoch0_vs_epoch1_treated_changed": bool(not torch.equal(epoch0_a["target_embedding_index"], epoch1["target_embedding_index"])),
        "within_set_unique": bool(
            len(torch.unique(epoch0_a["source_embedding_index"])) == SET_SIZE
            and len(torch.unique(epoch0_a["target_embedding_index"])) == SET_SIZE
        ),
    }


def run_smoke(overwrite: bool) -> dict[str, Any] | None:
    rank, local_rank, device = setup_distributed()
    try:
        verify_protocol()
        if rank == 0 and SMOKE_PATH.exists() and not overwrite:
            raise FileExistsError(SMOKE_PATH)
        seed_everything(SEED)
        dataset = Phase2STAv2Dataset(split="train", seed=SEED, epoch=0)
        sampling = sampling_smoke(dataset)
        sampler = DistributedSampler(
            dataset, num_replicas=WORLD_SIZE, rank=rank, shuffle=True, seed=SEED, drop_last=True
        )
        loader = make_dataloader(
            dataset,
            batch_size=BATCH_SIZE_PER_GPU,
            shuffle=False,
            seed=SEED,
            sampler=sampler,
            drop_last=True,
            num_workers=NUM_WORKERS_PER_RANK,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2,
        )
        raw_model, _ = build_stav2(SEED)
        architecture = architecture_audit(raw_model)
        if architecture["status"] != "pass":
            raise AssertionError(architecture)
        raw_model.to(device)
        model = DistributedDataParallel(raw_model, device_ids=[local_rank], output_device=local_rank)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        transformer_probe = next(
            parameter
            for parameter in raw_model.transformer_backbone.parameters()
            if parameter.requires_grad
        )
        before = {
            "drug_embedding": raw_model.drug_embedding.weight.detach().clone(),
            "transformer": transformer_probe.detach().clone(),
            "project_out": next(raw_model.project_out.parameters()).detach().clone(),
        }
        dataset.set_epoch(0)
        sampler.set_epoch(0)
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        gradient_audit: dict[str, Any] = {}
        prediction_audit: dict[str, Any] = {}
        started = time.perf_counter()
        for microbatch, raw_batch in enumerate(loader):
            if microbatch >= SMOKE_MICROBATCHES:
                break
            batch = move_batch(raw_batch, device)
            synchronize = (microbatch + 1) % GRADIENT_ACCUMULATION_STEPS == 0
            sync_context = contextlib.nullcontext() if synchronize else model.no_sync()
            with sync_context:
                prediction, loss, _ = forward_energy(model, batch)
                (loss / GRADIENT_ACCUMULATION_STEPS).backward()
            losses.append(float(loss.detach()))
            if not prediction_audit:
                core = model_core(model)
                conditioned, drug, drug_dose = core.conditioning(
                    batch["ctrl_cell_emb"], batch["drug_id"], batch["dose_scaled"]
                )
                zero_conditioned, _, zero_drug_dose = core.conditioning(
                    batch["ctrl_cell_emb"], batch["drug_id"], torch.zeros_like(batch["dose_scaled"])
                )
                prediction_audit = {
                    "ctrl_shape": list(batch["ctrl_cell_emb"].shape),
                    "target_shape": list(batch["pert_cell_emb"].shape),
                    "drug_id_shape": list(batch["drug_id"].shape),
                    "dose_scaled_shape": list(batch["dose_scaled"].shape),
                    "drug_embedding_shape": list(drug.shape),
                    "drug_dose_shape": list(drug_dose.shape),
                    "conditioned_shape": list(conditioned.shape),
                    "prediction_shape": list(prediction.shape),
                    "prediction_finite": bool(torch.isfinite(prediction).all()),
                    "prediction_signed": bool(torch.any(prediction < 0) and torch.any(prediction > 0)),
                    "dose_zero_drug_dose_exact_zero": bool(torch.count_nonzero(zero_drug_dose) == 0),
                    "dose_zero_zinput_equals_control": bool(torch.equal(zero_conditioned, batch["ctrl_cell_emb"])),
                }
            if synchronize:
                gradient_audit = branch_gradient_audit(model_core(model))
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), GRADIENT_CLIP_NORM, error_if_nonfinite=True
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if len(losses) != SMOKE_MICROBATCHES:
            raise AssertionError("Smoke did not execute the frozen number of microbatches")
        updates = {
            name: bool(not torch.equal(before[name], parameter.detach()))
            for name, parameter in (
                ("drug_embedding", raw_model.drug_embedding.weight),
                ("transformer", transformer_probe),
                ("project_out", next(raw_model.project_out.parameters())),
            )
        }
        sync_error = parameter_sync_error(model)
        local = {
            "rank": rank,
            "gpu": torch.cuda.get_device_name(device),
            "peak_allocated_GiB": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_GiB": torch.cuda.max_memory_reserved(device) / 2**30,
            "losses": losses,
        }
        gathered: list[dict[str, Any] | None] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, local)
        if rank != 0:
            return None
        checks = {
            "architecture": architecture["status"] == "pass",
            "real_data_shapes": prediction_audit.get("prediction_shape")
            == [BATCH_SIZE_PER_GPU, SET_SIZE, LATENT_DIM],
            "prediction_finite_signed": prediction_audit.get("prediction_finite") is True
            and prediction_audit.get("prediction_signed") is True,
            "loss_finite": all(math.isfinite(value) for item in gathered if item for value in item["losses"]),
            "gradients_finite_nonzero": all(
                value["has_gradient"] and value["finite"] and value["norm"] > 0
                for value in gradient_audit.values()
            ),
            "parameters_updated": all(updates.values()),
            "ddp_synchronized": sync_error == 0.0,
            "dose_zero": prediction_audit.get("dose_zero_drug_dose_exact_zero") is True
            and prediction_audit.get("dose_zero_zinput_equals_control") is True,
            "dynamic_sampling": all(sampling.values()),
        }
        result = {
            "schema": "phase2_stav2_exact_config_smoke_v1",
            "created_at_utc": utc_now(),
            "status": "pass" if all(checks.values()) else "fail",
            "scientific_result": False,
            "formal_training_started": False,
            "world_size": WORLD_SIZE,
            "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "optimizer_updates": SMOKE_MICROBATCHES // GRADIENT_ACCUMULATION_STEPS,
            "architecture": architecture,
            "sampling": sampling,
            "forward": prediction_audit,
            "gradients": gradient_audit,
            "parameters_updated": updates,
            "losses": losses,
            "ddp_parameter_moment_max_abs_error": sync_error,
            "rank_runtime": gathered,
            "elapsed_seconds": elapsed,
            "checks": checks,
            "protocol": artifact(PROTOCOL_PATH),
            "embedding_manifest": artifact(EMBEDDING_MANIFEST),
            "phase3_started": False,
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
def validate(model: nn.Module, dataset: Phase2STAv2Dataset, device: torch.device) -> dict[str, Any]:
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
    metadata = dataset.conditions.set_index("pair_id")
    records: list[dict[str, Any]] = []
    observed: list[str] = []
    model.eval()
    for raw_batch in loader:
        ids = list(raw_batch["condition_id"])
        observed.extend(ids)
        batch = move_batch(raw_batch, device)
        _, _, per_set = forward_energy(model, batch)
        for condition_id, value in zip(ids, per_set.cpu().numpy(), strict=True):
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
    expected = dataset.conditions["pair_id"].astype(str).tolist()
    if observed != expected:
        raise AssertionError("Validation was shuffled or dropped")
    raw = pd.DataFrame.from_records(records)
    edges, details = aggregate(raw)
    metric = float(edges["energy_distance"].mean())
    if len(raw) != len(dataset) or not math.isfinite(metric):
        raise AssertionError("Validation completeness/metric failed")
    return {
        "condition_energy_mean": float(raw["energy_distance"].mean()),
        "edge_energy_mean": metric,
        "conditions": len(raw),
        "edges": len(edges),
        "plate6_plate14_high_dose_replicate_groups": int(
            details["replicate_audit"].iloc[0]["plate6_plate14_groups"]
        ),
    }


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
    kind: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_metric: float,
    best_epoch: int,
    no_improvement: int,
    history: list[dict[str, Any]],
    rng_states: list[dict[str, Any]],
    terminal: bool,
) -> dict[str, Any]:
    return {
        "schema": "phase2_stav2_checkpoint_v1",
        "kind": kind,
        "created_at_utc": utc_now(),
        "model_state_dict": model_core(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_edge_energy": best_metric,
        "best_epoch": best_epoch,
        "no_improvement_epochs": no_improvement,
        "history": history,
        "rng_states": rng_states,
        "terminal": terminal,
        "seed": SEED,
        "model_kwargs": model_kwargs(),
        "protocol": artifact(PROTOCOL_PATH),
        "subset_manifest": artifact(SUBSET_MANIFEST),
        "cache_plan": artifact(CACHE_PLAN),
        "dose_normalization": artifact(DOSE_NORMALIZATION),
        "author_embedding_manifest": artifact(EMBEDDING_MANIFEST),
        "runner": artifact(SCRIPT_PATH),
        "dataset": artifact(DATASET_PATH),
        "model": artifact(MODEL_PATH),
        "phase3_started": False,
    }


def run_train(resume: bool) -> dict[str, Any] | None:
    rank, local_rank, device = setup_distributed()
    writer = None
    try:
        verify_protocol()
        smoke = json.loads(SMOKE_PATH.read_text(encoding="utf-8"))
        if smoke.get("status") != "pass":
            raise AssertionError("Exact-config real-data smoke is not PASS")
        if rank == 0:
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
            if not resume and (BEST_CHECKPOINT.exists() or LAST_CHECKPOINT.exists() or TRAINING_RESULT.exists()):
                raise FileExistsError("Phase-II formal outputs exist; pass --resume-last")
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=str(TENSORBOARD_DIR))
        seed_everything(SEED)
        train_dataset = Phase2STAv2Dataset(split="train", seed=SEED, epoch=0)
        val_dataset = Phase2STAv2Dataset(split="val", seed=SEED, epoch=0) if rank == 0 else None
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=WORLD_SIZE,
            rank=rank,
            shuffle=True,
            seed=SEED,
            drop_last=True,
        )
        loader = make_dataloader(
            train_dataset,
            batch_size=BATCH_SIZE_PER_GPU,
            shuffle=False,
            seed=SEED,
            sampler=sampler,
            drop_last=True,
            num_workers=NUM_WORKERS_PER_RANK,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2,
        )
        usable_microbatches = (len(loader) // GRADIENT_ACCUMULATION_STEPS) * GRADIENT_ACCUMULATION_STEPS
        if usable_microbatches < GRADIENT_ACCUMULATION_STEPS:
            raise AssertionError("Training loader is too short")
        raw_model, _ = build_stav2(SEED)
        if architecture_audit(raw_model)["status"] != "pass":
            raise AssertionError("Runtime model architecture changed")
        raw_model.to(device)
        model = DistributedDataParallel(raw_model, device_ids=[local_rank], output_device=local_rank)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        start_epoch = 0
        global_step = 0
        best_metric = float("inf")
        best_epoch = -1
        no_improvement = 0
        history: list[dict[str, Any]] = []
        if resume:
            checkpoint = torch.load(LAST_CHECKPOINT, map_location="cpu", weights_only=False)
            if checkpoint.get("schema") != "phase2_stav2_checkpoint_v1" or checkpoint.get("kind") != "last":
                raise AssertionError("Not a Phase-II ST-A v2 last checkpoint")
            raw_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = int(checkpoint["epoch"]) + 1
            global_step = int(checkpoint["global_step"])
            best_metric = float(checkpoint["best_val_edge_energy"])
            best_epoch = int(checkpoint["best_epoch"])
            no_improvement = int(checkpoint["no_improvement_epochs"])
            history = list(checkpoint["history"])
            restore_rng(checkpoint["rng_states"][rank])
            if checkpoint.get("terminal"):
                raise RuntimeError("Last checkpoint is already terminal")
        dist.barrier()
        training_started = time.perf_counter()
        stop_reason = "max_epochs"
        rank_peak_allocated = 0.0
        rank_peak_reserved = 0.0
        for epoch in range(start_epoch, MAX_EPOCHS):
            train_dataset.set_epoch(epoch)
            sampler.set_epoch(epoch)
            model.train()
            torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            epoch_energy_sum = 0.0
            epoch_conditions = 0
            epoch_started = time.perf_counter()
            optimizer_updates = 0
            for microbatch, raw_batch in enumerate(loader):
                if microbatch >= usable_microbatches:
                    break
                batch = move_batch(raw_batch, device)
                synchronize = (microbatch + 1) % GRADIENT_ACCUMULATION_STEPS == 0
                sync_context = contextlib.nullcontext() if synchronize else model.no_sync()
                with sync_context:
                    _, loss, per_set = forward_energy(model, batch)
                    (loss / GRADIENT_ACCUMULATION_STEPS).backward()
                epoch_energy_sum += float(per_set.detach().sum())
                epoch_conditions += len(per_set)
                if synchronize:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), GRADIENT_CLIP_NORM, error_if_nonfinite=True
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    optimizer_updates += 1
                    if rank == 0 and writer is not None:
                        writer.add_scalar("train/energy_step", float(loss.detach()), global_step)
                        writer.add_scalar("train/grad_norm_before_clip", float(grad_norm), global_step)
            torch.cuda.synchronize(device)
            rank_peak_allocated = max(rank_peak_allocated, torch.cuda.max_memory_allocated(device) / 2**30)
            rank_peak_reserved = max(rank_peak_reserved, torch.cuda.max_memory_reserved(device) / 2**30)
            totals = torch.tensor([epoch_energy_sum, epoch_conditions], device=device, dtype=torch.float64)
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            train_energy = float(totals[0] / totals[1])
            dist.barrier()
            validation: dict[str, Any] | None = None
            if rank == 0:
                assert val_dataset is not None
                validation = validate(raw_model, val_dataset, device)
            payload: list[Any] = [validation]
            dist.broadcast_object_list(payload, src=0)
            validation = payload[0]
            val_metric = float(validation["edge_energy_mean"])
            improved = val_metric < best_metric
            if improved:
                best_metric = val_metric
                best_epoch = epoch
                no_improvement = 0
            else:
                no_improvement += 1
            epoch_elapsed = time.perf_counter() - epoch_started
            row = {
                "epoch": epoch,
                "optimizer_updates": optimizer_updates,
                "global_step": global_step,
                "train_energy_mean": train_energy,
                "val_condition_energy_mean": float(validation["condition_energy_mean"]),
                "val_edge_energy_mean": val_metric,
                "val_edges": int(validation["edges"]),
                "improved": improved,
                "best_epoch": best_epoch,
                "best_val_edge_energy": best_metric,
                "no_improvement_epochs": no_improvement,
                "elapsed_seconds": epoch_elapsed,
            }
            history.append(row)
            terminal = epoch + 1 >= MAX_EPOCHS or no_improvement >= EARLY_STOPPING_PATIENCE
            if no_improvement >= EARLY_STOPPING_PATIENCE:
                stop_reason = "early_stopping_patience_5"
            rng = capture_rng()
            rng_states: list[dict[str, Any] | None] = [None] * WORLD_SIZE
            dist.all_gather_object(rng_states, rng)
            if rank == 0:
                common = dict(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    best_metric=best_metric,
                    best_epoch=best_epoch,
                    no_improvement=no_improvement,
                    history=history,
                    rng_states=[value for value in rng_states if value is not None],
                    terminal=terminal,
                )
                if improved:
                    atomic_torch(BEST_CHECKPOINT, checkpoint_payload(kind="best", **common))
                atomic_torch(LAST_CHECKPOINT, checkpoint_payload(kind="last", **common))
                if writer is not None:
                    writer.add_scalar("train/energy_epoch", train_energy, epoch)
                    writer.add_scalar("val/condition_energy", validation["condition_energy_mean"], epoch)
                    writer.add_scalar("val/edge_energy", val_metric, epoch)
                    writer.add_scalar("best/val_edge_energy", best_metric, epoch)
                    writer.add_scalar("best/epoch", best_epoch, epoch)
                    writer.flush()
                print(
                    f"phase2 epoch={epoch} step={global_step} train_energy={train_energy:.8f} "
                    f"val_edge_energy={val_metric:.8f} best_epoch={best_epoch} "
                    f"early_stop_counter={no_improvement}",
                    flush=True,
                )
            dist.barrier()
            if terminal:
                break
        sync_error = parameter_sync_error(model)
        local_runtime = {
            "rank": rank,
            "peak_allocated_GiB": rank_peak_allocated,
            "peak_reserved_GiB": rank_peak_reserved,
        }
        runtimes: list[dict[str, Any] | None] = [None] * WORLD_SIZE
        dist.all_gather_object(runtimes, local_runtime)
        if rank != 0:
            return None
        elapsed = time.perf_counter() - training_started
        result = {
            "schema": "phase2_stav2_training_result_v1",
            "created_at_utc": utc_now(),
            "status": "pass",
            "epochs_completed": len(history),
            "stop_reason": stop_reason,
            "best_epoch": best_epoch,
            "best_val_edge_energy": best_metric,
            "optimizer_steps": global_step,
            "effective_global_batch": EFFECTIVE_GLOBAL_BATCH,
            "training_conditions": len(train_dataset),
            "microbatches_per_epoch_used": usable_microbatches,
            "conditions_used_per_epoch": usable_microbatches * BATCH_SIZE_PER_GPU * WORLD_SIZE,
            "history": history,
            "elapsed_seconds": elapsed,
            "rank_runtime": runtimes,
            "ddp_parameter_moment_max_abs_error": sync_error,
            "checkpoints": {"best": artifact(BEST_CHECKPOINT), "last": artifact(LAST_CHECKPOINT)},
            "protocol": artifact(PROTOCOL_PATH),
            "smoke": artifact(SMOKE_PATH),
            "test_dataset_constructed_during_training": False,
            "phase3_started": False,
        }
        if sync_error != 0.0:
            raise AssertionError("Formal DDP parameters are not synchronized")
        atomic_json(TRAINING_RESULT, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    finally:
        if writer is not None:
            writer.close()
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def cosine_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    numerator = (left * right).sum(dim=-1)
    denominator = left.norm(dim=-1) * right.norm(dim=-1)
    return numerator / denominator.clamp_min(1e-12)


def descriptive(values: pd.Series | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


@torch.inference_mode()
def evaluate_latent(batch_size: int) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or torch.cuda.device_count() != 1:
        raise RuntimeError("Latent evaluation requires CUDA_VISIBLE_DEVICES=0")
    for path in (LATENT_RESULT, LATENT_REPEATS, LATENT_CONDITIONS, TEST_SAMPLING_INDICES):
        if path.exists():
            raise FileExistsError(path)
    verify_protocol()
    training = json.loads(TRAINING_RESULT.read_text(encoding="utf-8"))
    if training.get("status") != "pass":
        raise AssertionError("Formal Phase-II training result is not PASS")
    checkpoint = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "phase2_stav2_checkpoint_v1" or checkpoint.get("kind") != "best":
        raise AssertionError("Invalid Phase-II best checkpoint")
    device = torch.device("cuda:0")
    model, _ = build_stav2(SEED)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().requires_grad_(False).to(device)
    dataset = Phase2STAv2Dataset(split="test", seed=SEED, epoch=0)
    metadata = dataset.conditions.set_index("pair_id")
    rows = len(dataset) * len(FINAL_REPEAT_EPOCHS)
    control_indices = np.empty((rows, SET_SIZE), dtype=np.int64)
    treated_indices = np.empty((rows, SET_SIZE), dtype=np.int64)
    condition_ids: list[str] = []
    repeat_ids = np.empty(rows, dtype=np.int8)
    records: list[dict[str, Any]] = []
    cursor = 0
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for repeat_epoch in FINAL_REPEAT_EPOCHS:
        dataset.set_epoch(repeat_epoch)
        loader = make_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            seed=SEED,
            drop_last=False,
            num_workers=NUM_WORKERS_PER_RANK,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2,
        )
        observed: list[str] = []
        for raw_batch in loader:
            ids = list(raw_batch["condition_id"])
            observed.extend(ids)
            batch = move_batch(raw_batch, device)
            prediction, _, per_set = forward_energy(model, batch)
            control = batch["ctrl_cell_emb"].float()
            target = batch["pert_cell_emb"].float()
            delta_pred = prediction.float().mean(dim=1) - control.mean(dim=1)
            delta_true = target.mean(dim=1) - control.mean(dim=1)
            cosines = cosine_rows(delta_pred, delta_true).cpu().numpy()
            errors = (delta_pred - delta_true).norm(dim=-1).cpu().numpy()
            energies = per_set.cpu().numpy()
            source = raw_batch["source_embedding_index"].numpy().astype(np.int64)
            treated = raw_batch["target_embedding_index"].numpy().astype(np.int64)
            count = len(ids)
            control_indices[cursor : cursor + count] = source
            treated_indices[cursor : cursor + count] = treated
            repeat_ids[cursor : cursor + count] = repeat_epoch
            condition_ids.extend(ids)
            for condition_id, energy, cosine, error in zip(
                ids, energies, cosines, errors, strict=True
            ):
                row = metadata.loc[condition_id]
                records.append(
                    {
                        "condition_id": condition_id,
                        "edge_id": str(row["edge_id"]),
                        "cell_line_id": str(row["cell_line_id"]),
                        "drug": str(row["drug"]),
                        "dose_uM": float(row["dose_uM"]),
                        "plate": str(row["plate"]),
                        "repeat_epoch": repeat_epoch,
                        "energy_distance": float(energy),
                        "centroid_delta_cosine": float(cosine),
                        "centroid_delta_l2_error": float(error),
                    }
                )
            cursor += count
        if observed != dataset.conditions["pair_id"].astype(str).tolist():
            raise AssertionError(f"Test repeat {repeat_epoch} was shuffled or dropped")
        print(f"phase2 latent repeat={repeat_epoch} conditions={len(observed)}", flush=True)
    if cursor != rows:
        raise AssertionError("Latent evaluation row count mismatch")
    raw = pd.DataFrame.from_records(records)
    if raw.duplicated(["condition_id", "repeat_epoch"]).any() or not np.isfinite(
        raw[["energy_distance", "centroid_delta_cosine", "centroid_delta_l2_error"]]
    ).all().all():
        raise AssertionError("Latent condition/repeat output is duplicate or non-finite")
    atomic_csv(LATENT_REPEATS, raw)
    condition_aggregation = (
        raw.groupby(
            ["condition_id", "edge_id", "cell_line_id", "drug", "dose_uM", "plate"],
            as_index=False,
            sort=False,
        )[["energy_distance", "centroid_delta_cosine", "centroid_delta_l2_error"]]
        .mean()
    )
    atomic_csv(LATENT_CONDITIONS, condition_aggregation)
    temporary_npz = TEST_SAMPLING_INDICES.with_name(TEST_SAMPLING_INDICES.stem + ".tmp.npz")
    np.savez_compressed(
        temporary_npz,
        condition_id=np.asarray(condition_ids),
        repeat_epoch=repeat_ids,
        control_embedding_index=control_indices,
        treated_embedding_index=treated_indices,
    )
    os.replace(temporary_npz, TEST_SAMPLING_INDICES)
    repeat_summaries: list[dict[str, Any]] = []
    for repeat_epoch in FINAL_REPEAT_EPOCHS:
        subset = raw.loc[raw["repeat_epoch"].eq(repeat_epoch)]
        edge, _ = aggregate(subset)
        repeat_summaries.append(
            {
                "repeat_epoch": repeat_epoch,
                "conditions": len(subset),
                "edges": len(edge),
                "edge_energy_mean": float(edge["energy_distance"].mean()),
                "condition_energy_mean": float(subset["energy_distance"].mean()),
                "condition_centroid_delta_cosine_mean": float(subset["centroid_delta_cosine"].mean()),
                "condition_centroid_delta_l2_error_mean": float(subset["centroid_delta_l2_error"].mean()),
            }
        )
    final_edges, aggregation = aggregate(raw)
    repeat_energy = np.asarray([row["edge_energy_mean"] for row in repeat_summaries])
    result = {
        "schema": "phase2_stav2_latent_evaluation_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "test_conditions": len(dataset),
        "final_test_repeats": len(FINAL_REPEAT_EPOCHS),
        "sampling_epochs": FINAL_REPEAT_EPOCHS,
        "seed": SEED,
        "set_size": SET_SIZE,
        "raw_rows": len(raw),
        "repeat_summaries": repeat_summaries,
        "edge_energy_5_repeat_mean": float(repeat_energy.mean()),
        "edge_energy_5_repeat_std": float(repeat_energy.std(ddof=1)),
        "final_aggregated_edges": len(final_edges),
        "final_edge_energy": descriptive(final_edges["energy_distance"]),
        "condition_metrics": {
            "energy_distance": descriptive(condition_aggregation["energy_distance"]),
            "centroid_delta_cosine": descriptive(condition_aggregation["centroid_delta_cosine"]),
            "centroid_delta_l2_error": descriptive(condition_aggregation["centroid_delta_l2_error"]),
        },
        "plate6_plate14_rule": "only dose_uM == 5.0 high-dose groups are paired",
        "plate6_plate14_replicate_groups": int(
            aggregation["replicate_audit"].iloc[0]["plate6_plate14_groups"]
        ),
        "sampling_indices": {
            **artifact(TEST_SAMPLING_INDICES),
            "rows": rows,
            "control_shape": list(control_indices.shape),
            "treated_shape": list(treated_indices.shape),
            "same_indices_required_by_decoder_only_and_full_pipeline": True,
        },
        "outputs": {
            "condition_repeat": artifact(LATENT_REPEATS),
            "condition_aggregation": artifact(LATENT_CONDITIONS),
        },
        "best_checkpoint": artifact(BEST_CHECKPOINT),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_GiB": torch.cuda.max_memory_allocated(device) / 2**30,
        "phase3_started": False,
    }
    atomic_json(LATENT_RESULT, result)
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
    train_parser.add_argument("--resume-last", action="store_true")
    evaluate_parser = commands.add_parser("evaluate-latent")
    evaluate_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_PER_GPU)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args.rewrite)
    elif args.command == "smoke":
        run_smoke(args.overwrite)
    elif args.command == "train":
        run_train(args.resume_last)
    else:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        evaluate_latent(args.batch_size)


if __name__ == "__main__":
    main()
