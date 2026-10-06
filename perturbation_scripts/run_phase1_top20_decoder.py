#!/usr/bin/env python3
"""Train, evaluate, and compare the two frozen Phase-I Top20 decoders."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn as nn
from state.tx.models.base import LatentToGeneDecoder
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
TASK = PROJECT_ROOT.parent / "20260927新路线总纲.md"
SUBSET_MANIFEST = RESULTS / "phase1_top20_subset_manifest.json"
SUBSET_CELLS = RESULTS / "phase1_top20_subset_cells.parquet"
SUBSET_CONDITIONS = RESULTS / "phase1_top20_subset_conditions.csv"
SUBSET_CONTEXTS = RESULTS / "phase1_top20_subset_contexts.csv"
PANEL = RESULTS / "phase1_top20_gene_panel.csv"
PANEL_MANIFEST = RESULTS / "phase1_top20_gene_panel.json"
TARGETS = RESULTS / "phase1_top20_targets.npy"
TARGET_MANIFEST = RESULTS / "phase1_top20_target_manifest.json"
OUR_EMBEDDINGS = RESULTS / "phase1_our_genejepa_embeddings.npy"
OUR_MANIFEST = RESULTS / "phase1_our_genejepa_embedding_manifest.json"
AUTHOR_EMBEDDINGS = RESULTS / "phase1_author_genejepa_embeddings.npy"
AUTHOR_MANIFEST = RESULTS / "phase1_author_genejepa_embedding_manifest.json"
SMOKE_RESULT = RESULTS / "phase1_top20_decoder_exact_config_smoke.json"
COMPARISON_CSV = RESULTS / "phase1_genejepa_top20_comparison.csv"
COMPARISON_JSON = RESULTS / "phase1_genejepa_top20_comparison.json"
HANDOFF = RESULTS / "phase1_genejepa_top20_handoff.md"

TASK_SHA256 = "5f2b177f00c309ba6409a9fea907153253a09de73c6efabacf1d42adf6dcae75"
REPRESENTATIONS = {"our": OUR_EMBEDDINGS, "author": AUTHOR_EMBEDDINGS}
MANIFESTS = {"our": OUR_MANIFEST, "author": AUTHOR_MANIFEST}
TOTAL_CELLS = 101_120
LATENT_DIM = 768
GENE_DIM = 20
WORLD_SIZE = 2
BATCH_SIZE_PER_GPU = 1_024
GLOBAL_BATCH = 2_048
NUM_WORKERS_PER_RANK = 4
PREFETCH_FACTOR = 2
SEED = 42
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3
EXPECTED_PARAMETER_COUNT = 2_377_236
CONTROL_PERT = "__CONTROL__"
EVALUATION_LABEL = (
    "ARC VCC 2025 Generalist metrics adapted to the frozen Phase-I Top20 space"
)
DISCLAIMER = "Adapted Tahoe Top20 diagnostics; this is not an official ARC leaderboard result."
PRIMARY_WINNER_METRICS = (
    "DES",
    "PDS",
    "Pearson delta",
    "Spearman logFC",
    "AUPRC",
    "Spearman effect size",
)


def utc_now() -> str:
    from datetime import datetime, timezone

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
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
        float_format="%.10g",
        na_rep="NaN",
    )
    os.replace(temporary, path)


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def artifact(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[str, Any] = {"path": relative(path), "size_bytes": path.stat().st_size}
    if hash_file:
        result["sha256"] = sha256_file(path)
    return result


def rep_dir(representation: str) -> Path:
    return RESULTS / f"phase1_{representation}_top20_decoder"


def rep_paths(representation: str) -> dict[str, Path]:
    root = rep_dir(representation)
    return {
        "root": root,
        "best": root / "best.pt",
        "last": root / "last.pt",
        "config": root / "training_config.json",
        "result": root / "training_result.json",
        "tensorboard": RESULTS / f"tensorboard/phase1_{representation}_top20_decoder",
        "evaluation": RESULTS / f"phase1_{representation}_top20_evaluation.json",
        "condition_metrics": RESULTS / f"phase1_{representation}_top20_condition_metrics.csv",
        "cell_eval": RESULTS / f"phase1_{representation}_top20_cell_eval",
    }


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_decoder() -> LatentToGeneDecoder:
    seed_everything()
    model = LatentToGeneDecoder(
        latent_dim=LATENT_DIM,
        gene_dim=GENE_DIM,
        hidden_dims=[1024, 1024, 512],
        dropout=0.1,
        residual_decoder=False,
    )
    modules = list(model.decoder.children())
    if not modules or not isinstance(modules[-1], nn.ReLU):
        raise AssertionError("STATE decoder terminal ReLU contract changed")
    model.decoder = nn.Sequential(*modules[:-1])
    assert_decoder(model)
    return model


def assert_decoder(model: LatentToGeneDecoder) -> None:
    expected = [
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
    ]
    modules = list(model.decoder.children())
    if len(modules) != len(expected) or any(
        not isinstance(module, kind) for module, kind in zip(modules, expected, strict=True)
    ):
        raise AssertionError("Phase-I Decoder module sequence changed")
    dimensions = [
        (module.in_features, module.out_features)
        for module in modules
        if isinstance(module, nn.Linear)
    ]
    if dimensions != [(768, 1024), (1024, 1024), (1024, 512), (512, 20)]:
        raise AssertionError(f"Phase-I Decoder dimensions changed: {dimensions}")
    if not isinstance(modules[-1], nn.Linear):
        raise AssertionError("Phase-I Decoder does not end in Linear")
    if any(isinstance(module, (nn.ReLU, nn.Softplus, nn.Sigmoid, nn.Tanh)) for module in modules):
        raise AssertionError("Phase-I Decoder contains a forbidden output activation")
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != EXPECTED_PARAMETER_COUNT:
        raise AssertionError(f"Phase-I Decoder parameter count changed: {count}")


def load_contract() -> dict[str, Any]:
    required = (
        TASK,
        SUBSET_MANIFEST,
        SUBSET_CELLS,
        SUBSET_CONDITIONS,
        SUBSET_CONTEXTS,
        PANEL,
        PANEL_MANIFEST,
        TARGETS,
        TARGET_MANIFEST,
        OUR_EMBEDDINGS,
        OUR_MANIFEST,
        AUTHOR_EMBEDDINGS,
        AUTHOR_MANIFEST,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen Phase-I inputs: {missing}")
    if sha256_file(TASK) != TASK_SHA256:
        raise AssertionError("Frozen Task 1 document changed")
    subset = read_json(SUBSET_MANIFEST)
    panel_manifest = read_json(PANEL_MANIFEST)
    target_manifest = read_json(TARGET_MANIFEST)
    manifests = {name: read_json(path) for name, path in MANIFESTS.items()}
    if subset.get("status") != "pass" or subset["counts"]["total_unique_cells"] != TOTAL_CELLS:
        raise AssertionError("Phase-I subset manifest changed")
    if panel_manifest.get("status") != "frozen" or panel_manifest.get("genes") != GENE_DIM:
        raise AssertionError("Phase-I Top20 panel is not frozen")
    if target_manifest.get("status") != "pass":
        raise AssertionError("Phase-I target manifest is not PASS")
    for name, manifest in manifests.items():
        output = manifest.get("output", {})
        row_contract = output.get("row_equals_phase1_cell_index")
        if name == "our":
            row_contract = manifest.get("row_contract") is not None
        if (
            manifest.get("status") != "pass"
            or output.get("shape") != [TOTAL_CELLS, LATENT_DIM]
            or output.get("dtype") != "float32"
            or not row_contract
        ):
            raise AssertionError(f"{name} embedding manifest contract changed")
    arrays = {
        "target": np.load(TARGETS, mmap_mode="r"),
        "our": np.load(OUR_EMBEDDINGS, mmap_mode="r"),
        "author": np.load(AUTHOR_EMBEDDINGS, mmap_mode="r"),
    }
    if arrays["target"].shape != (TOTAL_CELLS, GENE_DIM) or arrays["target"].dtype != np.float32:
        raise AssertionError("Phase-I target array shape/dtype changed")
    for name in REPRESENTATIONS:
        if arrays[name].shape != (TOTAL_CELLS, LATENT_DIM) or arrays[name].dtype != np.float32:
            raise AssertionError(f"{name} embedding array shape/dtype changed")
    cells = pq.read_table(
        SUBSET_CELLS,
        columns=["phase1_cell_index", "split", "role", "stable_physical_locator"],
    ).to_pandas()
    if not np.array_equal(cells["phase1_cell_index"].to_numpy(np.int64), np.arange(TOTAL_CELLS)):
        raise AssertionError("phase1_cell_index is not row-contiguous")
    if cells["stable_physical_locator"].duplicated().any():
        raise AssertionError("Phase-I physical locators are duplicated")
    split_indices = {
        split: cells.loc[cells["split"].eq(split), "phase1_cell_index"].to_numpy(np.int64)
        for split in ("train", "val", "test")
    }
    if {key: len(value) for key, value in split_indices.items()} != {
        "train": 67_840,
        "val": 14_080,
        "test": 19_200,
    }:
        raise AssertionError("Phase-I split cell counts changed")
    return {
        "subset": subset,
        "panel_manifest": panel_manifest,
        "target_manifest": target_manifest,
        "embedding_manifests": manifests,
        "cells": cells,
        "split_indices": split_indices,
    }


class LatentTargetDataset(Dataset[dict[str, Any]]):
    def __init__(self, latent_path: Path, indices: np.ndarray) -> None:
        self.latent_path = latent_path
        self.indices = np.asarray(indices, dtype=np.int64)
        self._latent: np.memmap | None = None
        self._target: np.memmap | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_latent"] = None
        state["_target"] = None
        return state

    def _open(self) -> None:
        if self._latent is None:
            self._latent = np.load(self.latent_path, mmap_mode="r")
            self._target = np.load(TARGETS, mmap_mode="r")

    def __getitem__(self, item: int) -> dict[str, Any]:
        self._open()
        index = int(self.indices[item])
        return {
            "phase1_cell_index": index,
            "latent": np.asarray(self._latent[index], dtype=np.float32).copy(),
            "target": np.asarray(self._target[index], dtype=np.float32).copy(),
        }


def seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(
    dataset: Dataset[dict[str, Any]],
    *,
    rank: int,
    train: bool,
) -> tuple[DataLoader, DistributedSampler]:
    sampler = DistributedSampler(
        dataset,
        num_replicas=WORLD_SIZE,
        rank=rank,
        shuffle=train,
        seed=SEED,
        drop_last=False,
    )
    generator = torch.Generator().manual_seed(SEED + rank)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_PER_GPU,
        sampler=sampler,
        shuffle=False,
        drop_last=train,
        num_workers=NUM_WORKERS_PER_RANK,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return loader, sampler


def setup_distributed() -> tuple[int, int, torch.device]:
    if int(os.environ.get("WORLD_SIZE", "1")) != WORLD_SIZE:
        raise RuntimeError("Phase-I Decoder smoke/training requires torchrun with two ranks")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    if dist.get_world_size() != WORLD_SIZE:
        raise RuntimeError("Expected exactly two DDP ranks")
    return dist.get_rank(), local_rank, device


def forward_mse(
    model: nn.Module,
    latent: torch.Tensor,
    target: torch.Tensor,
    *,
    audit_dtypes: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    dtypes: list[str] = []
    hooks = []
    if audit_dtypes:
        hooks = [
            module.register_forward_hook(
                lambda _module, _inputs, output: dtypes.append(str(output.dtype))
            )
            for module in model.modules()
            if isinstance(module, nn.Linear)
        ]
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(latent)
    finally:
        for hook in hooks:
            hook.remove()
    prediction_fp32 = prediction.float()
    target_fp32 = target.float()
    if prediction_fp32.shape != target_fp32.shape or prediction_fp32.shape[1:] != (GENE_DIM,):
        raise AssertionError("Phase-I Decoder output/target is not [B,20]")
    if not torch.isfinite(prediction_fp32).all() or not torch.isfinite(target_fp32).all():
        raise AssertionError("Non-finite Phase-I Decoder prediction/target")
    loss = torch.mean((prediction_fp32 - target_fp32).square())
    if not torch.isfinite(loss):
        raise AssertionError("Non-finite FP32 MSE")
    return prediction_fp32, loss, dtypes


def train_epoch(
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    sampler: DistributedSampler,
    device: torch.device,
    epoch: int,
    *,
    max_steps: int | None = None,
) -> dict[str, Any]:
    sampler.set_epoch(epoch)
    model.train()
    local_sse = torch.zeros((), dtype=torch.float64, device=device)
    local_elements = torch.zeros((), dtype=torch.int64, device=device)
    indices: list[int] = []
    steps = 0
    first_dtypes: list[str] = []
    started = time.perf_counter()
    for batch in loader:
        if max_steps is not None and steps >= max_steps:
            break
        latent = batch["latent"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prediction, loss, dtypes = forward_mse(
            model, latent, target, audit_dtypes=steps == 0
        )
        loss.backward()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise AssertionError("Phase-I Decoder gradient became non-finite")
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRADIENT_CLIP_NORM, error_if_nonfinite=True
        )
        optimizer.step()
        difference = prediction.detach() - target.float()
        local_sse += difference.square().sum(dtype=torch.float64)
        local_elements += difference.numel()
        if max_steps is not None:
            indices.extend(batch["phase1_cell_index"].to(torch.int64).tolist())
        if steps == 0:
            first_dtypes = dtypes
        steps += 1
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    dist.all_reduce(local_sse, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_elements, op=dist.ReduceOp.SUM)
    cells = int(local_elements.item()) // GENE_DIM
    return {
        "mse": float(local_sse.item() / local_elements.item()),
        "steps": steps,
        "global_cells": cells,
        "elapsed_seconds": elapsed,
        "global_cells_per_second": cells / elapsed,
        "indices": indices,
        "linear_output_dtypes": first_dtypes,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    sampler: DistributedSampler,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> dict[str, Any]:
    sampler.set_epoch(0)
    model.eval()
    local_sse = torch.zeros((), dtype=torch.float64, device=device)
    local_elements = torch.zeros((), dtype=torch.int64, device=device)
    batches = 0
    for batch in loader:
        if max_batches is not None and batches >= max_batches:
            break
        latent = batch["latent"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        prediction, _loss, _dtypes = forward_mse(model, latent, target)
        difference = prediction - target.float()
        local_sse += difference.square().sum(dtype=torch.float64)
        local_elements += difference.numel()
        batches += 1
    dist.all_reduce(local_sse, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_elements, op=dist.ReduceOp.SUM)
    return {
        "mse": float(local_sse.item() / local_elements.item()),
        "cells": int(local_elements.item()) // GENE_DIM,
        "batches_per_rank": batches,
        "finite": True,
        "drop_last": False,
        "shuffle": False,
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


def gather_rng() -> list[dict[str, Any]]:
    gathered: list[dict[str, Any] | None] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, capture_rng())
    return [item for item in gathered if item is not None]


def training_config(representation: str, contract: dict[str, Any]) -> dict[str, Any]:
    model = build_decoder()
    config = {
        "schema": "phase1_top20_decoder_training_config_v1",
        "created_at_utc": utc_now(),
        "status": "frozen",
        "representation": representation,
        "input": {
            "embedding": artifact(REPRESENTATIONS[representation], hash_file=False),
            "embedding_manifest": artifact(MANIFESTS[representation]),
            "target": artifact(TARGETS, hash_file=False),
            "target_manifest": artifact(TARGET_MANIFEST),
            "subset_manifest": artifact(SUBSET_MANIFEST),
            "row_contract": "phase1_cell_index is identical for latent and target",
        },
        "architecture": {
            "source": "state.tx.models.base.LatentToGeneDecoder",
            "dimensions": [768, 1024, 1024, 512, 20],
            "dropout": 0.1,
            "final_module": "Linear",
            "terminal_activation": None,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "training": {
            "seed": SEED,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "loss": "elementwise MSE",
            "forward_precision": "BF16 autocast",
            "prediction_target_loss_precision": "FP32",
            "gpus": 2,
            "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
            "global_batch": GLOBAL_BATCH,
            "gradient_accumulation": 1,
            "dataloader_workers_per_rank": NUM_WORKERS_PER_RANK,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "scheduler": None,
        },
        "cells": {
            split: len(indices) for split, indices in contract["split_indices"].items()
        },
        "fairness": {
            "same_physical_cells": True,
            "same_targets": True,
            "same_top20": True,
            "same_initialization_seed": True,
            "same_training_settings": True,
            "only_input_latent_differs": True,
        },
    }
    del model
    return config


def ensure_configs(contract: dict[str, Any]) -> None:
    for representation in REPRESENTATIONS:
        paths = rep_paths(representation)
        paths["root"].mkdir(parents=True, exist_ok=True)
        expected = training_config(representation, contract)
        if paths["config"].exists():
            observed = read_json(paths["config"])
            comparable = {key: value for key, value in observed.items() if key != "created_at_utc"}
            target = {key: value for key, value in expected.items() if key != "created_at_utc"}
            if comparable != target:
                raise AssertionError(f"Frozen {representation} training config changed")
        else:
            atomic_json(paths["config"], expected)


def exact_sync_error(model: nn.Module, device: torch.device) -> float:
    maximum = torch.zeros((), device=device)
    for parameter in model.parameters():
        reference = parameter.detach().clone()
        dist.broadcast(reference, src=0)
        maximum = torch.maximum(maximum, (parameter.detach() - reference).abs().max())
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return float(maximum.item())


def command_smoke(steps: int) -> dict[str, Any] | None:
    rank, local_rank, device = setup_distributed()
    try:
        contract = load_contract()
        if rank == 0:
            if SMOKE_RESULT.exists():
                raise FileExistsError(SMOKE_RESULT)
            ensure_configs(contract)
        dist.barrier()
        results: dict[str, Any] = {}
        reference_indices: list[int] | None = None
        initial_hash: str | None = None
        for representation in REPRESENTATIONS:
            seed_everything()
            raw_model = build_decoder().to(device)
            current_hash = model_fingerprint(raw_model)
            if initial_hash is None:
                initial_hash = current_hash
            elif current_hash != initial_hash:
                raise AssertionError("Our/Author Decoder initial weights differ")
            model = DistributedDataParallel(raw_model, device_ids=[local_rank])
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
            dataset = LatentTargetDataset(
                REPRESENTATIONS[representation], contract["split_indices"]["train"]
            )
            loader, sampler = make_loader(dataset, rank=rank, train=True)
            torch.cuda.reset_peak_memory_stats(device)
            before = model_fingerprint(raw_model)
            train = train_epoch(
                model, optimizer, loader, sampler, device, 0, max_steps=steps
            )
            after = model_fingerprint(raw_model)
            if before == after:
                raise AssertionError("Smoke parameters did not update")
            if reference_indices is None:
                reference_indices = train["indices"]
            elif train["indices"] != reference_indices:
                raise AssertionError("Our/Author smoke sampled different physical cells")
            val_dataset = LatentTargetDataset(
                REPRESENTATIONS[representation], contract["split_indices"]["val"]
            )
            val_loader, val_sampler = make_loader(val_dataset, rank=rank, train=False)
            validation = validate(model, val_loader, val_sampler, device, max_batches=1)
            if train["linear_output_dtypes"] != ["torch.bfloat16"] * 4:
                raise AssertionError("Decoder Linear path did not run in BF16")
            gathered_indices: list[list[int] | None] = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_indices, train["indices"])
            rank_overlap = len(set(gathered_indices[0] or []) & set(gathered_indices[1] or []))
            if rank_overlap:
                raise AssertionError("Smoke train ranks sampled overlapping physical cells")
            results[representation] = {
                "initial_model_sha256": current_hash,
                "initial_mse": train["mse"],
                "validation_mse": validation["mse"],
                "steps_per_rank": train["steps"],
                "global_cells": train["global_cells"],
                "forward_shape": [BATCH_SIZE_PER_GPU, GENE_DIM],
                "loss_finite": math.isfinite(train["mse"]),
                "backward_parameter_update": True,
                "linear_output_dtypes": train["linear_output_dtypes"],
                "rank_index_overlap": rank_overlap,
                "peak_allocated_GiB_per_rank": torch.cuda.max_memory_allocated(device) / 2**30,
                "global_cells_per_second": train["global_cells_per_second"],
            }
            del model, raw_model, optimizer, loader, dataset, val_loader, val_dataset
            torch.cuda.empty_cache()
            dist.barrier()
        if rank != 0:
            return None
        payload = {
            "schema": "phase1_top20_decoder_exact_config_smoke_v1",
            "created_at_utc": utc_now(),
            "status": "pass",
            "configuration": {
                "gpus": 2,
                "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
                "global_batch": GLOBAL_BATCH,
                "dataloader_workers_per_rank": NUM_WORKERS_PER_RANK,
                "BF16_forward": True,
                "FP32_MSE": True,
                "DDP": True,
            },
            "architecture": {
                "dimensions": [768, 1024, 1024, 512, 20],
                "final_linear": True,
                "softplus": False,
                "relu_output": False,
                "parameter_count": EXPECTED_PARAMETER_COUNT,
            },
            "physical_alignment": {
                "row_equals_phase1_cell_index": True,
                "our_author_targets_same_rows": True,
                "our_author_smoke_indices_identical": True,
            },
            "representations": results,
            "ready_for_formal_training": True,
            "phase2_started": False,
        }
        atomic_json(SMOKE_RESULT, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def checkpoint_payload(
    representation: str,
    kind: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_epoch: int,
    best_val_mse: float,
    early_stop_counter: int,
    history: list[dict[str, Any]],
    rng_states: list[dict[str, Any]],
    config_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": "phase1_top20_decoder_checkpoint_v1",
        "representation": representation,
        "checkpoint_kind": kind,
        "epoch": epoch,
        "epoch_complete": True,
        "global_step": global_step,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "early_stop_counter": early_stop_counter,
        "history": history,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_states": rng_states,
        "training_config_sha256": config_sha256,
        "architecture": [768, 1024, 1024, 512, 20],
        "final_activation": None,
    }


def command_train(representation: str, resume: bool) -> dict[str, Any] | None:
    rank, local_rank, device = setup_distributed()
    writer: Any | None = None
    try:
        contract = load_contract()
        paths = rep_paths(representation)
        if read_json(SMOKE_RESULT).get("status") != "pass":
            raise RuntimeError("Exact-config Decoder smoke has not passed")
        if rank == 0:
            ensure_configs(contract)
            if resume and not paths["last"].is_file():
                raise FileNotFoundError(paths["last"])
            if not resume and any(
                path.exists()
                for path in (
                    paths["best"],
                    paths["last"],
                    paths["result"],
                    paths["tensorboard"],
                )
            ):
                raise FileExistsError(f"Fresh {representation} training artifacts exist")
        dist.barrier()
        config_sha = sha256_file(paths["config"])
        seed_everything()
        raw_model = build_decoder().to(device)
        optimizer = torch.optim.AdamW(
            raw_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        if resume:
            checkpoint = torch.load(paths["last"], map_location="cpu", weights_only=False)
            if (
                checkpoint.get("schema") != "phase1_top20_decoder_checkpoint_v1"
                or checkpoint.get("representation") != representation
                or checkpoint.get("training_config_sha256") != config_sha
                or not checkpoint.get("epoch_complete")
            ):
                raise AssertionError("Resume checkpoint contract changed")
            raw_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
            restore_rng(checkpoint["rng_states"][rank])
            start_epoch = int(checkpoint["epoch"]) + 1
            global_step = int(checkpoint["global_step"])
            best_epoch = int(checkpoint["best_epoch"])
            best_val_mse = float(checkpoint["best_val_mse"])
            early_stop_counter = int(checkpoint["early_stop_counter"])
            history = list(checkpoint["history"])
        else:
            start_epoch = 0
            global_step = 0
            best_epoch = -1
            best_val_mse = float("inf")
            early_stop_counter = 0
            history: list[dict[str, Any]] = []
        model = DistributedDataParallel(raw_model, device_ids=[local_rank])
        train_dataset = LatentTargetDataset(
            REPRESENTATIONS[representation], contract["split_indices"]["train"]
        )
        val_dataset = LatentTargetDataset(
            REPRESENTATIONS[representation], contract["split_indices"]["val"]
        )
        train_loader, train_sampler = make_loader(train_dataset, rank=rank, train=True)
        val_loader, val_sampler = make_loader(val_dataset, rank=rank, train=False)
        if rank == 0:
            from torch.utils.tensorboard import SummaryWriter

            paths["tensorboard"].mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(
                log_dir=str(paths["tensorboard"]), purge_step=global_step if resume else None
            )
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        stop_reason = "max_epochs"
        for epoch in range(start_epoch, MAX_EPOCHS):
            train = train_epoch(
                model, optimizer, train_loader, train_sampler, device, epoch
            )
            global_step += int(train["steps"])
            validation = validate(model, val_loader, val_sampler, device)
            val_mse = float(validation["mse"])
            improved = val_mse < best_val_mse
            if improved:
                best_epoch = epoch
                best_val_mse = val_mse
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "train_mse": train["mse"],
                "val_mse": val_mse,
                "strict_improvement": improved,
                "best_epoch": best_epoch,
                "best_val_mse": best_val_mse,
                "early_stop_counter": early_stop_counter,
                "global_cells_per_second": train["global_cells_per_second"],
            }
            history.append(row)
            rng_states = gather_rng()
            if rank == 0:
                common = dict(
                    representation=representation,
                    model=raw_model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    best_epoch=best_epoch,
                    best_val_mse=best_val_mse,
                    early_stop_counter=early_stop_counter,
                    history=history,
                    rng_states=rng_states,
                    config_sha256=config_sha,
                )
                if improved:
                    atomic_torch(paths["best"], checkpoint_payload(kind="best", **common))
                atomic_torch(paths["last"], checkpoint_payload(kind="last", **common))
                if writer is not None:
                    writer.add_scalar("train/mse", train["mse"], global_step)
                    writer.add_scalar("val/mse", val_mse, global_step)
                    writer.flush()
                print(
                    f"phase1-{representation} epoch={epoch} step={global_step} "
                    f"train_mse={train['mse']:.8f} val_mse={val_mse:.8f} "
                    f"best_epoch={best_epoch} early_stop_counter={early_stop_counter}",
                    flush=True,
                )
            dist.barrier()
            if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                stop_reason = "early_stopping"
                break
        sync_error = exact_sync_error(raw_model, device)
        if sync_error != 0.0:
            raise AssertionError("DDP Decoder parameters differ across ranks")
        if rank != 0:
            dist.barrier()
            return None
        result = {
            "schema": "phase1_top20_decoder_training_result_v1",
            "created_at_utc": utc_now(),
            "status": "pass",
            "representation": representation,
            "stop_reason": stop_reason,
            "epochs_completed": len(history),
            "global_optimizer_steps": global_step,
            "best_epoch": best_epoch,
            "best_val_mse": best_val_mse,
            "history": history,
            "configuration": artifact(paths["config"]),
            "checkpoints": {
                "best": artifact(paths["best"]),
                "last": artifact(paths["last"]),
            },
            "peak_allocated_GiB_rank0": torch.cuda.max_memory_allocated(device) / 2**30,
            "DDP_parameter_max_abs_error": sync_error,
            "elapsed_seconds": time.perf_counter() - started,
            "phase2_started": False,
        }
        atomic_json(paths["result"], result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        dist.barrier()
        return result
    finally:
        if writer is not None:
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(pd.Series(x).rank(method="average").to_numpy(), pd.Series(y).rank(method="average").to_numpy())


def descriptive(values: pd.Series | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)) if len(finite) else None,
        "median": float(np.median(finite)) if len(finite) else None,
        "std": float(np.std(finite)) if len(finite) else None,
        "p05": float(np.quantile(finite, 0.05)) if len(finite) else None,
        "p95": float(np.quantile(finite, 0.95)) if len(finite) else None,
        "min": float(np.min(finite)) if len(finite) else None,
        "max": float(np.max(finite)) if len(finite) else None,
    }


def make_obs(selected: pd.DataFrame, *, predicted: bool) -> pd.DataFrame:
    prefix = "pred" if predicted else "real"
    records: list[dict[str, Any]] = []
    first = selected.iloc[0]
    for cell in range(256):
        records.append(
            {
                "obs_name": f"{prefix}_control_{cell:03d}",
                "perturbation": CONTROL_PERT,
                "pair_id": "",
                "drug": "DMSO",
                "dose_uM": np.nan,
                "cell_line_id": first["cell_line_id"],
                "plate": first["plate"],
                "set_cell_index": cell,
            }
        )
    for row in selected.itertuples(index=False):
        for cell in range(256):
            records.append(
                {
                    "obs_name": f"{prefix}_{row.pair_id}_{cell:03d}",
                    "perturbation": row.perturbation,
                    "pair_id": row.pair_id,
                    "drug": row.drug,
                    "dose_uM": row.dose_uM,
                    "cell_line_id": row.cell_line_id,
                    "plate": row.plate,
                    "set_cell_index": cell,
                }
            )
    return pd.DataFrame.from_records(records).set_index("obs_name", verify_integrity=True)


def write_pair_h5ad(
    path: Path,
    matrix: np.ndarray,
    obs: pd.DataFrame,
    panel: pd.DataFrame,
) -> None:
    if matrix.dtype != np.float32 or matrix.shape[1] != GENE_DIM or not np.isfinite(matrix).all():
        raise AssertionError("Phase-I evaluation matrix contract failed")
    var = panel[["gene_symbol", "genejepa_index", "panel_rank"]].copy()
    var.index = pd.Index(panel["ensembl_id"].astype(str), name="ensembl_id")
    data = ad.AnnData(X=matrix, obs=obs, var=var)
    data.uns["evaluation_label"] = EVALUATION_LABEL
    data.uns["disclaimer"] = DISCLAIMER
    data.uns["expression_space"] = "log1p(CP10000)"
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    data.write_h5ad(temporary, compression="lzf")
    del data
    os.replace(temporary, path)


@torch.no_grad()
def decode_test(
    representation: str,
    model: nn.Module,
    test_indices: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    dataset = LatentTargetDataset(REPRESENTATIONS[representation], test_indices)
    loader = DataLoader(
        dataset,
        batch_size=4096,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS_PER_RANK,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR,
    )
    decoded = np.full((TOTAL_CELLS, GENE_DIM), np.nan, dtype=np.float32)
    targets = np.load(TARGETS, mmap_mode="r")
    sse = 0.0
    elements = 0
    model.eval()
    for batch in loader:
        latent = batch["latent"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(latent)
        values = prediction.float().cpu().numpy()
        indices = batch["phase1_cell_index"].to(torch.int64).numpy()
        if not np.isfinite(values).all() or np.isfinite(decoded[indices]).any():
            raise AssertionError("Invalid or duplicate decoded test rows")
        decoded[indices] = values
        difference = values.astype(np.float64) - np.asarray(targets[indices], dtype=np.float64)
        sse += float(np.square(difference).sum())
        elements += difference.size
    if not np.isfinite(decoded[test_indices]).all():
        raise AssertionError("Decoded test rows are incomplete")
    return decoded, sse / elements


def command_evaluate(representation: str, threads: int | None) -> dict[str, Any]:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Evaluation is single-process; use CUDA_VISIBLE_DEVICES=0")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or torch.cuda.device_count() != 1:
        raise RuntimeError("Evaluation requires CUDA_VISIBLE_DEVICES=0")
    contract = load_contract()
    paths = rep_paths(representation)
    if paths["evaluation"].exists() or paths["condition_metrics"].exists() or paths["cell_eval"].exists():
        raise FileExistsError(f"{representation} evaluation outputs already exist")
    training = read_json(paths["result"])
    if training.get("status") != "pass":
        raise AssertionError(f"{representation} formal training is not PASS")
    checkpoint = torch.load(paths["best"], map_location="cpu", weights_only=False)
    model = build_decoder()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda:0")
    model.to(device).eval()
    decoded, test_mse = decode_test(
        representation, model, contract["split_indices"]["test"], device
    )
    targets = np.load(TARGETS, mmap_mode="r")
    conditions = pd.read_csv(SUBSET_CONDITIONS, encoding="utf-8-sig", keep_default_na=False)
    contexts = pd.read_csv(SUBSET_CONTEXTS, encoding="utf-8-sig", keep_default_na=False)
    conditions = conditions.loc[conditions["split"].eq("test")].copy()
    contexts = contexts.loc[contexts["split"].eq("test")].copy()
    if len(conditions) != 60 or len(contexts) != 15:
        raise AssertionError("Frozen test condition/context count changed")
    panel = pd.read_csv(PANEL, encoding="utf-8-sig", keep_default_na=False)
    paths["cell_eval"].mkdir(parents=True)
    from run_genejepa_decoder_v1_arc7_100 import METRICS, available_threads, run_cell_eval

    threads = available_threads() if threads is None else min(threads, available_threads())
    condition_frames: list[pd.DataFrame] = []
    context_summaries: list[pd.DataFrame] = []
    failures: dict[str, str] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    started = time.perf_counter()
    for context in contexts.itertuples(index=False):
        selected = conditions.loc[conditions["context_index"].eq(context.context_index)].copy()
        if len(selected) != 4:
            raise AssertionError("Each test context must contain four conditions")
        selected["perturbation"] = [
            f"drug={row.drug}|dose_uM={format(float(row.dose_uM), '.15g')}|pair_id={row.pair_id}"
            for row in selected.itertuples(index=False)
        ]
        control = np.arange(
            int(context.phase1_control_start),
            int(context.phase1_control_stop_exclusive),
            dtype=np.int64,
        )
        true_chunks = [np.asarray(targets[control], dtype=np.float32)]
        pred_chunks = [np.asarray(decoded[control], dtype=np.float32)]
        true_control_mean = true_chunks[0].mean(axis=0)
        pred_control_mean = pred_chunks[0].mean(axis=0)
        for row in selected.itertuples(index=False):
            treated = np.arange(
                int(row.phase1_treated_start),
                int(row.phase1_treated_stop_exclusive),
                dtype=np.int64,
            )
            true_treated = np.asarray(targets[treated], dtype=np.float32)
            pred_treated = np.asarray(decoded[treated], dtype=np.float32)
            true_chunks.append(true_treated)
            pred_chunks.append(pred_treated)
            true_mean = true_treated.mean(axis=0)
            pred_mean = pred_treated.mean(axis=0)
            diagnostics[str(row.pair_id)] = {
                "absolute_pearson": pearson(true_mean, pred_mean),
                "absolute_spearman": spearman(true_mean, pred_mean),
                "delta_pearson_direct": pearson(
                    true_mean - true_control_mean, pred_mean - pred_control_mean
                ),
                "delta_spearman_direct": spearman(
                    true_mean - true_control_mean, pred_mean - pred_control_mean
                ),
            }
        true_matrix = np.concatenate(true_chunks).astype(np.float32, copy=False)
        pred_matrix = np.concatenate(pred_chunks).astype(np.float32, copy=False)
        context_dir = paths["cell_eval"] / f"context_{int(context.context_index):03d}"
        context_dir.mkdir()
        with tempfile.TemporaryDirectory(prefix="phase1_top20_eval_") as temporary:
            temporary_root = Path(temporary)
            real_path = temporary_root / "real.h5ad"
            pred_path = temporary_root / "pred.h5ad"
            write_pair_h5ad(real_path, true_matrix, make_obs(selected, predicted=False), panel)
            write_pair_h5ad(pred_path, pred_matrix, make_obs(selected, predicted=True), panel)
            summary, per_condition, _official, runtime = run_cell_eval(
                real_path, pred_path, selected, context_dir, threads
            )
        if runtime["metric_failures"]:
            failures[str(context.context_id)] = json.dumps(
                runtime["metric_failures"], ensure_ascii=False
            )
        per_condition["context_id"] = str(context.context_id)
        per_condition["context_index"] = int(context.context_index)
        per_condition["cell_line_id"] = str(context.cell_line_id)
        per_condition["plate"] = str(context.plate)
        for column in ("edge_id", "control_pool_id"):
            mapping = selected.set_index("pair_id")[column].astype(str)
            per_condition[column] = per_condition["pair_id"].map(mapping)
        for name in next(iter(diagnostics.values())):
            per_condition[name] = per_condition["pair_id"].map(
                lambda pair: diagnostics[str(pair)][name]
            )
        condition_frames.append(per_condition)
        summary["context_id"] = str(context.context_id)
        summary["context_index"] = int(context.context_index)
        context_summaries.append(summary)
    condition_metrics = pd.concat(condition_frames, ignore_index=True)
    if len(condition_metrics) != 60 or condition_metrics["pair_id"].nunique() != 60:
        raise AssertionError("Phase-I evaluation condition rows changed")
    context_summary = pd.concat(context_summaries, ignore_index=True)
    seven_metrics: list[dict[str, Any]] = []
    undefined_metrics: dict[str, Any] = {}
    for spec in METRICS:
        label = str(spec["label"])
        column = str(spec["column"])
        if spec["scope"] == "global_across_conditions":
            values = context_summary.loc[context_summary["metric"].eq(label), "value"]
            aggregation = "macro mean across 15 matched-control contexts"
            expected_count = 15
        else:
            values = condition_metrics[column]
            aggregation = "macro mean across 60 test conditions"
            expected_count = 60
        stats = descriptive(values)
        if stats["count"] != expected_count:
            numeric = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
            invalid = ~np.isfinite(numeric)
            undefined = condition_metrics.loc[invalid] if spec["scope"] != "global_across_conditions" else None
            expected_undefined = (
                label in {"Spearman logFC", "AUPRC"}
                and undefined is not None
                and len(undefined) == expected_count - int(stats["count"])
                and bool(undefined["true_DEG_count"].eq(0).all())
            )
            if expected_undefined:
                reason = (
                    "no significant true genes to rank"
                    if label == "Spearman logFC"
                    else "no positive true-DEG class"
                )
                undefined_metrics[label] = {
                    "count": int(len(undefined)),
                    "reason": reason,
                    "handling": "retained as NaN and excluded from the finite macro mean; no imputation",
                    "conditions": [
                        {
                            "pair_id": str(row.pair_id),
                            "drug": str(row.drug),
                            "dose_uM": float(row.dose_uM),
                            "true_DEG_count": int(row.true_DEG_count),
                        }
                        for row in undefined.itertuples(index=False)
                    ],
                }
                aggregation = (
                    f"macro mean across {stats['count']} mathematically evaluable test conditions; "
                    f"{len(undefined)} zero-true-DEG condition excluded without imputation"
                )
            else:
                failures[label] = "unexpected non-finite or missing metric values"
        seven_metrics.append(
            {
                "metric": label,
                "column": column,
                "direction": str(spec["direction"]),
                "aggregation": aggregation,
                **stats,
            }
        )
    atomic_csv(paths["condition_metrics"], condition_metrics)
    result = {
        "schema": "phase1_top20_evaluation_v1",
        "created_at_utc": utc_now(),
        "status": "pass" if not failures else "fail",
        "representation": representation,
        "evaluation_label": EVALUATION_LABEL,
        "disclaimer": DISCLAIMER,
        "checkpoint": artifact(paths["best"]),
        "best_epoch": training["best_epoch"],
        "best_val_mse": training["best_val_mse"],
        "test_cell_mse": test_mse,
        "test_cells": len(contract["split_indices"]["test"]),
        "evaluation_conditions": 60,
        "matched_control_contexts": 15,
        "genes": GENE_DIM,
        "seven_metrics": seven_metrics,
        "true_DEG_distribution": descriptive(condition_metrics["true_DEG_count"]),
        "predicted_DEG_distribution": descriptive(condition_metrics["pred_DEG_count"]),
        "diagnostics": {
            "absolute_pearson": descriptive(condition_metrics["absolute_pearson"]),
            "absolute_spearman": descriptive(condition_metrics["absolute_spearman"]),
            "delta_pearson_direct": descriptive(condition_metrics["delta_pearson_direct"]),
            "delta_spearman_direct": descriptive(condition_metrics["delta_spearman_direct"]),
        },
        "undefined_metric_policy": {
            "policy": "retain mathematically undefined values as NaN; aggregate only finite values; do not impute",
            "metrics": undefined_metrics,
        },
        "metric_failures": failures,
        "contracts": {
            "decoder_only": True,
            "ST_used": False,
            "predicted_delta": "decoded treated - decoded control",
            "true_delta": "real treated Top20 - real control Top20",
            "same_physical_cells": True,
            "expression_space": "full mapped library CP10000 -> log1p -> Top20",
        },
        "provenance": {
            "subset_manifest": artifact(SUBSET_MANIFEST),
            "panel": artifact(PANEL),
            "target_manifest": artifact(TARGET_MANIFEST),
            "embedding_manifest": artifact(MANIFESTS[representation]),
            "training_result": artifact(paths["result"]),
            "implementation": artifact(Path(__file__)),
        },
        "outputs": {"condition_metrics": artifact(paths["condition_metrics"])},
        "elapsed_seconds": time.perf_counter() - started,
        "phase2_started": False,
    }
    atomic_json(paths["evaluation"], result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if result["status"] != "pass":
        raise RuntimeError(f"{representation} evaluation metric failures: {failures}")
    return result


def metric_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["metric"]: row for row in result["seven_metrics"]}


def command_compare() -> dict[str, Any]:
    if any(path.exists() for path in (COMPARISON_CSV, COMPARISON_JSON, HANDOFF)):
        raise FileExistsError("Phase-I comparison output exists")
    contract = load_contract()
    evaluations = {
        name: read_json(rep_paths(name)["evaluation"]) for name in REPRESENTATIONS
    }
    training = {name: read_json(rep_paths(name)["result"]) for name in REPRESENTATIONS}
    if any(payload.get("status") != "pass" for payload in (*evaluations.values(), *training.values())):
        raise AssertionError("Both formal training and evaluation results must PASS")
    maps = {name: metric_map(payload) for name, payload in evaluations.items()}
    rows: list[dict[str, Any]] = []
    primary_wins = {"our": 0, "author": 0}
    for label in [*PRIMARY_WINNER_METRICS, "MAE"]:
        ours = float(maps["our"][label]["mean"])
        author = float(maps["author"][label]["mean"])
        direction = maps["our"][label]["direction"]
        if math.isclose(ours, author, rel_tol=0.0, abs_tol=1e-12):
            winner = "tie"
        elif (direction == "↑" and ours > author) or (direction == "↓" and ours < author):
            winner = "our"
        else:
            winner = "author"
        if label in PRIMARY_WINNER_METRICS and winner != "tie":
            primary_wins[winner] += 1
        rows.append(
            {
                "metric": label,
                "direction": direction,
                "our": ours,
                "author": author,
                "winner": winner,
                "primary_winner_metric": label in PRIMARY_WINNER_METRICS,
            }
        )
    if primary_wins["our"] != primary_wins["author"]:
        selected = max(primary_wins, key=primary_wins.get)
        tie_break = None
    else:
        our_mae = float(maps["our"]["MAE"]["mean"])
        author_mae = float(maps["author"]["MAE"]["mean"])
        if not math.isclose(our_mae, author_mae, rel_tol=0.0, abs_tol=1e-12):
            selected = "our" if our_mae < author_mae else "author"
            tie_break = "lower MAE"
        else:
            selected = (
                "our"
                if evaluations["our"]["test_cell_mse"]
                < evaluations["author"]["test_cell_mse"]
                else "author"
            )
            tie_break = "lower test cell MSE"
    comparison = pd.DataFrame.from_records(rows)
    atomic_csv(COMPARISON_CSV, comparison)
    result = {
        "schema": "phase1_genejepa_top20_comparison_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "phase1_selected_genejepa": selected,
        "winner_rule": {
            "primary": (
                "majority of six perturbation-sensitive metrics: DES, PDS, Pearson delta, "
                "Spearman logFC, AUPRC, Spearman effect size"
            ),
            "primary_wins": primary_wins,
            "tie_break": tie_break,
            "composite_score": False,
        },
        "fairness": {
            "physical_cells_identical": True,
            "phase1_cell_index_identical": True,
            "Top20_identical": True,
            "target_identical": True,
            "decoder_architecture_identical": True,
            "training_settings_identical": True,
            "only_input_latent_differs": True,
        },
        "subset": contract["subset"]["counts"],
        "models": {
            name: {
                "best_epoch": training[name]["best_epoch"],
                "best_val_mse": training[name]["best_val_mse"],
                "test_cell_mse": evaluations[name]["test_cell_mse"],
                "seven_metrics": evaluations[name]["seven_metrics"],
                "true_DEG_distribution": evaluations[name]["true_DEG_distribution"],
                "predicted_DEG_distribution": evaluations[name]["predicted_DEG_distribution"],
                "diagnostics": evaluations[name]["diagnostics"],
                "parameter_count": EXPECTED_PARAMETER_COUNT,
                "training_cell_count": 67_840,
                "validation_cell_count": 14_080,
                "evaluation_cell_count": 19_200,
                "evaluation_condition_count": 60,
            }
            for name in REPRESENTATIONS
        },
        "conclusion_boundary": (
            "Selection applies only to decodability in this frozen Top20 readout experiment; "
            "model depth, attention heads, and checkpoint epoch also differ."
        ),
        "comparison_csv": artifact(COMPARISON_CSV),
        "phase1_complete": True,
        "phase2_started": False,
    }
    atomic_json(COMPARISON_JSON, result)
    lines = [
        "# Phase I GeneJEPA Top20 final handoff",
        "",
        f"Status: PASS. Phase I selected GeneJEPA: **{selected}**.",
        "",
        "## Frozen subset",
        "",
        f"- Unique physical cells: {contract['subset']['counts']['total_unique_cells']:,}",
        f"- Treated/control: {contract['subset']['counts']['treated_cells']:,} / {contract['subset']['counts']['control_cells']:,}",
        f"- Cell lines / conditions / drugs: {contract['subset']['counts']['cell_lines']} / {contract['subset']['counts']['conditions']} / {contract['subset']['counts']['drugs']}",
        "- Split cells: train 67,840; val 14,080; test 19,200.",
        "- Selection was deterministic (seed 42), stratified across five high-coverage cell lines, all three doses, and frozen train/val/test splits; every condition/control contributes 256 unique cells.",
        "- Our, Author, and Top20 targets share the same contiguous `phase1_cell_index`; duplicate physical locators = 0.",
        "",
        "## Top20 and Author inference",
        "",
        "- Top20 is exactly zero-based HD100 panel ranks 0-19 (human ranks 1-20).",
        f"- Frozen Top20 panel SHA-256: `{sha256_file(PANEL)}`.",
        "- Author epoch49 EMA Teacher exposes `[B,512,768]` after `final_norm`; manual mean parity passed.",
        "- Formal Author cache stores only signed FP32 `[N,768]`; token tensors were not persisted.",
        "",
        "## Decoder contract",
        "",
        f"- Architecture: `768 -> 1024 -> 1024 -> 512 -> 20`; final Linear; no Softplus/ReLU; {EXPECTED_PARAMETER_COUNT:,} parameters.",
        f"- Our: best epoch {training['our']['best_epoch']}, val MSE {training['our']['best_val_mse']:.8g}, test MSE {evaluations['our']['test_cell_mse']:.8g}.",
        f"- Author: best epoch {training['author']['best_epoch']}, val MSE {training['author']['best_val_mse']:.8g}, test MSE {evaluations['author']['test_cell_mse']:.8g}.",
        f"- Our true/pred DEG means: {evaluations['our']['true_DEG_distribution']['mean']:.6g} / {evaluations['our']['predicted_DEG_distribution']['mean']:.6g}.",
        f"- Author true/pred DEG means: {evaluations['author']['true_DEG_distribution']['mean']:.6g} / {evaluations['author']['predicted_DEG_distribution']['mean']:.6g}.",
        "",
        "## Seven adapted Top20 metrics",
        "",
        "| Metric | Direction | Our | Author | Winner |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric']} | {row['direction']} | {row['our']:.6g} | {row['author']:.6g} | {row['winner']} |"
        )
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"Primary metric wins: Our {primary_wins['our']}, Author {primary_wins['author']}. Tie-break: {tie_break or 'not needed'}.",
            f"Phase I 后续选择：**{selected} GeneJEPA**.",
            "",
            result["conclusion_boundary"],
            "",
            "Phase II was not started.",
        ]
    )
    temporary = HANDOFF.with_name(HANDOFF.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, HANDOFF)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit-data")
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--steps", type=int, default=2)
    train = commands.add_parser("train")
    train.add_argument("--representation", choices=tuple(REPRESENTATIONS), required=True)
    train.add_argument("--resume", action="store_true")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--representation", choices=tuple(REPRESENTATIONS), required=True)
    evaluate.add_argument("--threads", type=int)
    commands.add_parser("compare")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "audit-data":
        contract = load_contract()
        ensure_configs(contract)
        result = {
            "status": "pass",
            "cells": TOTAL_CELLS,
            "split_cells": {
                key: len(value) for key, value in contract["split_indices"].items()
            },
            "physical_alignment": "phase1_cell_index shared by Our, Author, target",
            "architecture": [768, 1024, 1024, 512, 20],
            "final_activation": None,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "smoke":
        if args.steps < 1 or args.steps > 4:
            raise ValueError("Smoke --steps must be in [1,4]")
        command_smoke(args.steps)
    elif args.command == "train":
        command_train(args.representation, args.resume)
    elif args.command == "evaluate":
        command_evaluate(args.representation, args.threads)
    elif args.command == "compare":
        command_compare()


if __name__ == "__main__":
    main()
