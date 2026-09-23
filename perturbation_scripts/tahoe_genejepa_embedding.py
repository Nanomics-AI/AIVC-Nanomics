#!/usr/bin/env python3
"""Shared Tahoe preprocessing and frozen GeneJEPA embedding helpers."""

from __future__ import annotations

import hashlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from genejepa.configs import DataConfig, ExperimentConfig
from genejepa.data import Tahoe100MDataModule, Tahoe100MDataset


DEFAULT_LOCAL_MANIFEST = PROJECT_ROOT / "hf_data_cache" / "local_file_manifest.json"
DEFAULT_STATS = PROJECT_ROOT / "hf_data_cache" / "global_stats.json"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "genejepa_quarter_d12_h6_700k_e30_seed42_run1"
    / "scjepa-epoch=25-val_loss=0.179.ckpt"
)
EMBEDDING_DIM = 768
PARQUET_COLUMNS = [
    "genes",
    "expressions",
    "plate",
    "sample",
    "drug",
    "cell_line_id",
    "BARCODE_SUB_LIB_ID",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_metadata_path(local_manifest_path: Path) -> Path:
    with local_manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    configured = Path(manifest["metadata_file"])
    if configured.is_file():
        return configured
    local_copy = PROJECT_ROOT / "hf_data_cache" / "metadata" / "metadata" / configured.name
    if local_copy.is_file():
        return local_copy
    raise FileNotFoundError(configured)


def build_official_preprocessor(
    local_manifest_path: Path,
    stats_path: Path,
) -> tuple[Tahoe100MDataModule, Path]:
    """Load the same gene mapping and normalization used for GeneJEPA training."""
    metadata_path = resolve_metadata_path(local_manifest_path)
    datamodule = Tahoe100MDataModule(DataConfig(), ExperimentConfig())
    datamodule.stats_path = str(stats_path)
    datamodule._load_stats()
    datamodule._load_metadata(str(metadata_path))
    if not datamodule.gene_map:
        raise RuntimeError("Tahoe gene_map is empty")
    return datamodule, metadata_path


def preprocess_once(
    raw_cells: list[dict[str, object]],
    datamodule: Tahoe100MDataModule,
    audit: Any,
    inverse_check: bool,
) -> dict[str, torch.Tensor | list[dict[str, str]]]:
    """Apply sentinel removal, gene mapping, log1p, and global normalization once."""
    if datamodule.gene_map is None:
        raise RuntimeError("gene_map is not initialized")

    expected_mapped_counts: list[int] = []
    for raw in raw_cells:
        genes = raw["genes"]
        expressions = raw["expressions"]
        if not isinstance(genes, list) or not isinstance(expressions, list):
            raise TypeError("Parquet genes/expressions must decode to lists")
        if not genes or not expressions or len(genes) != len(expressions):
            raise ValueError("Empty or length-mismatched genes/expressions")
        sentinel = expressions[0] < 0
        effective_genes = genes[1:] if sentinel else genes
        effective_expressions = expressions[1:] if sentinel else expressions
        values = np.asarray(effective_expressions, dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Non-finite or negative expression after sentinel handling")
        mapped_count = sum(gene in datamodule.gene_map for gene in effective_genes)
        if mapped_count == 0:
            raise ValueError("Cell has no genes in the Tahoe vocabulary")
        expected_mapped_counts.append(mapped_count)
        audit.add_raw(len(genes), len(effective_genes), mapped_count, sentinel)

    processed = list(Tahoe100MDataset(raw_cells, datamodule.gene_map))
    if len(processed) != len(raw_cells):
        raise AssertionError("Tahoe dataset dropped a selected cell")
    for item, expected_count in zip(processed, expected_mapped_counts, strict=True):
        if len(item["gene_indices"]) != expected_count:
            raise AssertionError("Gene mapping count differs from the preprocessing audit")

    batch = datamodule._collate_fn(processed)
    values = batch["values"]
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Tahoe collate produced non-finite values")
    audit.add_normalized(values)

    if inverse_check:
        raw_mapped = torch.cat(
            [torch.from_numpy(item["counts"]).float() for item in processed]
        )
        restored = torch.expm1(
            values * (float(datamodule.global_std) + 1e-6)
            + float(datamodule.global_mean)
        )
        error = float(torch.max(torch.abs(restored - raw_mapped)).item())
        audit.inverse_recovery_checked_cells += len(raw_cells)
        audit.inverse_recovery_max_abs_error = max(
            audit.inverse_recovery_max_abs_error, error
        )
        if not torch.allclose(restored, raw_mapped, rtol=1e-5, atol=1e-4):
            raise AssertionError("Normalized values do not invert to mapped raw counts")

    return batch


def load_frozen_model(checkpoint: Path, device: torch.device):
    """Load the frozen Epoch25 model and select its EMA Teacher for inference."""
    from genejepa.train import JepaLightningModule

    module = JepaLightningModule.load_from_checkpoint(str(checkpoint), map_location="cpu")
    module.eval()
    module.requires_grad_(False)
    module.to(device)
    module.model.teacher_encoder.ema_model.eval()
    if module.model.teacher_encoder.ema_model.training:
        raise AssertionError("EMA Teacher is not in eval mode")
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise AssertionError("Checkpoint module is not frozen")
    return module


def infer_teacher(
    module: Any,
    batch: dict[str, torch.Tensor | list[dict[str, str]]],
    device: torch.device,
) -> np.ndarray:
    """Return one mean-pooled 768-dimensional EMA-Teacher embedding per cell."""
    indices = batch["indices"].to(device, non_blocking=True)
    values = batch["values"].to(device, non_blocking=True)
    offsets = batch["offsets"].to(device, non_blocking=True)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        embeddings = module.model.get_embedding(
            indices=indices,
            values=values,
            offsets=offsets,
            use_teacher=True,
        )
    result = embeddings.float().cpu().numpy()
    if result.ndim != 2 or result.shape[1] != EMBEDDING_DIM:
        raise AssertionError(f"Unexpected GeneJEPA embedding shape: {result.shape}")
    return result
