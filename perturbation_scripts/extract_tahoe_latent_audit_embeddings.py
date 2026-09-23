#!/usr/bin/env python3
"""Audit Tahoe preprocessing and extract Epoch25 EMA-teacher embeddings.

The stable locators in ``tahoe_latent_audit_unique_cells.csv`` are read in
physical parquet order.  Embeddings are scattered back by ``embedding_index``.
Tahoe sentinel removal, gene mapping, log1p, and normalization are delegated to
the production GeneJEPA data classes rather than reimplemented here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from genejepa.configs import DataConfig, ExperimentConfig
from genejepa.data import Tahoe100MDataModule, Tahoe100MDataset


DEFAULT_UNIQUE_CELLS = PROJECT_ROOT / "results" / "tahoe_latent_audit_unique_cells.csv"
DEFAULT_CELL_INDEX_SUMMARY = (
    PROJECT_ROOT / "results" / "tahoe_latent_audit_cell_index_summary.json"
)
DEFAULT_LOCAL_MANIFEST = PROJECT_ROOT / "hf_data_cache" / "local_file_manifest.json"
DEFAULT_STATS = PROJECT_ROOT / "hf_data_cache" / "global_stats.json"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "genejepa_quarter_d12_h6_700k_e30_seed42_run1"
    / "scjepa-epoch=25-val_loss=0.179.ckpt"
)

LOCATOR_VERSION = "tahoe_parquet_row_v1"
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
CSV_COLUMNS = {
    "embedding_index",
    "cell_id",
    "locator_version",
    "cell_locator",
    "shard_path",
    "row_group_index",
    "row_index_in_row_group",
    "row_index_in_shard",
    "plate",
    "sample",
    "drug",
    "cell_line_id",
    "BARCODE_SUB_LIB_ID",
}


@dataclass(frozen=True, slots=True)
class CellLocator:
    embedding_index: int
    cell_id: str
    locator_version: str
    cell_locator: str
    shard_path: str
    row_group_index: int
    row_index_in_row_group: int
    row_index_in_shard: int
    plate: str
    sample: str
    drug: str
    cell_line_id: str
    barcode_sub_lib_id: str


class PreprocessingAudit:
    def __init__(self) -> None:
        self.cells = 0
        self.sentinel_cells = 0
        self.raw_gene_counts: list[int] = []
        self.post_sentinel_gene_counts: list[int] = []
        self.mapped_gene_counts: list[int] = []
        self.unmapped_gene_counts: list[int] = []
        self.normalized_value_count = 0
        self.normalized_min = math.inf
        self.normalized_max = -math.inf
        self.normalized_sum = 0.0
        self.normalized_sumsq = 0.0
        self.normalized_finite = True
        self.inverse_recovery_checked_cells = 0
        self.inverse_recovery_max_abs_error = 0.0

    def add_raw(
        self,
        raw_gene_count: int,
        post_sentinel_count: int,
        mapped_count: int,
        sentinel: bool,
    ) -> None:
        self.cells += 1
        self.sentinel_cells += int(sentinel)
        self.raw_gene_counts.append(raw_gene_count)
        self.post_sentinel_gene_counts.append(post_sentinel_count)
        self.mapped_gene_counts.append(mapped_count)
        self.unmapped_gene_counts.append(post_sentinel_count - mapped_count)

    def add_normalized(self, values: torch.Tensor) -> None:
        array = values.detach().cpu().numpy().astype(np.float64, copy=False)
        self.normalized_finite &= bool(np.isfinite(array).all())
        if array.size:
            self.normalized_value_count += int(array.size)
            self.normalized_min = min(self.normalized_min, float(array.min()))
            self.normalized_max = max(self.normalized_max, float(array.max()))
            self.normalized_sum += float(array.sum(dtype=np.float64))
            self.normalized_sumsq += float(np.square(array).sum(dtype=np.float64))

    @staticmethod
    def _describe(values: list[int]) -> dict[str, float | int]:
        return {
            "min": min(values),
            "median": float(statistics.median(values)),
            "mean": float(statistics.fmean(values)),
            "max": max(values),
        }

    def to_dict(self) -> dict[str, object]:
        mean = self.normalized_sum / self.normalized_value_count
        variance = max(
            0.0,
            self.normalized_sumsq / self.normalized_value_count - mean * mean,
        )
        return {
            "cells": self.cells,
            "sentinel_cells": self.sentinel_cells,
            "raw_gene_count": self._describe(self.raw_gene_counts),
            "post_sentinel_gene_count": self._describe(
                self.post_sentinel_gene_counts
            ),
            "mapped_gene_count": self._describe(self.mapped_gene_counts),
            "unmapped_gene_count": self._describe(self.unmapped_gene_counts),
            "normalized_values": {
                "count": self.normalized_value_count,
                "finite": self.normalized_finite,
                "min": self.normalized_min,
                "max": self.normalized_max,
                "mean": mean,
                "std": math.sqrt(variance),
            },
            "inverse_recovery_checked_cells": self.inverse_recovery_checked_cells,
            "inverse_recovery_max_abs_error": (
                self.inverse_recovery_max_abs_error
                if self.inverse_recovery_checked_cells
                else None
            ),
        }


class EmbeddingAudit:
    def __init__(self) -> None:
        self.cells = 0
        self.value_count = 0
        self.value_min = math.inf
        self.value_max = -math.inf
        self.value_sum = 0.0
        self.value_sumsq = 0.0
        self.negative_count = 0
        self.finite = True
        self.norms: list[float] = []

    def add(self, embeddings: np.ndarray) -> None:
        values = embeddings.astype(np.float64, copy=False)
        self.cells += int(values.shape[0])
        self.value_count += int(values.size)
        self.finite &= bool(np.isfinite(values).all())
        self.value_min = min(self.value_min, float(values.min()))
        self.value_max = max(self.value_max, float(values.max()))
        self.value_sum += float(values.sum(dtype=np.float64))
        self.value_sumsq += float(np.square(values).sum(dtype=np.float64))
        self.negative_count += int(np.count_nonzero(values < 0))
        self.norms.extend(np.linalg.norm(values, axis=1).tolist())

    def to_dict(self) -> dict[str, object]:
        mean = self.value_sum / self.value_count
        variance = max(0.0, self.value_sumsq / self.value_count - mean * mean)
        norms = np.asarray(self.norms, dtype=np.float64)
        return {
            "shape": [self.cells, EMBEDDING_DIM],
            "dtype": "float32",
            "finite": self.finite,
            "min": self.value_min,
            "max": self.value_max,
            "mean": mean,
            "std": math.sqrt(variance),
            "negative_ratio": self.negative_count / self.value_count,
            "norm": {
                "min": float(norms.min()),
                "median": float(np.median(norms)),
                "mean": float(norms.mean()),
                "std": float(norms.std()),
                "max": float(norms.max()),
            },
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def parse_cell(row: dict[str, str]) -> CellLocator:
    return CellLocator(
        embedding_index=int(row["embedding_index"]),
        cell_id=row["cell_id"],
        locator_version=row["locator_version"],
        cell_locator=row["cell_locator"],
        shard_path=row["shard_path"],
        row_group_index=int(row["row_group_index"]),
        row_index_in_row_group=int(row["row_index_in_row_group"]),
        row_index_in_shard=int(row["row_index_in_shard"]),
        plate=row["plate"],
        sample=row["sample"],
        drug=row["drug"],
        cell_line_id=row["cell_line_id"],
        barcode_sub_lib_id=row["BARCODE_SUB_LIB_ID"],
    )


def load_cell_plan(
    unique_cells_path: Path,
    cell_index_summary_path: Path,
    max_cells: int,
) -> tuple[list[CellLocator], int]:
    with unique_cells_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = CSV_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"unique_cells.csv missing columns: {sorted(missing)}")
        all_cells = [parse_cell(row) for row in reader]

    all_cells.sort(key=lambda cell: cell.embedding_index)
    expected_indices = range(len(all_cells))
    if any(cell.embedding_index != expected for cell, expected in zip(all_cells, expected_indices)):
        raise AssertionError("embedding_index is not contiguous from 0")
    if len({cell.cell_id for cell in all_cells}) != len(all_cells):
        raise AssertionError("Duplicate cell_id in unique_cells.csv")
    if len({cell.cell_locator for cell in all_cells}) != len(all_cells):
        raise AssertionError("Duplicate cell_locator in unique_cells.csv")
    if any(cell.locator_version != LOCATOR_VERSION for cell in all_cells):
        raise AssertionError("Unexpected locator_version")

    with cell_index_summary_path.open("r", encoding="utf-8") as handle:
        cell_index_summary = json.load(handle)
    expected_total = int(cell_index_summary["unique_cell_union"]["total"])
    expected_csv_hash = cell_index_summary["outputs"]["unique_cells_sha256"]
    observed_csv_hash = sha256_file(unique_cells_path)
    if len(all_cells) != expected_total:
        raise AssertionError(
            f"unique_cells row count {len(all_cells)} != audited total {expected_total}"
        )
    if observed_csv_hash != expected_csv_hash:
        raise AssertionError("unique_cells.csv SHA-256 differs from cell-index summary")

    selected_count = len(all_cells) if max_cells == 0 else min(max_cells, len(all_cells))
    selected = all_cells[:selected_count]
    return selected, len(all_cells)


def expected_locator(cell: CellLocator) -> str:
    return (
        f"{cell.shard_path}::row_group={cell.row_group_index}"
        f"::row={cell.row_index_in_row_group}"
    )


def as_text(value: object) -> str:
    return "" if value is None else str(value)


def iter_cells_in_physical_order(
    cells: list[CellLocator],
    read_audit: dict[str, int],
) -> Iterator[tuple[CellLocator, dict[str, object]]]:
    physical = sorted(
        cells,
        key=lambda cell: (
            cell.shard_path,
            cell.row_group_index,
            cell.row_index_in_row_group,
        ),
    )
    for shard_path, shard_group_iter in groupby(
        physical, key=lambda cell: cell.shard_path
    ):
        shard_cells = list(shard_group_iter)
        absolute_path = (PROJECT_ROOT / shard_path).resolve()
        if not absolute_path.is_relative_to(PROJECT_ROOT):
            raise ValueError(f"Shard escapes project root: {shard_path}")
        if not absolute_path.is_file():
            raise FileNotFoundError(absolute_path)

        parquet = pq.ParquetFile(absolute_path)
        missing = set(PARQUET_COLUMNS) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{shard_path} missing columns: {sorted(missing)}")
        row_group_offsets = []
        offset = 0
        for row_group_index in range(parquet.num_row_groups):
            row_group_offsets.append(offset)
            offset += parquet.metadata.row_group(row_group_index).num_rows
        read_audit["shards"] += 1

        for row_group_index, row_group_iter in groupby(
            shard_cells, key=lambda cell: cell.row_group_index
        ):
            if not 0 <= row_group_index < parquet.num_row_groups:
                raise AssertionError(f"Row group out of range in {shard_path}")
            selected = list(row_group_iter)
            table = parquet.read_row_group(row_group_index, columns=PARQUET_COLUMNS)
            read_audit["row_groups"] += 1

            for cell in selected:
                row_index = cell.row_index_in_row_group
                if not 0 <= row_index < table.num_rows:
                    raise AssertionError(f"Row index out of range: {cell.cell_locator}")
                locator = expected_locator(cell)
                rebuilt_id = hashlib.sha256(
                    f"{LOCATOR_VERSION}|{locator}".encode("utf-8")
                ).hexdigest()
                if locator != cell.cell_locator or rebuilt_id != cell.cell_id:
                    raise AssertionError(f"Stable locator mismatch: {cell.cell_locator}")
                if (
                    row_group_offsets[row_group_index] + row_index
                    != cell.row_index_in_shard
                ):
                    raise AssertionError(f"Shard row mismatch: {cell.cell_locator}")

                raw = {
                    column: table[column][row_index].as_py()
                    for column in PARQUET_COLUMNS
                }
                expected_metadata = {
                    "plate": cell.plate,
                    "sample": cell.sample,
                    "drug": cell.drug,
                    "cell_line_id": cell.cell_line_id,
                    "BARCODE_SUB_LIB_ID": cell.barcode_sub_lib_id,
                }
                observed_metadata = {
                    key: as_text(raw[key]) for key in expected_metadata
                }
                if observed_metadata != expected_metadata:
                    raise AssertionError(f"Metadata mismatch: {cell.cell_locator}")
                read_audit["locator_metadata_verified_cells"] += 1
                yield cell, raw


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
    metadata_path = resolve_metadata_path(local_manifest_path)
    datamodule = Tahoe100MDataModule(DataConfig(), ExperimentConfig())
    datamodule.stats_path = str(stats_path)
    datamodule._load_stats()
    datamodule._load_metadata(str(metadata_path))
    if not datamodule.gene_map:
        raise RuntimeError("Official Tahoe gene_map is empty")
    return datamodule, metadata_path


def preprocess_once(
    raw_cells: list[dict[str, object]],
    datamodule: Tahoe100MDataModule,
    audit: PreprocessingAudit,
    inverse_check: bool,
) -> dict[str, torch.Tensor | list[dict[str, str]]]:
    if datamodule.gene_map is None:
        raise RuntimeError("gene_map is not initialized")

    expected_mapped_counts = []
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
            raise ValueError("Cell has no genes in the official Tahoe vocabulary")
        expected_mapped_counts.append(mapped_count)
        audit.add_raw(len(genes), len(effective_genes), mapped_count, sentinel)

    # Production transformations happen only in these two existing components.
    processed = list(Tahoe100MDataset(raw_cells, datamodule.gene_map))
    if len(processed) != len(raw_cells):
        raise AssertionError("Official Tahoe dataset dropped a selected cell")
    for item, expected_count in zip(processed, expected_mapped_counts):
        if len(item["gene_indices"]) != expected_count:
            raise AssertionError("Official Tahoe gene mapping count differs from audit")

    batch = datamodule._collate_fn(processed)
    values = batch["values"]
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Official Tahoe collate produced non-finite values")
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
            raise AssertionError("Normalized values do not invert to raw mapped counts")

    return batch


def common_inputs(
    args: argparse.Namespace,
    metadata_path: Path,
    total_unique_cells: int,
) -> dict[str, object]:
    with args.stats.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)
    return {
        "unique_cells_csv": display_path(args.unique_cells),
        "unique_cells_sha256": sha256_file(args.unique_cells),
        "cell_index_summary": display_path(args.cell_index_summary),
        "cell_index_summary_sha256": sha256_file(args.cell_index_summary),
        "local_manifest": display_path(args.local_manifest),
        "local_manifest_sha256": sha256_file(args.local_manifest),
        "gene_metadata": display_path(metadata_path),
        "gene_metadata_sha256": sha256_file(metadata_path),
        "global_stats_file": display_path(args.stats),
        "global_stats_sha256": sha256_file(args.stats),
        "global_mean": float(stats["mean"]),
        "global_std": float(stats["std"]),
        "total_unique_cells": total_unique_cells,
    }


def run_dry_run(args: argparse.Namespace) -> None:
    cells, total_unique_cells = load_cell_plan(
        args.unique_cells, args.cell_index_summary, args.max_cells
    )
    datamodule, metadata_path = build_official_preprocessor(
        args.local_manifest, args.stats
    )
    read_audit = {"shards": 0, "row_groups": 0, "locator_metadata_verified_cells": 0}
    preprocessing_audit = PreprocessingAudit()

    buffered_raw: list[dict[str, object]] = []
    for _, raw in iter_cells_in_physical_order(cells, read_audit):
        buffered_raw.append(raw)
        if len(buffered_raw) == args.batch_size:
            preprocess_once(buffered_raw, datamodule, preprocessing_audit, True)
            buffered_raw = []
    if buffered_raw:
        preprocess_once(buffered_raw, datamodule, preprocessing_audit, True)

    if preprocessing_audit.cells != len(cells):
        raise AssertionError("Dry-run did not preprocess every selected cell")
    result = {
        "created_at_utc": utc_now(),
        "status": "pass",
        "mode": "cpu_preprocessing_dry_run",
        "selected_cells": len(cells),
        "device": "cpu",
        "model_or_checkpoint_loaded": False,
        "production_components_reused": [
            "genejepa.data.Tahoe100MDataset.__iter__",
            "genejepa.data.Tahoe100MDataModule._collate_fn",
        ],
        "transformation_contract": {
            "sentinel_handling": "once in Tahoe100MDataset.__iter__",
            "gene_token_mapping": "once in Tahoe100MDataset.__iter__",
            "log1p": "once in Tahoe100MDataModule._collate_fn",
            "global_normalization": "once in Tahoe100MDataModule._collate_fn",
            "inverse_audit_is_model_input": False,
        },
        "inputs": common_inputs(args, metadata_path, total_unique_cells),
        "physical_read": {
            "order": "shard_path + row_group_index + row_index_in_row_group",
            **read_audit,
        },
        "preprocessing": preprocessing_audit.to_dict(),
        "success_count": len(cells),
        "failure_count": 0,
    }
    write_json(args.summary, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nDry-run summary: {display_path(args.summary)}")


def load_frozen_model(checkpoint: Path, device: torch.device):
    from genejepa.train import JepaLightningModule

    module = JepaLightningModule.load_from_checkpoint(
        str(checkpoint), map_location="cpu"
    )
    module.eval()
    module.requires_grad_(False)
    module.to(device)
    module.model.teacher_encoder.ema_model.eval()
    if module.model.teacher_encoder.ema_model.training:
        raise AssertionError("EMA teacher is not in eval mode")
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise AssertionError("Checkpoint module is not frozen")
    return module


def infer_teacher(
    module,
    batch: dict[str, torch.Tensor | list[dict[str, str]]],
    device: torch.device,
) -> np.ndarray:
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
    return embeddings.float().cpu().numpy()


def run_extraction(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.mode == "full" and args.max_cells != 0:
        raise ValueError("full mode requires --max-cells 0")
    if args.mode == "smoke" and not 256 <= args.max_cells <= 1024:
        raise ValueError("smoke mode requires 256 <= --max-cells <= 1024")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("smoke/full extraction requires an available CUDA device")

    cells, total_unique_cells = load_cell_plan(
        args.unique_cells, args.cell_index_summary, args.max_cells
    )
    if [cell.embedding_index for cell in cells] != list(range(len(cells))):
        raise AssertionError("Selected extraction rows must be an embedding_index prefix")
    datamodule, metadata_path = build_official_preprocessor(
        args.local_manifest, args.stats
    )

    output_path = args.output
    partial_path = output_path.with_name(output_path.name + ".partial")
    progress_path = output_path.with_name(output_path.stem + "_progress.json")
    occupied = [path for path in (output_path, partial_path, args.summary) if path.exists()]
    if occupied and not args.overwrite:
        raise FileExistsError(
            "Output exists; inspect it or rerun with --overwrite: "
            + ", ".join(map(str, occupied))
        )
    if args.overwrite:
        for path in occupied:
            path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Hashing checkpoint: {display_path(args.checkpoint)}", flush=True)
    checkpoint_sha256 = sha256_file(args.checkpoint)
    print(f"Loading frozen EMA-teacher checkpoint on {device}...", flush=True)
    module = load_frozen_model(args.checkpoint, device)
    print("Checkpoint loaded; starting grouped parquet extraction.", flush=True)

    output = np.lib.format.open_memmap(
        partial_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(cells), EMBEDDING_DIM),
    )
    completed = np.zeros(len(cells), dtype=bool)
    read_audit = {"shards": 0, "row_groups": 0, "locator_metadata_verified_cells": 0}
    preprocessing_audit = PreprocessingAudit()
    embedding_audit = EmbeddingAudit()
    repeat_exact = True
    repeat_allclose = True
    repeat_max_abs_diff = 0.0
    started = time.time()
    batch_number = 0

    buffered_cells: list[CellLocator] = []
    buffered_raw: list[dict[str, object]] = []

    def process_buffer() -> None:
        nonlocal batch_number, repeat_exact, repeat_allclose, repeat_max_abs_diff
        if not buffered_cells:
            return
        batch_number += 1
        batch = preprocess_once(
            buffered_raw, datamodule, preprocessing_audit, inverse_check=False
        )
        first = infer_teacher(module, batch, device)
        expected_shape = (len(buffered_cells), EMBEDDING_DIM)
        if first.shape != expected_shape:
            raise AssertionError(
                f"Embedding shape {first.shape} != expected {expected_shape}"
            )
        if not np.isfinite(first).all():
            raise ValueError(f"Non-finite embedding in batch {batch_number}")

        if args.mode == "smoke":
            second = infer_teacher(module, batch, device)
            difference = float(np.max(np.abs(first - second)))
            repeat_exact &= bool(np.array_equal(first, second))
            repeat_allclose &= bool(np.allclose(first, second, rtol=0.0, atol=0.0))
            repeat_max_abs_diff = max(repeat_max_abs_diff, difference)

        indices = np.fromiter(
            (cell.embedding_index for cell in buffered_cells),
            dtype=np.int64,
            count=len(buffered_cells),
        )
        if completed[indices].any():
            raise AssertionError("A unique cell would be embedded more than once")
        output[indices] = first.astype(np.float32, copy=False)
        completed[indices] = True
        embedding_audit.add(first)

        if batch_number % 10 == 0 or completed.all():
            output.flush()
            elapsed = time.time() - started
            processed = int(completed.sum())
            rate = processed / elapsed if elapsed else 0.0
            remaining = (len(cells) - processed) / rate if rate else None
            write_json(
                progress_path,
                {
                    "updated_at_utc": utc_now(),
                    "mode": args.mode,
                    "processed_cells": processed,
                    "total_cells": len(cells),
                    "batch": batch_number,
                    "cells_per_second": rate,
                    "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": remaining,
                    "partial_embedding_file": display_path(partial_path),
                },
            )
            eta = "unknown" if remaining is None else f"{remaining / 60:.1f} min"
            print(
                f"batch={batch_number} cells={processed}/{len(cells)} "
                f"rate={rate:.2f}/s ETA={eta}",
                flush=True,
            )

    for cell, raw in iter_cells_in_physical_order(cells, read_audit):
        buffered_cells.append(cell)
        buffered_raw.append(raw)
        if len(buffered_cells) == args.batch_size:
            process_buffer()
            buffered_cells.clear()
            buffered_raw.clear()
    process_buffer()

    output.flush()
    if not completed.all():
        raise AssertionError(f"Only {int(completed.sum())}/{len(cells)} cells completed")
    if embedding_audit.cells != len(cells):
        raise AssertionError("Embedding audit count does not match selected cells")
    if args.mode == "smoke" and not repeat_exact:
        raise AssertionError("Repeated eval/EMA-teacher inference was not exactly identical")

    del output
    os.replace(partial_path, output_path)
    elapsed = time.time() - started
    result = {
        "created_at_utc": utc_now(),
        "status": "pass",
        "mode": args.mode,
        "checkpoint": {
            "path": display_path(args.checkpoint),
            "sha256": checkpoint_sha256,
            "size_bytes": args.checkpoint.stat().st_size,
            "ema_teacher": True,
            "frozen": True,
            "eval_mode": True,
            "get_embedding_use_teacher": True,
        },
        "inputs": common_inputs(args, metadata_path, total_unique_cells),
        "preprocessing_contract": {
            "sentinel_and_gene_mapping": "Tahoe100MDataset.__iter__, once per cell",
            "log1p_and_global_normalization": "Tahoe100MDataModule._collate_fn, once per cell",
            "model_input_reused_for_repeat_check": args.mode == "smoke",
        },
        "physical_read": {
            "order": "shard_path + row_group_index + row_index_in_row_group",
            "scatter_key": "embedding_index",
            **read_audit,
        },
        "embedding_index_contract": {
            "output_row_equals_embedding_index": True,
            "first_index": 0,
            "last_index": len(cells) - 1,
            "all_indices_written_once": True,
        },
        "output": {
            "embedding_file": display_path(output_path),
            "embedding_sha256": sha256_file(output_path),
            "format": "NumPy .npy",
            "shape": [len(cells), EMBEDDING_DIM],
            "dtype": "float32",
        },
        "preprocessing": preprocessing_audit.to_dict(),
        "embedding_statistics": embedding_audit.to_dict(),
        "repeat_inference": (
            {
                "performed": True,
                "same_preprocessed_inputs": True,
                "exactly_equal": repeat_exact,
                "allclose_rtol_0_atol_0": repeat_allclose,
                "max_abs_difference": repeat_max_abs_diff,
            }
            if args.mode == "smoke"
            else {"performed": False}
        ),
        "inference": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "autocast": "bfloat16",
            "batch_size": args.batch_size,
            "elapsed_seconds": elapsed,
            "cells_per_second": len(cells) / elapsed,
        },
        "success_count": len(cells),
        "failure_count": 0,
    }
    write_json(args.summary, result)
    if progress_path.exists():
        progress_path.unlink()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nEmbedding file: {display_path(output_path)}")
    print(f"Manifest: {display_path(args.summary)}")


def resolve_cli_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dry-run", "smoke", "full"))
    parser.add_argument("--unique-cells", type=resolve_cli_path, default=DEFAULT_UNIQUE_CELLS)
    parser.add_argument(
        "--cell-index-summary",
        type=resolve_cli_path,
        default=DEFAULT_CELL_INDEX_SUMMARY,
    )
    parser.add_argument(
        "--local-manifest", type=resolve_cli_path, default=DEFAULT_LOCAL_MANIFEST
    )
    parser.add_argument("--stats", type=resolve_cli_path, default=DEFAULT_STATS)
    parser.add_argument("--checkpoint", type=resolve_cli_path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=resolve_cli_path)
    parser.add_argument("--summary", type=resolve_cli_path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.max_cells is None:
        args.max_cells = {"dry-run": 32, "smoke": 256, "full": 0}[args.mode]
    if args.max_cells < 0 or args.batch_size < 1:
        raise ValueError("--max-cells must be >= 0 and --batch-size must be >= 1")
    if args.mode == "dry-run":
        if args.max_cells == 0:
            raise ValueError("dry-run requires a small positive --max-cells")
        args.summary = args.summary or (
            PROJECT_ROOT / "results" / "tahoe_latent_audit_preprocessing_dry_run.json"
        )
        if args.output is not None:
            raise ValueError("dry-run does not create an embedding --output")
    elif args.mode == "smoke":
        stem = f"tahoe_latent_audit_epoch25_smoke_{args.max_cells}"
        args.output = args.output or PROJECT_ROOT / "results" / f"{stem}_embeddings.npy"
        args.summary = args.summary or PROJECT_ROOT / "results" / f"{stem}_manifest.json"
    else:
        args.output = args.output or (
            PROJECT_ROOT / "results" / "tahoe_latent_audit_epoch25_embeddings.npy"
        )
        args.summary = args.summary or (
            PROJECT_ROOT / "results" / "tahoe_latent_audit_epoch25_manifest.json"
        )
    return args


def main() -> None:
    args = finalize_args(build_parser().parse_args())
    required_inputs = (
        args.unique_cells,
        args.cell_index_summary,
        args.local_manifest,
        args.stats,
    )
    for path in required_inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.mode == "dry-run":
        run_dry_run(args)
    else:
        run_extraction(args)


if __name__ == "__main__":
    main()
