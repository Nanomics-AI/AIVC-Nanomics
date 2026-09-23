#!/usr/bin/env python3
"""Paired Tahoe GeneJEPA-latent/expression data for Decoder v1.

The formal path streams the frozen Experiment 1 locator plans in physical
Tahoe shard/row-group order and looks up the already-extracted latent by its
global embedding_index.  It never reruns GeneJEPA and never materializes a
dense all-cell expression matrix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
LATENT_DIM = 768
CP_SCALE = 10_000.0
OWNER_TREATED = 0
FORMAL_PREFIX = RESULTS / "tahoe_experiment1_cache_cap512_all_dmso"
FORMAL_EMBEDDINGS = Path(str(FORMAL_PREFIX) + "_embeddings.npy")
FORMAL_EMBEDDING_MANIFEST = Path(str(FORMAL_PREFIX) + "_embedding_manifest.json")
FORMAL_CACHE_SUMMARY = Path(str(FORMAL_PREFIX) + "_summary.json")
FORMAL_CONDITION_INDEX = Path(str(FORMAL_PREFIX) + "_condition_index.csv")
FORMAL_PLANS = tuple(
    Path(str(FORMAL_PREFIX) + f"_worker{worker}_plan.parquet") for worker in range(2)
)
GENE_METADATA = PROJECT_ROOT / "hf_data_cache/metadata/metadata/gene_metadata.parquet"
DEFAULT_PANEL = RESULTS / "genejepa_decoder_v1_gene_panel.csv"
DEFAULT_PANEL_SUMMARY = RESULTS / "genejepa_decoder_v1_gene_panel.json"
DEFAULT_PANEL_STATE = RESULTS / "genejepa_decoder_v1_gene_panel_builder_state.npz"
DEFAULT_PANEL_PROGRESS = RESULTS / "genejepa_decoder_v1_gene_panel_builder_progress.json"
PLAN_COLUMNS = (
    "embedding_index",
    "shard_path",
    "row_group_index",
    "row_index_in_row_group",
    "row_index_in_shard",
    "owner_type",
    "owner_index",
)
RAW_COLUMNS = (
    "genes",
    "expressions",
    "drug",
    "sample",
    "BARCODE_SUB_LIB_ID",
    "cell_line_id",
    "plate",
)
SPLIT_TO_CODE = {"train": 0, "val": 1, "test": 2}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def load_gene_universe(
    metadata_path: Path = GENE_METADATA,
) -> tuple[pd.DataFrame, np.ndarray]:
    genes = pq.read_table(metadata_path).to_pandas()
    required = {"gene_symbol", "ensembl_id", "token_id"}
    if set(genes.columns) != required:
        raise AssertionError(f"Unexpected gene metadata columns: {list(genes.columns)}")
    genes = genes.sort_values("token_id", kind="stable").reset_index(drop=True)
    if genes["token_id"].duplicated().any():
        raise AssertionError("Gene metadata has duplicate token_id")
    token_ids = genes["token_id"].to_numpy(np.int64)
    if len(token_ids) == 0 or token_ids.min() < 0:
        raise AssertionError("Gene metadata has no usable non-negative token IDs")
    lookup = np.full(int(token_ids.max()) + 1, -1, dtype=np.int32)
    lookup[token_ids] = np.arange(len(token_ids), dtype=np.int32)
    genes.insert(0, "genejepa_index", np.arange(len(genes), dtype=np.int32))
    return genes, lookup


def load_conditions(path: Path = FORMAL_CONDITION_INDEX) -> pd.DataFrame:
    required = {
        "pair_id",
        "cache_condition_index",
        "edge_id",
        "split",
        "plate",
        "cell_line_id",
        "drug",
        "dose_uM",
        "treated_samples",
        "treated_cached_cell_count",
        "treated_embedding_start",
        "treated_embedding_stop_exclusive",
    }
    conditions = pd.read_csv(
        path,
        usecols=sorted(required),
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    missing = required - set(conditions.columns)
    if missing:
        raise AssertionError(f"Condition index is missing columns: {sorted(missing)}")
    conditions = conditions.sort_values("cache_condition_index").reset_index(drop=True)
    expected = np.arange(len(conditions), dtype=np.int64)
    if not np.array_equal(conditions["cache_condition_index"].to_numpy(np.int64), expected):
        raise AssertionError("cache_condition_index is not contiguous")
    if conditions["pair_id"].duplicated().any() or set(conditions["split"]) != set(SPLIT_TO_CODE):
        raise AssertionError("Condition IDs or split labels are invalid")
    return conditions


def prepare_mapped_counts(
    genes: list[int],
    expressions: list[float],
    token_lookup: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply the production sentinel/map contract, but no encoder normalization."""
    gene_values = np.asarray(genes, dtype=np.int64)
    count_values = np.asarray(expressions, dtype=np.float64)
    if gene_values.ndim != 1 or count_values.ndim != 1 or len(gene_values) != len(count_values):
        raise ValueError("genes/expressions must be equal-length one-dimensional lists")
    if len(gene_values) == 0:
        raise ValueError("Empty genes/expressions")

    sentinel = bool(count_values[0] < 0)
    if sentinel:
        gene_values = gene_values[1:]
        count_values = count_values[1:]
    if len(gene_values) == 0:
        raise ValueError("No genes remain after sentinel removal")
    if not np.isfinite(count_values).all() or np.any(count_values < 0):
        raise ValueError("Non-finite or negative expression after sentinel removal")

    in_lookup = (gene_values >= 0) & (gene_values < len(token_lookup))
    mapped = np.full(len(gene_values), -1, dtype=np.int32)
    mapped[in_lookup] = token_lookup[gene_values[in_lookup]]
    keep = mapped >= 0
    mapped = mapped[keep]
    mapped_counts = count_values[keep]
    if len(mapped) == 0:
        raise ValueError("Cell has no genes in the GeneJEPA vocabulary")

    noninteger_count_entries = int(
        np.count_nonzero(mapped_counts != np.rint(mapped_counts))
    )
    duplicate_gene_entries = 0
    if len(mapped) > 1 and np.any(mapped[1:] <= mapped[:-1]):
        unique, inverse = np.unique(mapped, return_inverse=True)
        collapsed = np.zeros(len(unique), dtype=np.float64)
        np.add.at(collapsed, inverse, mapped_counts)
        duplicate_gene_entries = int(len(mapped) - len(unique))
        mapped, mapped_counts = unique.astype(np.int32, copy=False), collapsed

    library_size = float(mapped_counts.sum(dtype=np.float64))
    if not math.isfinite(library_size) or library_size <= 0:
        raise ValueError("Mapped-gene library size is not positive and finite")
    audit = {
        "sentinel_removed": sentinel,
        "raw_entries": int(len(genes)),
        "post_sentinel_entries": int(len(gene_values)),
        "mapped_entries": int(len(mapped)),
        "unmapped_entries": int(np.count_nonzero(~keep)),
        "duplicate_gene_entries_collapsed": duplicate_gene_entries,
        "noninteger_count_entries": noninteger_count_entries,
        "library_size": library_size,
    }
    return mapped, mapped_counts, audit


def log1p_cp10k_target(
    genes: list[int],
    expressions: list[float],
    token_lookup: np.ndarray,
    panel_indices: np.ndarray,
    *,
    panel_lookup: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Full mapped library -> CP10K -> log1p -> panel select."""
    mapped, counts, audit = prepare_mapped_counts(genes, expressions, token_lookup)
    panel_indices = np.asarray(panel_indices, dtype=np.int64)
    if panel_lookup is None:
        panel_lookup = build_panel_lookup(
            panel_indices, vocabulary_size=int(token_lookup.max()) + 1
        )
    else:
        panel_lookup = np.asarray(panel_lookup)
        if panel_lookup.dtype != np.int32 or panel_lookup.ndim != 1:
            raise ValueError("panel_lookup must be a one-dimensional int32 array")
    positions = panel_lookup[mapped]
    selected = positions >= 0
    dense_counts = np.zeros(len(panel_indices), dtype=np.float64)
    dense_counts[positions[selected]] = counts[selected]
    target = np.log1p(CP_SCALE * dense_counts / audit["library_size"]).astype(
        np.float32
    )
    if not np.isfinite(target).all() or np.any(target < 0):
        raise AssertionError("Decoder target is not finite and non-negative")
    audit["panel_nonzero"] = int(np.count_nonzero(dense_counts))
    return target, audit


def build_panel_lookup(
    panel_indices: np.ndarray, *, vocabulary_size: int
) -> np.ndarray:
    """Build the GeneJEPA-index -> panel-position lookup once."""
    panel_indices = np.asarray(panel_indices, dtype=np.int64)
    if panel_indices.ndim != 1 or len(panel_indices) == 0:
        raise ValueError("panel_indices must be a non-empty one-dimensional array")
    if panel_indices.min() < 0 or panel_indices.max() >= vocabulary_size:
        raise ValueError("panel_indices contains an out-of-range GeneJEPA index")
    if len(np.unique(panel_indices)) != len(panel_indices):
        raise ValueError("panel_indices contains duplicates")
    panel_lookup = np.full(vocabulary_size, -1, dtype=np.int32)
    panel_lookup[panel_indices] = np.arange(len(panel_indices), dtype=np.int32)
    return panel_lookup


def load_panel(path: Path, genes: pd.DataFrame) -> np.ndarray:
    panel = pd.read_csv(path, keep_default_na=False, encoding="utf-8-sig")
    required = {"panel_rank", "genejepa_index", "token_id", "gene_symbol", "ensembl_id"}
    missing = required - set(panel.columns)
    if missing:
        raise AssertionError(f"Panel is missing columns: {sorted(missing)}")
    panel = panel.sort_values("panel_rank").reset_index(drop=True)
    if not np.array_equal(panel["panel_rank"].to_numpy(np.int64), np.arange(len(panel))):
        raise AssertionError("panel_rank is not contiguous from zero")
    indices = panel["genejepa_index"].to_numpy(np.int64)
    if len(indices) == 0 or len(np.unique(indices)) != len(indices):
        raise AssertionError("Panel is empty or has duplicate GeneJEPA indices")
    if indices.min() < 0 or indices.max() >= len(genes):
        raise AssertionError("Panel has an out-of-range GeneJEPA index")
    expected = genes.iloc[indices].reset_index(drop=True)
    for column in ("token_id", "gene_symbol", "ensembl_id"):
        if not panel[column].astype(str).eq(expected[column].astype(str)).all():
            raise AssertionError(f"Panel metadata disagrees with gene universe: {column}")
    return indices


def plan_descriptors(plan_paths: tuple[Path, ...] = FORMAL_PLANS) -> list[tuple[int, int]]:
    descriptors: list[tuple[int, int]] = []
    for plan_index, path in enumerate(plan_paths):
        parquet = pq.ParquetFile(path)
        descriptors.extend((plan_index, row_group) for row_group in range(parquet.num_row_groups))
    return descriptors


def _partition() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
    else:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    worker = get_worker_info()
    workers = worker.num_workers if worker is not None else 1
    worker_id = worker.id if worker is not None else 0
    return rank * workers + worker_id, world_size * workers


def iter_descriptor_records(
    *,
    plan: pq.ParquetFile,
    plan_row_group: int,
    condition_records: list[dict[str, Any]],
    condition_split_codes: np.ndarray,
    split: str,
    max_cells_per_shard: int | None = None,
    max_cells_per_source_row_group: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Read one Tahoe shard once, and each selected source row group once."""
    table = plan.read_row_group(plan_row_group, columns=PLAN_COLUMNS)
    owner_type = table["owner_type"].to_numpy(zero_copy_only=False)
    owner_index = table["owner_index"].to_numpy(zero_copy_only=False).astype(
        np.int64, copy=False
    )
    if np.any((owner_index < 0) | (owner_index >= len(condition_records))):
        # DMSO owner indices use a different namespace and are removed below.
        bad_treated = (owner_type == OWNER_TREATED) & (
            (owner_index < 0) | (owner_index >= len(condition_records))
        )
        if np.any(bad_treated):
            raise AssertionError("Treated owner_index is out of range")
    keep = owner_type == OWNER_TREATED
    keep[keep] &= condition_split_codes[owner_index[keep]] == SPLIT_TO_CODE[split]
    positions = np.flatnonzero(keep)
    if len(positions) == 0:
        return

    shard_paths = table["shard_path"].unique().to_pylist()
    if len(shard_paths) != 1:
        raise AssertionError("A plan row group must describe exactly one Tahoe shard")
    shard_path = str(shard_paths[0])
    source = pq.ParquetFile(PROJECT_ROOT / shard_path)
    source_groups = table["row_group_index"].to_numpy(zero_copy_only=False).astype(
        np.int64, copy=False
    )
    rows_in_group = table["row_index_in_row_group"].to_numpy(
        zero_copy_only=False
    ).astype(np.int64, copy=False)
    rows_in_shard = table["row_index_in_shard"].to_numpy(zero_copy_only=False).astype(
        np.int64, copy=False
    )
    embedding_indices = table["embedding_index"].to_numpy(zero_copy_only=False).astype(
        np.int64, copy=False
    )

    emitted_in_shard = 0
    for source_group in np.unique(source_groups[positions]):
        group_positions = positions[source_groups[positions] == source_group]
        if max_cells_per_source_row_group is not None:
            group_positions = group_positions[:max_cells_per_source_row_group]
        if max_cells_per_shard is not None:
            remaining = max_cells_per_shard - emitted_in_shard
            if remaining <= 0:
                return
            group_positions = group_positions[:remaining]
        raw = source.read_row_group(int(source_group), columns=RAW_COLUMNS)
        for position in group_positions:
            row = int(rows_in_group[position])
            if row < 0 or row >= raw.num_rows:
                raise AssertionError("row_index_in_row_group is out of range")
            condition_index = int(owner_index[position])
            condition = condition_records[condition_index]
            metadata = {column: raw[column][row].as_py() for column in RAW_COLUMNS[2:]}
            if (
                metadata["plate"] != condition["plate"]
                or metadata["cell_line_id"] != condition["cell_line_id"]
                or metadata["drug"] != condition["drug"]
                or metadata["sample"] not in str(condition["treated_samples"]).split("|")
            ):
                raise AssertionError("Raw Tahoe metadata disagrees with condition owner")
            yield {
                "genes": raw["genes"][row].as_py(),
                "expressions": raw["expressions"][row].as_py(),
                "embedding_index": int(embedding_indices[position]),
                "condition_index": condition_index,
                "condition": condition,
                "shard_path": shard_path,
                "row_group_index": int(source_group),
                "row_index_in_row_group": row,
                "row_index_in_shard": int(rows_in_shard[position]),
                **metadata,
            }
            emitted_in_shard += 1


class TahoeDecoderV1PairedDataset(IterableDataset[dict[str, Any]]):
    """Stream exact `(cached latent, raw-derived target)` treated-cell pairs."""

    def __init__(
        self,
        *,
        split: str,
        panel_path: Path,
        embeddings_path: Path = FORMAL_EMBEDDINGS,
        embedding_manifest_path: Path = FORMAL_EMBEDDING_MANIFEST,
        cache_summary_path: Path = FORMAL_CACHE_SUMMARY,
        condition_index_path: Path = FORMAL_CONDITION_INDEX,
        plan_paths: tuple[Path, ...] = FORMAL_PLANS,
        gene_metadata_path: Path = GENE_METADATA,
        shuffle_shards: bool = False,
        seed: int = 42,
        epoch: int = 0,
        max_cells: int | None = None,
        max_cells_per_shard: int | None = None,
        max_cells_per_source_row_group: int | None = None,
    ) -> None:
        super().__init__()
        if split not in SPLIT_TO_CODE:
            raise ValueError("split must be train, val, or test")
        if epoch < 0 or (max_cells is not None and max_cells < 1):
            raise ValueError("epoch/max_cells is invalid")
        self.split = split
        self.panel_path = Path(panel_path)
        self.embeddings_path = Path(embeddings_path)
        self.plan_paths = tuple(Path(path) for path in plan_paths)
        self.shuffle_shards = bool(shuffle_shards)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.max_cells = max_cells
        self.max_cells_per_shard = max_cells_per_shard
        self.max_cells_per_source_row_group = max_cells_per_source_row_group
        self._embeddings: np.memmap | None = None

        manifest = json.loads(Path(embedding_manifest_path).read_text(encoding="utf-8"))
        summary = json.loads(Path(cache_summary_path).read_text(encoding="utf-8"))
        output = manifest["output"]
        integrity = manifest["integrity"]
        if manifest.get("status") != "pass" or any(
            (integrity["missing"], integrity["duplicate"], integrity["worker_overlap"])
        ):
            raise AssertionError("Formal merged cache has not passed integrity checks")
        if integrity.get("finite") is not True or not output.get(
            "row_equals_global_embedding_index"
        ):
            raise AssertionError("Formal cache finite/index contract is missing")
        if self.embeddings_path.stat().st_size != int(output["size_bytes"]):
            raise AssertionError("Formal cache size differs from its pass manifest")
        if tuple(output["shape"]) != (30_839_089, LATENT_DIM) or output["dtype"] != "float32":
            raise AssertionError("Unexpected formal cache shape/dtype")
        if int(summary["embedding_index"]["total_cells"]) != int(output["shape"][0]):
            raise AssertionError("Cache summary and merge manifest disagree")

        self.conditions = load_conditions(Path(condition_index_path))
        self.condition_records = self.conditions.to_dict("records")
        self.condition_split_codes = self.conditions["split"].map(SPLIT_TO_CODE).to_numpy(
            np.int8
        )
        self.genes, self.token_lookup = load_gene_universe(Path(gene_metadata_path))
        self.panel_indices = load_panel(self.panel_path, self.genes)
        self.panel_lookup = build_panel_lookup(
            self.panel_indices, vocabulary_size=len(self.genes)
        )
        self.descriptors = plan_descriptors(self.plan_paths)
        self.expected_cells = int(
            self.conditions.loc[self.conditions["split"].eq(split), "treated_cached_cell_count"].sum()
        )
        self.provenance = {
            "cache_sha256_from_pass_manifest": output["sha256"],
            "cache_hash_recomputed": False,
            "row_equals_global_embedding_index": True,
            "condition_index_sha256": sha256_file(Path(condition_index_path)),
            "gene_metadata_sha256": sha256_file(Path(gene_metadata_path)),
            "panel_sha256": sha256_file(self.panel_path),
            "latent_transforms": [],
            "target_transform": "full mapped library -> CP10000 -> log1p -> panel select",
        }

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.expected_cells if self.max_cells is None else min(
            self.expected_cells, self.max_cells
        )

    def _mmap(self) -> np.memmap:
        if self._embeddings is None:
            embeddings = np.load(self.embeddings_path, mmap_mode="r")
            if not isinstance(embeddings, np.memmap) or embeddings.dtype != np.float32:
                raise AssertionError("Latent cache did not open as float32 memmap")
            self._embeddings = embeddings
        return self._embeddings

    def __iter__(self) -> Iterator[dict[str, Any]]:
        partition_id, partition_count = _partition()
        descriptors = list(self.descriptors)
        if self.shuffle_shards:
            rng = np.random.Generator(np.random.PCG64(self.seed + self.epoch))
            rng.shuffle(descriptors)
        descriptors = descriptors[partition_id::partition_count]
        plans = {index: pq.ParquetFile(path) for index, path in enumerate(self.plan_paths)}
        embeddings = self._mmap()
        emitted = 0
        for plan_index, plan_row_group in descriptors:
            records = iter_descriptor_records(
                plan=plans[plan_index],
                plan_row_group=plan_row_group,
                condition_records=self.condition_records,
                condition_split_codes=self.condition_split_codes,
                split=self.split,
                max_cells_per_shard=self.max_cells_per_shard,
                max_cells_per_source_row_group=self.max_cells_per_source_row_group,
            )
            for record in records:
                embedding_index = record["embedding_index"]
                if embedding_index < 0 or embedding_index >= len(embeddings):
                    raise AssertionError("global embedding_index is out of range")
                latent = np.array(embeddings[embedding_index], dtype=np.float32, copy=True)
                target, target_audit = log1p_cp10k_target(
                    record["genes"],
                    record["expressions"],
                    self.token_lookup,
                    self.panel_indices,
                    panel_lookup=self.panel_lookup,
                )
                condition = record["condition"]
                yield {
                    "latent": torch.from_numpy(latent),
                    "target_expression": torch.from_numpy(target),
                    "global_embedding_index": embedding_index,
                    "split": self.split,
                    "condition_id": str(condition["pair_id"]),
                    "pair_id": str(condition["pair_id"]),
                    "edge_id": str(condition["edge_id"]),
                    "cell_line_id": str(condition["cell_line_id"]),
                    "drug": str(condition["drug"]),
                    "dose_uM": float(condition["dose_uM"]),
                    "plate": str(condition["plate"]),
                    "sample": str(record["sample"]),
                    "BARCODE_SUB_LIB_ID": str(record["BARCODE_SUB_LIB_ID"]),
                    "shard_path": str(record["shard_path"]),
                    "row_group_index": int(record["row_group_index"]),
                    "row_index_in_row_group": int(record["row_index_in_row_group"]),
                    "row_index_in_shard": int(record["row_index_in_shard"]),
                    "target_library_size": float(target_audit["library_size"]),
                }
                emitted += 1
                if self.max_cells is not None and emitted >= self.max_cells:
                    return


def _builder_fingerprint(
    *,
    top_k: int,
    max_cells: int | None,
    plan_paths: tuple[Path, ...],
    condition_path: Path,
    metadata_path: Path,
) -> str:
    payload = {
        "schema": "genejepa_decoder_v1_panel_builder_v1",
        "top_k": top_k,
        "max_cells": max_cells,
        "expression_space": "log1p(CP10000 over all mapped GeneJEPA genes)",
        "split": "train treated physical cells only",
        "plans": [sha256_file(path) for path in plan_paths],
        "condition_index": sha256_file(condition_path),
        "gene_metadata": sha256_file(metadata_path),
        "builder_code": sha256_file(Path(__file__)),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _save_builder_state(
    path: Path,
    *,
    fingerprint: str,
    next_descriptor: int,
    cells: int,
    value_sum: np.ndarray,
    value_sumsq: np.ndarray,
    audit_counts: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            fingerprint=np.asarray(fingerprint),
            next_descriptor=np.asarray(next_descriptor, dtype=np.int64),
            cells=np.asarray(cells, dtype=np.int64),
            value_sum=value_sum,
            value_sumsq=value_sumsq,
            audit_counts=audit_counts,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_panel(
    path: Path,
    genes: pd.DataFrame,
    variance: np.ndarray,
    mean: np.ndarray,
    top_k: int,
) -> None:
    indices = np.lexsort((np.arange(len(variance), dtype=np.int64), -variance))[:top_k]
    selected = genes.iloc[indices].copy().reset_index(drop=True)
    selected.insert(0, "panel_rank", np.arange(len(selected), dtype=np.int32))
    selected["mean_log1p_cp10k"] = mean[indices]
    selected["population_variance_log1p_cp10k"] = variance[indices]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    selected.to_csv(temporary, index=False, encoding="utf-8-sig", lineterminator="\n")
    os.replace(temporary, path)


def build_panel(args: argparse.Namespace) -> None:
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    if args.max_cells is not None and args.max_cells < 1:
        raise ValueError("--max-cells must be positive when supplied")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Panel statistics builder is single-process and must not use torchrun")
    if args.max_cells is not None:
        smoke_stem = f"genejepa_decoder_v1_gene_panel_smoke_only_{args.max_cells}"
        if args.output == DEFAULT_PANEL:
            args.output = RESULTS / f"{smoke_stem}.csv"
        if args.summary == DEFAULT_PANEL_SUMMARY:
            args.summary = RESULTS / f"{smoke_stem}.json"
        if args.state == DEFAULT_PANEL_STATE:
            args.state = RESULTS / f"{smoke_stem}_state.npz"
        if args.progress == DEFAULT_PANEL_PROGRESS:
            args.progress = RESULTS / f"{smoke_stem}_progress.json"

    conditions = load_conditions(args.condition_index)
    condition_records = conditions.to_dict("records")
    condition_split_codes = conditions["split"].map(SPLIT_TO_CODE).to_numpy(np.int8)
    genes, token_lookup = load_gene_universe(args.gene_metadata)
    if args.top_k > len(genes):
        raise ValueError("--top-k exceeds the GeneJEPA vocabulary")
    descriptors = plan_descriptors(args.plans)
    fingerprint = _builder_fingerprint(
        top_k=args.top_k,
        max_cells=args.max_cells,
        plan_paths=args.plans,
        condition_path=args.condition_index,
        metadata_path=args.gene_metadata,
    )
    expected_train_cells = int(
        conditions.loc[conditions["split"].eq("train"), "treated_cached_cell_count"].sum()
    )

    if args.state.exists():
        with np.load(args.state, allow_pickle=False) as state:
            if str(state["fingerprint"].item()) != fingerprint:
                raise AssertionError("Builder state belongs to a different input contract")
            next_descriptor = int(state["next_descriptor"])
            cells = int(state["cells"])
            value_sum = state["value_sum"].astype(np.float64, copy=True)
            value_sumsq = state["value_sumsq"].astype(np.float64, copy=True)
            audit_counts = state["audit_counts"].astype(np.int64, copy=True)
    else:
        next_descriptor, cells = 0, 0
        value_sum = np.zeros(len(genes), dtype=np.float64)
        value_sumsq = np.zeros(len(genes), dtype=np.float64)
        # sentinel, unmapped, noninteger, duplicates, zero_library/errors
        audit_counts = np.zeros(5, dtype=np.int64)

    plans = {index: pq.ParquetFile(path) for index, path in enumerate(args.plans)}
    started = time.perf_counter()
    stopped_early = args.max_cells is not None and cells >= args.max_cells
    next_descriptor_after = next_descriptor
    for descriptor_index in range(next_descriptor, len(descriptors)):
        if stopped_early:
            break
        plan_index, plan_row_group = descriptors[descriptor_index]
        for record in iter_descriptor_records(
            plan=plans[plan_index],
            plan_row_group=plan_row_group,
            condition_records=condition_records,
            condition_split_codes=condition_split_codes,
            split="train",
        ):
            try:
                mapped, counts, audit = prepare_mapped_counts(
                    record["genes"], record["expressions"], token_lookup
                )
            except ValueError:
                audit_counts[4] += 1
                raise
            values = np.log1p(CP_SCALE * counts / audit["library_size"])
            np.add.at(value_sum, mapped, values)
            np.add.at(value_sumsq, mapped, values * values)
            audit_counts[0] += int(audit["sentinel_removed"])
            audit_counts[1] += int(audit["unmapped_entries"])
            audit_counts[2] += int(audit["noninteger_count_entries"])
            audit_counts[3] += int(audit["duplicate_gene_entries_collapsed"])
            cells += 1
            if args.max_cells is not None and cells >= args.max_cells:
                stopped_early = True
                break
        completed_descriptor = descriptor_index + 1
        next_descriptor_after = completed_descriptor
        if stopped_early or completed_descriptor % args.checkpoint_every_shards == 0:
            _save_builder_state(
                args.state,
                fingerprint=fingerprint,
                next_descriptor=completed_descriptor,
                cells=cells,
                value_sum=value_sum,
                value_sumsq=value_sumsq,
                audit_counts=audit_counts,
            )
            atomic_write_json(
                args.progress,
                {
                    "updated_at_utc": utc_now(),
                    "status": "smoke_complete" if stopped_early else "running",
                    "processed_train_cells": cells,
                    "expected_train_cells": expected_train_cells,
                    "next_descriptor": completed_descriptor,
                    "total_descriptors": len(descriptors),
                    "cells_per_second": cells / max(time.perf_counter() - started, 1e-9),
                    "state": display_path(args.state),
                },
            )
        if stopped_early:
            break

    full_complete = not stopped_early and cells == expected_train_cells
    if not full_complete and args.max_cells is None:
        raise AssertionError(
            f"Exact builder expected {expected_train_cells} train cells, observed {cells}"
        )
    _save_builder_state(
        args.state,
        fingerprint=fingerprint,
        next_descriptor=next_descriptor_after,
        cells=cells,
        value_sum=value_sum,
        value_sumsq=value_sumsq,
        audit_counts=audit_counts,
    )
    mean = value_sum / cells
    variance = np.maximum(value_sumsq / cells - mean * mean, 0.0)
    write_panel(args.output, genes, variance, mean, args.top_k)
    result = {
        "schema": "genejepa_decoder_v1_panel_statistics_v1",
        "created_at_utc": utc_now(),
        "status": "formal_complete" if full_complete else "smoke_only",
        "formal_panel": full_complete,
        "processed_train_treated_cells": cells,
        "expected_train_treated_cells": expected_train_cells,
        "candidate_universe": len(genes),
        "top_k": args.top_k,
        "statistics_scope": "Decoder train treated physical cells only",
        "expression_space": "full mapped counts -> CP10000 -> log1p",
        "zeros_included": True,
        "variance": "population, ddof=0",
        "ranking": "population variance descending",
        "tie_breaker": "genejepa_index ascending",
        "audit": {
            "sentinel_cells": int(audit_counts[0]),
            "unmapped_entries": int(audit_counts[1]),
            "noninteger_count_entries": int(audit_counts[2]),
            "duplicate_gene_entries_collapsed": int(audit_counts[3]),
            "invalid_or_zero_library_cells": int(audit_counts[4]),
        },
        "output": {
            "path": display_path(args.output),
            "sha256": sha256_file(args.output),
            "encoding": "utf-8-sig",
        },
        "builder": {
            "path": display_path(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
            "fingerprint": fingerprint,
            "resume_state": display_path(args.state),
        },
    }
    atomic_write_json(args.summary, result)
    atomic_write_json(args.progress, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-panel")
    build.add_argument("--top-k", type=int, default=5000)
    build.add_argument("--max-cells", type=int)
    build.add_argument(
        "--output",
        type=resolve_path,
        default=DEFAULT_PANEL,
    )
    build.add_argument(
        "--summary",
        type=resolve_path,
        default=DEFAULT_PANEL_SUMMARY,
    )
    build.add_argument(
        "--state",
        type=resolve_path,
        default=DEFAULT_PANEL_STATE,
    )
    build.add_argument(
        "--progress",
        type=resolve_path,
        default=DEFAULT_PANEL_PROGRESS,
    )
    build.add_argument("--checkpoint-every-shards", type=int, default=10)
    build.add_argument("--condition-index", type=resolve_path, default=FORMAL_CONDITION_INDEX)
    build.add_argument("--gene-metadata", type=resolve_path, default=GENE_METADATA)
    build.add_argument(
        "--plans", type=resolve_path, nargs=2, default=FORMAL_PLANS, metavar=("WORKER0", "WORKER1")
    )
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "build-panel":
        args.plans = tuple(args.plans)
        if args.checkpoint_every_shards < 1:
            raise ValueError("--checkpoint-every-shards must be positive")
        build_panel(args)


if __name__ == "__main__":
    main()
