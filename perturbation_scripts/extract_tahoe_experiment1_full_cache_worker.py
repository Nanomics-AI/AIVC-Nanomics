#!/usr/bin/env python3
"""Resume-safe worker for the frozen Tahoe Experiment 1 GeneJEPA cache plan."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq
import torch

try:
    from extract_tahoe_latent_audit_embeddings import (
        DEFAULT_CHECKPOINT,
        DEFAULT_LOCAL_MANIFEST,
        DEFAULT_STATS,
        EMBEDDING_DIM,
        PARQUET_COLUMNS,
        build_official_preprocessor,
        infer_teacher,
        load_frozen_model,
        preprocess_once,
        resolve_metadata_path,
        sha256_file,
    )
except ModuleNotFoundError:  # Support import as perturbation_scripts.<module>.
    from perturbation_scripts.extract_tahoe_latent_audit_embeddings import (
        DEFAULT_CHECKPOINT,
        DEFAULT_LOCAL_MANIFEST,
        DEFAULT_STATS,
        EMBEDDING_DIM,
        PARQUET_COLUMNS,
        build_official_preprocessor,
        infer_teacher,
        load_frozen_model,
        preprocess_once,
        resolve_metadata_path,
        sha256_file,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
PREFIX = RESULTS / "tahoe_experiment1_cache_cap512_all_dmso"
PLAN_SUMMARY = Path(str(PREFIX) + "_summary.json")
E0_MANIFEST = RESULTS / "tahoe_latent_audit_epoch25_manifest.json"
CONDITION_INDEX = Path(str(PREFIX) + "_condition_index.csv")
CONTROL_INDEX = Path(str(PREFIX) + "_control_pool_index.csv")
REFERENCE_SINGLE_RATE = 85.47608052384246
FORMAL_BATCH_SIZE = 64
EXPECTED_TOTAL = 30_839_089
EXPECTED_TREATED = 28_639_959
EXPECTED_DMSO = 2_199_130
OWNER_TREATED = 0
OWNER_DMSO = 1
PROGRESS_SCHEMA = "tahoe_experiment1_worker_progress_v1"
RUNTIME_MANIFEST_SCHEMA = "tahoe_experiment1_worker_extraction_v1"
PLAN_COLUMNS = [
    "part_index",
    "embedding_index",
    "shard_index",
    "shard_path",
    "row_group_index",
    "row_index_in_row_group",
    "row_index_in_shard",
    "owner_type",
    "owner_index",
]


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def fsync_path(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def allocated_bytes(path: Path) -> int:
    stat = path.stat()
    return int(getattr(stat, "st_blocks", 0) * 512 or stat.st_size)


@dataclass(frozen=True, slots=True)
class WorkerPaths:
    worker_id: int
    plan: Path
    plan_manifest: Path
    progress: Path
    completed: Path
    partial: Path
    final: Path
    runtime_manifest: Path
    lock: Path


def worker_paths(worker_id: int) -> WorkerPaths:
    stem = str(PREFIX) + f"_worker{worker_id}"
    return WorkerPaths(
        worker_id=worker_id,
        plan=Path(stem + "_plan.parquet"),
        plan_manifest=Path(stem + "_manifest.json"),
        progress=Path(stem + "_progress.json"),
        completed=Path(stem + "_completed.npy"),
        partial=Path(stem + "_embeddings.npy.partial"),
        final=Path(stem + "_embeddings.npy"),
        runtime_manifest=Path(stem + "_extraction_manifest.json"),
        lock=Path(stem + "_extraction.lock"),
    )


@contextmanager
def exclusive_worker_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process already holds {path}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started_at_utc={utc_now()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class RunningStats:
    def __init__(self, state: dict[str, Any] | None = None) -> None:
        state = state or {}
        self.count = int(state.get("count", 0))
        minimum = state.get("min")
        maximum = state.get("max")
        self.minimum = math.inf if minimum is None else float(minimum)
        self.maximum = -math.inf if maximum is None else float(maximum)
        self.total = float(state.get("sum", 0.0))
        self.total_squares = float(state.get("sum_squares", 0.0))

    def add_array(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        if not array.size:
            return
        if not np.isfinite(array).all():
            raise ValueError("Non-finite value entered running statistics")
        self.count += int(array.size)
        self.minimum = min(self.minimum, float(array.min()))
        self.maximum = max(self.maximum, float(array.max()))
        self.total += float(array.sum(dtype=np.float64))
        self.total_squares += float(np.square(array).sum(dtype=np.float64))

    def add_scalar(self, value: float) -> None:
        self.add_array(np.asarray([value], dtype=np.float64))

    def merge(self, other: "RunningStats") -> None:
        if not other.count:
            return
        self.count += other.count
        self.minimum = min(self.minimum, other.minimum)
        self.maximum = max(self.maximum, other.maximum)
        self.total += other.total
        self.total_squares += other.total_squares

    def state(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "min": None if not self.count else self.minimum,
            "max": None if not self.count else self.maximum,
            "sum": self.total,
            "sum_squares": self.total_squares,
        }

    def summary(self) -> dict[str, Any]:
        if not self.count:
            return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
        mean = self.total / self.count
        variance = max(0.0, self.total_squares / self.count - mean * mean)
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": mean,
            "std": math.sqrt(variance),
        }


class StreamingPreprocessingAudit:
    def __init__(self, state: dict[str, Any] | None = None) -> None:
        state = state or {}
        self.cells = int(state.get("cells", 0))
        self.sentinel_cells = int(state.get("sentinel_cells", 0))
        self.raw_gene_count = RunningStats(state.get("raw_gene_count"))
        self.post_sentinel_gene_count = RunningStats(
            state.get("post_sentinel_gene_count")
        )
        self.mapped_gene_count = RunningStats(state.get("mapped_gene_count"))
        self.unmapped_gene_count = RunningStats(state.get("unmapped_gene_count"))
        self.normalized_values = RunningStats(state.get("normalized_values"))
        self.inverse_recovery_checked_cells = int(
            state.get("inverse_recovery_checked_cells", 0)
        )
        self.inverse_recovery_max_abs_error = float(
            state.get("inverse_recovery_max_abs_error", 0.0)
        )

    def add_raw(
        self,
        raw_gene_count: int,
        post_sentinel_count: int,
        mapped_count: int,
        sentinel: bool,
    ) -> None:
        self.cells += 1
        self.sentinel_cells += int(sentinel)
        self.raw_gene_count.add_scalar(raw_gene_count)
        self.post_sentinel_gene_count.add_scalar(post_sentinel_count)
        self.mapped_gene_count.add_scalar(mapped_count)
        self.unmapped_gene_count.add_scalar(post_sentinel_count - mapped_count)

    def add_normalized(self, values: torch.Tensor) -> None:
        self.normalized_values.add_array(
            values.detach().cpu().numpy().astype(np.float64, copy=False)
        )

    def merge(self, other: "StreamingPreprocessingAudit") -> None:
        self.cells += other.cells
        self.sentinel_cells += other.sentinel_cells
        for name in (
            "raw_gene_count",
            "post_sentinel_gene_count",
            "mapped_gene_count",
            "unmapped_gene_count",
            "normalized_values",
        ):
            getattr(self, name).merge(getattr(other, name))
        self.inverse_recovery_checked_cells += other.inverse_recovery_checked_cells
        self.inverse_recovery_max_abs_error = max(
            self.inverse_recovery_max_abs_error,
            other.inverse_recovery_max_abs_error,
        )

    def state(self) -> dict[str, Any]:
        return {
            "cells": self.cells,
            "sentinel_cells": self.sentinel_cells,
            "raw_gene_count": self.raw_gene_count.state(),
            "post_sentinel_gene_count": self.post_sentinel_gene_count.state(),
            "mapped_gene_count": self.mapped_gene_count.state(),
            "unmapped_gene_count": self.unmapped_gene_count.state(),
            "normalized_values": self.normalized_values.state(),
            "inverse_recovery_checked_cells": self.inverse_recovery_checked_cells,
            "inverse_recovery_max_abs_error": self.inverse_recovery_max_abs_error,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "cells": self.cells,
            "sentinel_cells": self.sentinel_cells,
            "raw_gene_count": self.raw_gene_count.summary(),
            "post_sentinel_gene_count": self.post_sentinel_gene_count.summary(),
            "mapped_gene_count": self.mapped_gene_count.summary(),
            "unmapped_gene_count": self.unmapped_gene_count.summary(),
            "normalized_values": {
                **self.normalized_values.summary(),
                "finite": True,
            },
            "inverse_recovery_checked_cells": self.inverse_recovery_checked_cells,
            "inverse_recovery_max_abs_error": (
                self.inverse_recovery_max_abs_error
                if self.inverse_recovery_checked_cells
                else None
            ),
        }


class StreamingEmbeddingAudit:
    def __init__(self, state: dict[str, Any] | None = None) -> None:
        state = state or {}
        self.cells = int(state.get("cells", 0))
        self.values = RunningStats(state.get("values"))
        self.norms = RunningStats(state.get("norms"))
        self.negative_count = int(state.get("negative_count", 0))
        self.positive_count = int(state.get("positive_count", 0))

    def add(self, embeddings: np.ndarray) -> None:
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != EMBEDDING_DIM:
            raise ValueError(f"Unexpected embedding shape: {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("Non-finite embedding")
        self.cells += int(array.shape[0])
        self.values.add_array(array)
        self.norms.add_array(np.linalg.norm(array.astype(np.float64), axis=1))
        self.negative_count += int(np.count_nonzero(array < 0))
        self.positive_count += int(np.count_nonzero(array > 0))

    def merge(self, other: "StreamingEmbeddingAudit") -> None:
        self.cells += other.cells
        self.values.merge(other.values)
        self.norms.merge(other.norms)
        self.negative_count += other.negative_count
        self.positive_count += other.positive_count

    def state(self) -> dict[str, Any]:
        return {
            "cells": self.cells,
            "values": self.values.state(),
            "norms": self.norms.state(),
            "negative_count": self.negative_count,
            "positive_count": self.positive_count,
        }

    def summary(self) -> dict[str, Any]:
        value_summary = self.values.summary()
        return {
            "shape": [self.cells, EMBEDDING_DIM],
            "dtype": "float32",
            "finite": True,
            **{key: value_summary[key] for key in ("min", "max", "mean", "std")},
            "negative_ratio": (
                self.negative_count / self.values.count if self.values.count else None
            ),
            "positive_ratio": (
                self.positive_count / self.values.count if self.values.count else None
            ),
            "norm": self.norms.summary(),
        }


@dataclass(frozen=True, slots=True)
class PlanCell:
    part_index: int
    embedding_index: int
    shard_index: int
    shard_path: str
    row_group_index: int
    row_index_in_row_group: int
    row_index_in_shard: int
    owner_type: int
    owner_index: int


@dataclass(slots=True)
class RawBatch:
    cells: list[PlanCell]
    raw_cells: list[dict[str, Any]]
    read_seconds: float
    source_shards_opened: int
    source_row_groups_read: int


def load_metadata_maps() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    treated: list[dict[str, Any]] = []
    with CONDITION_INDEX.open("r", encoding="utf-8-sig", newline="") as handle:
        for expected_index, row in enumerate(csv.DictReader(handle)):
            if int(row["cache_condition_index"]) != expected_index:
                raise AssertionError("cache_condition_index is not contiguous")
            treated.append(
                {
                    "plate": row["plate"],
                    "cell_line_id": row["cell_line_id"],
                    "drug": row["drug"],
                    "samples": frozenset(row["treated_samples"].split("|")),
                }
            )
    controls: list[dict[str, Any]] = []
    with CONTROL_INDEX.open("r", encoding="utf-8-sig", newline="") as handle:
        for expected_index, row in enumerate(csv.DictReader(handle)):
            if int(row["control_pool_index"]) != expected_index:
                raise AssertionError("control_pool_index is not contiguous")
            controls.append(
                {
                    "plate": row["plate"],
                    "cell_line_id": row["cell_line_id"],
                    "drug": row["control_drug"],
                    "samples": frozenset(row["control_samples"].split("|")),
                }
            )
    if len(treated) != 56_993 or len(controls) != 643:
        raise AssertionError("Frozen condition/control index row count changed")
    return treated, controls


def expected_metadata(
    cell: PlanCell,
    treated: list[dict[str, Any]],
    controls: list[dict[str, Any]],
) -> dict[str, Any]:
    if cell.owner_type == OWNER_TREATED:
        return treated[cell.owner_index]
    if cell.owner_type == OWNER_DMSO:
        return controls[cell.owner_index]
    raise AssertionError(f"Unknown owner_type={cell.owner_type}")


def plan_cells_from_table(table) -> list[PlanCell]:
    arrays = {
        column: table[column].to_numpy(zero_copy_only=False) for column in PLAN_COLUMNS
    }
    paths = table["shard_path"].to_pylist()
    return [
        PlanCell(
            part_index=int(arrays["part_index"][index]),
            embedding_index=int(arrays["embedding_index"][index]),
            shard_index=int(arrays["shard_index"][index]),
            shard_path=str(paths[index]),
            row_group_index=int(arrays["row_group_index"][index]),
            row_index_in_row_group=int(arrays["row_index_in_row_group"][index]),
            row_index_in_shard=int(arrays["row_index_in_shard"][index]),
            owner_type=int(arrays["owner_type"][index]),
            owner_index=int(arrays["owner_index"][index]),
        )
        for index in range(table.num_rows)
    ]


def iter_raw_batches(
    plan_path: Path,
    *,
    start: int,
    stop: int,
    batch_size: int,
    treated: list[dict[str, Any]],
    controls: list[dict[str, Any]],
) -> Iterator[RawBatch]:
    plan = pq.ParquetFile(plan_path)
    buffer_cells: list[PlanCell] = []
    buffer_raw: list[dict[str, Any]] = []
    buffer_read_seconds = 0.0
    buffer_shards = 0
    buffer_row_groups = 0
    expected_part = start

    def emit() -> RawBatch:
        nonlocal buffer_cells, buffer_raw, buffer_read_seconds
        nonlocal buffer_shards, buffer_row_groups
        result = RawBatch(
            cells=buffer_cells,
            raw_cells=buffer_raw,
            read_seconds=buffer_read_seconds,
            source_shards_opened=buffer_shards,
            source_row_groups_read=buffer_row_groups,
        )
        buffer_cells = []
        buffer_raw = []
        buffer_read_seconds = 0.0
        buffer_shards = 0
        buffer_row_groups = 0
        return result

    for plan_row_group in range(plan.num_row_groups):
        statistics = plan.metadata.row_group(plan_row_group).column(0).statistics
        if statistics is not None:
            if int(statistics.max) < start:
                continue
            if int(statistics.min) >= stop:
                break
        table = plan.read_row_group(plan_row_group, columns=PLAN_COLUMNS)
        cells = plan_cells_from_table(table)
        cells = [cell for cell in cells if start <= cell.part_index < stop]
        if not cells:
            continue
        if any(
            cell.part_index != expected_part + offset
            for offset, cell in enumerate(cells)
        ):
            raise AssertionError("Worker plan part_index is not a contiguous prefix")

        shard_paths = {cell.shard_path for cell in cells}
        if len(shard_paths) != 1:
            raise AssertionError("A plan row group spans more than one source shard")
        source_path = (PROJECT_ROOT / cells[0].shard_path).resolve()
        if not source_path.is_relative_to(PROJECT_ROOT) or not source_path.is_file():
            raise FileNotFoundError(source_path)
        opened_at = time.perf_counter()
        source = pq.ParquetFile(source_path)
        buffer_read_seconds += time.perf_counter() - opened_at
        buffer_shards += 1
        missing = set(PARQUET_COLUMNS) - set(source.schema_arrow.names)
        if missing:
            raise ValueError(f"{source_path.name} missing columns: {sorted(missing)}")
        row_group_offsets = np.concatenate(
            (
                np.zeros(1, dtype=np.int64),
                np.cumsum(
                    [
                        source.metadata.row_group(index).num_rows
                        for index in range(source.num_row_groups)
                    ],
                    dtype=np.int64,
                )[:-1],
            )
        )

        cursor = 0
        while cursor < len(cells):
            source_row_group = cells[cursor].row_group_index
            group_stop = cursor + 1
            while (
                group_stop < len(cells)
                and cells[group_stop].row_group_index == source_row_group
            ):
                group_stop += 1
            selected = cells[cursor:group_stop]
            read_at = time.perf_counter()
            source_table = source.read_row_group(
                source_row_group, columns=PARQUET_COLUMNS
            )
            buffer_read_seconds += time.perf_counter() - read_at
            buffer_row_groups += 1
            for cell in selected:
                if cell.part_index != expected_part:
                    raise AssertionError("Worker plan yielded a part_index gap")
                row = cell.row_index_in_row_group
                if not 0 <= row < source_table.num_rows:
                    raise AssertionError("Stable row locator is out of range")
                if int(row_group_offsets[source_row_group]) + row != cell.row_index_in_shard:
                    raise AssertionError("row_index_in_shard disagrees with stable locator")
                raw = {
                    column: source_table[column][row].as_py()
                    for column in PARQUET_COLUMNS
                }
                expected = expected_metadata(cell, treated, controls)
                observed = {
                    "plate": "" if raw["plate"] is None else str(raw["plate"]),
                    "cell_line_id": (
                        "" if raw["cell_line_id"] is None else str(raw["cell_line_id"])
                    ),
                    "drug": "" if raw["drug"] is None else str(raw["drug"]),
                }
                if observed != {key: expected[key] for key in observed}:
                    raise AssertionError(
                        f"Locator metadata mismatch at part_index={cell.part_index}"
                    )
                if str(raw["sample"]) not in expected["samples"]:
                    raise AssertionError(
                        f"Locator sample mismatch at part_index={cell.part_index}"
                    )
                buffer_cells.append(cell)
                buffer_raw.append(raw)
                expected_part += 1
                if len(buffer_cells) == batch_size:
                    yield emit()
            cursor = group_stop

    if buffer_cells:
        yield emit()
    if expected_part != stop:
        raise AssertionError(
            f"Plan iterator stopped at {expected_part}, expected {stop}"
        )


class GpuSampler:
    def __init__(self, physical_gpu: int, interval_seconds: float) -> None:
        self.physical_gpu = physical_gpu
        self.interval_seconds = interval_seconds
        self.utilization: list[float] = []
        self.memory_mib: list[float] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.physical_gpu}",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                utilization, memory = result.stdout.strip().split(",")
                self.utilization.append(float(utilization.strip()))
                self.memory_mib.append(float(memory.strip()))
            except Exception as error:  # Monitoring must never stop extraction.
                self.error = f"{type(error).__name__}: {error}"
                return
            self._stop.wait(self.interval_seconds)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 6)

    @staticmethod
    def _summary(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"samples": 0, "min": None, "mean": None, "max": None}
        return {
            "samples": len(values),
            "min": min(values),
            "mean": sum(values) / len(values),
            "max": max(values),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "physical_gpu_index": self.physical_gpu,
            "utilization_percent": self._summary(self.utilization),
            "memory_used_MiB": self._summary(self.memory_mib),
            "error": self.error,
        }


def verify_file_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise AssertionError(f"{label} SHA-256 changed: {observed} != {expected}")
    return observed


def build_provenance(paths: WorkerPaths) -> tuple[dict[str, Any], int]:
    plan_summary = read_json(PLAN_SUMMARY)
    if plan_summary.get("status") != "pass":
        raise AssertionError("Frozen cache plan summary is not pass")
    selection = plan_summary["selection"]
    if (
        int(selection["treated"]["conditions"]) != 56_993
        or int(selection["treated"]["cached_cells"]) != EXPECTED_TREATED
        or int(selection["dmso"]["cached_cells"]) != EXPECTED_DMSO
        or int(plan_summary["embedding_index"]["total_cells"]) != EXPECTED_TOTAL
    ):
        raise AssertionError("Frozen Experiment 1 cache population changed")

    plan_manifest = read_json(paths.plan_manifest)
    plan_key = f"worker{paths.worker_id}_plans"
    expected_plan_sha = plan_summary["outputs"][plan_key]["sha256"]
    plan_sha = verify_file_hash(paths.plan, expected_plan_sha, "worker plan")
    if plan_manifest["plan"]["sha256"] != plan_sha:
        raise AssertionError("Worker plan manifest SHA-256 disagrees with plan summary")
    plan_rows = int(plan_manifest["plan"]["rows"])
    parquet = pq.ParquetFile(paths.plan)
    if parquet.metadata.num_rows != plan_rows:
        raise AssertionError("Worker plan row count changed")
    missing = set(PLAN_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Worker plan missing columns: {sorted(missing)}")

    e0 = read_json(E0_MANIFEST)
    if e0.get("status") != "pass" or not e0["checkpoint"]["ema_teacher"]:
        raise AssertionError("Experiment 0 extraction provenance is not pass/EMA teacher")
    checkpoint_sha = verify_file_hash(
        DEFAULT_CHECKPOINT, e0["checkpoint"]["sha256"], "Epoch25 checkpoint"
    )
    local_manifest_sha = verify_file_hash(
        DEFAULT_LOCAL_MANIFEST,
        e0["inputs"]["local_manifest_sha256"],
        "local Tahoe manifest",
    )
    stats_sha = verify_file_hash(
        DEFAULT_STATS, e0["inputs"]["global_stats_sha256"], "global stats"
    )
    metadata_path = resolve_metadata_path(DEFAULT_LOCAL_MANIFEST)
    metadata_sha = verify_file_hash(
        metadata_path, e0["inputs"]["gene_metadata_sha256"], "gene metadata"
    )
    for name in ("condition_index", "control_pool_index"):
        entry = plan_summary["outputs"][name]
        verify_file_hash(PROJECT_ROOT / entry["path"], entry["sha256"], name)

    global_stats = read_json(DEFAULT_STATS)
    if not math.isclose(float(global_stats["mean"]), float(e0["inputs"]["global_mean"])):
        raise AssertionError("Global mean changed")
    if not math.isclose(float(global_stats["std"]), float(e0["inputs"]["global_std"])):
        raise AssertionError("Global std changed")

    reused_extractor = Path(__file__).with_name(
        "extract_tahoe_latent_audit_embeddings.py"
    )
    provenance = {
        "policy": "treated cap=512 + all eligible DMSO; frozen v1",
        "plan_summary": {
            "path": relative(PLAN_SUMMARY),
            "sha256": sha256_file(PLAN_SUMMARY),
        },
        "worker_plan": {
            "path": relative(paths.plan),
            "sha256": plan_sha,
            "rows": plan_rows,
        },
        "worker_plan_manifest": {
            "path": relative(paths.plan_manifest),
            "sha256": sha256_file(paths.plan_manifest),
        },
        "checkpoint": {
            "path": relative(DEFAULT_CHECKPOINT),
            "sha256": checkpoint_sha,
            "ema_teacher": True,
            "frozen": True,
        },
        "preprocessing": {
            "implementation": relative(reused_extractor),
            "implementation_sha256": sha256_file(reused_extractor),
            "genejepa_data_sha256": sha256_file(PROJECT_ROOT / "genejepa" / "data.py"),
            "genejepa_models_sha256": sha256_file(
                PROJECT_ROOT / "genejepa" / "models.py"
            ),
            "sentinel_and_gene_mapping": "Tahoe100MDataset.__iter__, once per extraction inference",
            "log1p_and_global_normalization": "Tahoe100MDataModule._collate_fn, once per extraction inference",
            "centering": False,
            "whitening": False,
            "rectification": False,
            "raw_signed_float32_output": True,
            "formal_inference_batch_size": FORMAL_BATCH_SIZE,
            "canonical_batch_partition": (
                "worker-local consecutive part_index blocks of 64; final block may be shorter"
            ),
        },
        "local_manifest": {
            "path": relative(DEFAULT_LOCAL_MANIFEST),
            "sha256": local_manifest_sha,
        },
        "gene_metadata": {
            "path": relative(metadata_path),
            "sha256": metadata_sha,
        },
        "global_stats": {
            "path": relative(DEFAULT_STATS),
            "sha256": stats_sha,
            "mean": float(global_stats["mean"]),
            "std": float(global_stats["std"]),
        },
        "condition_index_sha256": sha256_file(CONDITION_INDEX),
        "control_pool_index_sha256": sha256_file(CONTROL_INDEX),
        "worker_code_sha256": sha256_file(Path(__file__)),
        "output_contract": {
            "row": "part_index",
            "global_scatter_key": "embedding_index",
            "shape": [plan_rows, EMBEDDING_DIM],
            "dtype": "float32",
        },
    }
    return provenance, plan_rows


def initial_statistics() -> dict[str, Any]:
    return {
        "preprocessing": StreamingPreprocessingAudit().state(),
        "embedding": StreamingEmbeddingAudit().state(),
    }


def initial_timing() -> dict[str, float]:
    return {
        "active_extraction_seconds": 0.0,
        "parquet_read_seconds": 0.0,
        "preprocessing_seconds": 0.0,
        "inference_seconds": 0.0,
        "output_write_seconds": 0.0,
        "commit_sync_seconds": 0.0,
        "resume_integrity_seconds": 0.0,
        "model_load_seconds": 0.0,
    }


def initialize_or_resume(
    paths: WorkerPaths,
    provenance: dict[str, Any],
    total_cells: int,
    *,
    verify_completed_finite: bool,
) -> tuple[np.memmap, np.memmap, dict[str, Any], int]:
    fingerprint = canonical_hash(provenance)
    progress = read_json(paths.progress)
    placeholder = progress.get("status") == "not_started" and not progress.get(
        "embedding_extraction_started", False
    )
    partial_exists = paths.partial.exists()
    final_exists = paths.final.exists()
    completed_exists = paths.completed.exists()
    if partial_exists and final_exists:
        raise RuntimeError("Both partial and final worker arrays exist")

    if placeholder:
        if partial_exists or final_exists or completed_exists:
            raise RuntimeError(
                "Plan placeholder says not_started but extraction state files exist"
            )
        output = np.lib.format.open_memmap(
            paths.partial,
            mode="w+",
            dtype=np.float32,
            shape=(total_cells, EMBEDDING_DIM),
        )
        output.flush()
        fsync_path(paths.partial)
        completed = np.lib.format.open_memmap(
            paths.completed, mode="w+", dtype=np.bool_, shape=(total_cells,)
        )
        completed[:] = False
        completed.flush()
        fsync_path(paths.completed)
        progress = {
            "schema": PROGRESS_SCHEMA,
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "status": "ready",
            "embedding_extraction_started": True,
            "worker_id": paths.worker_id,
            "provenance_fingerprint": fingerprint,
            "processed_cells": 0,
            "remaining_cells": total_cells,
            "total_cells": total_cells,
            "committed_prefix_stop": 0,
            "sessions": [],
            "resume_count": 0,
            "failed_attempts": 0,
            "retry_sessions": 0,
            "permanent_failed_cells": 0,
            "timing": initial_timing(),
            "statistics": initial_statistics(),
            "physical_read": {
                "source_shard_open_operations": 0,
                "source_row_group_read_operations": 0,
                "locator_metadata_verified_cells": 0,
            },
            "current": None,
            "partial_embedding_file": relative(paths.partial),
            "completed_bitmap_file": relative(paths.completed),
        }
        atomic_write_json(paths.progress, progress)
        return output, completed, progress, 0

    if progress.get("schema") != PROGRESS_SCHEMA:
        raise RuntimeError("Existing progress is not a resumable v1 worker state")
    if progress.get("provenance_fingerprint") != fingerprint:
        raise RuntimeError("Extraction provenance changed; refusing unsafe resume")
    if int(progress["total_cells"]) != total_cells:
        raise RuntimeError("Progress total_cells changed")
    committed = int(progress["committed_prefix_stop"])
    if not 0 <= committed <= total_cells:
        raise RuntimeError("Committed prefix is out of range")
    if progress.get("permanent_failed_cells", 0) != 0:
        raise RuntimeError("Progress contains permanent failed cells")
    if not completed_exists:
        raise RuntimeError("Completed bitmap is missing; file size is not completion")
    array_path = paths.final if final_exists else paths.partial
    if not array_path.exists():
        raise RuntimeError("Embedding array is missing; bitmap alone is not completion")
    if final_exists and committed != total_cells:
        raise RuntimeError("Final array exists before all cells were committed")

    output = np.lib.format.open_memmap(array_path, mode="r+")
    completed = np.lib.format.open_memmap(paths.completed, mode="r+")
    if output.shape != (total_cells, EMBEDDING_DIM) or output.dtype != np.float32:
        raise RuntimeError("Worker embedding array shape/dtype changed")
    if completed.shape != (total_cells,) or completed.dtype != np.bool_:
        raise RuntimeError("Completed bitmap shape/dtype changed")

    checked_at = time.perf_counter()
    # Progress is the atomic commit marker. Bitmap can only lag if a process died
    # after progress replace; repairing that lag is safe because output was fsynced first.
    if committed and not bool(np.asarray(completed[:committed]).all()):
        completed[:committed] = True
        completed.flush()
        fsync_path(paths.completed)
    if committed < total_cells and bool(np.asarray(completed[committed:]).any()):
        raise RuntimeError("Completed bitmap extends beyond atomic committed prefix")
    if int(progress["processed_cells"]) != committed:
        raise RuntimeError("processed_cells disagrees with committed prefix")
    if int(progress["statistics"]["embedding"]["cells"]) != committed:
        raise RuntimeError("Embedding statistics disagree with committed prefix")
    if int(progress["statistics"]["preprocessing"]["cells"]) != committed:
        raise RuntimeError("Preprocessing statistics disagree with committed prefix")
    if verify_completed_finite:
        for start in range(0, committed, 8192):
            chunk = np.asarray(output[start : min(start + 8192, committed)])
            if not np.isfinite(chunk).all():
                raise RuntimeError(f"Non-finite committed output near part_index={start}")
    progress["timing"]["resume_integrity_seconds"] += (
        time.perf_counter() - checked_at
    )
    return output, completed, progress, committed


def progress_payload(
    progress: dict[str, Any],
    paths: WorkerPaths,
    *,
    status: str,
    committed: int,
    total_cells: int,
    session: dict[str, Any],
    active_session_seconds: float,
    gpu: dict[str, Any] | None,
) -> dict[str, Any]:
    timing = progress["timing"]
    active = float(timing["active_extraction_seconds"]) + active_session_seconds
    rate = committed / active if active else 0.0
    remaining = total_cells - committed
    eta = remaining / rate if rate else None
    progress.update(
        {
            "updated_at_utc": utc_now(),
            "status": status,
            "processed_cells": committed,
            "remaining_cells": remaining,
            "committed_prefix_stop": committed,
            "cells_per_active_second": rate,
            "effective_speedup_vs_experiment0_single": (
                rate / REFERENCE_SINGLE_RATE if rate else 0.0
            ),
            "estimated_remaining_seconds": eta,
            "current": session.get("current"),
            "gpu": gpu,
            "partial_logical_bytes": (
                paths.partial.stat().st_size if paths.partial.exists() else None
            ),
            "partial_allocated_bytes": (
                allocated_bytes(paths.partial) if paths.partial.exists() else None
            ),
        }
    )
    return progress


def write_runtime_state(
    paths: WorkerPaths,
    provenance: dict[str, Any],
    progress: dict[str, Any],
    *,
    status: str,
) -> None:
    atomic_write_json(
        paths.runtime_manifest,
        {
            "schema": RUNTIME_MANIFEST_SCHEMA,
            "updated_at_utc": utc_now(),
            "status": status,
            "worker_id": paths.worker_id,
            "provenance": provenance,
            "provenance_fingerprint": canonical_hash(provenance),
            "output": {
                "partial_path": relative(paths.partial),
                "final_path": relative(paths.final),
                **provenance["output_contract"],
            },
            "progress": {
                "path": relative(paths.progress),
                "processed_cells": int(progress["processed_cells"]),
                "remaining_cells": int(progress["remaining_cells"]),
                "failed_attempts": int(progress["failed_attempts"]),
                "permanent_failed_cells": int(progress["permanent_failed_cells"]),
            },
        },
    )


def commit(
    *,
    output: np.memmap,
    completed: np.memmap,
    paths: WorkerPaths,
    progress: dict[str, Any],
    old_committed: int,
    new_committed: int,
    pending_preprocessing: StreamingPreprocessingAudit,
    pending_embeddings: StreamingEmbeddingAudit,
    cumulative_preprocessing: StreamingPreprocessingAudit,
    cumulative_embeddings: StreamingEmbeddingAudit,
    session: dict[str, Any],
    session_started: float,
    status: str,
    gpu: dict[str, Any] | None,
) -> int:
    if new_committed == old_committed:
        return old_committed
    if pending_embeddings.cells != new_committed - old_committed:
        raise AssertionError("Pending embedding count does not match commit interval")
    if pending_preprocessing.cells != pending_embeddings.cells:
        raise AssertionError("Pending preprocessing and embedding counts differ")

    sync_started = time.perf_counter()
    output.flush()
    fsync_path(paths.partial)
    progress["timing"]["commit_sync_seconds"] += time.perf_counter() - sync_started
    cumulative_preprocessing.merge(pending_preprocessing)
    cumulative_embeddings.merge(pending_embeddings)
    progress["statistics"] = {
        "preprocessing": cumulative_preprocessing.state(),
        "embedding": cumulative_embeddings.state(),
    }
    progress_payload(
        progress,
        paths,
        status=status,
        committed=new_committed,
        total_cells=int(progress["total_cells"]),
        session=session,
        active_session_seconds=time.perf_counter() - session_started,
        gpu=gpu,
    )
    # This atomic file is the commit marker and is written only after output fsync.
    atomic_write_json(paths.progress, progress)
    completed[old_committed:new_committed] = True
    completed.flush()
    fsync_path(paths.completed)
    return new_committed


def verify_full_finite(output: np.memmap) -> float:
    started = time.perf_counter()
    for start in range(0, output.shape[0], 8192):
        if not np.isfinite(np.asarray(output[start : start + 8192])).all():
            raise RuntimeError(f"Final non-finite output near part_index={start}")
    return time.perf_counter() - started


def finish_worker(
    *,
    output: np.memmap,
    completed: np.memmap,
    paths: WorkerPaths,
    progress: dict[str, Any],
    provenance: dict[str, Any],
    total_cells: int,
) -> dict[str, Any]:
    if int(progress["committed_prefix_stop"]) != total_cells:
        raise RuntimeError("Cannot finish an incomplete worker")
    if not bool(np.asarray(completed).all()):
        raise RuntimeError("Cannot finish: completed bitmap is not all true")
    finite_scan_seconds = verify_full_finite(output)
    progress["timing"]["final_finite_scan_seconds"] = finite_scan_seconds
    output.flush()
    current_path = Path(output.filename)
    del output
    if current_path == paths.partial:
        os.replace(paths.partial, paths.final)
    elif current_path != paths.final:
        raise RuntimeError(f"Unexpected worker array path: {current_path}")
    hash_started = time.perf_counter()
    output_sha = sha256_file(paths.final)
    hash_seconds = time.perf_counter() - hash_started
    final_array = np.lib.format.open_memmap(paths.final, mode="r")
    if final_array.shape != (total_cells, EMBEDDING_DIM) or final_array.dtype != np.float32:
        raise RuntimeError("Final worker array shape/dtype changed after rename")
    del final_array
    manifest = {
        "schema": RUNTIME_MANIFEST_SCHEMA,
        "created_at_utc": utc_now(),
        "status": "pass",
        "worker_id": paths.worker_id,
        "provenance": provenance,
        "provenance_fingerprint": canonical_hash(provenance),
        "output": {
            "path": relative(paths.final),
            "sha256": output_sha,
            "size_bytes": paths.final.stat().st_size,
            "shape": [total_cells, EMBEDDING_DIM],
            "dtype": "float32",
            "row_equals": "worker-local part_index",
            "global_scatter_key": "embedding_index from frozen worker plan",
        },
        "completed": {
            "cells": total_cells,
            "remaining": 0,
            "failed": int(progress["permanent_failed_cells"]),
            "bitmap_path": relative(paths.completed),
            "bitmap_all_true": True,
        },
        "statistics": {
            "preprocessing": StreamingPreprocessingAudit(
                progress["statistics"]["preprocessing"]
            ).summary(),
            "embedding": StreamingEmbeddingAudit(
                progress["statistics"]["embedding"]
            ).summary(),
        },
        "timing": {**progress["timing"], "output_sha256_seconds": hash_seconds},
        "sessions": progress["sessions"],
    }
    atomic_write_json(paths.runtime_manifest, manifest)
    progress.update(
        {
            "updated_at_utc": utc_now(),
            "status": "finished",
            "remaining_cells": 0,
            "estimated_remaining_seconds": 0.0,
            "partial_embedding_file": None,
            "final_embedding_file": relative(paths.final),
            "final_embedding_sha256": output_sha,
            "runtime_manifest": relative(paths.runtime_manifest),
        }
    )
    atomic_write_json(paths.progress, progress)
    return manifest


class StopController:
    def __init__(self) -> None:
        self.requested = False
        self.signal_name: str | None = None

    def handler(self, signum: int, _frame: Any) -> None:
        self.requested = True
        self.signal_name = signal.Signals(signum).name
        print(
            f"Received {self.signal_name}; stopping after the current batch and commit.",
            flush=True,
        )


def run_worker(args: argparse.Namespace) -> None:
    paths = worker_paths(args.worker_id)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(args.worker_id):
        raise RuntimeError(
            f"worker{args.worker_id} requires CUDA_VISIBLE_DEVICES={args.worker_id}; got {visible!r}"
        )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("Experiment 1 extraction workers must not use DDP/NCCL")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Worker requires exactly one visible CUDA GPU")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    with exclusive_worker_lock(paths.lock):
        provenance, total_cells = build_provenance(paths)
        output, completed, progress, committed = initialize_or_resume(
            paths,
            provenance,
            total_cells,
            verify_completed_finite=not args.skip_resume_finite_scan,
        )
        if progress.get("status") == "finished":
            print(f"worker{args.worker_id} is already finished; nothing to do.")
            return
        if committed == total_cells:
            manifest = finish_worker(
                output=output,
                completed=completed,
                paths=paths,
                progress=progress,
                provenance=provenance,
                total_cells=total_cells,
            )
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
            return
        if args.batch_size != FORMAL_BATCH_SIZE:
            raise RuntimeError(
                f"Formal extraction batch size is frozen at {FORMAL_BATCH_SIZE}; "
                "the batch probe rejected changing it"
            )
        if committed % FORMAL_BATCH_SIZE != 0:
            raise RuntimeError("Incomplete worker resume prefix is not a canonical batch boundary")
        stop_at = total_cells
        if args.max_new_cells:
            stop_at = min(total_cells, committed + args.max_new_cells)
        if stop_at < total_cells and stop_at % FORMAL_BATCH_SIZE != 0:
            raise RuntimeError(
                "--max-new-cells must stop on a canonical worker-local 64-cell boundary"
            )

        session_id = len(progress["sessions"]) + 1
        previous_status = progress.get("status")
        session = {
            "session_id": session_id,
            "started_at_utc": utc_now(),
            "status": "loading_model",
            "start_committed_cells": committed,
            "new_cells_processed": 0,
            "batch_size": args.batch_size,
            "commit_every_batches": args.commit_every_batches,
            "max_new_cells": args.max_new_cells,
            "previous_status": previous_status,
            "current": None,
        }
        if progress["sessions"]:
            progress["resume_count"] += 1
        if previous_status == "error":
            progress["retry_sessions"] += 1
        progress["sessions"].append(session)
        progress_payload(
            progress,
            paths,
            status="loading_model",
            committed=committed,
            total_cells=total_cells,
            session=session,
            active_session_seconds=0.0,
            gpu=None,
        )
        atomic_write_json(paths.progress, progress)
        write_runtime_state(
            paths, provenance, progress, status="loading_model"
        )

        model_load_started = time.perf_counter()
        print(f"worker{args.worker_id}: loading frozen Epoch25 EMA Teacher...", flush=True)
        try:
            datamodule, _metadata_path = build_official_preprocessor(
                DEFAULT_LOCAL_MANIFEST, DEFAULT_STATS
            )
            module = load_frozen_model(DEFAULT_CHECKPOINT, device)
        except BaseException as error:
            progress["timing"]["model_load_seconds"] += (
                time.perf_counter() - model_load_started
            )
            progress["failed_attempts"] += 1
            session.update(
                {
                    "finished_at_utc": utc_now(),
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "uncommitted_cells_to_retry": 0,
                }
            )
            progress["status"] = "error"
            progress["updated_at_utc"] = utc_now()
            progress["last_error"] = {
                "at_utc": utc_now(),
                "type": type(error).__name__,
                "message": str(error),
                "uncommitted_cells_to_retry": 0,
            }
            atomic_write_json(paths.progress, progress)
            write_runtime_state(paths, provenance, progress, status="error")
            del output, completed
            raise
        progress["timing"]["model_load_seconds"] += (
            time.perf_counter() - model_load_started
        )
        torch.cuda.reset_peak_memory_stats(device)
        print(
            f"worker{args.worker_id}: resume prefix={committed:,}/{total_cells:,}; "
            f"batch={args.batch_size}",
            flush=True,
        )

        treated, controls = load_metadata_maps()
        stop = StopController()
        previous_sigint = signal.signal(signal.SIGINT, stop.handler)
        previous_sigterm = signal.signal(signal.SIGTERM, stop.handler)
        sampler = GpuSampler(args.worker_id, args.gpu_sample_interval)
        sampler.start()
        session_started = time.perf_counter()
        cumulative_preprocessing = StreamingPreprocessingAudit(
            progress["statistics"]["preprocessing"]
        )
        cumulative_embeddings = StreamingEmbeddingAudit(
            progress["statistics"]["embedding"]
        )
        pending_preprocessing = StreamingPreprocessingAudit()
        pending_embeddings = StreamingEmbeddingAudit()
        pending_start = committed
        inferred_stop = committed
        session_batches = 0

        try:
            for raw_batch in iter_raw_batches(
                paths.plan,
                start=committed,
                stop=stop_at,
                batch_size=args.batch_size,
                treated=treated,
                controls=controls,
            ):
                progress["timing"]["parquet_read_seconds"] += raw_batch.read_seconds
                progress["physical_read"]["source_shard_open_operations"] += (
                    raw_batch.source_shards_opened
                )
                progress["physical_read"]["source_row_group_read_operations"] += (
                    raw_batch.source_row_groups_read
                )
                progress["physical_read"]["locator_metadata_verified_cells"] += len(
                    raw_batch.cells
                )

                preprocessing_started = time.perf_counter()
                model_batch = preprocess_once(
                    raw_batch.raw_cells,
                    datamodule,
                    pending_preprocessing,
                    inverse_check=False,
                )
                progress["timing"]["preprocessing_seconds"] += (
                    time.perf_counter() - preprocessing_started
                )

                inference_started = time.perf_counter()
                embeddings = infer_teacher(module, model_batch, device)
                progress["timing"]["inference_seconds"] += (
                    time.perf_counter() - inference_started
                )
                if embeddings.shape != (len(raw_batch.cells), EMBEDDING_DIM):
                    raise AssertionError(f"Unexpected embedding shape {embeddings.shape}")
                if not np.isfinite(embeddings).all():
                    raise ValueError("Non-finite GeneJEPA output")
                if not np.any(embeddings < 0) or not np.any(embeddings > 0):
                    raise ValueError("GeneJEPA output did not preserve signed coordinates")

                part_indices = np.fromiter(
                    (cell.part_index for cell in raw_batch.cells),
                    dtype=np.int64,
                    count=len(raw_batch.cells),
                )
                global_indices = np.fromiter(
                    (cell.embedding_index for cell in raw_batch.cells),
                    dtype=np.int64,
                    count=len(raw_batch.cells),
                )
                if part_indices[0] != inferred_stop or not np.array_equal(
                    part_indices,
                    np.arange(inferred_stop, inferred_stop + len(part_indices)),
                ):
                    raise AssertionError("Inference would violate part-index prefix order")
                if bool(np.asarray(completed[part_indices]).any()):
                    raise AssertionError("Inference attempted an already committed cell")
                write_started = time.perf_counter()
                output[part_indices] = embeddings.astype(np.float32, copy=False)
                progress["timing"]["output_write_seconds"] += (
                    time.perf_counter() - write_started
                )
                pending_embeddings.add(embeddings)
                inferred_stop += len(part_indices)
                session_batches += 1
                last = raw_batch.cells[-1]
                session["current"] = {
                    "part_index": last.part_index,
                    "embedding_index": last.embedding_index,
                    "shard_index": last.shard_index,
                    "shard_path": last.shard_path,
                    "row_group_index": last.row_group_index,
                    "row_index_in_row_group": last.row_index_in_row_group,
                }
                session["last_batch_global_embedding_index_min"] = int(
                    global_indices.min()
                )
                session["last_batch_global_embedding_index_max"] = int(
                    global_indices.max()
                )

                should_commit = (
                    session_batches % args.commit_every_batches == 0
                    or inferred_stop == stop_at
                    or stop.requested
                )
                if should_commit:
                    committed = commit(
                        output=output,
                        completed=completed,
                        paths=paths,
                        progress=progress,
                        old_committed=pending_start,
                        new_committed=inferred_stop,
                        pending_preprocessing=pending_preprocessing,
                        pending_embeddings=pending_embeddings,
                        cumulative_preprocessing=cumulative_preprocessing,
                        cumulative_embeddings=cumulative_embeddings,
                        session=session,
                        session_started=session_started,
                        status="running",
                        gpu=sampler.summary(),
                    )
                    pending_start = committed
                    pending_preprocessing = StreamingPreprocessingAudit()
                    pending_embeddings = StreamingEmbeddingAudit()
                    processed = committed
                    elapsed = time.perf_counter() - session_started
                    session_rate = (processed - session["start_committed_cells"]) / elapsed
                    print(
                        f"worker{args.worker_id} cells={processed:,}/{total_cells:,} "
                        f"session_rate={session_rate:.2f}/s "
                        f"global_index={last.embedding_index:,}",
                        flush=True,
                    )
                if stop.requested:
                    break

            if inferred_stop != pending_start:
                committed = commit(
                    output=output,
                    completed=completed,
                    paths=paths,
                    progress=progress,
                    old_committed=pending_start,
                    new_committed=inferred_stop,
                    pending_preprocessing=pending_preprocessing,
                    pending_embeddings=pending_embeddings,
                    cumulative_preprocessing=cumulative_preprocessing,
                    cumulative_embeddings=cumulative_embeddings,
                    session=session,
                    session_started=session_started,
                    status="running",
                    gpu=sampler.summary(),
                )

            sampler.stop()
            session_elapsed = time.perf_counter() - session_started
            progress["timing"]["active_extraction_seconds"] += session_elapsed
            session.update(
                {
                    "finished_at_utc": utc_now(),
                    "status": "complete" if committed == total_cells else "paused",
                    "stop_reason": (
                        stop.signal_name
                        if stop.requested
                        else "max_new_cells"
                        if committed < total_cells
                        else "worker_complete"
                    ),
                    "new_cells_processed": committed
                    - int(session["start_committed_cells"]),
                    "active_seconds": session_elapsed,
                    "cells_per_second": (
                        (committed - int(session["start_committed_cells"]))
                        / session_elapsed
                        if session_elapsed
                        else 0.0
                    ),
                    "batches": session_batches,
                    "gpu": sampler.summary(),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                }
            )
            progress_payload(
                progress,
                paths,
                status="finalizing" if committed == total_cells else "paused",
                committed=committed,
                total_cells=total_cells,
                session=session,
                active_session_seconds=0.0,
                gpu=sampler.summary(),
            )
            atomic_write_json(paths.progress, progress)
            if committed == total_cells:
                manifest = finish_worker(
                    output=output,
                    completed=completed,
                    paths=paths,
                    progress=progress,
                    provenance=provenance,
                    total_cells=total_cells,
                )
                print(json.dumps(manifest, ensure_ascii=False, indent=2))
            else:
                write_runtime_state(paths, provenance, progress, status="paused")
                del output, completed
                print(
                    json.dumps(
                        {
                            "status": "paused",
                            "worker_id": args.worker_id,
                            "processed_cells": committed,
                            "remaining_cells": total_cells - committed,
                            "resume_command_uses_same_plan_and_output": True,
                            "progress": relative(paths.progress),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
        except BaseException as error:
            sampler.stop()
            session_elapsed = time.perf_counter() - session_started
            progress["timing"]["active_extraction_seconds"] += session_elapsed
            progress["failed_attempts"] += 1
            session.update(
                {
                    "finished_at_utc": utc_now(),
                    "status": "error",
                    "new_cells_processed": committed
                    - int(session["start_committed_cells"]),
                    "active_seconds": session_elapsed,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "uncommitted_cells_to_retry": inferred_stop - committed,
                    "gpu": sampler.summary(),
                }
            )
            progress["last_error"] = {
                "at_utc": utc_now(),
                "type": type(error).__name__,
                "message": str(error),
                "uncommitted_cells_to_retry": inferred_stop - committed,
            }
            progress_payload(
                progress,
                paths,
                status="error",
                committed=committed,
                total_cells=total_cells,
                session=session,
                active_session_seconds=0.0,
                gpu=sampler.summary(),
            )
            atomic_write_json(paths.progress, progress)
            write_runtime_state(paths, provenance, progress, status="error")
            del output, completed
            raise
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)


def status_payload() -> dict[str, Any]:
    workers = []
    for worker_id in (0, 1):
        paths = worker_paths(worker_id)
        progress = read_json(paths.progress)
        workers.append(
            {
                "worker_id": worker_id,
                "status": progress.get("status"),
                "processed_cells": int(progress.get("processed_cells", 0)),
                "remaining_cells": int(
                    progress.get("remaining_cells", progress.get("total_cells", 0))
                ),
                "cells_per_active_second": float(
                    progress.get("cells_per_active_second", 0.0)
                ),
                "estimated_remaining_seconds": progress.get(
                    "estimated_remaining_seconds"
                ),
                "failed_attempts": int(progress.get("failed_attempts", 0)),
                "permanent_failed_cells": int(
                    progress.get("permanent_failed_cells", 0)
                ),
                "current": progress.get("current"),
                "gpu": progress.get("gpu"),
                "progress_path": relative(paths.progress),
            }
        )
    combined_rate = sum(row["cells_per_active_second"] for row in workers)
    remaining = sum(row["remaining_cells"] for row in workers)
    worker_etas = [
        row["remaining_cells"] / row["cells_per_active_second"]
        for row in workers
        if row["cells_per_active_second"] > 0
    ]
    return {
        "created_at_utc": utc_now(),
        "status": "pass"
        if all(row["permanent_failed_cells"] == 0 for row in workers)
        else "fail",
        "workers": workers,
        "combined": {
            "processed_cells": sum(row["processed_cells"] for row in workers),
            "remaining_cells": remaining,
            "cells_per_active_second_sum": combined_rate,
            "effective_total_speedup_vs_experiment0_single": (
                combined_rate / REFERENCE_SINGLE_RATE if combined_rate else 0.0
            ),
            "parallel_wall_eta_seconds": max(worker_etas) if len(worker_etas) == 2 else None,
            "eta_note": "Valid once both workers have non-zero measured rates.",
        },
    }


def run_preflight(worker_id: int, cells: int) -> dict[str, Any]:
    paths = worker_paths(worker_id)
    provenance, total = build_provenance(paths)
    treated, controls = load_metadata_maps()
    datamodule, metadata_path = build_official_preprocessor(
        DEFAULT_LOCAL_MANIFEST, DEFAULT_STATS
    )
    audit = StreamingPreprocessingAudit()
    observed_parts: list[int] = []
    for batch in iter_raw_batches(
        paths.plan,
        start=0,
        stop=cells,
        batch_size=min(cells, 8),
        treated=treated,
        controls=controls,
    ):
        model_batch = preprocess_once(
            batch.raw_cells, datamodule, audit, inverse_check=True
        )
        if model_batch["offsets"].numel() != len(batch.cells) + 1:
            raise AssertionError("Official collate offsets do not match the cells")
        observed_parts.extend(cell.part_index for cell in batch.cells)
    if observed_parts != list(range(cells)) or audit.cells != cells:
        raise AssertionError("Preflight did not consume the requested plan prefix")
    return {
        "created_at_utc": utc_now(),
        "status": "pass",
        "worker_id": worker_id,
        "worker_plan_rows": total,
        "worker_plan_sha256": provenance["worker_plan"]["sha256"],
        "checked_cells": cells,
        "part_index_contiguous": True,
        "stable_locator_and_owner_metadata": "pass",
        "official_preprocessing_inverse_check": "pass",
        "gene_metadata": relative(metadata_path),
        "preprocessing": audit.summary(),
        "model_or_checkpoint_loaded": False,
        "formal_output_created": False,
    }


def run_batch_probe(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or torch.cuda.device_count() != 1:
        raise RuntimeError("Batch probe requires CUDA_VISIBLE_DEVICES=0")
    output_path = RESULTS / "tahoe_experiment1_genejepa_batch_probe.json"
    if output_path.exists():
        raise FileExistsError(f"Batch probe already exists: {output_path}")
    paths = worker_paths(0)
    provenance, _total = build_provenance(paths)
    treated, controls = load_metadata_maps()
    raw_cells: list[dict[str, Any]] = []
    plan_cells: list[PlanCell] = []
    for batch in iter_raw_batches(
        paths.plan,
        start=0,
        stop=args.cells,
        batch_size=args.cells,
        treated=treated,
        controls=controls,
    ):
        raw_cells.extend(batch.raw_cells)
        plan_cells.extend(batch.cells)
    if len(raw_cells) != args.cells:
        raise AssertionError("Batch probe did not read the requested formal-plan cells")

    device = torch.device("cuda:0")
    datamodule, _ = build_official_preprocessor(DEFAULT_LOCAL_MANIFEST, DEFAULT_STATS)
    module = load_frozen_model(DEFAULT_CHECKPOINT, device)
    results = []
    reference_outputs: dict[int, np.ndarray] = {}
    for batch_size in args.batch_sizes:
        repeated_rates = []
        representative: np.ndarray | None = None
        peak_allocated = 0
        peak_reserved = 0
        for repeat in range(args.repeats):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            warmup_audit = StreamingPreprocessingAudit()
            warmup = preprocess_once(
                raw_cells[:batch_size], datamodule, warmup_audit, inverse_check=False
            )
            infer_teacher(module, warmup, device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            outputs = []
            for start in range(0, len(raw_cells), batch_size):
                audit = StreamingPreprocessingAudit()
                model_batch = preprocess_once(
                    raw_cells[start : start + batch_size],
                    datamodule,
                    audit,
                    inverse_check=False,
                )
                output = infer_teacher(module, model_batch, device)
                outputs.append(output)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            combined = np.concatenate(outputs)
            if combined.shape != (args.cells, EMBEDDING_DIM):
                raise AssertionError("Batch probe output shape mismatch")
            if not np.isfinite(combined).all() or not np.any(combined < 0):
                raise ValueError("Batch probe output is non-finite or unsigned")
            if representative is None:
                representative = combined
            elif not np.array_equal(representative, combined):
                raise AssertionError("Repeated inference changed at the same batch size")
            repeated_rates.append(args.cells / elapsed)
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
        assert representative is not None
        reference_outputs[batch_size] = representative
        results.append(
            {
                "batch_size": batch_size,
                "cells": args.cells,
                "repeats": args.repeats,
                "cells_per_second": repeated_rates,
                "cells_per_second_mean": sum(repeated_rates) / len(repeated_rates),
                "finite": True,
                "signed": True,
                "repeat_exact": True,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "peak_reserved_fraction": peak_reserved
                / torch.cuda.get_device_properties(device).total_memory,
            }
        )
    baseline = reference_outputs[args.batch_sizes[0]]
    cross_batch = []
    for batch_size in args.batch_sizes[1:]:
        other = reference_outputs[batch_size]
        difference = float(np.max(np.abs(baseline - other)))
        allclose = bool(np.allclose(baseline, other, rtol=1e-5, atol=1e-4))
        cross_batch.append(
            {
                "baseline_batch_size": args.batch_sizes[0],
                "other_batch_size": batch_size,
                "rtol": 1e-5,
                "atol": 1e-4,
                "allclose": allclose,
                "max_abs_difference": difference,
            }
        )
    fastest = max(results, key=lambda row: row["cells_per_second_mean"])
    baseline_result = results[0]
    improvement = (
        fastest["cells_per_second_mean"]
        / baseline_result["cells_per_second_mean"]
        - 1.0
    )
    cross_batch_consistent = all(row["allclose"] for row in cross_batch)
    recommendation = (
        fastest["batch_size"]
        if (
            improvement >= 0.10
            and fastest["peak_reserved_fraction"] < 0.8
            and cross_batch_consistent
        )
        else baseline_result["batch_size"]
    )
    payload = {
        "created_at_utc": utc_now(),
        "status": "pass" if cross_batch_consistent else "pass_keep_baseline",
        "scope": "Very short inference-only check on cells from the frozen formal worker0 plan; no cache population or output was created.",
        "provenance_fingerprint": canonical_hash(provenance),
        "formal_plan_cells": {
            "worker_id": 0,
            "part_index_start": plan_cells[0].part_index,
            "part_index_stop_exclusive": plan_cells[-1].part_index + 1,
            "count": len(plan_cells),
        },
        "results": results,
        "cross_batch_consistency": cross_batch,
        "selection_rule": "Use the faster safe batch only for >=10% mean throughput gain and cross-batch allclose; otherwise keep the Experiment 0 batch=64.",
        "recommended_extraction_batch_size": recommendation,
        "fastest_vs_baseline_gain": improvement,
        "interpretation": (
            "Cross-batch numerical inconsistency rejects the larger batch and freezes "
            "worker-local 64-cell batch boundaries for extraction and direct re-inference."
            if not cross_batch_consistent
            else "All tested batch sizes are numerically consistent."
        ),
        "formal_cache_rows_written": 0,
    }
    atomic_write_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Start or resume one formal worker")
    run.add_argument("--worker-id", type=int, choices=(0, 1), required=True)
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--commit-every-batches", type=int, default=10)
    run.add_argument(
        "--max-new-cells",
        type=int,
        default=0,
        help="Safe-stop after this many new cells; 0 means all remaining cells",
    )
    run.add_argument("--gpu-sample-interval", type=float, default=2.0)
    run.add_argument(
        "--skip-resume-finite-scan",
        action="store_true",
        help="Diagnostic escape hatch only; formal runs should not use it",
    )
    preflight = subparsers.add_parser(
        "preflight", help="Read and preprocess a tiny frozen-plan prefix without GPU/output"
    )
    preflight.add_argument("--worker-id", type=int, choices=(0, 1), required=True)
    preflight.add_argument("--cells", type=int, default=8)
    probe = subparsers.add_parser(
        "probe-batch", help="Very short real-plan GeneJEPA inference batch check"
    )
    probe.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128])
    probe.add_argument("--cells", type=int, default=512)
    probe.add_argument("--repeats", type=int, default=2)
    subparsers.add_parser("status", help="Summarize both independent workers")
    args = parser.parse_args()
    if args.command == "run":
        if (
            args.batch_size != FORMAL_BATCH_SIZE
            or args.commit_every_batches < 1
            or args.max_new_cells < 0
            or args.gpu_sample_interval <= 0
        ):
            parser.error(
                f"Formal batch must be {FORMAL_BATCH_SIZE}; commit/interval must be "
                "positive and max-new-cells >= 0"
            )
        run_worker(args)
    elif args.command == "preflight":
        if not 1 <= args.cells <= 1024:
            parser.error("preflight --cells must be in [1, 1024]")
        payload = run_preflight(args.worker_id, args.cells)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "probe-batch":
        if (
            args.cells < max(args.batch_sizes)
            or args.cells > 4096
            or any(size < 1 for size in args.batch_sizes)
            or args.repeats < 1
        ):
            parser.error("Probe requires positive batch sizes/repeats and max(batch) <= cells <= 4096")
        payload = run_batch_probe(args)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        payload = status_payload()
        atomic_write_json(
            RESULTS / "tahoe_experiment1_cache_cap512_all_dmso_extraction_status.json",
            payload,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
