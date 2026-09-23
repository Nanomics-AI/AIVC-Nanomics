#!/usr/bin/env python3
"""Snapshot and verify an in-place resume of the formal Experiment 1 cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from extract_tahoe_experiment1_full_cache_worker import (
    PLAN_COLUMNS,
    PROJECT_ROOT,
    atomic_write_json,
    read_json,
    relative,
    utc_now,
    worker_paths,
)


DEFAULT_BEFORE = (
    PROJECT_ROOT
    / "results/tahoe_experiment1_cache_cap512_all_dmso_resume_before.json"
)
DEFAULT_AUDIT = (
    PROJECT_ROOT
    / "results/tahoe_experiment1_cache_cap512_all_dmso_resume_audit.json"
)


def sha256_array_prefix(array: np.ndarray, stop: int) -> str:
    digest = hashlib.sha256()
    for start in range(0, stop, 4096):
        block = np.ascontiguousarray(array[start : min(start + 4096, stop)])
        digest.update(block.tobytes(order="C"))
    return digest.hexdigest()


def plan_prefix(path: Path, stop: int) -> dict[str, Any]:
    columns: dict[str, list[Any]] = {name: [] for name in PLAN_COLUMNS}
    remaining = stop
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=min(65536, max(stop, 1)), columns=PLAN_COLUMNS):
        take = min(remaining, batch.num_rows)
        for name in PLAN_COLUMNS:
            columns[name].extend(batch.column(name).slice(0, take).to_pylist())
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise AssertionError(f"Plan {path} has fewer than {stop} rows")

    rows_digest = hashlib.sha256()
    for row in zip(*(columns[name] for name in PLAN_COLUMNS), strict=True):
        rows_digest.update(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        rows_digest.update(b"\n")
    embedding_indices = np.asarray(columns["embedding_index"], dtype=np.int64)
    part_indices = np.asarray(columns["part_index"], dtype=np.int64)
    if not np.array_equal(part_indices, np.arange(stop, dtype=np.int64)):
        raise AssertionError(f"Non-contiguous part_index prefix in {path}")
    return {
        "rows": stop,
        "rows_sha256": rows_digest.hexdigest(),
        "embedding_indices_sha256": hashlib.sha256(
            embedding_indices.tobytes(order="C")
        ).hexdigest(),
        "embedding_indices": embedding_indices,
        "embedding_index_min": int(embedding_indices.min()) if stop else None,
        "embedding_index_max": int(embedding_indices.max()) if stop else None,
    }


def worker_snapshot(worker_id: int, stop: int | None = None) -> dict[str, Any]:
    paths = worker_paths(worker_id)
    progress = read_json(paths.progress)
    committed = int(progress["committed_prefix_stop"])
    if stop is None:
        stop = committed
    if not 0 <= stop <= committed:
        raise AssertionError(f"Requested prefix {stop} is outside committed prefix {committed}")

    embeddings = np.load(paths.partial, mmap_mode="r")
    completed = np.load(paths.completed, mmap_mode="r")
    if embeddings.shape[0] != int(progress["total_cells"]):
        raise AssertionError("Embedding row count does not match progress")
    if completed.shape != (int(progress["total_cells"]),):
        raise AssertionError("Completed bitmap shape does not match progress")
    if not bool(np.asarray(completed[:committed]).all()):
        raise AssertionError("Completed bitmap has a false value inside committed prefix")
    if bool(np.asarray(completed[committed:]).any()):
        raise AssertionError("Completed bitmap is ahead of committed prefix")
    if not bool(np.isfinite(embeddings[:stop]).all()):
        raise AssertionError("Non-finite embedding inside audited prefix")

    plan = plan_prefix(paths.plan, stop)
    embedding_indices = plan.pop("embedding_indices")
    return {
        "worker_id": worker_id,
        "progress_file": relative(paths.progress),
        "provenance_fingerprint": progress["provenance_fingerprint"],
        "status": progress["status"],
        "committed_prefix_stop": committed,
        "audited_prefix_stop": stop,
        "embedding_prefix_sha256": sha256_array_prefix(embeddings, stop),
        "completed_prefix_sha256": sha256_array_prefix(completed, stop),
        "failed_attempts": int(progress["failed_attempts"]),
        "permanent_failed_cells": int(progress["permanent_failed_cells"]),
        "sessions": len(progress["sessions"]),
        "plan_prefix": plan,
        "_embedding_indices": embedding_indices,
    }


def public_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in snapshot.items() if not key.startswith("_")}


def snapshot_command(output: Path) -> None:
    workers = [worker_snapshot(0), worker_snapshot(1)]
    overlap = np.intersect1d(
        workers[0]["_embedding_indices"], workers[1]["_embedding_indices"]
    )
    if overlap.size:
        raise AssertionError("Worker committed prefixes overlap in global embedding_index")
    payload = {
        "schema": "tahoe_experiment1_resume_before_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "workers": [public_snapshot(worker) for worker in workers],
        "committed_prefix_global_embedding_index_overlap": 0,
    }
    atomic_write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def verify_command(before_path: Path, output: Path) -> None:
    before = read_json(before_path)
    results: list[dict[str, Any]] = []
    current_snapshots: list[dict[str, Any]] = []
    for previous in before["workers"]:
        worker_id = int(previous["worker_id"])
        old_stop = int(previous["committed_prefix_stop"])
        current = worker_snapshot(worker_id, stop=old_stop)
        current_snapshots.append(current)
        progress = read_json(worker_paths(worker_id).progress)
        last_session = progress["sessions"][-1]
        checks = {
            "committed_prefix_advanced": int(progress["committed_prefix_stop"]) > old_stop,
            "old_embedding_prefix_unchanged": current["embedding_prefix_sha256"]
            == previous["embedding_prefix_sha256"],
            "old_completed_prefix_unchanged": current["completed_prefix_sha256"]
            == previous["completed_prefix_sha256"],
            "plan_prefix_unchanged": current["plan_prefix"] == previous["plan_prefix"],
            "provenance_unchanged": current["provenance_fingerprint"]
            == previous["provenance_fingerprint"],
            "resume_started_at_old_prefix": int(last_session["start_committed_cells"])
            == old_stop,
            "resume_processed_new_cells": int(last_session["new_cells_processed"]) > 0,
            "failed_attempts_unchanged": int(progress["failed_attempts"])
            == int(previous["failed_attempts"]),
            "permanent_failed_cells_zero": int(progress["permanent_failed_cells"]) == 0,
        }
        results.append(
            {
                "worker_id": worker_id,
                "before_committed_prefix_stop": old_stop,
                "after_committed_prefix_stop": int(progress["committed_prefix_stop"]),
                "new_cells": int(progress["committed_prefix_stop"]) - old_stop,
                "last_session_id": int(last_session["session_id"]),
                "checks": checks,
                "status": "pass" if all(checks.values()) else "fail",
            }
        )

    current_full = [worker_snapshot(0), worker_snapshot(1)]
    overlap = np.intersect1d(
        current_full[0]["_embedding_indices"], current_full[1]["_embedding_indices"]
    )
    overall = all(result["status"] == "pass" for result in results) and not overlap.size
    payload = {
        "schema": "tahoe_experiment1_resume_audit_v1",
        "created_at_utc": utc_now(),
        "status": "pass" if overall else "fail",
        "before_snapshot": relative(before_path),
        "workers": results,
        "current_committed_prefix_global_embedding_index_overlap": int(overlap.size),
        "interpretation": (
            "Committed rows from before the restart were unchanged; each resumed session "
            "started at the prior committed prefix and appended only new rows."
            if overall
            else "One or more resume invariants failed."
        ),
    }
    atomic_write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not overall:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    before = subparsers.add_parser("snapshot")
    before.add_argument("--output", type=Path, default=DEFAULT_BEFORE)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
    verify.add_argument("--output", type=Path, default=DEFAULT_AUDIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "snapshot":
        snapshot_command(args.output.resolve())
    else:
        verify_command(args.before.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
