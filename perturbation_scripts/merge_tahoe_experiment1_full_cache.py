#!/usr/bin/env python3
"""Rehearse or execute the frozen Experiment 1 worker-part scatter merge."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from extract_tahoe_experiment1_full_cache_worker import (
    DEFAULT_CHECKPOINT,
    DEFAULT_LOCAL_MANIFEST,
    DEFAULT_STATS,
    EMBEDDING_DIM,
    EXPECTED_TOTAL,
    FORMAL_BATCH_SIZE,
    PROJECT_ROOT,
    RESULTS,
    StreamingEmbeddingAudit,
    StreamingPreprocessingAudit,
    atomic_write_json,
    build_official_preprocessor,
    build_provenance,
    canonical_hash,
    infer_teacher,
    iter_raw_batches,
    load_frozen_model,
    load_metadata_maps,
    plan_cells_from_table,
    preprocess_once,
    relative,
    sha256_file,
    utc_now,
    worker_paths,
)


PREFIX = RESULTS / "tahoe_experiment1_cache_cap512_all_dmso"
REHEARSAL_OUTPUT = Path(str(PREFIX) + "_merge_rehearsal_embeddings.npy")
REHEARSAL_INDEX = Path(str(PREFIX) + "_merge_rehearsal_index.csv")
REHEARSAL_AUDIT = Path(str(PREFIX) + "_merge_rehearsal_audit.json")
FULL_OUTPUT = Path(str(PREFIX) + "_embeddings.npy")
FULL_PARTIAL = Path(str(FULL_OUTPUT) + ".partial")
FULL_MANIFEST = Path(str(PREFIX) + "_embedding_manifest.json")
REINFERENCE_SEED = 42
REINFERENCE_RTOL = 1e-5
REINFERENCE_ATOL = 1e-4


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_plan_cells(plan_path: Path, targets: np.ndarray) -> list[Any]:
    remaining = set(map(int, targets.tolist()))
    found: dict[int, Any] = {}
    plan = pq.ParquetFile(plan_path)
    for row_group in range(plan.num_row_groups):
        if not remaining:
            break
        statistics = plan.metadata.row_group(row_group).column(0).statistics
        if statistics is not None and (
            int(statistics.max) < min(remaining)
            or int(statistics.min) > max(remaining)
        ):
            continue
        table = plan.read_row_group(row_group)
        for cell in plan_cells_from_table(table):
            if cell.part_index in remaining:
                found[cell.part_index] = cell
                remaining.remove(cell.part_index)
    if remaining:
        raise AssertionError(f"Plan did not contain requested part indices: {remaining}")
    return [found[int(index)] for index in targets]


def scatter_chunk(
    destination: np.ndarray,
    written: np.ndarray,
    destination_rows: np.ndarray,
    embeddings: np.ndarray,
) -> None:
    rows = np.asarray(destination_rows, dtype=np.int64)
    values = np.asarray(embeddings, dtype=np.float32)
    if values.shape != (len(rows), EMBEDDING_DIM):
        raise AssertionError("Scatter embedding shape mismatch")
    if len(np.unique(rows)) != len(rows) or bool(written[rows].any()):
        raise AssertionError("Duplicate scatter destination")
    if not np.isfinite(values).all():
        raise ValueError("Non-finite worker embedding before scatter")
    destination[rows] = values
    written[rows] = True


def worker_state(worker_id: int, require_finished: bool) -> dict[str, Any]:
    paths = worker_paths(worker_id)
    provenance, total = build_provenance(paths)
    progress = read_json(paths.progress)
    committed = int(progress.get("committed_prefix_stop", 0))
    if progress.get("provenance_fingerprint") != canonical_hash(provenance):
        raise AssertionError(f"worker{worker_id} progress provenance changed")
    if int(progress.get("permanent_failed_cells", 0)) != 0:
        raise AssertionError(f"worker{worker_id} has permanent failed cells")
    if require_finished:
        if progress.get("status") != "finished" or committed != total:
            raise RuntimeError(f"worker{worker_id} is not finished")
        runtime = read_json(paths.runtime_manifest)
        if runtime.get("status") != "pass":
            raise RuntimeError(f"worker{worker_id} runtime manifest is not pass")
        array_path = paths.final
        if sha256_file(array_path) != runtime["output"]["sha256"]:
            raise AssertionError(f"worker{worker_id} final array hash changed")
    else:
        if committed <= 0:
            raise RuntimeError(f"worker{worker_id} has no committed formal rows")
        array_path = paths.final if paths.final.exists() else paths.partial
    completed = np.lib.format.open_memmap(paths.completed, mode="r")
    if completed.shape != (total,) or not bool(np.asarray(completed[:committed]).all()):
        raise AssertionError(f"worker{worker_id} completed bitmap/prefix mismatch")
    if committed < total and bool(np.asarray(completed[committed:]).any()):
        raise AssertionError(f"worker{worker_id} bitmap extends past commit marker")
    del completed
    array = np.lib.format.open_memmap(array_path, mode="r")
    if array.shape != (total, EMBEDDING_DIM) or array.dtype != np.float32:
        raise AssertionError(f"worker{worker_id} part shape/dtype mismatch")
    return {
        "worker_id": worker_id,
        "paths": paths,
        "provenance": provenance,
        "total": total,
        "committed": committed,
        "array_path": array_path,
        "array": array,
        "progress": progress,
    }


def run_reinference(
    selected: list[tuple[int, Any]],
    expected: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    treated, controls = load_metadata_maps()
    locators = []
    grouped: dict[tuple[int, int], list[tuple[int, Any]]] = {}
    for selected_index, (worker_id, cell) in enumerate(selected):
        batch_start = (cell.part_index // FORMAL_BATCH_SIZE) * FORMAL_BATCH_SIZE
        grouped.setdefault((worker_id, batch_start), []).append(
            (selected_index, cell)
        )
        locators.append(
            {
                "worker_id": worker_id,
                "part_index": cell.part_index,
                "embedding_index": cell.embedding_index,
                "shard_path": cell.shard_path,
                "row_group_index": cell.row_group_index,
                "row_index_in_row_group": cell.row_index_in_row_group,
            }
        )

    datamodule, _ = build_official_preprocessor(DEFAULT_LOCAL_MANIFEST, DEFAULT_STATS)
    preprocessing = StreamingPreprocessingAudit()
    module = load_frozen_model(DEFAULT_CHECKPOINT, device)
    observed = np.empty_like(expected, dtype=np.float32)
    context_cells = 0
    for (worker_id, batch_start), group in sorted(grouped.items()):
        paths = worker_paths(worker_id)
        worker_total = pq.ParquetFile(paths.plan).metadata.num_rows
        batch_stop = min(batch_start + FORMAL_BATCH_SIZE, worker_total)
        batches = list(
            iter_raw_batches(
                paths.plan,
                start=batch_start,
                stop=batch_stop,
                batch_size=FORMAL_BATCH_SIZE,
                treated=treated,
                controls=controls,
            )
        )
        if len(batches) != 1 or len(batches[0].cells) != batch_stop - batch_start:
            raise AssertionError("Could not reconstruct a canonical extraction batch")
        model_batch = preprocess_once(
            batches[0].raw_cells, datamodule, preprocessing, inverse_check=False
        )
        batch_output = infer_teacher(module, model_batch, device)
        context_cells += len(batches[0].cells)
        for selected_index, cell in group:
            reread = batches[0].cells[cell.part_index - batch_start]
            if reread.embedding_index != cell.embedding_index:
                raise AssertionError("Locator/global embedding index changed on re-read")
            observed[selected_index] = batch_output[cell.part_index - batch_start]
    reference = np.asarray(expected, dtype=np.float32)
    difference = np.abs(observed - reference)
    maximum = float(difference.max())
    allclose = bool(
        np.allclose(
            observed,
            reference,
            rtol=REINFERENCE_RTOL,
            atol=REINFERENCE_ATOL,
        )
    )
    if not allclose:
        raise AssertionError(
            f"Direct Epoch25 re-inference differs from cache: max_abs={maximum}"
        )
    return {
        "performed": True,
        "cells": len(selected),
        "canonical_context_cells_inferred": context_cells,
        "canonical_batch_size": FORMAL_BATCH_SIZE,
        "seed": REINFERENCE_SEED,
        "rtol": REINFERENCE_RTOL,
        "atol": REINFERENCE_ATOL,
        "allclose": allclose,
        "max_abs_difference": maximum,
        "finite": bool(np.isfinite(observed).all()),
        "locators": locators,
        "preprocessing": preprocessing.summary(),
    }


def rehearsal(args: argparse.Namespace) -> None:
    occupied = [
        path for path in (REHEARSAL_OUTPUT, REHEARSAL_INDEX, REHEARSAL_AUDIT) if path.exists()
    ]
    if occupied:
        raise FileExistsError(
            "Rehearsal outputs already exist; inspect them instead of overwriting: "
            + ", ".join(map(str, occupied))
        )
    states = [worker_state(worker_id, require_finished=False) for worker_id in (0, 1)]
    selected_by_worker: dict[int, list[Any]] = {}
    for state in states:
        if state["committed"] < args.cells_per_worker:
            raise RuntimeError(
                f"worker{state['worker_id']} needs at least {args.cells_per_worker} committed rows"
            )
        targets = np.linspace(
            0,
            state["committed"] - 1,
            args.cells_per_worker,
            dtype=np.int64,
        )
        selected_by_worker[state["worker_id"]] = read_plan_cells(
            state["paths"].plan, targets
        )

    all_cells = [
        (worker_id, cell)
        for worker_id, cells in selected_by_worker.items()
        for cell in cells
    ]
    global_indices = np.array(
        [cell.embedding_index for _, cell in all_cells], dtype=np.int64
    )
    if len(np.unique(global_indices)) != len(global_indices):
        raise AssertionError("Rehearsal worker sample has overlapping global indices")
    sorted_global = np.sort(global_indices)
    compact_lookup = {int(value): index for index, value in enumerate(sorted_global)}
    destination = np.empty((len(all_cells), EMBEDDING_DIM), dtype=np.float32)
    written = np.zeros(len(all_cells), dtype=bool)
    index_rows = []
    for state in states:
        cells = selected_by_worker[state["worker_id"]]
        part_indices = np.array([cell.part_index for cell in cells], dtype=np.int64)
        destinations = np.array(
            [compact_lookup[cell.embedding_index] for cell in cells], dtype=np.int64
        )
        values = np.asarray(state["array"][part_indices])
        scatter_chunk(destination, written, destinations, values)
        for cell, cache_row in zip(cells, destinations, strict=True):
            index_rows.append(
                {
                    "cache_row": int(cache_row),
                    "embedding_index": cell.embedding_index,
                    "worker_id": state["worker_id"],
                    "part_index": cell.part_index,
                    "shard_path": cell.shard_path,
                    "row_group_index": cell.row_group_index,
                    "row_index_in_row_group": cell.row_index_in_row_group,
                    "owner_type": cell.owner_type,
                    "owner_index": cell.owner_index,
                }
            )
    if not bool(written.all()):
        raise AssertionError("Rehearsal scatter has missing rows")
    index_rows.sort(key=lambda row: row["cache_row"])
    if [row["embedding_index"] for row in index_rows] != sorted_global.tolist():
        raise AssertionError("Rehearsal cache row/global index alignment failed")
    if not np.isfinite(destination).all():
        raise AssertionError("Rehearsal output is not fully finite")

    rng = np.random.default_rng(REINFERENCE_SEED)
    selection = np.sort(
        rng.choice(len(all_cells), size=min(args.reinference_cells, len(all_cells)), replace=False)
    )
    validation_cells = [all_cells[int(index)] for index in selection]
    validation_expected = np.stack(
        [
            np.asarray(states[worker_id]["array"][cell.part_index])
            for worker_id, cell in validation_cells
        ]
    )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or torch.cuda.device_count() != 1:
        raise RuntimeError("Rehearsal re-inference requires CUDA_VISIBLE_DEVICES=0")
    reinference = run_reinference(
        validation_cells, validation_expected, torch.device("cuda:0")
    )

    temporary = REHEARSAL_OUTPUT.with_name(REHEARSAL_OUTPUT.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, destination, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, REHEARSAL_OUTPUT)
    atomic_write_csv(REHEARSAL_INDEX, index_rows)
    audit = StreamingEmbeddingAudit()
    audit.add(destination)
    result = {
        "created_at_utc": utc_now(),
        "status": "pass",
        "scope": "Small formal-part scatter rehearsal; no pilot cache was created or deleted.",
        "scatter_implementation": {
            "script": relative(Path(__file__)),
            "script_sha256": sha256_file(Path(__file__)),
            "same_scatter_function_as_full_merge": True,
            "global_key": "embedding_index",
        },
        "workers": [
            {
                "worker_id": state["worker_id"],
                "committed_prefix": state["committed"],
                "sampled_rows": args.cells_per_worker,
                "plan_sha256": state["provenance"]["worker_plan"]["sha256"],
                "source_array": relative(state["array_path"]),
            }
            for state in states
        ],
        "scatter": {
            "rows": len(all_cells),
            "shape": list(destination.shape),
            "dtype": str(destination.dtype),
            "missing": 0,
            "duplicate": 0,
            "worker_overlap": 0,
            "finite_ratio": 1.0,
            "locator_embedding_index_cache_row_alignment": "pass",
        },
        "embedding_statistics": audit.summary(),
        "direct_reinference": reinference,
        "outputs": {
            "embedding": {
                "path": relative(REHEARSAL_OUTPUT),
                "sha256": sha256_file(REHEARSAL_OUTPUT),
            },
            "index": {
                "path": relative(REHEARSAL_INDEX),
                "sha256": sha256_file(REHEARSAL_INDEX),
                "encoding": "utf-8-sig",
            },
        },
    }
    atomic_write_json(REHEARSAL_AUDIT, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def full_merge(args: argparse.Namespace) -> None:
    if FULL_OUTPUT.exists() or FULL_PARTIAL.exists() or FULL_MANIFEST.exists():
        raise FileExistsError("Full merge output exists; never infer completion from size or overwrite it")
    states = [worker_state(worker_id, require_finished=True) for worker_id in (0, 1)]
    destination: np.memmap | None = np.lib.format.open_memmap(
        FULL_PARTIAL,
        mode="w+",
        dtype=np.float32,
        shape=(EXPECTED_TOTAL, EMBEDDING_DIM),
    )
    written = np.zeros(EXPECTED_TOTAL, dtype=bool)
    audit = StreamingEmbeddingAudit()
    started = time.perf_counter()
    try:
        for state in states:
            plan = pq.ParquetFile(state["paths"].plan)
            for batch in plan.iter_batches(
                batch_size=args.chunk_cells,
                columns=["part_index", "embedding_index"],
            ):
                part_indices = batch["part_index"].to_numpy(zero_copy_only=False)
                global_indices = batch["embedding_index"].to_numpy(
                    zero_copy_only=False
                )
                values = np.asarray(state["array"][part_indices])
                scatter_chunk(destination, written, global_indices, values)
                audit.add(values)
        destination.flush()
        if not bool(written.all()) or int(written.sum()) != EXPECTED_TOTAL:
            raise AssertionError("Full scatter has missing global embedding indices")
        del destination
        destination = None
        os.replace(FULL_PARTIAL, FULL_OUTPUT)
        output_sha = sha256_file(FULL_OUTPUT)
        result = {
            "created_at_utc": utc_now(),
            "status": "pass",
            "policy": "treated cap=512 + all eligible DMSO; frozen v1",
            "scatter_implementation": {
                "script": relative(Path(__file__)),
                "script_sha256": sha256_file(Path(__file__)),
                "function": "scatter_chunk",
                "rehearsal_audit": relative(REHEARSAL_AUDIT),
                "rehearsal_audit_sha256": sha256_file(REHEARSAL_AUDIT),
            },
            "worker_manifests": [
                {
                    "worker_id": state["worker_id"],
                    "path": relative(state["paths"].runtime_manifest),
                    "sha256": sha256_file(state["paths"].runtime_manifest),
                    "part_sha256": read_json(state["paths"].runtime_manifest)["output"][
                        "sha256"
                    ],
                }
                for state in states
            ],
            "output": {
                "path": relative(FULL_OUTPUT),
                "sha256": output_sha,
                "size_bytes": FULL_OUTPUT.stat().st_size,
                "shape": [EXPECTED_TOTAL, EMBEDDING_DIM],
                "dtype": "float32",
                "row_equals_global_embedding_index": True,
            },
            "integrity": {
                "missing": 0,
                "duplicate": 0,
                "finite": True,
                "cells": EXPECTED_TOTAL,
                "worker_overlap": 0,
            },
            "embedding_statistics": audit.summary(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_write_json(FULL_MANIFEST, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except BaseException:
        if destination is not None:
            destination.flush()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    rehearsal_parser = subparsers.add_parser("rehearsal")
    rehearsal_parser.add_argument("--cells-per-worker", type=int, default=256)
    rehearsal_parser.add_argument("--reinference-cells", type=int, default=16)
    full_parser = subparsers.add_parser("full")
    full_parser.add_argument("--chunk-cells", type=int, default=8192)
    args = parser.parse_args()
    if args.command == "rehearsal":
        if args.cells_per_worker < 1 or args.reinference_cells < 1:
            parser.error("Rehearsal counts must be positive")
        rehearsal(args)
    else:
        if args.chunk_cells < 1:
            parser.error("--chunk-cells must be positive")
        if not REHEARSAL_AUDIT.is_file() or read_json(REHEARSAL_AUDIT).get("status") != "pass":
            raise RuntimeError("A passing formal-part merge rehearsal is required first")
        full_merge(args)


if __name__ == "__main__":
    main()
