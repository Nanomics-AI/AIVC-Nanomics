#!/usr/bin/env python3
"""Extract and merge the Phase-II cache with Author Epoch49 EMA Teacher."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
SCRIPT_DIR = PROJECT_ROOT / "perturbation_scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import author_genejepa_epoch49_hd100_cache as author
import phase1_author_genejepa as phase1_author


CACHE_DIR = RESULTS / "phase2_author_genejepa_cache"
PREFIX = CACHE_DIR / "cache"
PLAN_SUMMARY = RESULTS / "phase2_stav2_cache_plan.json"
SUBSET_MANIFEST = RESULTS / "phase2_stav2_subset_manifest.json"
SOURCE_CONDITIONS = RESULTS / "tahoe_experiment1_cache_cap512_all_dmso_condition_index.csv"
SOURCE_CONTROLS = RESULTS / "tahoe_experiment1_cache_cap512_all_dmso_control_pool_index.csv"
PHASE1_PARITY = RESULTS / "phase1_author_manual_mean_parity.json"
MERGED_EMBEDDINGS = RESULTS / "phase2_author_genejepa_embeddings.npy"
MERGED_PARTIAL = Path(str(MERGED_EMBEDDINGS) + ".partial")
FINAL_MANIFEST = RESULTS / "phase2_author_genejepa_embedding_manifest.json"

EMBEDDING_DIM = 768
FORMAL_BATCH_SIZE = 64


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


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


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": relative(path),
        "sha256": author.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def expected_counts() -> tuple[int, int, int]:
    plan = read_json(PLAN_SUMMARY)
    return (
        int(plan["total_unique_cells"]),
        int(plan["treated_cells"]),
        int(plan["control_cells"]),
    )


def build_provenance(paths: Any) -> tuple[dict[str, Any], int]:
    subset = read_json(SUBSET_MANIFEST)
    plan = read_json(PLAN_SUMMARY)
    parity = read_json(PHASE1_PARITY)
    if subset.get("status") != "pass" or plan.get("status") != "pass" or parity.get("status") != "pass":
        raise AssertionError("Phase-II subset/plan and Phase-I Author pooling parity must PASS")
    worker = next(
        item for item in plan["worker_partition"] if int(item["worker_id"]) == paths.worker_id
    )
    plan_sha = author.sha256_file(paths.plan)
    if plan_sha != worker["plan_sha256"]:
        raise AssertionError("Frozen Phase-II worker plan changed")
    plan_manifest = read_json(paths.plan_manifest)
    rows = int(plan_manifest["plan"]["rows"])
    parquet = pq.ParquetFile(paths.plan)
    if rows != int(worker["rows"]) or parquet.metadata.num_rows != rows:
        raise AssertionError("Phase-II worker plan row count mismatch")
    base = configure_base()
    missing = set(base.PLAN_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise AssertionError(f"Phase-II worker plan missing columns: {sorted(missing)}")
    if author.sha256_file(author.AUTHOR_CHECKPOINT) != author.AUTHOR_CHECKPOINT_SHA256:
        raise AssertionError("Author checkpoint changed")
    commit = subprocess.run(
        ["git", "-C", str(author.AUTHOR_CODE), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != author.EXPECTED_AUTHOR_COMMIT:
        raise AssertionError("Author source commit changed")
    stats = read_json(author.GLOBAL_STATS)
    return {
        "policy": "Task 2 / Phase II selected pools; Author Epoch49 EMA Teacher",
        "subset_manifest": artifact(SUBSET_MANIFEST),
        "cache_plan": artifact(PLAN_SUMMARY),
        "worker_plan": {**artifact(paths.plan), "rows": rows},
        "worker_plan_manifest": artifact(paths.plan_manifest),
        "reused_phase1_manual_mean_parity": artifact(PHASE1_PARITY),
        "checkpoint": {
            **artifact(author.AUTHOR_CHECKPOINT),
            "ema_teacher": True,
            "branch": "teacher_encoder.ema_model",
            "frozen": True,
        },
        "author_code": {"path": relative(author.AUTHOR_CODE), "git_commit": commit},
        "preprocessing": {
            "local_manifest": artifact(author.LOCAL_MANIFEST),
            "gene_metadata": artifact(author.GENE_METADATA),
            "global_stats": {
                **artifact(author.GLOBAL_STATS),
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
            },
            "sentinel_gene_mapping_log1p_normalization_once": True,
            "centering": False,
            "whitening": False,
            "l2_normalization": False,
            "rectification": False,
        },
        "pooling": {
            "captured_tensor": "EMA Teacher final_norm [B,512,768]",
            "operation": "manual mean(dim=1)",
            "saved_output": "raw signed float32 [B,768] only",
            "tokens_persisted": False,
        },
        "condition_index": artifact(SOURCE_CONDITIONS),
        "control_pool_index": artifact(SOURCE_CONTROLS),
        "worker_code": artifact(Path(__file__)),
        "reused_resume_worker_code": artifact(
            PROJECT_ROOT / "perturbation_scripts/extract_tahoe_experiment1_full_cache_worker.py"
        ),
        "output_contract": {
            "row": "worker-local part_index",
            "global_scatter_key": "phase2_cell_index stored as embedding_index",
            "shape": [rows, EMBEDDING_DIM],
            "dtype": "float32",
        },
    }, rows


def configure_base() -> Any:
    total, treated, control = expected_counts()
    base = author.configure_base(require_contract=False)
    base.PREFIX = PREFIX
    base.PLAN_SUMMARY = PLAN_SUMMARY
    # Owner IDs in Phase-II plans deliberately retain source Experiment-1 owner IDs.
    base.CONDITION_INDEX = SOURCE_CONDITIONS
    base.CONTROL_INDEX = SOURCE_CONTROLS
    base.DEFAULT_CHECKPOINT = author.AUTHOR_CHECKPOINT
    base.DEFAULT_LOCAL_MANIFEST = author.LOCAL_MANIFEST
    base.DEFAULT_STATS = author.GLOBAL_STATS
    base.FORMAL_BATCH_SIZE = FORMAL_BATCH_SIZE
    base.EXPECTED_TOTAL = total
    base.EXPECTED_TREATED = treated
    base.EXPECTED_DMSO = control
    base.PROGRESS_SCHEMA = "phase2_author_genejepa_worker_progress_v1"
    base.RUNTIME_MANIFEST_SCHEMA = "phase2_author_genejepa_worker_extraction_v1"
    base.build_provenance = build_provenance
    base.load_frozen_model = author.load_author_model
    base.infer_teacher = phase1_author.infer_manual_mean
    return base


def run_worker(args: argparse.Namespace) -> None:
    if read_json(PHASE1_PARITY).get("status") != "pass":
        raise AssertionError("Phase-I Author manual-mean parity artifact is not PASS")
    if args.batch_size != FORMAL_BATCH_SIZE:
        raise ValueError(f"Frozen Phase-II Author extraction batch size is {FORMAL_BATCH_SIZE}")
    configure_base().run_worker(args)


def status() -> dict[str, Any]:
    result = configure_base().status_payload()
    result["scope"] = "Task 2 / Phase II selected physical cells only"
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def audit() -> dict[str, Any]:
    base = configure_base()
    workers: list[dict[str, Any]] = []
    for worker_id in (0, 1):
        paths = base.worker_paths(worker_id)
        provenance, rows = build_provenance(paths)
        workers.append(
            {
                "worker_id": worker_id,
                "rows": rows,
                "plan_sha256": provenance["worker_plan"]["sha256"],
                "checkpoint_sha256": provenance["checkpoint"]["sha256"],
                "ema_teacher": provenance["checkpoint"]["ema_teacher"],
                "manual_mean_parity_reused": provenance["reused_phase1_manual_mean_parity"],
            }
        )
    result = {
        "status": "pass",
        "workers": workers,
        "worker_union_cells": sum(row["rows"] for row in workers),
        "expected_total_cells": expected_counts()[0],
        "formal_batch_size": FORMAL_BATCH_SIZE,
        "tokens_persisted": False,
    }
    if result["worker_union_cells"] != result["expected_total_cells"]:
        raise AssertionError(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def merge(chunk_cells: int) -> dict[str, Any]:
    if MERGED_EMBEDDINGS.exists() or MERGED_PARTIAL.exists() or FINAL_MANIFEST.exists():
        raise FileExistsError("Phase-II Author merged output exists; refusing overwrite")
    base = configure_base()
    total, treated, control = expected_counts()
    destination = np.lib.format.open_memmap(
        MERGED_PARTIAL, mode="w+", dtype=np.float32, shape=(total, EMBEDDING_DIM)
    )
    written = np.zeros(total, dtype=np.bool_)
    statistics = base.StreamingEmbeddingAudit()
    workers: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for worker_id in (0, 1):
            paths = base.worker_paths(worker_id)
            progress = read_json(paths.progress)
            if (
                progress.get("status") != "finished"
                or int(progress.get("processed_cells", -1)) != int(progress.get("total_cells", -2))
                or int(progress.get("permanent_failed_cells", 0)) != 0
            ):
                raise RuntimeError(f"worker{worker_id} is not complete and clean")
            runtime = read_json(paths.runtime_manifest)
            if runtime.get("status") != "pass" or not paths.final.is_file():
                raise RuntimeError(f"worker{worker_id} final output is not audited")
            source = np.load(paths.final, mmap_mode="r")
            plan = pq.ParquetFile(paths.plan)
            if source.shape != (plan.metadata.num_rows, EMBEDDING_DIM):
                raise AssertionError(f"worker{worker_id} output shape changed")
            for batch in plan.iter_batches(
                batch_size=chunk_cells, columns=["part_index", "embedding_index"]
            ):
                part = batch["part_index"].to_numpy(zero_copy_only=False).astype(np.int64)
                phase = batch["embedding_index"].to_numpy(zero_copy_only=False).astype(np.int64)
                values = np.asarray(source[part], dtype=np.float32)
                if (
                    np.any(phase < 0)
                    or np.any(phase >= total)
                    or bool(written[phase].any())
                    or not np.isfinite(values).all()
                ):
                    raise AssertionError("Invalid, duplicate, or non-finite Phase-II scatter chunk")
                destination[phase] = values
                written[phase] = True
                statistics.add(values)
            workers.append(
                {
                    "worker_id": worker_id,
                    "plan": artifact(paths.plan),
                    "manifest": artifact(paths.runtime_manifest),
                    "embedding": artifact(paths.final),
                }
            )
        if not written.all() or int(written.sum()) != total:
            raise AssertionError("Phase-II Author merge has missing rows")
        destination.flush()
        del destination
        destination = None
        os.replace(MERGED_PARTIAL, MERGED_EMBEDDINGS)
        summary = statistics.summary()
        if not summary["finite"] or not (summary["min"] < 0 < summary["max"]):
            raise AssertionError("Merged Author cache is non-finite or lost signed coordinates")
        result = {
            "schema": "phase2_author_genejepa_embedding_manifest_v1",
            "created_at_utc": utc_now(),
            "status": "pass",
            "scope": "Task 2 / Phase II exact selected physical-cell cache",
            "subset_manifest": artifact(SUBSET_MANIFEST),
            "cache_plan": artifact(PLAN_SUMMARY),
            "reused_phase1_manual_mean_parity": artifact(PHASE1_PARITY),
            "checkpoint": {
                **artifact(author.AUTHOR_CHECKPOINT),
                "ema_teacher": True,
                "branch": "teacher_encoder.ema_model",
                "frozen": True,
            },
            "pooling": {
                "source": "EMA Teacher final_norm [B,512,768]",
                "operation": "manual mean(dim=1)",
                "tokens_persisted": False,
            },
            "workers": workers,
            "output": {
                **artifact(MERGED_EMBEDDINGS),
                "shape": [total, EMBEDDING_DIM],
                "dtype": "float32",
                "row_equals_phase2_cell_index": True,
            },
            "integrity": {
                "cells": total,
                "treated_cells": treated,
                "control_cells": control,
                "missing": 0,
                "duplicate": 0,
                "finite": True,
                "signed": True,
                "worker_overlap": 0,
            },
            "embedding_statistics": summary,
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_json(FINAL_MANIFEST, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    except BaseException:
        if destination is not None:
            destination.flush()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-worker")
    run.add_argument("--worker-id", type=int, choices=(0, 1), required=True)
    run.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    run.add_argument("--commit-every-batches", type=int, default=10)
    run.add_argument("--max-new-cells", type=int, default=0)
    run.add_argument("--gpu-sample-interval", type=float, default=2.0)
    run.add_argument("--skip-resume-finite-scan", action="store_true")
    commands.add_parser("status")
    commands.add_parser("audit")
    merge_parser = commands.add_parser("merge")
    merge_parser.add_argument("--chunk-cells", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run-worker":
        run_worker(args)
    elif args.command == "status":
        status()
    elif args.command == "audit":
        audit()
    else:
        if args.chunk_cells < 1:
            raise ValueError("--chunk-cells must be positive")
        merge(args.chunk_cells)


if __name__ == "__main__":
    main()
