#!/usr/bin/env python3
"""Extract the frozen Phase-I subset with the Author epoch49 EMA teacher."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
SCRIPT_DIR = PROJECT_ROOT / "perturbation_scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import author_genejepa_epoch49_hd100_cache as old_author


CACHE_DIR = RESULTS / "phase1_author_genejepa_cache"
PREFIX = CACHE_DIR / "cache"
PLAN_SUMMARY = CACHE_DIR / "plan_summary.json"
SUBSET_MANIFEST = RESULTS / "phase1_top20_subset_manifest.json"
SUBSET_CELLS = RESULTS / "phase1_top20_subset_cells.parquet"
SOURCE_CONDITIONS = RESULTS / "tahoe_experiment1_cache_cap512_all_dmso_condition_index.csv"
SOURCE_CONTROLS = RESULTS / "tahoe_experiment1_cache_cap512_all_dmso_control_pool_index.csv"
PARITY_AUDIT = RESULTS / "phase1_author_manual_mean_parity.json"
MERGED_EMBEDDINGS = RESULTS / "phase1_author_genejepa_embeddings.npy"
MERGED_PARTIAL = Path(str(MERGED_EMBEDDINGS) + ".partial")
FINAL_MANIFEST = RESULTS / "phase1_author_genejepa_embedding_manifest.json"

EXPECTED_TOTAL = 101_120
EXPECTED_TREATED = 89_600
EXPECTED_CONTROL = 11_520
LATENT_TOKENS = 512
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
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": relative(path),
        "sha256": old_author.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def worker_path(worker_id: int, suffix: str) -> Path:
    return Path(str(PREFIX) + f"_worker{worker_id}_{suffix}")


def manual_mean_outputs(
    module: Any,
    batch: dict[str, torch.Tensor | list[dict[str, str]]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    captured: list[torch.Tensor] = []

    def capture_final_norm(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        captured.append(output)

    teacher = module.model.teacher_encoder.ema_model
    hook = teacher.final_norm.register_forward_hook(capture_final_norm)
    indices = batch["indices"].to(device, non_blocking=True)
    values = batch["values"].to(device, non_blocking=True)
    offsets = batch["offsets"].to(device, non_blocking=True)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    try:
        with torch.inference_mode(), autocast:
            official = module.model.get_embedding(
                indices=indices,
                values=values,
                offsets=offsets,
                use_teacher=True,
            )
            if len(captured) != 1:
                raise AssertionError(f"Expected one final_norm output, observed {len(captured)}")
            tokens = captured[0]
            expected_shape = (official.shape[0], LATENT_TOKENS, EMBEDDING_DIM)
            if tuple(tokens.shape) != expected_shape:
                raise AssertionError(
                    f"Author pre-pool shape changed: {tuple(tokens.shape)} != {expected_shape}"
                )
            manual = tokens.mean(dim=1)
    finally:
        hook.remove()
    difference = (manual.float() - official.float()).abs()
    max_abs = float(difference.max().item())
    if max_abs > 1e-6:
        raise AssertionError(f"Manual mean differs from get_embedding: max_abs={max_abs}")
    return manual, official, {
        "pre_pool_shape": list(tokens.shape),
        "pre_pool_dtype": str(tokens.dtype),
        "manual_shape": list(manual.shape),
        "official_shape": list(official.shape),
        "max_abs_difference": max_abs,
        "exact_equal": bool(torch.equal(manual, official)),
    }


def infer_manual_mean(
    module: Any,
    batch: dict[str, torch.Tensor | list[dict[str, str]]],
    device: torch.device,
) -> np.ndarray:
    manual, _official, _audit = manual_mean_outputs(module, batch, device)
    return manual.float().cpu().numpy()


def build_provenance(paths: Any) -> tuple[dict[str, Any], int]:
    subset = read_json(SUBSET_MANIFEST)
    plan_summary = read_json(PLAN_SUMMARY)
    parity = read_json(PARITY_AUDIT)
    if any(payload.get("status") != "pass" for payload in (subset, plan_summary, parity)):
        raise AssertionError("Subset, Author plan, and manual-mean parity must all PASS")
    if int(subset["counts"]["total_unique_cells"]) != EXPECTED_TOTAL:
        raise AssertionError("Frozen Phase-I subset size changed")
    expected_worker = next(
        row
        for row in plan_summary["worker_partition"]
        if int(row["worker_id"]) == paths.worker_id
    )
    plan_sha = old_author.sha256_file(paths.plan)
    if plan_sha != expected_worker["plan_sha256"]:
        raise AssertionError("Phase-I Author worker plan changed")
    plan_manifest = read_json(paths.plan_manifest)
    plan_rows = int(plan_manifest["plan"]["rows"])
    if plan_rows != int(expected_worker["rows"]):
        raise AssertionError("Worker row count disagrees with plan summary")
    parquet = pq.ParquetFile(paths.plan)
    if parquet.metadata.num_rows != plan_rows:
        raise AssertionError("Worker plan parquet row count changed")
    missing = set(configure_base(require_parity=False).PLAN_COLUMNS) - set(
        parquet.schema_arrow.names
    )
    if missing:
        raise AssertionError(f"Worker plan missing columns: {sorted(missing)}")
    checkpoint_sha = old_author.sha256_file(old_author.AUTHOR_CHECKPOINT)
    if checkpoint_sha != old_author.AUTHOR_CHECKPOINT_SHA256:
        raise AssertionError("Author checkpoint changed")
    commit = subprocess.run(
        ["git", "-C", str(old_author.AUTHOR_CODE), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != old_author.EXPECTED_AUTHOR_COMMIT:
        raise AssertionError("Author source commit changed")
    stats = read_json(old_author.GLOBAL_STATS)
    return {
        "policy": "Task 1 Phase I exact physical-cell subset; Author epoch49 EMA teacher",
        "subset_manifest": artifact(SUBSET_MANIFEST),
        "plan_summary": artifact(PLAN_SUMMARY),
        "worker_plan": {**artifact(paths.plan), "rows": plan_rows},
        "worker_plan_manifest": artifact(paths.plan_manifest),
        "manual_mean_parity": artifact(PARITY_AUDIT),
        "checkpoint": {
            **artifact(old_author.AUTHOR_CHECKPOINT),
            "ema_teacher": True,
            "branch": "teacher_encoder.ema_model",
            "frozen": True,
        },
        "author_code": {"path": relative(old_author.AUTHOR_CODE), "git_commit": commit},
        "preprocessing": {
            "local_manifest": artifact(old_author.LOCAL_MANIFEST),
            "gene_metadata": artifact(old_author.GENE_METADATA),
            "global_stats": {
                **artifact(old_author.GLOBAL_STATS),
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
            },
            "sentinel_gene_mapping_log1p_normalization_once": True,
            "centering": False,
            "whitening": False,
            "rectification": False,
        },
        "pooling": {
            "captured_tensor": "EMA teacher final_norm output [B,512,768]",
            "operation": "manual torch.mean(dim=1)",
            "saved_output": "raw signed float32 [B,768] only",
            "tokens_persisted": False,
        },
        "condition_index_sha256": old_author.sha256_file(SOURCE_CONDITIONS),
        "control_pool_index_sha256": old_author.sha256_file(SOURCE_CONTROLS),
        "worker_code_sha256": old_author.sha256_file(Path(__file__)),
        "output_contract": {
            "row": "worker-local part_index",
            "global_scatter_key": "phase1_cell_index stored as embedding_index",
            "shape": [plan_rows, EMBEDDING_DIM],
            "dtype": "float32",
        },
    }, plan_rows


def configure_base(*, require_parity: bool) -> Any:
    if require_parity and (
        not PARITY_AUDIT.is_file() or read_json(PARITY_AUDIT).get("status") != "pass"
    ):
        raise FileNotFoundError("Run the Phase-I manual-mean parity check first")
    base = old_author.configure_base(require_contract=False)
    base.PREFIX = PREFIX
    base.PLAN_SUMMARY = PLAN_SUMMARY
    base.CONDITION_INDEX = SOURCE_CONDITIONS
    base.CONTROL_INDEX = SOURCE_CONTROLS
    base.DEFAULT_CHECKPOINT = old_author.AUTHOR_CHECKPOINT
    base.DEFAULT_LOCAL_MANIFEST = old_author.LOCAL_MANIFEST
    base.DEFAULT_STATS = old_author.GLOBAL_STATS
    base.FORMAL_BATCH_SIZE = FORMAL_BATCH_SIZE
    base.EXPECTED_TOTAL = EXPECTED_TOTAL
    base.EXPECTED_TREATED = EXPECTED_TREATED
    base.EXPECTED_DMSO = EXPECTED_CONTROL
    base.PROGRESS_SCHEMA = "phase1_author_genejepa_worker_progress_v1"
    base.RUNTIME_MANIFEST_SCHEMA = "phase1_author_genejepa_worker_extraction_v1"
    base.build_provenance = build_provenance
    base.load_frozen_model = old_author.load_author_model
    base.infer_teacher = infer_manual_mean
    return base


def parity(cells: int) -> dict[str, Any]:
    if PARITY_AUDIT.exists():
        raise FileExistsError(PARITY_AUDIT)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or torch.cuda.device_count() != 1:
        raise RuntimeError("Parity requires CUDA_VISIBLE_DEVICES=0")
    base = configure_base(require_parity=False)
    paths = base.worker_paths(0)
    total = pq.ParquetFile(paths.plan).metadata.num_rows
    if not 1 <= cells <= min(32, total):
        raise ValueError("--cells must be in [1,32]")
    treated, controls = base.load_metadata_maps()
    batches = list(
        base.iter_raw_batches(
            paths.plan,
            start=0,
            stop=cells,
            batch_size=cells,
            treated=treated,
            controls=controls,
        )
    )
    if len(batches) != 1:
        raise AssertionError("Parity prefix did not form one batch")
    preprocessing = base.StreamingPreprocessingAudit()
    datamodule, _ = base.build_official_preprocessor(
        old_author.LOCAL_MANIFEST, old_author.GLOBAL_STATS
    )
    model_batch = base.preprocess_once(
        batches[0].raw_cells, datamodule, preprocessing, inverse_check=True
    )
    device = torch.device("cuda:0")
    module = old_author.load_author_model(old_author.AUTHOR_CHECKPOINT, device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    manual, official, pooling = manual_mean_outputs(module, model_batch, device)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    output = manual.float().cpu().numpy()
    indices = [int(cell.embedding_index) for cell in batches[0].cells]
    subset = pq.read_table(
        SUBSET_CELLS,
        columns=["phase1_cell_index", "source_worker_id", "worker_part_index"],
        filters=[("phase1_cell_index", "in", indices)],
    ).to_pandas().set_index("phase1_cell_index")
    aligned = all(
        int(subset.loc[index, "source_worker_id"]) == 0
        and int(subset.loc[index, "worker_part_index"]) == part_index
        for part_index, index in enumerate(indices)
    )
    result = {
        "schema": "phase1_author_manual_mean_parity_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "cells": cells,
        "phase1_cell_indices": indices,
        "physical_locator_order_verified": aligned and len(set(indices)) == cells,
        "checkpoint": artifact(old_author.AUTHOR_CHECKPOINT),
        "ema_teacher": True,
        "pooling": pooling,
        "output": {
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "finite": bool(np.isfinite(output).all()),
            "signed": bool(np.any(output < 0) and np.any(output > 0)),
            "min": float(output.min()),
            "max": float(output.max()),
            "negative_ratio": float(np.mean(output < 0)),
        },
        "official_get_embedding_shape": list(official.shape),
        "preprocessing": preprocessing.summary(),
        "tokens_saved": False,
        "elapsed_seconds": elapsed,
        "peak_allocated_GiB": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    if not all(
        (
            result["physical_locator_order_verified"],
            result["output"]["finite"],
            result["output"]["signed"],
            pooling["max_abs_difference"] <= 1e-6,
        )
    ):
        raise AssertionError(result)
    atomic_json(PARITY_AUDIT, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def run_worker(args: argparse.Namespace) -> None:
    base = configure_base(require_parity=True)
    if args.batch_size != FORMAL_BATCH_SIZE:
        raise ValueError(f"Frozen Phase-I extraction batch is {FORMAL_BATCH_SIZE}")
    base.run_worker(args)


def status() -> dict[str, Any]:
    base = configure_base(require_parity=False)
    result = base.status_payload()
    result["scope"] = "Task 1 / Phase I Author subset only"
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def merge(chunk_cells: int) -> dict[str, Any]:
    if MERGED_EMBEDDINGS.exists() or MERGED_PARTIAL.exists() or FINAL_MANIFEST.exists():
        raise FileExistsError("Phase-I Author merged output exists; refusing overwrite")
    base = configure_base(require_parity=True)
    subset = pq.read_table(
        SUBSET_CELLS,
        columns=["phase1_cell_index", "source_worker_id", "worker_part_index"],
    ).to_pandas()
    if not np.array_equal(
        subset["phase1_cell_index"].to_numpy(np.int64), np.arange(EXPECTED_TOTAL)
    ):
        raise AssertionError("Subset row order changed")
    subset_worker = subset["source_worker_id"].to_numpy(np.int8)
    subset_part = subset["worker_part_index"].to_numpy(np.int64)
    destination = np.lib.format.open_memmap(
        MERGED_PARTIAL, mode="w+", dtype=np.float32, shape=(EXPECTED_TOTAL, EMBEDDING_DIM)
    )
    written = np.zeros(EXPECTED_TOTAL, dtype=np.bool_)
    statistics = base.StreamingEmbeddingAudit()
    worker_artifacts: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for worker_id in (0, 1):
            paths = base.worker_paths(worker_id)
            progress = read_json(paths.progress)
            if (
                progress.get("status") != "finished"
                or int(progress.get("processed_cells", -1))
                != int(progress.get("total_cells", -2))
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
                phase = batch["embedding_index"].to_numpy(zero_copy_only=False).astype(
                    np.int64
                )
                values = np.asarray(source[part], dtype=np.float32)
                if (
                    np.any(phase < 0)
                    or np.any(phase >= EXPECTED_TOTAL)
                    or bool(written[phase].any())
                    or not np.isfinite(values).all()
                ):
                    raise AssertionError("Invalid, duplicate, or non-finite Author scatter chunk")
                if not np.all(subset_worker[phase] == worker_id) or not np.array_equal(
                    subset_part[phase], part
                ):
                    raise AssertionError("Worker plan disagrees with frozen physical-cell mapping")
                destination[phase] = values
                written[phase] = True
                statistics.add(values)
            worker_artifacts.append(
                {
                    "worker_id": worker_id,
                    "plan": artifact(paths.plan),
                    "manifest": artifact(paths.runtime_manifest),
                    "embedding": artifact(paths.final),
                }
            )
        if not written.all() or int(written.sum()) != EXPECTED_TOTAL:
            raise AssertionError("Phase-I Author merge has missing rows")
        destination.flush()
        del destination
        destination = None
        os.replace(MERGED_PARTIAL, MERGED_EMBEDDINGS)
        summary = statistics.summary()
        result = {
            "schema": "phase1_author_genejepa_embedding_manifest_v1",
            "created_at_utc": utc_now(),
            "status": "pass",
            "scope": "Task 1 / Phase I exact shared physical-cell subset",
            "subset_manifest": artifact(SUBSET_MANIFEST),
            "plan_summary": artifact(PLAN_SUMMARY),
            "manual_mean_parity": artifact(PARITY_AUDIT),
            "checkpoint": {
                **artifact(old_author.AUTHOR_CHECKPOINT),
                "ema_teacher": True,
                "branch": "teacher_encoder.ema_model",
                "frozen": True,
            },
            "pooling": {
                "source": "EMA teacher final_norm [B,512,768]",
                "operation": "manual mean(dim=1)",
                "tokens_persisted": False,
            },
            "workers": worker_artifacts,
            "output": {
                **artifact(MERGED_EMBEDDINGS),
                "shape": [EXPECTED_TOTAL, EMBEDDING_DIM],
                "dtype": "float32",
                "row_equals_phase1_cell_index": True,
            },
            "integrity": {
                "cells": EXPECTED_TOTAL,
                "missing": 0,
                "duplicate": 0,
                "finite": True,
                "worker_overlap": 0,
                "same_physical_cell_order_as_our_and_target": True,
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
    parity_parser = commands.add_parser("parity")
    parity_parser.add_argument("--cells", type=int, default=8)
    run = commands.add_parser("run-worker")
    run.add_argument("--worker-id", type=int, choices=(0, 1), required=True)
    run.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    run.add_argument("--commit-every-batches", type=int, default=10)
    run.add_argument("--max-new-cells", type=int, default=0)
    run.add_argument("--gpu-sample-interval", type=float, default=2.0)
    run.add_argument("--skip-resume-finite-scan", action="store_true")
    commands.add_parser("status")
    merge_parser = commands.add_parser("merge")
    merge_parser.add_argument("--chunk-cells", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "parity":
        parity(args.cells)
    elif args.command == "run-worker":
        run_worker(args)
    elif args.command == "status":
        status()
    elif args.command == "merge":
        if args.chunk_cells < 1:
            raise ValueError("--chunk-cells must be positive")
        merge(args.chunk_cells)


if __name__ == "__main__":
    main()
