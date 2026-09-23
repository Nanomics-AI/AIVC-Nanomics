#!/usr/bin/env python3
"""Plan, extract, and merge the Author epoch49 cache needed by HD100.

The implementation deliberately reuses the frozen Experiment 1 locator plans
and the proven resume-safe worker.  It does not resample Tahoe cells.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
CACHE_DIR = RESULTS / "author_genejepa_epoch49_hd100_cache"
PREFIX = CACHE_DIR / "cache"
PLAN_SUMMARY = CACHE_DIR / "plan_summary.json"
EXTRACTION_CONTRACT = CACHE_DIR / "extraction_contract.json"
BATCH_PROBE = CACHE_DIR / "batch_probe.json"
PARITY_AUDIT = CACHE_DIR / "eight_cell_production_parity.json"
REHEARSAL_AUDIT = CACHE_DIR / "merge_rehearsal.json"
MERGED_EMBEDDINGS = CACHE_DIR / "embeddings.npy"
MERGED_PARTIAL = Path(str(MERGED_EMBEDDINGS) + ".partial")
FINAL_MANIFEST = RESULTS / "author_genejepa_epoch49_hd100_cache_manifest.json"

OLD_PREFIX = RESULTS / "tahoe_experiment1_cache_cap512_all_dmso"
OLD_SUMMARY = Path(str(OLD_PREFIX) + "_summary.json")
CONDITION_INDEX = Path(str(OLD_PREFIX) + "_condition_index.csv")
CONTROL_INDEX = Path(str(OLD_PREFIX) + "_control_pool_index.csv")
OLD_PLANS = tuple(Path(str(OLD_PREFIX) + f"_worker{i}_plan.parquet") for i in range(2))
ARC7_PLAN = RESULTS / "genejepa_decoder_v1_arc7_100_plan.json"
ARC7_CONDITIONS = RESULTS / "genejepa_decoder_v1_arc7_100_conditions.csv"

AUTHOR_CHECKPOINT = PROJECT_ROOT / "external/author_genejepa/genejepa-epoch=49.ckpt"
AUTHOR_CODE = PROJECT_ROOT / "external/author_genejepa_code"
AUTHOR_AUDIT = RESULTS / "author_genejepa_checkpoint_audit.json"
AUTHOR_SMOKE = RESULTS / "author_genejepa_8cell_smoke_embeddings.npy"
LOCAL_MANIFEST = PROJECT_ROOT / "hf_data_cache/local_file_manifest.json"
GLOBAL_STATS = PROJECT_ROOT / "hf_data_cache/global_stats.json"
GENE_METADATA = PROJECT_ROOT / "hf_data_cache/metadata/metadata/gene_metadata.parquet"
TASK_PATH = PROJECT_ROOT.parent / "当前任务.txt"

AUTHOR_CHECKPOINT_SHA256 = "5db5c5750aeecc09955fefcaf143b349d9d8e1e877fe4a1073400ba66ef56f05"
EXPECTED_AUTHOR_COMMIT = "a2f4d7218b17f2f52cc5f1cc94420c8ef1ae3265"
NORMALIZATION_POLICY = (
    "reuse current project stats; author/current differences are below float32 effect "
    "in audited smoke"
)
EMBEDDING_DIM = 768
TRAIN_CELLS = 22_941_936
VAL_CELLS = 2_841_724
ARC7_CONTROL_CELLS = 256
ARC7_TREATED_CELLS = 23 * 256
ARC7_CELLS = ARC7_CONTROL_CELLS + ARC7_TREATED_CELLS
EXPECTED_TOTAL = TRAIN_CELLS + VAL_CELLS + ARC7_CELLS
SECTION_BOUNDS = {
    "train": (0, TRAIN_CELLS),
    "val": (TRAIN_CELLS, TRAIN_CELLS + VAL_CELLS),
    "arc7_control": (TRAIN_CELLS + VAL_CELLS, TRAIN_CELLS + VAL_CELLS + 256),
    "arc7_treated": (TRAIN_CELLS + VAL_CELLS + 256, EXPECTED_TOTAL),
}
SECTION_TO_CODE = {name: index for index, name in enumerate(SECTION_BOUNDS)}
CODE_TO_SECTION = {value: key for key, value in SECTION_TO_CODE.items()}
OWNER_TREATED = 0
OWNER_CONTROL = 1
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
    "cache_section",
]


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def artifact(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[str, Any] = {"path": relative(path), "size_bytes": path.stat().st_size}
    if hash_file:
        result["sha256"] = sha256_file(path)
    return result


def worker_paths(worker_id: int) -> dict[str, Path]:
    stem = str(PREFIX) + f"_worker{worker_id}"
    return {
        "plan": Path(stem + "_plan.parquet"),
        "plan_manifest": Path(stem + "_manifest.json"),
        "progress": Path(stem + "_progress.json"),
        "completed": Path(stem + "_completed.npy"),
        "partial": Path(stem + "_embeddings.npy.partial"),
        "final": Path(stem + "_embeddings.npy"),
        "runtime_manifest": Path(stem + "_extraction_manifest.json"),
        "lock": Path(stem + "_extraction.lock"),
    }


def source_plan_record(path: Path, expected: dict[str, Any]) -> None:
    if path.stat().st_size != int(expected["size_bytes"]):
        raise AssertionError(f"Frozen source plan size changed: {path}")
    if sha256_file(path) != expected["sha256"]:
        raise AssertionError(f"Frozen source plan SHA-256 changed: {path}")


def reconstruct_arc7() -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    # Reuse the exact frozen sampler; no new sampling implementation is introduced.
    from run_genejepa_decoder_v1_arc7_100 import (
        make_sampling_plan,
        select_context,
        selection_fingerprint,
    )

    dataset, selected, context = select_context()
    control, treated, sampling = make_sampling_plan(dataset, selected)
    del dataset
    frozen = read_json(ARC7_PLAN)
    if (
        frozen.get("status") != "pass"
        or frozen["selection_fingerprint"]
        != selection_fingerprint(selected, control, treated)
        or selected["pair_id"].astype(str).tolist()
        != pd.read_csv(ARC7_CONDITIONS, encoding="utf-8-sig")["pair_id"].astype(str).tolist()
    ):
        raise AssertionError("Frozen ARC7 selection/sampling changed")
    if len(control) != 256 or sum(map(len, treated.values())) != ARC7_TREATED_CELLS:
        raise AssertionError("ARC7 cell count changed")
    return selected, control, treated, {"context": context, "sampling": sampling, "frozen": frozen}


def arc7_index_contract(
    selected: pd.DataFrame, control: np.ndarray, treated: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    old_indices: list[int] = []
    new_indices: list[int] = []
    owner_types: list[int] = []
    owner_indices: list[int] = []
    arc_start = SECTION_BOUNDS["arc7_control"][0]
    control_owner = int(selected.iloc[0]["control_pool_index"])
    for offset, old in enumerate(control):
        old_indices.append(int(old))
        new_indices.append(arc_start + offset)
        owner_types.append(OWNER_CONTROL)
        owner_indices.append(control_owner)
    treated_start = SECTION_BOUNDS["arc7_treated"][0]
    for condition_offset, row in selected.iterrows():
        condition_index = int(row["cache_condition_index"])
        for cell_offset, old in enumerate(treated[str(row["pair_id"])]):
            old_indices.append(int(old))
            new_indices.append(treated_start + condition_offset * 256 + cell_offset)
            owner_types.append(OWNER_TREATED)
            owner_indices.append(condition_index)
    order = np.argsort(np.asarray(old_indices, dtype=np.int64), kind="stable")
    return tuple(
        np.asarray(values, dtype=np.int64)[order]
        for values in (old_indices, new_indices, owner_types, owner_indices)
    )  # type: ignore[return-value]


def _arc_lookup(
    old_embedding_indices: np.ndarray,
    arc_old: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(arc_old, old_embedding_indices)
    valid = positions < len(arc_old)
    clipped = np.minimum(positions, len(arc_old) - 1)
    valid &= arc_old[clipped] == old_embedding_indices
    return valid, clipped


def build_plan() -> dict[str, Any]:
    outputs = [PLAN_SUMMARY]
    for worker_id in (0, 1):
        outputs.extend(worker_paths(worker_id).values())
    existing = [relative(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite Author cache state: {existing}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    old_summary = read_json(OLD_SUMMARY)
    if old_summary.get("status") != "pass":
        raise AssertionError("Frozen Experiment 1 plan summary is not PASS")
    for worker_id, source in enumerate(OLD_PLANS):
        source_plan_record(source, old_summary["outputs"][f"worker{worker_id}_plans"])
    conditions = pd.read_csv(CONDITION_INDEX, encoding="utf-8-sig", keep_default_na=False)
    if len(conditions) != 56_993 or not np.array_equal(
        conditions["cache_condition_index"].to_numpy(np.int64), np.arange(len(conditions))
    ):
        raise AssertionError("Frozen condition index changed")

    split_codes = conditions["split"].map({"train": 0, "val": 1, "test": 2}).to_numpy(np.int8)
    counts = conditions["treated_cached_cell_count"].to_numpy(np.int64)
    old_starts = conditions["treated_embedding_start"].to_numpy(np.int64)
    old_stops = conditions["treated_embedding_stop_exclusive"].to_numpy(np.int64)
    if not np.array_equal(old_stops - old_starts, counts):
        raise AssertionError("Frozen condition ranges do not match cached counts")
    new_starts = np.full(len(conditions), -1, dtype=np.int64)
    cursor = 0
    for split, expected in (("train", TRAIN_CELLS), ("val", VAL_CELLS)):
        indices = np.flatnonzero(conditions["split"].eq(split).to_numpy())
        for condition_index in indices:
            new_starts[condition_index] = cursor
            cursor += int(counts[condition_index])
        if cursor != SECTION_BOUNDS[split][1]:
            raise AssertionError(f"{split} cache total changed: {cursor} != {SECTION_BOUNDS[split][1]}")

    selected, control, treated, arc7 = reconstruct_arc7()
    arc_old, arc_new, arc_owner_type, arc_owner_index = arc7_index_contract(
        selected, control, treated
    )
    if len(np.unique(arc_old)) != ARC7_CELLS or len(np.unique(arc_new)) != ARC7_CELLS:
        raise AssertionError("ARC7 locator indices are not unique")

    seen = np.zeros(EXPECTED_TOTAL, dtype=np.bool_)
    section_counts = np.zeros(len(SECTION_TO_CODE), dtype=np.int64)
    worker_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    schema = pa.schema(
        [
            pa.field("part_index", pa.int64()),
            pa.field("embedding_index", pa.int64()),
            pa.field("shard_index", pa.int32()),
            pa.field("shard_path", pa.string()),
            pa.field("row_group_index", pa.int16()),
            pa.field("row_index_in_row_group", pa.int32()),
            pa.field("row_index_in_shard", pa.int32()),
            pa.field("owner_type", pa.int8()),
            pa.field("owner_index", pa.int32()),
            pa.field("cache_section", pa.string()),
        ]
    )

    for worker_id, source_path in enumerate(OLD_PLANS):
        paths = worker_paths(worker_id)
        temporary = paths["plan"].with_name(paths["plan"].name + ".tmp")
        source = pq.ParquetFile(source_path)
        writer = pq.ParquetWriter(temporary, schema, compression="zstd")
        part_cursor = 0
        shards = 0
        try:
            for row_group in range(source.num_row_groups):
                table = source.read_row_group(row_group)
                owner_type = table["owner_type"].to_numpy(zero_copy_only=False).astype(np.int8)
                owner_index = table["owner_index"].to_numpy(zero_copy_only=False).astype(np.int64)
                old_embedding = table["embedding_index"].to_numpy(zero_copy_only=False).astype(
                    np.int64
                )
                treated_rows = owner_type == OWNER_TREATED
                train_val = np.zeros(len(table), dtype=np.bool_)
                train_val[treated_rows] = split_codes[owner_index[treated_rows]] < 2
                arc_match, arc_position = _arc_lookup(old_embedding, arc_old)
                keep = train_val | arc_match
                positions = np.flatnonzero(keep)
                if not len(positions):
                    continue

                new_embedding = np.empty(len(positions), dtype=np.int64)
                section_code = np.empty(len(positions), dtype=np.int8)
                selected_train_val = train_val[positions]
                condition = owner_index[positions[selected_train_val]]
                local = old_embedding[positions[selected_train_val]] - old_starts[condition]
                if np.any((local < 0) | (local >= counts[condition])):
                    raise AssertionError("Treated row falls outside its frozen condition range")
                new_embedding[selected_train_val] = new_starts[condition] + local
                section_code[selected_train_val] = split_codes[condition]

                selected_arc = ~selected_train_val
                arc_positions = arc_position[positions[selected_arc]]
                if np.any(owner_type[positions[selected_arc]] != arc_owner_type[arc_positions]) or np.any(
                    owner_index[positions[selected_arc]] != arc_owner_index[arc_positions]
                ):
                    raise AssertionError("ARC7 stable locator ownership changed")
                new_embedding[selected_arc] = arc_new[arc_positions]
                section_code[selected_arc] = np.where(
                    arc_owner_type[arc_positions] == OWNER_CONTROL,
                    SECTION_TO_CODE["arc7_control"],
                    SECTION_TO_CODE["arc7_treated"],
                )
                if np.any(seen[new_embedding]):
                    raise AssertionError("New cache embedding_index would be duplicated")
                seen[new_embedding] = True
                section_counts += np.bincount(section_code, minlength=len(section_counts))

                output = pa.table(
                    {
                        "part_index": np.arange(
                            part_cursor, part_cursor + len(positions), dtype=np.int64
                        ),
                        "embedding_index": new_embedding,
                        "shard_index": table["shard_index"].take(pa.array(positions)),
                        "shard_path": table["shard_path"].take(pa.array(positions)),
                        "row_group_index": table["row_group_index"].take(pa.array(positions)),
                        "row_index_in_row_group": table["row_index_in_row_group"].take(
                            pa.array(positions)
                        ),
                        "row_index_in_shard": table["row_index_in_shard"].take(
                            pa.array(positions)
                        ),
                        "owner_type": table["owner_type"].take(pa.array(positions)),
                        "owner_index": table["owner_index"].take(pa.array(positions)),
                        "cache_section": pa.array(
                            [CODE_TO_SECTION[int(code)] for code in section_code], type=pa.string()
                        ),
                    },
                    schema=schema,
                )
                writer.write_table(output)
                part_cursor += len(positions)
                shards += 1
        finally:
            writer.close()
        os.replace(temporary, paths["plan"])
        plan_sha = sha256_file(paths["plan"])
        manifest = {
            "schema": "author_genejepa_epoch49_hd100_worker_plan_v1",
            "created_at_utc": utc_now(),
            "status": "planned_not_started",
            "worker_id": worker_id,
            "device_contract": f"CUDA_VISIBLE_DEVICES={worker_id}; local device cuda:0",
            "distributed": False,
            "plan": {
                **artifact(paths["plan"]),
                "rows": part_cursor,
                "part_index_start": 0,
                "part_index_stop_exclusive": part_cursor,
                "physical_order": "shard_path + row_group_index + row_index_in_row_group",
                "scatter_key": "embedding_index",
                "source_shards": shards,
            },
            "resume": {
                "progress_path": relative(paths["progress"]),
                "completed_bitmap_path": relative(paths["completed"]),
                "partial_embedding_path": relative(paths["partial"]),
                "final_embedding_path": relative(paths["final"]),
            },
            "output_contract": {
                "shape": [part_cursor, EMBEDDING_DIM],
                "dtype": "float32",
                "no_shared_output_file_between_workers": True,
            },
        }
        atomic_json(paths["plan_manifest"], manifest)
        atomic_json(
            paths["progress"],
            {
                "schema": "author_genejepa_epoch49_worker_progress_v1",
                "created_at_utc": utc_now(),
                "updated_at_utc": utc_now(),
                "status": "not_started",
                "embedding_extraction_started": False,
                "worker_id": worker_id,
                "processed_cells": 0,
                "remaining_cells": part_cursor,
                "total_cells": part_cursor,
                "committed_prefix_stop": 0,
            },
        )
        worker_records.append(
            {
                "worker_id": worker_id,
                "rows": part_cursor,
                "shards_with_selected_cells": shards,
                "plan_size_bytes": paths["plan"].stat().st_size,
                "plan_sha256": plan_sha,
            }
        )

    expected_sections = np.asarray(
        [TRAIN_CELLS, VAL_CELLS, ARC7_CONTROL_CELLS, ARC7_TREATED_CELLS], dtype=np.int64
    )
    if not seen.all() or int(seen.sum()) != EXPECTED_TOTAL or not np.array_equal(
        section_counts, expected_sections
    ):
        raise AssertionError(
            f"Plan union mismatch: seen={int(seen.sum())}, sections={section_counts.tolist()}"
        )
    if sum(record["rows"] for record in worker_records) != EXPECTED_TOTAL:
        raise AssertionError("Worker plan row totals do not cover the complete cache")

    summary = {
        "schema": "author_genejepa_epoch49_hd100_cache_plan_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "policy": "frozen Experiment 1 train/val treated cells plus frozen ARC7-C cells",
        "selection": {
            "source": artifact(OLD_SUMMARY),
            "new_sampling_performed": False,
            "train": {"cells": TRAIN_CELLS, "treated_only": True},
            "val": {"cells": VAL_CELLS, "treated_only": True},
            "arc7_control": {"cells": ARC7_CONTROL_CELLS},
            "arc7_treated": {"conditions": 23, "cells": ARC7_TREATED_CELLS},
            "arc7_selection_fingerprint": arc7["frozen"]["selection_fingerprint"],
            "arc7_sampling": arc7["sampling"],
        },
        "embedding_index": {
            "total_cells": EXPECTED_TOTAL,
            "first": 0,
            "last": EXPECTED_TOTAL - 1,
            "sections": {
                name: {"start": start, "stop_exclusive": stop, "cells": stop - start}
                for name, (start, stop) in SECTION_BOUNDS.items()
            },
        },
        "partition": {
            "workers": worker_records,
            "algorithm": "reuse frozen whole-shard worker ownership and physical order",
            "worker_rows": [record["rows"] for record in worker_records],
            "missing_embedding_indices": 0,
            "duplicate_embedding_indices": 0,
            "each_embedding_index_exactly_once": True,
            "worker_cell_overlap": 0,
            "worker_union_cells": EXPECTED_TOTAL,
            "worker_shard_overlap": 0,
        },
        "inputs": {
            "condition_index": artifact(CONDITION_INDEX),
            "control_index": artifact(CONTROL_INDEX),
            "arc7_plan": artifact(ARC7_PLAN),
            "arc7_conditions": artifact(ARC7_CONDITIONS),
            "source_worker_plans": [artifact(path) for path in OLD_PLANS],
            "task": artifact(TASK_PATH),
        },
        "outputs": {
            f"worker{worker_id}_plans": artifact(worker_paths(worker_id)["plan"])
            for worker_id in (0, 1)
        },
        "elapsed_seconds": time.perf_counter() - started,
        "embedding_extraction_started": False,
    }
    atomic_json(PLAN_SUMMARY, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


_BASE: Any | None = None
_AUTHOR_MODULE_CLASS: Any | None = None
_LAST_LOAD_AUDIT: dict[str, Any] | None = None


def configure_base(*, require_contract: bool) -> Any:
    """Load current preprocessing first, then switch only the model package to author code."""
    global _BASE, _AUTHOR_MODULE_CLASS
    if _BASE is None:
        import extract_tahoe_experiment1_full_cache_worker as base

        # base now holds references to the current, audited Tahoe preprocessing classes.
        for name in list(sys.modules):
            if name == "genejepa" or name.startswith("genejepa."):
                del sys.modules[name]
        from audit_author_genejepa_checkpoint import register_author_config_aliases

        register_author_config_aliases()
        from genejepa.train import JepaLightningModule

        _AUTHOR_MODULE_CLASS = JepaLightningModule
        _BASE = base
    base = _BASE
    selected_batch = read_json(EXTRACTION_CONTRACT)["formal_batch_size"] if require_contract else 64
    base.PREFIX = PREFIX
    base.PLAN_SUMMARY = PLAN_SUMMARY
    base.CONDITION_INDEX = CONDITION_INDEX
    base.CONTROL_INDEX = CONTROL_INDEX
    base.DEFAULT_CHECKPOINT = AUTHOR_CHECKPOINT
    base.DEFAULT_LOCAL_MANIFEST = LOCAL_MANIFEST
    base.DEFAULT_STATS = GLOBAL_STATS
    base.FORMAL_BATCH_SIZE = int(selected_batch)
    base.EXPECTED_TOTAL = EXPECTED_TOTAL
    base.EXPECTED_TREATED = TRAIN_CELLS + VAL_CELLS + ARC7_TREATED_CELLS
    base.EXPECTED_DMSO = ARC7_CONTROL_CELLS
    base.PROGRESS_SCHEMA = "author_genejepa_epoch49_worker_progress_v1"
    base.RUNTIME_MANIFEST_SCHEMA = "author_genejepa_epoch49_worker_extraction_v1"
    base.build_provenance = build_provenance
    base.load_frozen_model = load_author_model
    return base


def load_author_model(checkpoint: Path, device: torch.device) -> Any:
    global _LAST_LOAD_AUDIT
    if _AUTHOR_MODULE_CLASS is None:
        raise RuntimeError("configure_base() must run before loading Author GeneJEPA")
    try:
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_keys = set(checkpoint_payload["state_dict"])
    del checkpoint_payload
    gc.collect()
    module = _AUTHOR_MODULE_CLASS.load_from_checkpoint(
        str(checkpoint), map_location="cpu", strict=False
    )
    loaded_keys = set(module.state_dict())
    missing = sorted(loaded_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - loaded_keys)
    expected_unexpected = sorted(
        ["teacher_center", "model.local_desc_proj.weight", "model.mask_desc_embed.weight"]
    )
    if missing or unexpected != expected_unexpected:
        raise AssertionError(
            f"Author checkpoint key contract failed: missing={missing}, unexpected={unexpected}"
        )
    module.eval().requires_grad_(False).to(device)
    module.model.teacher_encoder.ema_model.eval()
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise AssertionError("Author GeneJEPA was not frozen")
    _LAST_LOAD_AUDIT = {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "use_teacher": True,
        "branch": "teacher_encoder.ema_model",
    }
    return module


def build_provenance(paths: Any) -> tuple[dict[str, Any], int]:
    summary = read_json(PLAN_SUMMARY)
    contract = read_json(EXTRACTION_CONTRACT)
    parity = read_json(PARITY_AUDIT)
    probe = read_json(BATCH_PROBE)
    if any(payload.get("status") != "pass" for payload in (summary, contract, parity, probe)):
        raise AssertionError("Plan, batch probe, extraction contract, and parity must all PASS")
    expected_worker = summary["partition"]["workers"][paths.worker_id]
    if sha256_file(paths.plan) != expected_worker["plan_sha256"]:
        raise AssertionError("Author worker plan SHA-256 changed")
    plan_manifest = read_json(paths.plan_manifest)
    total = int(plan_manifest["plan"]["rows"])
    if pq.ParquetFile(paths.plan).metadata.num_rows != total:
        raise AssertionError("Author worker plan row count changed")
    checkpoint_sha = sha256_file(AUTHOR_CHECKPOINT)
    if checkpoint_sha != AUTHOR_CHECKPOINT_SHA256:
        raise AssertionError("Author checkpoint SHA-256 changed")
    audit = read_json(AUTHOR_AUDIT)
    if audit.get("status") != "pass_with_required_loading_adaptation":
        raise AssertionError("Author checkpoint audit is not PASS")
    commit = subprocess.run(
        ["git", "-C", str(AUTHOR_CODE), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != EXPECTED_AUTHOR_COMMIT:
        raise AssertionError("Author source commit changed")
    stats = read_json(GLOBAL_STATS)
    return {
        "policy": "Author epoch49 EMA teacher; train/val treated plus frozen ARC7-C only",
        "plan_summary": artifact(PLAN_SUMMARY),
        "worker_plan": {**artifact(paths.plan), "rows": total},
        "worker_plan_manifest": artifact(paths.plan_manifest),
        "extraction_contract": artifact(EXTRACTION_CONTRACT),
        "checkpoint": {
            **artifact(AUTHOR_CHECKPOINT),
            "ema_teacher": True,
            "use_teacher": True,
            "inference_branch": "teacher_encoder.ema_model",
            "missing_keys_required": [],
            "unexpected_keys_required": [
                "teacher_center",
                "model.local_desc_proj.weight",
                "model.mask_desc_embed.weight",
            ],
        },
        "author_code": {"path": relative(AUTHOR_CODE), "git_commit": commit},
        "checkpoint_audit": artifact(AUTHOR_AUDIT),
        "preprocessing": {
            "implementation": relative(
                PROJECT_ROOT / "perturbation_scripts/extract_tahoe_latent_audit_embeddings.py"
            ),
            "sentinel_and_gene_mapping": "current Tahoe100MDataset.__iter__, once",
            "log1p_and_global_normalization": "current Tahoe100MDataModule._collate_fn, once",
            "normalization_stats_policy": NORMALIZATION_POLICY,
            "CP10K": False,
            "centering": False,
            "whitening": False,
            "L2_normalization": False,
            "rectification": False,
            "raw_signed_float32_output": True,
            "formal_inference_batch_size": int(contract["formal_batch_size"]),
        },
        "local_manifest": artifact(LOCAL_MANIFEST),
        "gene_metadata": artifact(GENE_METADATA),
        "global_stats": {
            **artifact(GLOBAL_STATS),
            "mean": float(stats["mean"]),
            "std": float(stats["std"]),
            "policy": NORMALIZATION_POLICY,
        },
        "condition_index": artifact(CONDITION_INDEX),
        "control_index": artifact(CONTROL_INDEX),
        "worker_code": artifact(Path(__file__)),
        "reused_resume_worker_code": artifact(
            PROJECT_ROOT / "perturbation_scripts/extract_tahoe_experiment1_full_cache_worker.py"
        ),
        "output_contract": {
            "row": "worker-local part_index",
            "global_scatter_key": "embedding_index",
            "shape": [total, EMBEDDING_DIM],
            "dtype": "float32",
        },
    }, total


def _read_probe_cells(base: Any, cells: int) -> tuple[list[Any], list[dict[str, Any]]]:
    treated, controls = base.load_metadata_maps()
    plan_cells: list[Any] = []
    raw_cells: list[dict[str, Any]] = []
    for batch in base.iter_raw_batches(
        worker_paths(0)["plan"],
        start=0,
        stop=cells,
        batch_size=cells,
        treated=treated,
        controls=controls,
    ):
        plan_cells.extend(batch.cells)
        raw_cells.extend(batch.raw_cells)
    if len(raw_cells) != cells:
        raise AssertionError("Probe did not read the requested formal-plan prefix")
    return plan_cells, raw_cells


def batch_probe(cells: int) -> dict[str, Any]:
    if BATCH_PROBE.exists() or EXTRACTION_CONTRACT.exists():
        raise FileExistsError("Batch probe/contract already exists; do not silently replace it")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or torch.cuda.device_count() != 1:
        raise RuntimeError("Batch probe requires CUDA_VISIBLE_DEVICES=0")
    base = configure_base(require_contract=False)
    plan_cells, raw_cells = _read_probe_cells(base, cells)
    device = torch.device("cuda:0")
    datamodule, _ = base.build_official_preprocessor(LOCAL_MANIFEST, GLOBAL_STATS)
    module = load_author_model(AUTHOR_CHECKPOINT, device)
    results: list[dict[str, Any]] = []
    selected: int | None = None
    for batch_size in (64, 32):
        if selected is not None:
            break
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        outputs: list[np.ndarray] = []
        try:
            for start in range(0, cells, batch_size):
                audit = base.StreamingPreprocessingAudit()
                model_batch = base.preprocess_once(
                    raw_cells[start : start + batch_size], datamodule, audit, inverse_check=False
                )
                outputs.append(base.infer_teacher(module, model_batch, device))
            torch.cuda.synchronize(device)
            output = np.concatenate(outputs)
            if output.shape != (cells, EMBEDDING_DIM) or not np.isfinite(output).all():
                raise AssertionError("Author batch probe output shape/finite check failed")
            if not np.any(output < 0) or not np.any(output > 0):
                raise AssertionError("Author batch probe did not preserve signed coordinates")
            elapsed = time.perf_counter() - started
            results.append(
                {
                    "batch_size": batch_size,
                    "status": "pass",
                    "cells": cells,
                    "elapsed_seconds": elapsed,
                    "cells_per_second": cells / elapsed,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    "finite": True,
                    "signed": True,
                }
            )
            selected = batch_size
        except torch.OutOfMemoryError as error:
            results.append(
                {"batch_size": batch_size, "status": "oom", "error": str(error)}
            )
            torch.cuda.empty_cache()
    if selected is None:
        raise RuntimeError("Author GeneJEPA OOM at both batch=64 and batch=32")
    payload = {
        "schema": "author_genejepa_epoch49_batch_probe_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "scope": "one minimal real-Tahoe production-path throughput/OOM check",
        "formal_plan_cells": {
            "worker": 0,
            "count": cells,
            "first_part_index": plan_cells[0].part_index,
            "last_part_index": plan_cells[-1].part_index,
        },
        "results": results,
        "selected_formal_batch_size": selected,
        "loader": _LAST_LOAD_AUDIT,
        "formal_cache_rows_written": 0,
    }
    atomic_json(BATCH_PROBE, payload)
    contract = {
        "schema": "author_genejepa_epoch49_extraction_contract_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "formal_batch_size": selected,
        "precision": "bfloat16 autocast inference; float32 output",
        "checkpoint_sha256": AUTHOR_CHECKPOINT_SHA256,
        "normalization_stats_policy": NORMALIZATION_POLICY,
        "batch_probe": artifact(BATCH_PROBE),
    }
    atomic_json(EXTRACTION_CONTRACT, contract)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def parity() -> dict[str, Any]:
    if PARITY_AUDIT.exists():
        raise FileExistsError("Production parity result already exists")
    if not EXTRACTION_CONTRACT.is_file():
        raise FileNotFoundError("Run probe-batch first")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or torch.cuda.device_count() != 1:
        raise RuntimeError("Parity check requires CUDA_VISIBLE_DEVICES=0")
    base = configure_base(require_contract=True)
    from audit_author_genejepa_checkpoint import read_fixed_cells

    plan, raw_cells, locator_audit = read_fixed_cells()
    datamodule, _ = base.build_official_preprocessor(LOCAL_MANIFEST, GLOBAL_STATS)
    preprocessing = base.StreamingPreprocessingAudit()
    batch = base.preprocess_once(raw_cells, datamodule, preprocessing, inverse_check=True)
    device = torch.device("cuda:0")
    module = load_author_model(AUTHOR_CHECKPOINT, device)
    observed = base.infer_teacher(module, batch, device).astype(
        np.float32, copy=False
    )
    with torch.inference_mode():
        observed_fp32 = (
            module.model.get_embedding(
                indices=batch["indices"].to(device),
                values=batch["values"].to(device),
                offsets=batch["offsets"].to(device),
                use_teacher=True,
            )
            .float()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
    expected = np.load(AUTHOR_SMOKE, allow_pickle=False)
    difference = np.abs(observed.astype(np.float64) - expected.astype(np.float64))
    fp32_difference = np.abs(observed_fp32.astype(np.float64) - expected.astype(np.float64))
    fp32_allclose = bool(np.allclose(observed_fp32, expected, rtol=1e-5, atol=1e-6))
    expected64 = expected.astype(np.float64)
    observed64 = observed.astype(np.float64)
    relative_l2 = np.linalg.norm(observed64 - expected64, axis=1) / np.maximum(
        np.linalg.norm(expected64, axis=1), 1e-12
    )
    cosine = np.sum(observed64 * expected64, axis=1) / np.maximum(
        np.linalg.norm(observed64, axis=1) * np.linalg.norm(expected64, axis=1), 1e-12
    )
    bf16_thresholds = {
        "max_abs_difference": 0.20,
        "mean_abs_difference": 0.03,
        "max_relative_l2": 0.06,
        "min_row_cosine": 0.998,
    }
    bf16_consistent = bool(
        float(difference.max()) <= bf16_thresholds["max_abs_difference"]
        and float(difference.mean()) <= bf16_thresholds["mean_abs_difference"]
        and float(relative_l2.max()) <= bf16_thresholds["max_relative_l2"]
        and float(cosine.min()) >= bf16_thresholds["min_row_cosine"]
    )
    if (
        observed.shape != (8, 768)
        or expected.shape != observed.shape
        or observed_fp32.shape != observed.shape
        or not np.isfinite(observed).all()
        or not np.isfinite(observed_fp32).all()
        or not fp32_allclose
        or not bf16_consistent
        or not locator_audit["metadata_verified"]
    ):
        raise AssertionError(
            "8-cell production parity failed: "
            f"shape={observed.shape}, fp32_max_abs={fp32_difference.max()}, "
            f"bf16_max_abs={difference.max()}, bf16_mean_abs={difference.mean()}, "
            f"bf16_max_relative_l2={relative_l2.max()}, min_cosine={cosine.min()}"
        )
    payload = {
        "schema": "author_genejepa_epoch49_production_parity_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "cells": 8,
        "embedding_indices": plan["embedding_index"].astype(int).tolist(),
        "same_physical_cells_and_order": True,
        "locator_metadata_verified": True,
        "shape": list(observed.shape),
        "dtype": str(observed.dtype),
        "finite": True,
        "signed": bool(np.any(observed < 0) and np.any(observed > 0)),
        "production_precision": "bfloat16 autocast -> float32 output",
        "audit_smoke_precision": "float32",
        "fp32_reference_parity": {
            "rtol": 1e-5,
            "atol": 1e-6,
            "allclose": fp32_allclose,
            "max_abs_difference": float(fp32_difference.max()),
            "mean_abs_difference": float(fp32_difference.mean()),
        },
        "production_bfloat16_consistency": {
            "status": "pass",
            "acceptance_thresholds": bf16_thresholds,
            "max_abs_difference": float(difference.max()),
            "mean_abs_difference": float(difference.mean()),
            "max_relative_l2": float(relative_l2.max()),
            "mean_relative_l2": float(relative_l2.mean()),
            "min_row_cosine": float(cosine.min()),
            "mean_row_cosine": float(cosine.mean()),
        },
        "loader": _LAST_LOAD_AUDIT,
        "preprocessing": preprocessing.summary(),
        "reference": artifact(AUTHOR_SMOKE),
        "normalization_stats_policy": NORMALIZATION_POLICY,
    }
    atomic_json(PARITY_AUDIT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def run_worker(args: argparse.Namespace) -> None:
    if not (PLAN_SUMMARY.is_file() and EXTRACTION_CONTRACT.is_file() and PARITY_AUDIT.is_file()):
        raise FileNotFoundError("Run plan, probe-batch, and parity before extraction")
    base = configure_base(require_contract=True)
    selected = int(read_json(EXTRACTION_CONTRACT)["formal_batch_size"])
    if args.batch_size is not None and args.batch_size != selected:
        raise ValueError(f"Frozen extraction batch is {selected}, not {args.batch_size}")
    args.batch_size = selected
    print(
        f"worker{args.worker_id}: Author epoch49 EMA teacher, frozen batch={selected}",
        flush=True,
    )
    base.run_worker(args)


def status() -> dict[str, Any]:
    rows = []
    for worker_id in (0, 1):
        paths = worker_paths(worker_id)
        progress = read_json(paths["progress"])
        rows.append(
            {
                "worker_id": worker_id,
                "status": progress.get("status"),
                "processed_cells": int(progress.get("processed_cells", 0)),
                "remaining_cells": int(progress.get("remaining_cells", 0)),
                "cells_per_active_second": float(progress.get("cells_per_active_second", 0.0)),
                "failed_attempts": int(progress.get("failed_attempts", 0)),
                "permanent_failed_cells": int(progress.get("permanent_failed_cells", 0)),
                "progress": relative(paths["progress"]),
            }
        )
    rates = [row["cells_per_active_second"] for row in rows]
    etas = [
        row["remaining_cells"] / row["cells_per_active_second"]
        for row in rows
        if row["cells_per_active_second"] > 0
    ]
    result = {
        "created_at_utc": utc_now(),
        "status": "pass" if all(row["permanent_failed_cells"] == 0 for row in rows) else "fail",
        "workers": rows,
        "combined": {
            "processed_cells": sum(row["processed_cells"] for row in rows),
            "remaining_cells": sum(row["remaining_cells"] for row in rows),
            "cells_per_second": sum(rates),
            "parallel_wall_eta_seconds": max(etas) if len(etas) == 2 else None,
        },
    }
    atomic_json(CACHE_DIR / "extraction_status.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def worker_state(worker_id: int, *, require_finished: bool) -> dict[str, Any]:
    base = configure_base(require_contract=True)
    raw = worker_paths(worker_id)
    paths = base.worker_paths(worker_id)
    provenance, total = build_provenance(paths)
    progress = read_json(raw["progress"])
    committed = int(progress.get("committed_prefix_stop", 0))
    if progress.get("provenance_fingerprint") != base.canonical_hash(provenance):
        raise AssertionError(f"worker{worker_id} provenance changed")
    if int(progress.get("permanent_failed_cells", 0)):
        raise AssertionError(f"worker{worker_id} has failed cells")
    if require_finished and (progress.get("status") != "finished" or committed != total):
        raise RuntimeError(f"worker{worker_id} is not finished")
    if not require_finished and committed <= 0:
        raise RuntimeError(f"worker{worker_id} has no committed cells")
    array_path = raw["final"] if raw["final"].is_file() else raw["partial"]
    array = np.load(array_path, mmap_mode="r")
    if array.shape != (total, EMBEDDING_DIM) or array.dtype != np.float32:
        raise AssertionError(f"worker{worker_id} array contract changed")
    if require_finished:
        runtime = read_json(raw["runtime_manifest"])
        if runtime.get("status") != "pass" or runtime["output"]["sha256"] != sha256_file(array_path):
            raise AssertionError(f"worker{worker_id} final manifest/hash failed")
    return {
        "worker_id": worker_id,
        "paths": raw,
        "array": array,
        "total": total,
        "committed": committed,
        "provenance": provenance,
    }


def _plan_prefix_cells(base: Any, worker_id: int, stop: int) -> list[Any]:
    treated, controls = base.load_metadata_maps()
    output: list[Any] = []
    for batch in base.iter_raw_batches(
        worker_paths(worker_id)["plan"],
        start=0,
        stop=stop,
        batch_size=stop,
        treated=treated,
        controls=controls,
    ):
        output.extend(batch.cells)
    return output


def merge_rehearsal(cells_per_worker: int) -> dict[str, Any]:
    if REHEARSAL_AUDIT.exists():
        raise FileExistsError("Merge rehearsal already exists")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or torch.cuda.device_count() != 1:
        raise RuntimeError("Merge rehearsal requires CUDA_VISIBLE_DEVICES=0")
    base = configure_base(require_contract=True)
    batch_size = int(read_json(EXTRACTION_CONTRACT)["formal_batch_size"])
    cells_per_worker = max(cells_per_worker, batch_size)
    states = [worker_state(i, require_finished=False) for i in (0, 1)]
    selected: list[tuple[int, Any]] = []
    cached: list[np.ndarray] = []
    for state in states:
        if state["committed"] < cells_per_worker:
            raise RuntimeError(f"worker{state['worker_id']} needs {cells_per_worker} committed cells")
        cells = _plan_prefix_cells(base, state["worker_id"], batch_size)
        picks = np.linspace(0, batch_size - 1, min(8, batch_size), dtype=np.int64)
        for pick in picks:
            cell = cells[int(pick)]
            selected.append((state["worker_id"], cell))
            cached.append(np.asarray(state["array"][cell.part_index], dtype=np.float32))

    global_indices = np.asarray([cell.embedding_index for _, cell in selected], dtype=np.int64)
    if len(np.unique(global_indices)) != len(global_indices):
        raise AssertionError("Rehearsal selected duplicate global indices")
    order = np.argsort(global_indices)
    compact = np.stack(cached)[order]
    if compact.shape != (len(selected), EMBEDDING_DIM) or not np.isfinite(compact).all():
        raise AssertionError("Rehearsal compact scatter failed")

    treated, controls = base.load_metadata_maps()
    datamodule, _ = base.build_official_preprocessor(LOCAL_MANIFEST, GLOBAL_STATS)
    module = load_author_model(AUTHOR_CHECKPOINT, torch.device("cuda:0"))
    observed = np.empty_like(np.stack(cached))
    for worker_id in (0, 1):
        batches = list(
            base.iter_raw_batches(
                worker_paths(worker_id)["plan"],
                start=0,
                stop=batch_size,
                batch_size=batch_size,
                treated=treated,
                controls=controls,
            )
        )
        if len(batches) != 1:
            raise AssertionError("Could not reconstruct canonical extraction batch")
        audit = base.StreamingPreprocessingAudit()
        model_batch = base.preprocess_once(
            batches[0].raw_cells, datamodule, audit, inverse_check=False
        )
        output = base.infer_teacher(module, model_batch, torch.device("cuda:0"))
        for selected_index, (owner_worker, cell) in enumerate(selected):
            if owner_worker == worker_id:
                observed[selected_index] = output[cell.part_index]
    expected = np.stack(cached)
    max_abs = float(np.max(np.abs(observed.astype(np.float64) - expected.astype(np.float64))))
    allclose = bool(np.allclose(observed, expected, rtol=1e-5, atol=1e-4))
    if not allclose:
        raise AssertionError(f"Rehearsal direct re-inference mismatch: {max_abs}")
    payload = {
        "schema": "author_genejepa_epoch49_merge_rehearsal_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "scatter": {
            "rows": len(selected),
            "shape": list(compact.shape),
            "missing": 0,
            "duplicate": 0,
            "finite": True,
            "locator_embedding_index_cache_row_alignment": "pass",
        },
        "direct_reinference": {
            "cells": len(selected),
            "canonical_batch_size": batch_size,
            "allclose": allclose,
            "rtol": 1e-5,
            "atol": 1e-4,
            "max_abs_difference": max_abs,
        },
        "workers": [
            {"worker_id": state["worker_id"], "committed": state["committed"]}
            for state in states
        ],
    }
    atomic_json(REHEARSAL_AUDIT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def _stats_with_median(audit: Any, norms: np.ndarray) -> dict[str, Any]:
    result = audit.summary()
    result["norm"]["median"] = float(np.median(norms))
    return result


def merge_full(chunk_cells: int) -> dict[str, Any]:
    if any(path.exists() for path in (MERGED_EMBEDDINGS, MERGED_PARTIAL, FINAL_MANIFEST)):
        raise FileExistsError("Merged Author cache output exists; refusing overwrite")
    if read_json(REHEARSAL_AUDIT).get("status") != "pass":
        raise AssertionError("A passing merge rehearsal is required")
    base = configure_base(require_contract=True)
    states = [worker_state(i, require_finished=True) for i in (0, 1)]
    destination: np.memmap | None = np.lib.format.open_memmap(
        MERGED_PARTIAL, mode="w+", dtype=np.float32, shape=(EXPECTED_TOTAL, EMBEDDING_DIM)
    )
    written = np.zeros(EXPECTED_TOTAL, dtype=np.bool_)
    norms = np.empty(EXPECTED_TOTAL, dtype=np.float32)
    overall = base.StreamingEmbeddingAudit()
    sections = {name: base.StreamingEmbeddingAudit() for name in SECTION_BOUNDS}
    started = time.perf_counter()
    try:
        for state in states:
            plan = pq.ParquetFile(state["paths"]["plan"])
            for batch in plan.iter_batches(
                batch_size=chunk_cells, columns=["part_index", "embedding_index"]
            ):
                part = batch["part_index"].to_numpy(zero_copy_only=False).astype(np.int64)
                global_index = batch["embedding_index"].to_numpy(zero_copy_only=False).astype(
                    np.int64
                )
                values = np.asarray(state["array"][part], dtype=np.float32)
                if bool(written[global_index].any()) or not np.isfinite(values).all():
                    raise AssertionError("Duplicate or non-finite value during Author cache scatter")
                destination[global_index] = values
                written[global_index] = True
                norms[global_index] = np.linalg.norm(values.astype(np.float64), axis=1).astype(
                    np.float32
                )
                overall.add(values)
                for name, (start, stop) in SECTION_BOUNDS.items():
                    mask = (global_index >= start) & (global_index < stop)
                    if mask.any():
                        sections[name].add(values[mask])
        destination.flush()
        if not written.all() or int(written.sum()) != EXPECTED_TOTAL:
            raise AssertionError("Merged Author cache has missing rows")
        del destination
        destination = None
        os.replace(MERGED_PARTIAL, MERGED_EMBEDDINGS)
        section_statistics = {
            name: _stats_with_median(audit, norms[start:stop])
            for name, audit in sections.items()
            for start, stop in [SECTION_BOUNDS[name]]
        }
        result = {
            "schema": "author_genejepa_epoch49_hd100_cache_manifest_v1",
            "created_at_utc": utc_now(),
            "status": "pass",
            "checkpoint": {
                **artifact(AUTHOR_CHECKPOINT),
                "use_teacher": True,
                "branch": "teacher_encoder.ema_model",
                "frozen": True,
            },
            "author_code": {
                "path": relative(AUTHOR_CODE),
                "git_commit": EXPECTED_AUTHOR_COMMIT,
            },
            "preprocessing": {
                "normalization_stats_policy": NORMALIZATION_POLICY,
                "global_stats": artifact(GLOBAL_STATS),
                "gene_metadata": artifact(GENE_METADATA),
                "sentinel_handling_once": True,
                "gene_mapping_once": True,
                "log1p_once": True,
                "global_normalization_once": True,
                "CP10K": False,
                "postprocessing": [],
            },
            "plan": artifact(PLAN_SUMMARY),
            "extraction_contract": artifact(EXTRACTION_CONTRACT),
            "workers": [artifact(state["paths"]["runtime_manifest"]) for state in states],
            "output": {
                **artifact(MERGED_EMBEDDINGS),
                "shape": [EXPECTED_TOTAL, EMBEDDING_DIM],
                "dtype": "float32",
                "row_equals_global_embedding_index": True,
            },
            "sections": {
                name: {
                    "start": start,
                    "stop_exclusive": stop,
                    "cells": stop - start,
                    "statistics": section_statistics[name],
                }
                for name, (start, stop) in SECTION_BOUNDS.items()
            },
            "hard_checks": {
                "train_cells": TRAIN_CELLS,
                "val_cells": VAL_CELLS,
                "arc7_c_cells": ARC7_CELLS,
                "embedding_dim": EMBEDDING_DIM,
                "dtype": "float32",
                "all_finite": True,
                "missing": 0,
                "duplicate": 0,
                "worker_overlap": 0,
            },
            "embedding_statistics": _stats_with_median(overall, norms),
            "merge_rehearsal": artifact(REHEARSAL_AUDIT),
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
    commands.add_parser("plan", help="Derive the exact new cache plan from frozen plans")
    probe = commands.add_parser("probe-batch", help="Try batch=64, falling back only on OOM")
    probe.add_argument("--cells", type=int, default=256)
    commands.add_parser("parity", help="Run the fixed 8-cell production-path parity check")
    run = commands.add_parser("run-worker", help="Start or resume one independent worker")
    run.add_argument("--worker-id", type=int, choices=(0, 1), required=True)
    run.add_argument("--batch-size", type=int)
    run.add_argument("--commit-every-batches", type=int, default=10)
    run.add_argument("--max-new-cells", type=int, default=0)
    run.add_argument("--gpu-sample-interval", type=float, default=2.0)
    run.add_argument("--skip-resume-finite-scan", action="store_true")
    commands.add_parser("status")
    rehearsal = commands.add_parser("merge-rehearsal")
    rehearsal.add_argument("--cells-per-worker", type=int, default=64)
    merge = commands.add_parser("merge")
    merge.add_argument("--chunk-cells", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "plan":
        build_plan()
    elif args.command == "probe-batch":
        if not 64 <= args.cells <= 4096:
            raise ValueError("--cells must be in [64,4096]")
        batch_probe(args.cells)
    elif args.command == "parity":
        parity()
    elif args.command == "run-worker":
        if args.commit_every_batches < 1 or args.max_new_cells < 0:
            raise ValueError("commit interval must be positive and max-new-cells non-negative")
        run_worker(args)
    elif args.command == "status":
        status()
    elif args.command == "merge-rehearsal":
        merge_rehearsal(args.cells_per_worker)
    elif args.command == "merge":
        if args.chunk_cells < 1:
            raise ValueError("--chunk-cells must be positive")
        merge_full(args.chunk_cells)


if __name__ == "__main__":
    main()
