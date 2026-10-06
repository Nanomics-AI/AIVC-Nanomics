#!/usr/bin/env python3
"""Build and consume the frozen Phase-III Top20 expression cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from phase2_stav2_dataset import Phase2STAv2Dataset
from tahoe_decoder_v1_data import (
    GENE_METADATA,
    RAW_COLUMNS,
    build_panel_lookup,
    load_gene_universe,
    log1p_cp10k_target,
)
from tahoe_experiment1_latent_data import make_dataloader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
TASK_PATH = PROJECT_ROOT.parent / "phase_III_FINAL_FROZEN.md"
TASK_SHA256 = "d38e07a653f2f7e2403d677b5ff4b90eb9e036bd2bc5e9e8d5a2abce58019bcc"
CONDITIONS_PATH = RESULTS / "phase2_stav2_conditions.csv"
CONTROLS_PATH = RESULTS / "phase2_stav2_control_pools.csv"
SUBSET_MANIFEST = RESULTS / "phase2_stav2_subset_manifest.json"
CACHE_PLAN = RESULTS / "phase2_stav2_cache_plan.json"
EMBEDDINGS_PATH = RESULTS / "phase2_author_genejepa_embeddings.npy"
EMBEDDING_MANIFEST = RESULTS / "phase2_author_genejepa_embedding_manifest.json"
PANEL_PATH = RESULTS / "phase1_top20_gene_panel.csv"
PANEL_MANIFEST = RESULTS / "phase1_top20_gene_panel.json"
PLAN_PATHS = tuple(
    RESULTS / f"phase2_author_genejepa_cache/cache_worker{worker}_plan.parquet"
    for worker in range(2)
)
EXPRESSION_CACHE = RESULTS / "phase3_top20_expression_cache.npy"
EXPRESSION_PARTIAL = RESULTS / "phase3_top20_expression_cache.npy.partial"
EXPRESSION_STATE = RESULTS / "phase3_top20_expression_cache_builder_state.npz"
EXPRESSION_PROGRESS = RESULTS / "phase3_top20_expression_cache_builder_progress.json"
EXPRESSION_MANIFEST = RESULTS / "phase3_top20_expression_cache_manifest.json"
SAMPLING_AUDIT = RESULTS / "phase3_set_decoder_sampling_fairness_audit.json"

SEED = 42
SET_SIZE = 256
LATENT_DIM = 768
GENE_DIM = 20
TOTAL_CELLS = 2_547_484
OWNER_TREATED = 0
OWNER_CONTROL = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wait_for_file(path: Path, timeout_seconds: float = 30.0) -> None:
    """Wait for a just-renamed file to become visible through WSL DrvFS."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            if path.is_file() and path.stat().st_size > 0:
                return
        except FileNotFoundError:
            pass
        if time.monotonic() >= deadline:
            raise FileNotFoundError(f"Renamed file did not become visible: {path}")
        time.sleep(0.25)


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def artifact(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value: dict[str, Any] = {"path": relative(path), "size_bytes": path.stat().st_size}
    if hash_file:
        value["sha256"] = sha256_file(path)
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_state(completed: np.ndarray, counters: dict[str, int]) -> None:
    temporary = EXPRESSION_STATE.with_name(EXPRESSION_STATE.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            completed=completed,
            **{name: np.asarray(value, dtype=np.int64) for name, value in counters.items()},
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, EXPRESSION_STATE)


def _load_panel() -> tuple[pd.DataFrame, np.ndarray, dict[int, int], np.ndarray]:
    manifest = json.loads(PANEL_MANIFEST.read_text(encoding="utf-8"))
    panel = pd.read_csv(PANEL_PATH, encoding="utf-8-sig", keep_default_na=False).sort_values(
        "panel_rank", kind="stable"
    )
    if (
        manifest.get("status") != "frozen"
        or manifest["output"]["sha256"] != sha256_file(PANEL_PATH)
        or len(panel) != GENE_DIM
        or not np.array_equal(panel["panel_rank"].to_numpy(np.int64), np.arange(GENE_DIM))
    ):
        raise AssertionError("Frozen Phase-I Top20 panel changed")
    genes, token_lookup = load_gene_universe(GENE_METADATA)
    panel_indices = panel["genejepa_index"].to_numpy(np.int64)
    panel_lookup = build_panel_lookup(panel_indices, vocabulary_size=len(genes))
    return panel, panel_indices, token_lookup, panel_lookup


def _owner_metadata() -> dict[tuple[int, int], tuple[str, str, str, frozenset[str]]]:
    source_conditions = pd.read_csv(
        RESULTS / "tahoe_experiment1_cache_cap512_all_dmso_condition_index.csv",
        encoding="utf-8-sig",
        keep_default_na=False,
        low_memory=False,
    )
    source_controls = pd.read_csv(
        RESULTS / "tahoe_experiment1_cache_cap512_all_dmso_control_pool_index.csv",
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    result: dict[tuple[int, int], tuple[str, str, str, frozenset[str]]] = {}
    for row in source_conditions.itertuples(index=False):
        result[(OWNER_TREATED, int(row.cache_condition_index))] = (
            str(row.plate),
            str(row.cell_line_id),
            str(row.drug),
            frozenset(str(row.treated_samples).split("|")),
        )
    for row in source_controls.itertuples(index=False):
        result[(OWNER_CONTROL, int(row.control_pool_index))] = (
            str(row.plate),
            str(row.cell_line_id),
            str(row.control_drug),
            frozenset(str(row.control_samples).split("|")),
        )
    return result


def audit_locator_plans() -> dict[str, Any]:
    plan = json.loads(CACHE_PLAN.read_text(encoding="utf-8"))
    if plan.get("status") != "pass" or int(plan["total_unique_cells"]) != TOTAL_CELLS:
        raise AssertionError("Frozen Phase-II cache plan changed")
    seen = np.zeros(TOTAL_CELLS, dtype=np.bool_)
    rows_by_worker: list[int] = []
    owner_counts = {"treated": 0, "control": 0}
    for worker_id, path in enumerate(PLAN_PATHS):
        declared = plan["worker_partition"][worker_id]
        if declared["plan_sha256"] != sha256_file(path):
            raise AssertionError(f"Phase-II worker{worker_id} plan SHA-256 changed")
        parquet = pq.ParquetFile(path)
        worker_rows = 0
        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group, columns=["embedding_index", "owner_type"])
            indices = table["embedding_index"].to_numpy(zero_copy_only=False).astype(np.int64)
            owners = table["owner_type"].to_numpy(zero_copy_only=False).astype(np.int8)
            if (
                np.any((indices < 0) | (indices >= TOTAL_CELLS))
                or len(np.unique(indices)) != len(indices)
                or seen[indices].any()
            ):
                raise AssertionError("Phase-II locator plan has out-of-range or duplicate rows")
            seen[indices] = True
            worker_rows += len(indices)
            owner_counts["treated"] += int(np.count_nonzero(owners == OWNER_TREATED))
            owner_counts["control"] += int(np.count_nonzero(owners == OWNER_CONTROL))
            if np.any((owners != OWNER_TREATED) & (owners != OWNER_CONTROL)):
                raise AssertionError("Unknown owner_type in Phase-II locator plan")
        if worker_rows != int(declared["rows"]):
            raise AssertionError(f"Phase-II worker{worker_id} plan row count changed")
        rows_by_worker.append(worker_rows)
    if not seen.all() or sum(rows_by_worker) != TOTAL_CELLS:
        raise AssertionError("Phase-II locator plan union is incomplete")
    if owner_counts != {
        "treated": int(plan["treated_cells"]),
        "control": int(plan["control_cells"]),
    }:
        raise AssertionError("Phase-II locator owner counts changed")
    return {
        "status": "pass",
        "worker_rows": rows_by_worker,
        "union": int(seen.sum()),
        "missing": int((~seen).sum()),
        "duplicate": 0,
        "owner_counts": owner_counts,
        "row_equals_phase2_cell_index": True,
    }


def _load_or_create_state() -> tuple[np.memmap, np.ndarray, dict[str, int]]:
    counter_names = (
        "sentinel_cells",
        "unmapped_entries",
        "noninteger_count_entries",
        "duplicate_gene_entries_collapsed",
        "invalid_library_cells",
        "metadata_verified_cells",
    )
    if EXPRESSION_CACHE.exists() and EXPRESSION_PARTIAL.exists():
        raise RuntimeError("Both final and partial Phase-III expression caches exist")
    if not EXPRESSION_CACHE.exists() and EXPRESSION_PARTIAL.exists() != EXPRESSION_STATE.exists():
        raise RuntimeError("Expression partial/state must either both exist or both be absent")
    if EXPRESSION_CACHE.exists():
        if not EXPRESSION_STATE.exists():
            raise RuntimeError("Final expression cache exists without completed-state provenance")
        destination = np.load(EXPRESSION_CACHE, mmap_mode="r")
        with np.load(EXPRESSION_STATE, allow_pickle=False) as state:
            completed = state["completed"].astype(np.bool_, copy=True)
            counters = {name: int(state[name]) for name in counter_names}
        if not completed.all():
            raise RuntimeError("Final expression cache exists but completed state is incomplete")
    elif EXPRESSION_PARTIAL.exists():
        destination = np.load(EXPRESSION_PARTIAL, mmap_mode="r+")
        with np.load(EXPRESSION_STATE, allow_pickle=False) as state:
            completed = state["completed"].astype(np.bool_, copy=True)
            counters = {name: int(state[name]) for name in counter_names}
    else:
        destination = np.lib.format.open_memmap(
            EXPRESSION_PARTIAL, mode="w+", dtype=np.float32, shape=(TOTAL_CELLS, GENE_DIM)
        )
        completed = np.zeros(TOTAL_CELLS, dtype=np.bool_)
        counters = {name: 0 for name in counter_names}
        destination.flush()
        _atomic_state(completed, counters)
    if (
        not isinstance(destination, np.memmap)
        or destination.shape != (TOTAL_CELLS, GENE_DIM)
        or destination.dtype != np.float32
        or completed.shape != (TOTAL_CELLS,)
    ):
        raise AssertionError("Expression cache resume state shape/dtype changed")
    if counters["metadata_verified_cells"] != int(completed.sum()):
        raise AssertionError("Resume counters disagree with completed bitmap")
    return destination, completed, counters


def _progress_payload(
    completed: np.ndarray,
    counters: dict[str, int],
    started: float,
    current: dict[str, Any],
) -> dict[str, Any]:
    elapsed = max(time.perf_counter() - started, 1e-9)
    session_processed = int(completed.sum()) - int(current["initial_completed"])
    rate = session_processed / elapsed
    remaining = TOTAL_CELLS - int(completed.sum())
    return {
        "schema": "phase3_top20_expression_cache_progress_v1",
        "updated_at_utc": utc_now(),
        "status": "running" if remaining else "complete_pending_manifest",
        "processed_cells": int(completed.sum()),
        "total_cells": TOTAL_CELLS,
        "remaining_cells": remaining,
        "session_cells_per_second": rate,
        "estimated_remaining_seconds": remaining / rate if rate > 0 else None,
        "current": {key: value for key, value in current.items() if key != "initial_completed"},
        "audit": counters,
        "partial": relative(EXPRESSION_PARTIAL),
        "state": relative(EXPRESSION_STATE),
    }


def build_expression_cache(commit_every_plan_groups: int = 20) -> dict[str, Any]:
    if commit_every_plan_groups < 1:
        raise ValueError("--commit-every-plan-groups must be positive")
    if EXPRESSION_MANIFEST.exists():
        existing = json.loads(EXPRESSION_MANIFEST.read_text(encoding="utf-8"))
        if existing.get("status") != "pass":
            raise AssertionError("Existing Phase-III expression manifest is not PASS")
        print(json.dumps(existing, ensure_ascii=False, indent=2), flush=True)
        return existing
    if sha256_file(TASK_PATH) != TASK_SHA256:
        raise AssertionError("Frozen Phase-III task SHA-256 changed")
    locator_audit = audit_locator_plans()
    panel, panel_indices, token_lookup, panel_lookup = _load_panel()
    owners = _owner_metadata()
    destination, completed, counters = _load_or_create_state()
    initial_completed = int(completed.sum())
    started = time.perf_counter()
    current: dict[str, Any] = {
        "worker_id": 0,
        "plan_row_group": 0,
        "shard_path": "",
        "raw_row_group": 0,
        "initial_completed": initial_completed,
    }
    processed_plan_groups = 0
    for worker_id, plan_path in enumerate(PLAN_PATHS):
        plan = pq.ParquetFile(plan_path)
        for plan_row_group in range(plan.num_row_groups):
            table = plan.read_row_group(plan_row_group)
            frame = table.to_pandas()
            indices = frame["embedding_index"].to_numpy(np.int64)
            pending = frame.loc[~completed[indices]].copy()
            if pending.empty:
                continue
            shard_values = pending["shard_path"].astype(str).unique()
            if len(shard_values) != 1:
                raise AssertionError("A Phase-II plan row group must map to one source shard")
            shard_path = str(shard_values[0])
            source = pq.ParquetFile(PROJECT_ROOT / shard_path)
            current.update(
                worker_id=worker_id,
                plan_row_group=plan_row_group,
                shard_path=shard_path,
            )
            for raw_row_group, group in pending.groupby("row_group_index", sort=True):
                raw_row_group = int(raw_row_group)
                current["raw_row_group"] = raw_row_group
                raw = source.read_row_group(raw_row_group, columns=RAW_COLUMNS)
                for locator in group.itertuples(index=False):
                    output_row = int(locator.embedding_index)
                    row = int(locator.row_index_in_row_group)
                    expected = owners.get((int(locator.owner_type), int(locator.owner_index)))
                    if expected is None:
                        raise AssertionError("Locator owner is absent from frozen Experiment-1 maps")
                    plate, cell_line, drug, samples = expected
                    if (
                        str(raw["plate"][row].as_py()) != plate
                        or str(raw["cell_line_id"][row].as_py()) != cell_line
                        or str(raw["drug"][row].as_py()) != drug
                        or str(raw["sample"][row].as_py()) not in samples
                    ):
                        raise AssertionError("Raw Tahoe metadata disagrees with stable locator ownership")
                    target, audit = log1p_cp10k_target(
                        raw["genes"][row].as_py(),
                        raw["expressions"][row].as_py(),
                        token_lookup,
                        panel_indices,
                        panel_lookup=panel_lookup,
                    )
                    if completed[output_row]:
                        raise AssertionError("Expression cache attempted to write a row twice")
                    destination[output_row] = target
                    completed[output_row] = True
                    counters["sentinel_cells"] += int(bool(audit["sentinel_removed"]))
                    counters["unmapped_entries"] += int(audit["unmapped_entries"])
                    counters["noninteger_count_entries"] += int(audit["noninteger_count_entries"])
                    counters["duplicate_gene_entries_collapsed"] += int(
                        audit["duplicate_gene_entries_collapsed"]
                    )
                    counters["invalid_library_cells"] += int(audit["library_size"] <= 0)
                    counters["metadata_verified_cells"] += 1
            del source
            processed_plan_groups += 1
            if processed_plan_groups % commit_every_plan_groups == 0:
                destination.flush()
                _atomic_state(completed, counters)
                atomic_json(
                    EXPRESSION_PROGRESS,
                    _progress_payload(completed, counters, started, current),
                )
                print(
                    f"phase3 expression cells={int(completed.sum()):,}/{TOTAL_CELLS:,} "
                    f"worker={worker_id} plan_group={plan_row_group}/{plan.num_row_groups}",
                    flush=True,
                )
    destination.flush()
    _atomic_state(completed, counters)
    atomic_json(EXPRESSION_PROGRESS, _progress_payload(completed, counters, started, current))
    if not completed.all() or counters["metadata_verified_cells"] != TOTAL_CELLS:
        raise AssertionError("Phase-III expression cache is incomplete")
    for start in range(0, TOTAL_CELLS, 100_000):
        block = np.asarray(destination[start : start + 100_000])
        if not np.isfinite(block).all() or np.any(block < 0):
            raise AssertionError("Phase-III expression cache is non-finite or negative")
    del destination
    if EXPRESSION_PARTIAL.exists():
        os.replace(EXPRESSION_PARTIAL, EXPRESSION_CACHE)
        wait_for_file(EXPRESSION_CACHE)
    cache = np.load(EXPRESSION_CACHE, mmap_mode="r")
    spot_indices = np.unique(np.linspace(0, TOTAL_CELLS - 1, 32, dtype=np.int64))
    spot_max_abs_error = 0.0
    spot_found: dict[int, tuple[str, int, int]] = {}
    for plan_path in PLAN_PATHS:
        plan = pq.ParquetFile(plan_path)
        for row_group in range(plan.num_row_groups):
            table = plan.read_row_group(
                row_group,
                columns=["embedding_index", "shard_path", "row_group_index", "row_index_in_row_group"],
            ).to_pandas()
            selected = table.loc[table["embedding_index"].isin(spot_indices)]
            for row in selected.itertuples(index=False):
                spot_found[int(row.embedding_index)] = (
                    str(row.shard_path), int(row.row_group_index), int(row.row_index_in_row_group)
                )
    if set(spot_found) != set(spot_indices.tolist()):
        raise AssertionError("Could not recover all expression-cache spotcheck locators")
    for output_row, (shard_path, raw_group, row) in spot_found.items():
        source = pq.ParquetFile(PROJECT_ROOT / shard_path)
        raw = source.read_row_group(raw_group, columns=["genes", "expressions"])
        expected, _ = log1p_cp10k_target(
            raw["genes"][row].as_py(), raw["expressions"][row].as_py(), token_lookup,
            panel_indices, panel_lookup=panel_lookup,
        )
        spot_max_abs_error = max(
            spot_max_abs_error, float(np.max(np.abs(np.asarray(cache[output_row]) - expected)))
        )
    if spot_max_abs_error != 0.0:
        raise AssertionError("Expression cache locator/row spotcheck was not exact")
    result = {
        "schema": "phase3_top20_expression_cache_manifest_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "task": {**artifact(TASK_PATH), "expected_sha256": TASK_SHA256},
        "output": {
            **artifact(EXPRESSION_CACHE),
            "shape": [TOTAL_CELLS, GENE_DIM],
            "dtype": "float32",
            "row_equals_phase2_cell_index": True,
            "physical_cell_identity_matches_author_latent_row": True,
        },
        "author_latent": {
            **artifact(EMBEDDINGS_PATH, hash_file=False),
            "sha256_from_pass_manifest": json.loads(
                EMBEDDING_MANIFEST.read_text(encoding="utf-8")
            )["output"]["sha256"],
            "sha256_recomputed": False,
        },
        "author_embedding_manifest": artifact(EMBEDDING_MANIFEST),
        "phase2_subset_manifest": artifact(SUBSET_MANIFEST),
        "phase2_cache_plan": artifact(CACHE_PLAN),
        "locator_plans": [artifact(path) for path in PLAN_PATHS],
        "panel": artifact(PANEL_PATH),
        "panel_manifest": artifact(PANEL_MANIFEST),
        "gene_metadata": artifact(GENE_METADATA),
        "target_contract": (
            "raw counts -> sentinel removal -> map all usable genes -> full mapped-gene "
            "library size -> CP10000 -> log1p -> frozen Phase-I Top20 selection"
        ),
        "integrity": {
            "shape_exact": list(cache.shape) == [TOTAL_CELLS, GENE_DIM],
            "dtype_float32": cache.dtype == np.float32,
            "finite": True,
            "nonnegative": True,
            "physical_cell_missing": locator_audit["missing"],
            "physical_cell_duplicate": locator_audit["duplicate"],
            "row_alignment_exact": True,
            "panel_exact": True,
            "unmapped_required_top20_genes": 0,
            "locator_spotcheck_cells": len(spot_indices),
            "locator_spotcheck_max_abs_error": spot_max_abs_error,
        },
        "preprocessing_audit": counters,
        "resume_state": artifact(EXPRESSION_STATE),
        "elapsed_seconds_this_session": time.perf_counter() - started,
        "initial_completed_cells": initial_completed,
        "newly_completed_cells": TOTAL_CELLS - initial_completed,
        "phase3_complete": False,
        "phase4_started": False,
    }
    atomic_json(EXPRESSION_MANIFEST, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


class Phase3SetDecoderDataset(Phase2STAv2Dataset):
    """Same frozen Phase-II memberships, canonically ordered, with aligned Top20 targets."""

    def __init__(self, *, split: str | None, seed: int = SEED, epoch: int = 0) -> None:
        super().__init__(split=split, seed=seed, epoch=epoch)
        manifest = json.loads(EXPRESSION_MANIFEST.read_text(encoding="utf-8"))
        output = manifest.get("output", {})
        integrity = manifest.get("integrity", {})
        if manifest.get("status") != "pass" or not all(
            integrity.get(name) is True
            for name in (
                "shape_exact", "dtype_float32", "finite", "nonnegative",
                "row_alignment_exact", "panel_exact",
            )
        ):
            raise AssertionError("Phase-III expression cache manifest is not PASS")
        if integrity.get("unmapped_required_top20_genes") != 0:
            raise AssertionError("Frozen Top20 panel is not fully mapped")
        if integrity.get("physical_cell_missing") != 0 or integrity.get("physical_cell_duplicate") != 0:
            raise AssertionError("Phase-III expression cache locator integrity failed")
        if output.get("row_equals_phase2_cell_index") is not True:
            raise AssertionError("Expression cache row/index contract is absent")
        if EXPRESSION_CACHE.stat().st_size != int(output["size_bytes"]):
            raise AssertionError("Phase-III expression cache size changed")
        self.expression = np.load(EXPRESSION_CACHE, mmap_mode="r")
        if (
            not isinstance(self.expression, np.memmap)
            or self.expression.shape != self.embeddings.shape[:1] + (GENE_DIM,)
            or self.expression.dtype != np.float32
        ):
            raise AssertionError("Phase-III expression cache is not aligned [N,20] float32 mmap")
        self.expression_transforms: tuple[str, ...] = ()
        self.provenance.update(
            expression_row_equals_phase2_cell_index=True,
            canonical_order="stable ascending phase2_cell_index after membership selection",
            expression_transforms=[],
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.conditions.iloc[index]
        source_unsorted = self._sample_range(
            int(row["control_embedding_start"]),
            int(row["control_embedding_stop_exclusive"]),
            str(row["pair_id"]),
            "control",
        )
        target_unsorted = self._sample_range(
            int(row["treated_embedding_start"]),
            int(row["treated_embedding_stop_exclusive"]),
            str(row["pair_id"]),
            "treated",
        )
        source = np.sort(source_unsorted, kind="stable")
        target = np.sort(target_unsorted, kind="stable")
        if len(np.unique(source)) != SET_SIZE or len(np.unique(target)) != SET_SIZE:
            raise AssertionError("Phase-III set membership is not unique")
        control_latent = np.ascontiguousarray(self.embeddings[source], dtype=np.float32)
        treated_latent = np.ascontiguousarray(self.embeddings[target], dtype=np.float32)
        control_expression = np.ascontiguousarray(self.expression[source], dtype=np.float32)
        treated_expression = np.ascontiguousarray(self.expression[target], dtype=np.float32)
        gt_delta = treated_expression.mean(axis=0) - control_expression.mean(axis=0)
        dose_uM = float(row["dose_uM"])
        dose_scaled = math.log1p(dose_uM) / self.dose_denominator
        if not 0 < dose_scaled <= 1:
            raise AssertionError("Positive Phase-II dose_scaled is outside (0,1]")
        return {
            "ctrl_cell_emb": torch.from_numpy(control_latent),
            "pert_cell_emb": torch.from_numpy(treated_latent),
            "ctrl_top20_expr": torch.from_numpy(control_expression),
            "pert_top20_expr": torch.from_numpy(treated_expression),
            "gt_delta": torch.from_numpy(np.asarray(gt_delta, dtype=np.float32)),
            "drug_id": torch.tensor(int(row["drug_id"]), dtype=torch.long),
            "dose_scaled": torch.tensor([dose_scaled], dtype=torch.float32),
            "source_embedding_index": torch.from_numpy(source.copy()),
            "target_embedding_index": torch.from_numpy(target.copy()),
            "condition_id": str(row["pair_id"]),
            "phase2_condition_index": int(row["phase2_condition_index"]),
            "edge_id": str(row["edge_id"]),
            "control_pool_id": str(row["control_pool_id"]),
            "split": str(row["split"]),
            "plate": str(row["plate"]),
            "cell_line_id": str(row["cell_line_id"]),
            "drug": str(row["drug"]),
            "dose_uM": dose_uM,
        }


def _membership_digest(
    left: Phase3SetDecoderDataset,
    right: Phase3SetDecoderDataset,
    epoch: int,
) -> dict[str, Any]:
    left.set_epoch(epoch)
    right.set_epoch(epoch)
    digests = {name: hashlib.sha256() for name in ("d1_control", "d1_treated", "d2_control", "d2_treated")}
    changed_from_unsorted = 0
    for row_index, row in left.conditions.iterrows():
        pair_id = str(row["pair_id"])
        arrays: dict[str, np.ndarray] = {}
        for label, dataset in (("d1", left), ("d2", right)):
            for side, start_column, stop_column in (
                ("control", "control_embedding_start", "control_embedding_stop_exclusive"),
                ("treated", "treated_embedding_start", "treated_embedding_stop_exclusive"),
            ):
                raw = dataset._sample_range(
                    int(row[start_column]), int(row[stop_column]), pair_id, side
                )
                ordered = np.sort(raw, kind="stable")
                if len(np.unique(ordered)) != SET_SIZE or np.any(ordered[1:] < ordered[:-1]):
                    raise AssertionError("Sampling uniqueness/canonical ordering failed")
                if not np.array_equal(np.sort(raw), ordered):
                    raise AssertionError("Canonical sorting changed membership")
                changed_from_unsorted += int(not np.array_equal(raw, ordered) and label == "d1")
                arrays[f"{label}_{side}"] = ordered
                digests[f"{label}_{side}"].update(pair_id.encode("utf-8"))
                digests[f"{label}_{side}"].update(ordered.astype("<i8", copy=False).tobytes())
        if not np.array_equal(arrays["d1_control"], arrays["d2_control"]) or not np.array_equal(
            arrays["d1_treated"], arrays["d2_treated"]
        ):
            raise AssertionError(f"D1/D2 membership differs for {pair_id}")
    return {
        "split": left.split,
        "epoch": epoch,
        "conditions": len(left),
        "hashes": {name: digest.hexdigest() for name, digest in digests.items()},
        "d1_d2_control_equal": digests["d1_control"].digest() == digests["d2_control"].digest(),
        "d1_d2_treated_equal": digests["d1_treated"].digest() == digests["d2_treated"].digest(),
        "canonical_sort_changed_row_order_conditions": changed_from_unsorted,
        "membership_preserved_by_sort": True,
        "within_set_unique": True,
        "canonical_ascending": True,
    }


def write_sampling_fairness_audit() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for split, epochs in (("train", (0, 1)), ("val", (0,)), ("test", (0, 1, 2, 3, 4))):
        left = Phase3SetDecoderDataset(split=split, seed=SEED, epoch=0)
        right = Phase3SetDecoderDataset(split=split, seed=SEED, epoch=0)
        for epoch in epochs:
            records.append(_membership_digest(left, right, epoch))
    train0, train1 = records[0], records[1]
    checks = {
        "same_epoch_reproducible": all(
            row["d1_d2_control_equal"] and row["d1_d2_treated_equal"] for row in records
        ),
        "dynamic_train_membership": train0["hashes"]["d1_control"] != train1["hashes"]["d1_control"]
        and train0["hashes"]["d1_treated"] != train1["hashes"]["d1_treated"],
        "within_set_unique_256": all(row["within_set_unique"] for row in records),
        "membership_unchanged_by_sorting": all(row["membership_preserved_by_sort"] for row in records),
        "canonical_phase2_cell_index_ascending": all(row["canonical_ascending"] for row in records),
        "d1_d2_exact_same_ordered_indices": all(
            row["hashes"]["d1_control"] == row["hashes"]["d2_control"]
            and row["hashes"]["d1_treated"] == row["hashes"]["d2_treated"]
            for row in records
        ),
    }
    result = {
        "schema": "phase3_set_decoder_sampling_fairness_audit_v1",
        "created_at_utc": utc_now(),
        "status": "pass" if all(checks.values()) else "fail",
        "sampling_rule": "Phase-II seed+epoch+pair_id+side SHA256/PCG64 without replacement",
        "post_membership_ordering": "stable sort phase2_cell_index ascending",
        "set_size": SET_SIZE,
        "seed": SEED,
        "checks": checks,
        "records": records,
        "full_pipeline_synthetic_treated_order": "inherits canonical control token order; no synthetic index or resort",
        "phase3_complete": False,
        "phase4_started": False,
    }
    if result["status"] != "pass":
        raise AssertionError(result)
    atomic_json(SAMPLING_AUDIT, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-expression-cache")
    build.add_argument("--commit-every-plan-groups", type=int, default=20)
    commands.add_parser("audit-sampling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build-expression-cache":
        build_expression_cache(args.commit_every_plan_groups)
    else:
        write_sampling_fairness_audit()


if __name__ == "__main__":
    main()


__all__ = [
    "EXPRESSION_CACHE",
    "EXPRESSION_MANIFEST",
    "GENE_DIM",
    "LATENT_DIM",
    "Phase3SetDecoderDataset",
    "SAMPLING_AUDIT",
    "SET_SIZE",
    "artifact",
    "atomic_json",
    "make_dataloader",
    "sha256_file",
    "write_sampling_fairness_audit",
]
