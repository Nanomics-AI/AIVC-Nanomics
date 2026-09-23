#!/usr/bin/env python3
"""Audit Decoder v1 target-space sparsity on the frozen train-treated cells.

Mean and population variance are reconstructed from the completed Decoder v1
panel-builder state.  The only full Tahoe scan performed here counts, for each
GeneJEPA gene, the number of train-treated cells with a positive mapped count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import tahoe_decoder_v1_data as decoder_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"

EXPECTED_CELLS = 22_941_936
EXPECTED_GENES = 62_710
EXPECTED_PANEL_GENES = 5_000
THRESHOLDS = (0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95)
TOP_NS = (500, 1_000, 2_000, 3_000, 5_000)
ALGORITHM_VERSION = "genejepa_decoder_detection_count_v1"

PANEL = RESULTS / "genejepa_decoder_v1_gene_panel.csv"
PANEL_SUMMARY = RESULTS / "genejepa_decoder_v1_gene_panel.json"
PANEL_STATE = RESULTS / "genejepa_decoder_v1_gene_panel_builder_state.npz"
DESCRIPTOR_COUNTS = RESULTS / "genejepa_decoder_v1_descriptor_counts.npz"

DETECTION_STATE = RESULTS / "genejepa_decoder_gene_detection_scan_state.npz"
DETECTION_PROGRESS = RESULTS / "genejepa_decoder_gene_detection_scan_progress.json"
SMOKE_RESULT = RESULTS / "genejepa_decoder_gene_sparsity_smoke.json"

STATS_CSV = RESULTS / "genejepa_decoder_gene_sparsity_train_stats.csv"
THRESHOLDS_CSV = RESULTS / "genejepa_decoder_gene_detection_thresholds.csv"
TOPN_CSV = RESULTS / "genejepa_decoder_gene_topn_summary.csv"
HIGH_EXPRESSION_CSV = RESULTS / "genejepa_decoder_high_expression_top5000.csv"
HIGH_DETECTION_CSV = RESULTS / "genejepa_decoder_high_detection_top5000.csv"
OVERLAP_CSV = RESULTS / "genejepa_decoder_gene_panel_overlap_summary.csv"
DETECTION_PNG = RESULTS / "genejepa_decoder_gene_detection_distribution.png"
EXPRESSION_DETECTION_PNG = RESULTS / "genejepa_decoder_gene_expression_vs_detection.png"
SUMMARY_JSON = RESULTS / "genejepa_decoder_gene_sparsity_audit.json"

AUDIT_LABELS = (
    "sentinel_cells",
    "unmapped_entries",
    "noninteger_count_entries",
    "duplicate_gene_entries_collapsed",
    "invalid_or_zero_library_cells",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def display_path(path: Path) -> str:
    return decoder_data.display_path(path)


def sha256_file(path: Path) -> str:
    return decoder_data.sha256_file(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    decoder_data.atomic_write_json(path, payload)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", lineterminator="\n")
    os.replace(temporary, path)


def atomic_save_figure(figure: Any, path: Path) -> None:
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    figure.savefig(temporary, dpi=160, bbox_inches="tight")
    os.replace(temporary, path)


def audit_dict(values: np.ndarray) -> dict[str, int]:
    return {name: int(value) for name, value in zip(AUDIT_LABELS, values, strict=True)}


def load_descriptor_counts(path: Path) -> dict[str, np.ndarray]:
    required = {
        "plan_index",
        "plan_row_group",
        "train_cells",
        "val_cells",
        "test_cells",
        "dmso_cells",
        "total_cells",
        "fingerprint",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise AssertionError(f"Unexpected descriptor-count keys: {archive.files}")
        result = {key: archive[key].copy() for key in archive.files}
    return result


def load_panel_builder_state(path: Path) -> dict[str, Any]:
    required = {
        "fingerprint",
        "next_descriptor",
        "cells",
        "value_sum",
        "value_sumsq",
        "audit_counts",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise AssertionError(f"Unexpected panel-builder state keys: {archive.files}")
        result = {
            "fingerprint": str(archive["fingerprint"].item()),
            "next_descriptor": int(archive["next_descriptor"]),
            "cells": int(archive["cells"]),
            "value_sum": archive["value_sum"].astype(np.float64, copy=True),
            "value_sumsq": archive["value_sumsq"].astype(np.float64, copy=True),
            "audit_counts": archive["audit_counts"].astype(np.int64, copy=True),
        }
    if result["cells"] != EXPECTED_CELLS:
        raise AssertionError("Panel-builder state does not contain the frozen cell count")
    if result["value_sum"].shape != (EXPECTED_GENES,):
        raise AssertionError("Panel-builder value_sum has the wrong shape")
    if result["value_sumsq"].shape != (EXPECTED_GENES,):
        raise AssertionError("Panel-builder value_sumsq has the wrong shape")
    if result["audit_counts"].shape != (len(AUDIT_LABELS),):
        raise AssertionError("Panel-builder audit_counts has the wrong shape")
    return result


def load_contract(
    *,
    plans: tuple[Path, Path],
    condition_index: Path,
    gene_metadata: Path,
    descriptor_counts_path: Path,
) -> dict[str, Any]:
    genes, token_lookup = decoder_data.load_gene_universe(gene_metadata)
    if len(genes) != EXPECTED_GENES:
        raise AssertionError(f"Expected {EXPECTED_GENES} genes, observed {len(genes)}")
    if not np.array_equal(genes["genejepa_index"].to_numpy(), np.arange(EXPECTED_GENES)):
        raise AssertionError("GeneJEPA indices are not contiguous")

    conditions = decoder_data.load_conditions(condition_index)
    expected_cells = int(
        conditions.loc[conditions["split"].eq("train"), "treated_cached_cell_count"].sum()
    )
    if expected_cells != EXPECTED_CELLS:
        raise AssertionError(
            f"Expected {EXPECTED_CELLS} train-treated cells, observed {expected_cells}"
        )
    condition_records = conditions.to_dict("records")
    condition_split_codes = conditions["split"].map(decoder_data.SPLIT_TO_CODE).to_numpy(
        np.int8
    )

    descriptors = decoder_data.plan_descriptors(plans)
    descriptor_counts = load_descriptor_counts(descriptor_counts_path)
    plan_indices = descriptor_counts["plan_index"].astype(np.int64, copy=False)
    plan_row_groups = descriptor_counts["plan_row_group"].astype(np.int64, copy=False)
    recorded_descriptors = list(zip(plan_indices.tolist(), plan_row_groups.tolist()))
    if descriptors != recorded_descriptors:
        raise AssertionError("Descriptor-count rows disagree with the frozen plan descriptors")
    train_cells = descriptor_counts["train_cells"].astype(np.int64, copy=False)
    if int(train_cells.sum()) != EXPECTED_CELLS or np.any(train_cells < 0):
        raise AssertionError("Descriptor train-cell counts do not match the frozen scope")

    fingerprint_payload = {
        "schema": ALGORITHM_VERSION,
        "scope": "Experiment 1 frozen train-treated physical cells only",
        "expected_cells": EXPECTED_CELLS,
        "expected_genes": EXPECTED_GENES,
        "plans": [sha256_file(path) for path in plans],
        "condition_index": sha256_file(condition_index),
        "gene_metadata": sha256_file(gene_metadata),
        "descriptor_counts": sha256_file(descriptor_counts_path),
        "production_helper": sha256_file(Path(decoder_data.__file__)),
        "audit_script": sha256_file(Path(__file__)),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "genes": genes,
        "token_lookup": token_lookup,
        "conditions": conditions,
        "condition_records": condition_records,
        "condition_split_codes": condition_split_codes,
        "descriptors": descriptors,
        "train_cells": train_cells,
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
    }


def save_detection_state(
    path: Path,
    *,
    fingerprint: str,
    next_descriptor: int,
    cells: int,
    nonzero_cell_count: np.ndarray,
    audit_counts: np.ndarray,
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            fingerprint=np.asarray(fingerprint),
            next_descriptor=np.asarray(next_descriptor, dtype=np.int64),
            cells=np.asarray(cells, dtype=np.int64),
            nonzero_cell_count=nonzero_cell_count.astype(np.uint64, copy=False),
            audit_counts=audit_counts.astype(np.int64, copy=False),
            elapsed_seconds=np.asarray(elapsed_seconds, dtype=np.float64),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_detection_state(
    path: Path,
    *,
    fingerprint: str,
    train_cells: np.ndarray,
) -> dict[str, Any]:
    required = {
        "fingerprint",
        "next_descriptor",
        "cells",
        "nonzero_cell_count",
        "audit_counts",
        "elapsed_seconds",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise AssertionError(f"Unexpected detection-state keys: {archive.files}")
        if str(archive["fingerprint"].item()) != fingerprint:
            raise AssertionError("Detection state belongs to a different input/code contract")
        state = {
            "next_descriptor": int(archive["next_descriptor"]),
            "cells": int(archive["cells"]),
            "nonzero_cell_count": archive["nonzero_cell_count"].astype(
                np.uint64, copy=True
            ),
            "audit_counts": archive["audit_counts"].astype(np.int64, copy=True),
            "elapsed_seconds": float(archive["elapsed_seconds"]),
        }
    next_descriptor = state["next_descriptor"]
    if not 0 <= next_descriptor <= len(train_cells):
        raise AssertionError("Detection-state descriptor cursor is out of range")
    expected_prefix_cells = int(train_cells[:next_descriptor].sum())
    if state["cells"] != expected_prefix_cells:
        raise AssertionError(
            "Detection-state cell count is inconsistent with its committed descriptor cursor"
        )
    counts = state["nonzero_cell_count"]
    if counts.shape != (EXPECTED_GENES,) or np.any(counts > state["cells"]):
        raise AssertionError("Detection-state nonzero counts are invalid")
    if state["audit_counts"].shape != (len(AUDIT_LABELS),):
        raise AssertionError("Detection-state audit counters have the wrong shape")
    if not np.isfinite(state["elapsed_seconds"]) or state["elapsed_seconds"] < 0:
        raise AssertionError("Detection-state elapsed time is invalid")
    return state


def reference_count_descriptor(
    *,
    plan: pq.ParquetFile,
    plan_row_group: int,
    condition_records: list[dict[str, Any]],
    condition_split_codes: np.ndarray,
    token_lookup: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    counts = np.zeros(EXPECTED_GENES, dtype=np.uint64)
    audit_counts = np.zeros(len(AUDIT_LABELS), dtype=np.int64)
    cells = 0
    for record in decoder_data.iter_descriptor_records(
        plan=plan,
        plan_row_group=plan_row_group,
        condition_records=condition_records,
        condition_split_codes=condition_split_codes,
        split="train",
    ):
        try:
            mapped, mapped_counts, audit = decoder_data.prepare_mapped_counts(
                record["genes"], record["expressions"], token_lookup
            )
        except ValueError:
            audit_counts[4] += 1
            raise
        positive = mapped_counts > 0
        np.add.at(counts, mapped[positive], 1)
        audit_counts[0] += int(audit["sentinel_removed"])
        audit_counts[1] += int(audit["unmapped_entries"])
        audit_counts[2] += int(audit["noninteger_count_entries"])
        audit_counts[3] += int(audit["duplicate_gene_entries_collapsed"])
        cells += 1
    return counts, audit_counts, cells


def reference_count_arrays(
    genes: pa.Array,
    expressions: pa.Array,
    token_lookup: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    counts = np.zeros(EXPECTED_GENES, dtype=np.uint64)
    audit_counts = np.zeros(len(AUDIT_LABELS), dtype=np.int64)
    for row in range(len(genes)):
        try:
            mapped, mapped_counts, audit = decoder_data.prepare_mapped_counts(
                genes[row].as_py(), expressions[row].as_py(), token_lookup
            )
        except ValueError:
            audit_counts[4] += 1
            raise
        np.add.at(counts, mapped[mapped_counts > 0], 1)
        audit_counts[0] += int(audit["sentinel_removed"])
        audit_counts[1] += int(audit["unmapped_entries"])
        audit_counts[2] += int(audit["noninteger_count_entries"])
        audit_counts[3] += int(audit["duplicate_gene_entries_collapsed"])
    return counts, audit_counts, len(genes)


def count_list_arrays_fast(
    genes: pa.Array,
    expressions: pa.Array,
    token_lookup: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Vectorized production-equivalent detection count for one Arrow batch."""
    if genes.null_count or expressions.null_count or len(genes) != len(expressions):
        raise AssertionError("Raw genes/expressions arrays are null or misaligned")
    gene_offsets = genes.offsets.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    expression_offsets = expressions.offsets.to_numpy(zero_copy_only=False).astype(
        np.int64, copy=False
    )
    if not np.array_equal(gene_offsets, expression_offsets):
        raise AssertionError("Raw genes/expressions list offsets disagree")
    lengths = np.diff(gene_offsets)
    if np.any(lengths <= 0):
        return (*reference_count_arrays(genes, expressions, token_lookup), True)

    gene_values = genes.values.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    expression_values = expressions.values.to_numpy(zero_copy_only=False).astype(
        np.float64, copy=False
    )
    if len(gene_values) != len(expression_values):
        raise AssertionError("Flattened genes/expressions disagree in length")
    if not np.isfinite(expression_values).all():
        return (*reference_count_arrays(genes, expressions, token_lookup), True)

    starts = gene_offsets[:-1]
    sentinel = expression_values[starts] < 0
    sentinel_positions = starts[sentinel]
    negative_positions = np.flatnonzero(expression_values < 0)
    if not np.array_equal(negative_positions, sentinel_positions):
        return (*reference_count_arrays(genes, expressions, token_lookup), True)

    regular = np.ones(len(gene_values), dtype=bool)
    regular[sentinel_positions] = False
    in_lookup = (gene_values >= 0) & (gene_values < len(token_lookup))
    mapped = np.full(len(gene_values), -1, dtype=np.int32)
    mapped[in_lookup] = token_lookup[gene_values[in_lookup]]
    mapped_entries = regular & (mapped >= 0)
    unmapped_entries = int(np.count_nonzero(regular & (mapped < 0)))
    if unmapped_entries:
        return (*reference_count_arrays(genes, expressions, token_lookup), True)

    adjacent_bad = mapped[1:] <= mapped[:-1]
    cross_cell = gene_offsets[1:-1] - 1
    adjacent_bad[cross_cell] = False
    sentinel_comparisons = sentinel_positions[sentinel_positions < len(mapped) - 1]
    adjacent_bad[sentinel_comparisons] = False
    if np.any(adjacent_bad):
        return (*reference_count_arrays(genes, expressions, token_lookup), True)

    library_values = np.where(mapped_entries, expression_values, 0.0)
    library_sizes = np.add.reduceat(library_values, starts)
    if np.any(~np.isfinite(library_sizes)) or np.any(library_sizes <= 0):
        return (*reference_count_arrays(genes, expressions, token_lookup), True)

    positive = mapped_entries & (expression_values > 0)
    counts = np.bincount(mapped[positive], minlength=EXPECTED_GENES).astype(
        np.uint64, copy=False
    )
    audit_counts = np.asarray(
        [
            int(sentinel.sum()),
            0,
            int(
                np.count_nonzero(
                    expression_values[mapped_entries]
                    != np.rint(expression_values[mapped_entries])
                )
            ),
            0,
            0,
        ],
        dtype=np.int64,
    )
    return counts, audit_counts, len(genes), False


def fast_count_descriptor(
    *,
    plan: pq.ParquetFile,
    plan_row_group: int,
    condition_split_codes: np.ndarray,
    token_lookup: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    table = plan.read_row_group(plan_row_group, columns=decoder_data.PLAN_COLUMNS)
    owner_type = table["owner_type"].to_numpy(zero_copy_only=False)
    owner_index = table["owner_index"].to_numpy(zero_copy_only=False).astype(
        np.int64, copy=False
    )
    treated = owner_type == decoder_data.OWNER_TREATED
    bad_treated = treated & ((owner_index < 0) | (owner_index >= len(condition_split_codes)))
    if np.any(bad_treated):
        raise AssertionError("Treated owner_index is out of range")
    keep = treated.copy()
    keep[keep] &= (
        condition_split_codes[owner_index[keep]] == decoder_data.SPLIT_TO_CODE["train"]
    )
    positions = np.flatnonzero(keep)
    if len(positions) == 0:
        return (
            np.zeros(EXPECTED_GENES, dtype=np.uint64),
            np.zeros(len(AUDIT_LABELS), dtype=np.int64),
            0,
            0,
        )

    shard_paths = table["shard_path"].unique().to_pylist()
    if len(shard_paths) != 1:
        raise AssertionError("A plan row group must describe exactly one Tahoe shard")
    source = pq.ParquetFile(PROJECT_ROOT / str(shard_paths[0]))
    source_groups = table["row_group_index"].to_numpy(zero_copy_only=False).astype(
        np.int64, copy=False
    )
    rows_in_group = table["row_index_in_row_group"].to_numpy(
        zero_copy_only=False
    ).astype(np.int64, copy=False)

    counts = np.zeros(EXPECTED_GENES, dtype=np.uint64)
    audit_counts = np.zeros(len(AUDIT_LABELS), dtype=np.int64)
    cells = 0
    fallback_batches = 0
    selected_groups = source_groups[positions]
    for source_group in np.unique(selected_groups):
        group_positions = positions[selected_groups == source_group]
        raw = source.read_row_group(int(source_group), columns=("genes", "expressions"))
        selected_rows = rows_in_group[group_positions]
        if np.any((selected_rows < 0) | (selected_rows >= raw.num_rows)):
            raise AssertionError("row_index_in_row_group is out of range")
        selected = raw.take(pa.array(selected_rows, type=pa.int64()))
        gene_lists = selected["genes"].combine_chunks()
        expression_lists = selected["expressions"].combine_chunks()
        batch_counts, batch_audit, batch_cells, fell_back = count_list_arrays_fast(
            gene_lists, expression_lists, token_lookup
        )
        counts += batch_counts
        audit_counts += batch_audit
        cells += batch_cells
        fallback_batches += int(fell_back)
    return counts, audit_counts, cells, fallback_batches


def write_progress(
    path: Path,
    *,
    status: str,
    state_path: Path,
    fingerprint: str,
    next_descriptor: int,
    total_descriptors: int,
    cells: int,
    session_seconds: float,
    prior_seconds: float,
    session_start_cells: int,
    fallback_batches: int,
    error: str | None = None,
) -> None:
    session_rate = (cells - session_start_cells) / max(session_seconds, 1e-9)
    total_seconds = prior_seconds + session_seconds
    total_rate = cells / max(total_seconds, 1e-9)
    remaining = EXPECTED_CELLS - cells
    payload: dict[str, Any] = {
        "updated_at_utc": utc_now(),
        "status": status,
        "scope": "Experiment 1 frozen train-treated physical cells only",
        "processed_train_treated_cells": cells,
        "expected_train_treated_cells": EXPECTED_CELLS,
        "remaining_cells": remaining,
        "next_descriptor": next_descriptor,
        "total_descriptors": total_descriptors,
        "session_cells_per_second": session_rate,
        "overall_cells_per_second": total_rate,
        "estimated_remaining_seconds": remaining / session_rate if session_rate > 0 else None,
        "elapsed_seconds": total_seconds,
        "fallback_arrow_batches": fallback_batches,
        "fingerprint": fingerprint,
        "state": display_path(state_path),
    }
    if error is not None:
        payload["error"] = error
    atomic_write_json(path, payload)


def validate_smoke(path: Path, fingerprint: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required smoke result is missing: {display_path(path)}; run the smoke command first"
        )
    with path.open("r", encoding="utf-8") as handle:
        smoke = json.load(handle)
    if smoke.get("status") != "pass" or smoke.get("contract_fingerprint") != fingerprint:
        raise AssertionError("Smoke result does not match the current formal scan contract")
    return smoke


def run_smoke(args: argparse.Namespace) -> None:
    if args.descriptors < 1:
        raise ValueError("--descriptors must be positive")
    contract = load_contract(
        plans=args.plans,
        condition_index=args.condition_index,
        gene_metadata=args.gene_metadata,
        descriptor_counts_path=args.descriptor_counts,
    )
    panel_state = load_panel_builder_state(PANEL_STATE)
    if panel_state["next_descriptor"] != len(contract["descriptors"]):
        raise AssertionError("Formal panel-builder state did not complete every descriptor")
    state_mean = panel_state["value_sum"] / EXPECTED_CELLS
    state_variance = np.maximum(
        panel_state["value_sumsq"] / EXPECTED_CELLS - state_mean * state_mean, 0.0
    )
    formal_panel = pd.read_csv(PANEL, encoding="utf-8-sig", keep_default_na=False)
    formal_indices = formal_panel["genejepa_index"].to_numpy(np.int64)
    expected_formal_indices = np.lexsort(
        (np.arange(EXPECTED_GENES, dtype=np.int64), -state_variance)
    )[:EXPECTED_PANEL_GENES]
    if len(formal_panel) != EXPECTED_PANEL_GENES or not np.array_equal(
        formal_indices, expected_formal_indices
    ):
        raise AssertionError("Formal panel does not match variance reconstructed from state")
    formal_variance = formal_panel["population_variance_log1p_cp10k"].to_numpy(
        np.float64
    )
    variance_difference = np.abs(formal_variance - state_variance[formal_indices])
    if not np.allclose(
        formal_variance, state_variance[formal_indices], rtol=1e-12, atol=1e-14
    ):
        raise AssertionError("State-reconstructed selected-gene variance does not round-trip")
    descriptor_total = min(args.descriptors, len(contract["descriptors"]))
    plans = {index: pq.ParquetFile(path) for index, path in enumerate(args.plans)}
    fast_total = np.zeros(EXPECTED_GENES, dtype=np.uint64)
    reference_total = np.zeros(EXPECTED_GENES, dtype=np.uint64)
    fast_audit_total = np.zeros(len(AUDIT_LABELS), dtype=np.int64)
    reference_audit_total = np.zeros(len(AUDIT_LABELS), dtype=np.int64)
    fast_cells_total = 0
    reference_cells_total = 0
    fallback_batches = 0
    descriptor_results: list[dict[str, Any]] = []
    started = time.perf_counter()

    for descriptor_index in range(descriptor_total):
        plan_index, plan_row_group = contract["descriptors"][descriptor_index]
        fast_started = time.perf_counter()
        fast_counts, fast_audit, fast_cells, fallbacks = fast_count_descriptor(
            plan=plans[plan_index],
            plan_row_group=plan_row_group,
            condition_split_codes=contract["condition_split_codes"],
            token_lookup=contract["token_lookup"],
        )
        fast_seconds = time.perf_counter() - fast_started
        reference_started = time.perf_counter()
        reference_counts, reference_audit, reference_cells = reference_count_descriptor(
            plan=plans[plan_index],
            plan_row_group=plan_row_group,
            condition_records=contract["condition_records"],
            condition_split_codes=contract["condition_split_codes"],
            token_lookup=contract["token_lookup"],
        )
        reference_seconds = time.perf_counter() - reference_started
        expected_cells = int(contract["train_cells"][descriptor_index])
        exact = (
            fast_cells == reference_cells == expected_cells
            and np.array_equal(fast_counts, reference_counts)
            and np.array_equal(fast_audit, reference_audit)
        )
        if not exact:
            raise AssertionError(f"Fast/reference mismatch at descriptor {descriptor_index}")
        fast_total += fast_counts
        reference_total += reference_counts
        fast_audit_total += fast_audit
        reference_audit_total += reference_audit
        fast_cells_total += fast_cells
        reference_cells_total += reference_cells
        fallback_batches += fallbacks
        descriptor_results.append(
            {
                "descriptor_index": descriptor_index,
                "plan_index": plan_index,
                "plan_row_group": plan_row_group,
                "cells": fast_cells,
                "fast_seconds": fast_seconds,
                "reference_seconds": reference_seconds,
                "speedup": reference_seconds / max(fast_seconds, 1e-9),
                "counts_exact": True,
                "audit_exact": True,
                "fallback_arrow_batches": fallbacks,
            }
        )

    with tempfile.TemporaryDirectory(prefix="gene_detection_smoke_", dir=RESULTS) as temporary:
        resume_path = Path(temporary) / "state.npz"
        first_cells = int(contract["train_cells"][0])
        first_fast = descriptor_results[0]
        first_plan_index, first_plan_row_group = contract["descriptors"][0]
        first_counts, first_audit, observed_first_cells, _ = fast_count_descriptor(
            plan=plans[first_plan_index],
            plan_row_group=first_plan_row_group,
            condition_split_codes=contract["condition_split_codes"],
            token_lookup=contract["token_lookup"],
        )
        if observed_first_cells != first_cells or first_fast["cells"] != first_cells:
            raise AssertionError("Smoke resume prefix has the wrong cell count")
        save_detection_state(
            resume_path,
            fingerprint=contract["fingerprint"],
            next_descriptor=1,
            cells=first_cells,
            nonzero_cell_count=first_counts,
            audit_counts=first_audit,
            elapsed_seconds=1.0,
        )
        resumed = load_detection_state(
            resume_path,
            fingerprint=contract["fingerprint"],
            train_cells=contract["train_cells"],
        )
        resume_prefix_exact = (
            resumed["next_descriptor"] == 1
            and resumed["cells"] == first_cells
            and np.array_equal(resumed["nonzero_cell_count"], first_counts)
            and np.array_equal(resumed["audit_counts"], first_audit)
        )
        if not resume_prefix_exact:
            raise AssertionError("Detection-state round trip failed")

    elapsed = time.perf_counter() - started
    exact_total = (
        fast_cells_total == reference_cells_total
        and np.array_equal(fast_total, reference_total)
        and np.array_equal(fast_audit_total, reference_audit_total)
    )
    if not exact_total:
        raise AssertionError("Smoke aggregate fast/reference comparison failed")
    fast_seconds_total = sum(row["fast_seconds"] for row in descriptor_results)
    reference_seconds_total = sum(row["reference_seconds"] for row in descriptor_results)
    payload = {
        "schema": "genejepa_decoder_gene_sparsity_smoke_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "scope": "real frozen train-treated plan descriptors",
        "descriptors": descriptor_total,
        "cells": fast_cells_total,
        "fast_reference_counts_exact": exact_total,
        "fast_reference_audit_exact": True,
        "resume_state_round_trip_exact": resume_prefix_exact,
        "panel_builder_state_audit": {
            "keys": [
                "fingerprint",
                "next_descriptor",
                "cells",
                "value_sum",
                "value_sumsq",
                "audit_counts",
            ],
            "cells": panel_state["cells"],
            "descriptors": panel_state["next_descriptor"],
            "genes": len(panel_state["value_sum"]),
            "mean_reusable": True,
            "variance_reusable": True,
            "nonzero_count_present": False,
            "formal_panel_membership_and_order_exact": True,
            "selected_variance_roundtrip_allclose": True,
            "selected_variance_max_absolute_difference": float(
                variance_difference.max(initial=0.0)
            ),
        },
        "fallback_arrow_batches": fallback_batches,
        "audit": audit_dict(fast_audit_total),
        "timing": {
            "fast_seconds": fast_seconds_total,
            "reference_seconds": reference_seconds_total,
            "fast_cells_per_second": fast_cells_total / max(fast_seconds_total, 1e-9),
            "reference_cells_per_second": fast_cells_total
            / max(reference_seconds_total, 1e-9),
            "speedup": reference_seconds_total / max(fast_seconds_total, 1e-9),
            "total_smoke_seconds": elapsed,
        },
        "contract_fingerprint": contract["fingerprint"],
        "fingerprint_payload": contract["fingerprint_payload"],
        "descriptor_results": descriptor_results,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_scan(args: argparse.Namespace) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Detection scan is single-process and must not use torchrun")
    if args.checkpoint_every_descriptors < 1:
        raise ValueError("--checkpoint-every-descriptors must be positive")
    contract = load_contract(
        plans=args.plans,
        condition_index=args.condition_index,
        gene_metadata=args.gene_metadata,
        descriptor_counts_path=args.descriptor_counts,
    )
    validate_smoke(args.smoke, contract["fingerprint"])

    if args.state.exists():
        state = load_detection_state(
            args.state,
            fingerprint=contract["fingerprint"],
            train_cells=contract["train_cells"],
        )
    else:
        state = {
            "next_descriptor": 0,
            "cells": 0,
            "nonzero_cell_count": np.zeros(EXPECTED_GENES, dtype=np.uint64),
            "audit_counts": np.zeros(len(AUDIT_LABELS), dtype=np.int64),
            "elapsed_seconds": 0.0,
        }

    if state["next_descriptor"] == len(contract["descriptors"]):
        if state["cells"] != EXPECTED_CELLS:
            raise AssertionError("Complete descriptor cursor has an incomplete cell count")
        print("Detection state is already complete; finalizing without rescanning.")
        finalize(args, contract=contract, detection_state=state)
        return

    plans = {index: pq.ParquetFile(path) for index, path in enumerate(args.plans)}
    session_start = time.perf_counter()
    session_start_cells = state["cells"]
    prior_seconds = state["elapsed_seconds"]
    fallback_batches = 0
    committed_descriptor = state["next_descriptor"]
    try:
        for descriptor_index in range(state["next_descriptor"], len(contract["descriptors"])):
            plan_index, plan_row_group = contract["descriptors"][descriptor_index]
            descriptor_counts, descriptor_audit, descriptor_cells, fallbacks = (
                fast_count_descriptor(
                    plan=plans[plan_index],
                    plan_row_group=plan_row_group,
                    condition_split_codes=contract["condition_split_codes"],
                    token_lookup=contract["token_lookup"],
                )
            )
            expected_descriptor_cells = int(contract["train_cells"][descriptor_index])
            if descriptor_cells != expected_descriptor_cells:
                raise AssertionError(
                    f"Descriptor {descriptor_index} expected {expected_descriptor_cells} "
                    f"train cells, observed {descriptor_cells}"
                )
            state["nonzero_cell_count"] += descriptor_counts
            state["audit_counts"] += descriptor_audit
            state["cells"] += descriptor_cells
            fallback_batches += fallbacks
            committed_descriptor = descriptor_index + 1
            state["next_descriptor"] = committed_descriptor

            checkpoint = (
                committed_descriptor % args.checkpoint_every_descriptors == 0
                or committed_descriptor == len(contract["descriptors"])
            )
            if checkpoint:
                session_seconds = time.perf_counter() - session_start
                elapsed_seconds = prior_seconds + session_seconds
                save_detection_state(
                    args.state,
                    fingerprint=contract["fingerprint"],
                    next_descriptor=committed_descriptor,
                    cells=state["cells"],
                    nonzero_cell_count=state["nonzero_cell_count"],
                    audit_counts=state["audit_counts"],
                    elapsed_seconds=elapsed_seconds,
                )
                write_progress(
                    args.progress,
                    status=(
                        "detection_complete"
                        if committed_descriptor == len(contract["descriptors"])
                        else "running"
                    ),
                    state_path=args.state,
                    fingerprint=contract["fingerprint"],
                    next_descriptor=committed_descriptor,
                    total_descriptors=len(contract["descriptors"]),
                    cells=state["cells"],
                    session_seconds=session_seconds,
                    prior_seconds=prior_seconds,
                    session_start_cells=session_start_cells,
                    fallback_batches=fallback_batches,
                )
                rate = (state["cells"] - session_start_cells) / max(session_seconds, 1e-9)
                eta_minutes = (EXPECTED_CELLS - state["cells"]) / max(rate, 1e-9) / 60.0
                print(
                    f"descriptor={committed_descriptor}/{len(contract['descriptors'])} "
                    f"cells={state['cells']:,}/{EXPECTED_CELLS:,} "
                    f"session_rate={rate:.2f}/s ETA={eta_minutes:.1f} min",
                    flush=True,
                )
    except BaseException as error:
        session_seconds = time.perf_counter() - session_start
        # Only the last atomically saved descriptor prefix is resumable.  Never
        # persist an in-memory descriptor after an exception.
        if args.state.exists():
            committed = load_detection_state(
                args.state,
                fingerprint=contract["fingerprint"],
                train_cells=contract["train_cells"],
            )
            write_progress(
                args.progress,
                status="interrupted_or_failed",
                state_path=args.state,
                fingerprint=contract["fingerprint"],
                next_descriptor=committed["next_descriptor"],
                total_descriptors=len(contract["descriptors"]),
                cells=committed["cells"],
                session_seconds=session_seconds,
                prior_seconds=prior_seconds,
                session_start_cells=session_start_cells,
                fallback_batches=fallback_batches,
                error=f"{type(error).__name__}: {error}",
            )
        raise

    state = load_detection_state(
        args.state,
        fingerprint=contract["fingerprint"],
        train_cells=contract["train_cells"],
    )
    finalize(args, contract=contract, detection_state=state)


def quantile_summary(values: np.ndarray, *, include_mean: bool = True) -> dict[str, float]:
    probabilities = (0.00, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00)
    labels = ("min", "p01", "p05", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max")
    result = {
        label: float(value)
        for label, value in zip(labels, np.quantile(values, probabilities), strict=True)
    }
    if include_mean:
        result["mean"] = float(np.mean(values))
    return result


def validate_complete_detection_state(
    state: dict[str, Any], contract: dict[str, Any], panel_state: dict[str, Any]
) -> None:
    if state["next_descriptor"] != len(contract["descriptors"]):
        raise AssertionError("Detection scan has not completed all descriptors")
    if state["cells"] != EXPECTED_CELLS:
        raise AssertionError("Detection scan has not completed all frozen train-treated cells")
    if np.any(state["nonzero_cell_count"] > EXPECTED_CELLS):
        raise AssertionError("A gene detection count exceeds the cell count")
    if not np.array_equal(state["audit_counts"], panel_state["audit_counts"]):
        raise AssertionError(
            "Detection scan audit counters disagree with the formal panel-builder state"
        )


def artifact_entry(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": display_path(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if rows is not None:
        result["rows"] = rows
        result["encoding"] = "utf-8-sig"
    return result


def finalize(
    args: argparse.Namespace,
    *,
    contract: dict[str, Any] | None = None,
    detection_state: dict[str, Any] | None = None,
) -> None:
    if contract is None:
        contract = load_contract(
            plans=args.plans,
            condition_index=args.condition_index,
            gene_metadata=args.gene_metadata,
            descriptor_counts_path=args.descriptor_counts,
        )
    validate_smoke(args.smoke, contract["fingerprint"])
    if detection_state is None:
        if not args.state.exists():
            raise FileNotFoundError("Detection state does not exist; run the scan command")
        detection_state = load_detection_state(
            args.state,
            fingerprint=contract["fingerprint"],
            train_cells=contract["train_cells"],
        )
    panel_state = load_panel_builder_state(args.panel_state)
    if panel_state["next_descriptor"] != len(contract["descriptors"]):
        raise AssertionError("Formal panel-builder state did not complete every descriptor")
    validate_complete_detection_state(detection_state, contract, panel_state)

    genes = contract["genes"].copy()
    value_sum = panel_state["value_sum"]
    mean = value_sum / EXPECTED_CELLS
    variance = np.maximum(
        panel_state["value_sumsq"] / EXPECTED_CELLS - mean * mean, 0.0
    )
    standard_deviation = np.sqrt(variance)
    nonzero_count = detection_state["nonzero_cell_count"].astype(np.uint64, copy=False)
    detection_rate = nonzero_count.astype(np.float64) / EXPECTED_CELLS
    zero_rate = 1.0 - detection_rate
    nonzero_mean = np.divide(
        value_sum,
        nonzero_count,
        out=np.zeros(EXPECTED_GENES, dtype=np.float64),
        where=nonzero_count > 0,
    )

    if not np.isfinite(mean).all() or np.any(mean < 0):
        raise AssertionError("Reconstructed mean_logcp10k is invalid")
    if not np.isfinite(variance).all() or np.any(variance < 0):
        raise AssertionError("Reconstructed variance_logcp10k is invalid")
    if np.any((detection_rate < 0) | (detection_rate > 1)):
        raise AssertionError("Detection rates are outside [0, 1]")
    if np.any((zero_rate < 0) | (zero_rate > 1)):
        raise AssertionError("Zero rates are outside [0, 1]")
    if not np.array_equal(detection_rate, nonzero_count / EXPECTED_CELLS):
        raise AssertionError("Detection rate formula check failed")

    panel = pd.read_csv(args.panel, encoding="utf-8-sig", keep_default_na=False)
    if len(panel) != EXPECTED_PANEL_GENES:
        raise AssertionError("Formal panel does not contain exactly 5,000 genes")
    if not np.array_equal(panel["panel_rank"].to_numpy(np.int64), np.arange(len(panel))):
        raise AssertionError("Formal panel rank is not contiguous from zero")
    panel_indices = panel["genejepa_index"].to_numpy(np.int64)
    variance_order = np.lexsort((np.arange(EXPECTED_GENES, dtype=np.int64), -variance))
    if not np.array_equal(panel_indices, variance_order[:EXPECTED_PANEL_GENES]):
        raise AssertionError("Formal panel membership/order disagrees with reconstructed variance")
    panel_variance = panel["population_variance_log1p_cp10k"].to_numpy(np.float64)
    panel_mean = panel["mean_log1p_cp10k"].to_numpy(np.float64)
    variance_difference = np.abs(panel_variance - variance[panel_indices])
    mean_difference = np.abs(panel_mean - mean[panel_indices])
    if not np.allclose(panel_variance, variance[panel_indices], rtol=1e-12, atol=1e-14):
        raise AssertionError("Panel CSV variance disagrees beyond float round-trip tolerance")
    if not np.allclose(panel_mean, mean[panel_indices], rtol=1e-12, atol=1e-14):
        raise AssertionError("Panel CSV mean disagrees beyond float round-trip tolerance")

    gene_index = np.arange(EXPECTED_GENES, dtype=np.int64)
    mean_order = np.lexsort((gene_index, -detection_rate, -mean))
    detection_order = np.lexsort((gene_index, -mean, -detection_rate))
    rank_by_mean = np.empty(EXPECTED_GENES, dtype=np.int32)
    rank_by_detection = np.empty(EXPECTED_GENES, dtype=np.int32)
    rank_by_mean[mean_order] = np.arange(1, EXPECTED_GENES + 1, dtype=np.int32)
    rank_by_detection[detection_order] = np.arange(1, EXPECTED_GENES + 1, dtype=np.int32)
    in_panel = np.zeros(EXPECTED_GENES, dtype=bool)
    in_panel[panel_indices] = True
    panel_rank = np.full(EXPECTED_GENES, -1, dtype=np.int32)
    panel_rank[panel_indices] = np.arange(EXPECTED_PANEL_GENES, dtype=np.int32)

    stats = genes.copy()
    stats["mean_logcp10k"] = mean
    stats["variance_logcp10k"] = variance
    stats["std_logcp10k"] = standard_deviation
    stats["nonzero_cell_count"] = nonzero_count
    stats["detection_rate"] = detection_rate
    stats["zero_rate"] = zero_rate
    stats["nonzero_mean_logcp10k"] = nonzero_mean
    stats["current_top5000"] = in_panel.astype(np.int8)
    stats["current_panel_rank"] = panel_rank
    stats["rank_by_mean_expression"] = rank_by_mean
    stats["rank_by_detection"] = rank_by_detection
    atomic_write_csv(args.stats_csv, stats)

    threshold_rows: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        selected = detection_rate >= threshold
        threshold_rows.append(
            {
                "detection_rate_threshold": threshold,
                "threshold_percent": int(round(threshold * 100)),
                "gene_count": int(selected.sum()),
                "fraction_of_62710": float(selected.mean()),
                "overlap_with_current_top5000": int(np.count_nonzero(selected & in_panel)),
            }
        )
    threshold_frame = pd.DataFrame(threshold_rows)
    atomic_write_csv(args.thresholds_csv, threshold_frame)

    topn_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    ranking_orders = {
        "high_expression": mean_order,
        "high_detection": detection_order,
    }
    for ranking_type, order in ranking_orders.items():
        for top_n in TOP_NS:
            selected_indices = order[:top_n]
            selected_panel = in_panel[selected_indices]
            overlap = int(selected_panel.sum())
            topn_rows.append(
                {
                    "N": top_n,
                    "ranking_type": ranking_type,
                    "mean_detection_rate": float(detection_rate[selected_indices].mean()),
                    "median_detection_rate": float(np.median(detection_rate[selected_indices])),
                    "min_detection_rate": float(detection_rate[selected_indices].min()),
                    "mean_mean_logcp10k": float(mean[selected_indices].mean()),
                    "median_mean_logcp10k": float(np.median(mean[selected_indices])),
                    "min_mean_logcp10k": float(mean[selected_indices].min()),
                    "overlap_count_with_current_variance_top5000": overlap,
                    "overlap_fraction_of_N": overlap / top_n,
                    "jaccard_with_current_top5000": overlap
                    / (top_n + EXPECTED_PANEL_GENES - overlap),
                }
            )
            overlap_rows.append(
                {
                    "candidate_type": ranking_type,
                    "candidate_N": top_n,
                    "overlap_count": overlap,
                    "candidate_overlap_fraction": overlap / top_n,
                    "current_panel_overlap_fraction": overlap / EXPECTED_PANEL_GENES,
                    "jaccard": overlap / (top_n + EXPECTED_PANEL_GENES - overlap),
                }
            )
    topn_frame = pd.DataFrame(topn_rows)
    overlap_frame = pd.DataFrame(overlap_rows)
    atomic_write_csv(args.topn_csv, topn_frame)
    atomic_write_csv(args.overlap_csv, overlap_frame)

    detail_columns = [
        "genejepa_index",
        "gene_symbol",
        "ensembl_id",
        "token_id",
        "mean_logcp10k",
        "variance_logcp10k",
        "detection_rate",
        "zero_rate",
        "current_top5000",
        "current_panel_rank",
    ]
    high_expression = stats.iloc[mean_order[:EXPECTED_PANEL_GENES]][detail_columns].copy()
    high_expression.insert(0, "rank", np.arange(1, EXPECTED_PANEL_GENES + 1))
    high_detection = stats.iloc[detection_order[:EXPECTED_PANEL_GENES]][detail_columns].copy()
    high_detection.insert(0, "rank", np.arange(1, EXPECTED_PANEL_GENES + 1))
    atomic_write_csv(args.high_expression_csv, high_expression)
    atomic_write_csv(args.high_detection_csv, high_detection)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    bins = np.linspace(0.0, 1.0, 51)
    axis.hist(
        detection_rate,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.6,
        label="All 62,710 genes",
    )
    axis.hist(
        detection_rate[panel_indices],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.6,
        label="Current variance Top5000",
    )
    axis.set(xlabel="Detection rate", ylabel="Density", xlim=(0, 1))
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    atomic_save_figure(figure, args.detection_png)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 5.2))
    axis.scatter(
        detection_rate[~in_panel],
        mean[~in_panel],
        s=5,
        alpha=0.18,
        linewidths=0,
        color="#808080",
        label="Other genes",
    )
    axis.scatter(
        detection_rate[in_panel],
        mean[in_panel],
        s=7,
        alpha=0.42,
        linewidths=0,
        color="#d62728",
        label="Current variance Top5000",
    )
    axis.set(xlabel="Detection rate", ylabel="Mean log1p(CP10000)", xlim=(0, 1))
    axis.legend(frameon=False, markerscale=2)
    axis.grid(alpha=0.2)
    atomic_save_figure(figure, args.expression_detection_png)
    plt.close(figure)

    panel_detection = detection_rate[panel_indices]
    below_thresholds = {
        f"lt_{int(round(threshold * 100)):02d}_percent": {
            "count": int(np.count_nonzero(panel_detection < threshold)),
            "fraction": float(np.mean(panel_detection < threshold)),
        }
        for threshold in THRESHOLDS
        if threshold <= 0.80
    }
    current_sparsity = {
        "detection_rate": quantile_summary(panel_detection),
        "below_thresholds": below_thresholds,
        "mean_logcp10k": quantile_summary(mean[panel_indices]),
        "zero_rate": quantile_summary(zero_rate[panel_indices]),
    }

    csv_expectations = {
        args.stats_csv: EXPECTED_GENES,
        args.thresholds_csv: len(THRESHOLDS),
        args.topn_csv: len(TOP_NS) * 2,
        args.high_expression_csv: EXPECTED_PANEL_GENES,
        args.high_detection_csv: EXPECTED_PANEL_GENES,
        args.overlap_csv: len(TOP_NS) * 2,
    }
    for path, expected_rows in csv_expectations.items():
        observed = len(pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False))
        if observed != expected_rows:
            raise AssertionError(f"CSV readback row mismatch for {display_path(path)}")
    for path in (args.detection_png, args.expression_detection_png):
        if not path.exists() or path.stat().st_size < 1_000:
            raise AssertionError(f"Plot output is missing or unexpectedly small: {path}")

    outputs = {
        "gene_statistics": artifact_entry(args.stats_csv, rows=EXPECTED_GENES),
        "detection_thresholds": artifact_entry(
            args.thresholds_csv, rows=len(THRESHOLDS)
        ),
        "topn_summary": artifact_entry(args.topn_csv, rows=len(TOP_NS) * 2),
        "high_expression_top5000": artifact_entry(
            args.high_expression_csv, rows=EXPECTED_PANEL_GENES
        ),
        "high_detection_top5000": artifact_entry(
            args.high_detection_csv, rows=EXPECTED_PANEL_GENES
        ),
        "panel_overlap_summary": artifact_entry(
            args.overlap_csv, rows=len(TOP_NS) * 2
        ),
        "detection_distribution_plot": artifact_entry(args.detection_png),
        "expression_vs_detection_plot": artifact_entry(args.expression_detection_png),
    }
    with args.panel_summary.open("r", encoding="utf-8") as handle:
        formal_panel_summary = json.load(handle)
    summary = {
        "schema": "genejepa_decoder_gene_sparsity_audit_v1",
        "status": "pass",
        "created_at_utc": utc_now(),
        "scope": {
            "split": "train",
            "treated_only": True,
            "cells": EXPECTED_CELLS,
            "genes": EXPECTED_GENES,
            "excluded": ["val", "test", "DMSO"],
            "expression_space": "raw mapped counts -> CP10000 over all mapped genes -> log1p",
            "implicit_zeros_included": True,
        },
        "provenance": {
            "formal_panel": {
                "path": display_path(args.panel),
                "sha256": sha256_file(args.panel),
            },
            "formal_panel_summary": {
                "path": display_path(args.panel_summary),
                "sha256": sha256_file(args.panel_summary),
                "builder_fingerprint": formal_panel_summary["builder"]["fingerprint"],
            },
            "panel_builder_state": {
                "path": display_path(args.panel_state),
                "sha256": sha256_file(args.panel_state),
            },
            "detection_state": {
                "path": display_path(args.state),
                "sha256": sha256_file(args.state),
                "contract_fingerprint": contract["fingerprint"],
            },
            "gene_metadata": {
                "path": display_path(args.gene_metadata),
                "sha256": sha256_file(args.gene_metadata),
            },
            "source_code": {
                "audit_script": {
                    "path": display_path(Path(__file__)),
                    "sha256": sha256_file(Path(__file__)),
                },
                "frozen_production_helper": {
                    "path": display_path(Path(decoder_data.__file__)),
                    "sha256": sha256_file(Path(decoder_data.__file__)),
                },
            },
        },
        "state_reuse": {
            "mean_reused": True,
            "variance_reused": True,
            "nonzero_count_reused": False,
            "new_detection_scan_required": True,
            "mean_variance_source": display_path(args.panel_state),
            "detection_scan_recomputed_cp10k_targets": False,
        },
        "ranking_definitions": {
            "high_expression": [
                "mean_logcp10k descending",
                "detection_rate descending",
                "genejepa_index ascending",
            ],
            "high_detection": [
                "detection_rate descending",
                "mean_logcp10k descending",
                "genejepa_index ascending",
            ],
            "rank_base": 1,
            "current_panel_rank_base": 0,
            "current_panel_rank_outside_panel": -1,
        },
        "current_top5000_sparsity": current_sparsity,
        "whole_gene_universe": {
            "detection_threshold_counts": threshold_rows,
        },
        "topn_summary": topn_rows,
        "panel_overlap_summary": overlap_rows,
        "sanity_checks": {
            "processed_train_treated_cells_exact": True,
            "gene_rows_exact": True,
            "rates_in_unit_interval": True,
            "mean_nonnegative": True,
            "variance_nonnegative": True,
            "nonzero_counts_within_cell_count": True,
            "detection_formula_exact": True,
            "formal_panel_rows": EXPECTED_PANEL_GENES,
            "formal_panel_membership_and_order_exact": True,
            "selected_variance_roundtrip_allclose": True,
            "selected_variance_max_absolute_difference": float(
                variance_difference.max(initial=0.0)
            ),
            "selected_mean_roundtrip_allclose": True,
            "selected_mean_max_absolute_difference": float(mean_difference.max(initial=0.0)),
            "detection_audit_matches_panel_builder_audit": True,
            "audit": audit_dict(detection_state["audit_counts"]),
            "csv_readback_rows_exact": True,
            "plots_nonempty": True,
        },
        "nonzero_mean_zero_count_policy": "0.0 when nonzero_cell_count == 0",
        "interpretation_guardrail": (
            "High expression, high detection, high variance, and perturbation informativeness "
            "are distinct properties; this audit does not select or freeze a Decoder v2 panel."
        ),
        "outputs": outputs,
    }
    atomic_write_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def add_shared_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--plans",
        type=resolve_path,
        nargs=2,
        default=decoder_data.FORMAL_PLANS,
        metavar=("WORKER0", "WORKER1"),
    )
    command.add_argument(
        "--condition-index",
        type=resolve_path,
        default=decoder_data.FORMAL_CONDITION_INDEX,
    )
    command.add_argument(
        "--gene-metadata", type=resolve_path, default=decoder_data.GENE_METADATA
    )
    command.add_argument(
        "--descriptor-counts", type=resolve_path, default=DESCRIPTOR_COUNTS
    )
    command.add_argument("--smoke", type=resolve_path, default=SMOKE_RESULT)


def add_finalize_arguments(command: argparse.ArgumentParser) -> None:
    add_shared_arguments(command)
    command.add_argument("--state", type=resolve_path, default=DETECTION_STATE)
    command.add_argument("--panel", type=resolve_path, default=PANEL)
    command.add_argument("--panel-summary", type=resolve_path, default=PANEL_SUMMARY)
    command.add_argument("--panel-state", type=resolve_path, default=PANEL_STATE)
    command.add_argument("--stats-csv", type=resolve_path, default=STATS_CSV)
    command.add_argument("--thresholds-csv", type=resolve_path, default=THRESHOLDS_CSV)
    command.add_argument("--topn-csv", type=resolve_path, default=TOPN_CSV)
    command.add_argument(
        "--high-expression-csv", type=resolve_path, default=HIGH_EXPRESSION_CSV
    )
    command.add_argument(
        "--high-detection-csv", type=resolve_path, default=HIGH_DETECTION_CSV
    )
    command.add_argument("--overlap-csv", type=resolve_path, default=OVERLAP_CSV)
    command.add_argument("--detection-png", type=resolve_path, default=DETECTION_PNG)
    command.add_argument(
        "--expression-detection-png",
        type=resolve_path,
        default=EXPRESSION_DETECTION_PNG,
    )
    command.add_argument("--summary-json", type=resolve_path, default=SUMMARY_JSON)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    smoke = commands.add_parser(
        "smoke", help="Compare the vectorized detector with the frozen production helper"
    )
    add_shared_arguments(smoke)
    smoke.add_argument("--descriptors", type=int, default=2)
    smoke.add_argument("--output", type=resolve_path, default=SMOKE_RESULT)

    scan = commands.add_parser(
        "scan", help="Run or resume the formal detection-only streaming scan"
    )
    add_finalize_arguments(scan)
    scan.add_argument("--progress", type=resolve_path, default=DETECTION_PROGRESS)
    scan.add_argument("--checkpoint-every-descriptors", type=int, default=10)

    finalize_command = commands.add_parser(
        "finalize", help="Generate final tables and figures from a complete detection state"
    )
    add_finalize_arguments(finalize_command)
    return result


def main() -> None:
    args = parser().parse_args()
    args.plans = tuple(args.plans)
    if args.command == "smoke":
        run_smoke(args)
    elif args.command == "scan":
        run_scan(args)
    elif args.command == "finalize":
        finalize(args)


if __name__ == "__main__":
    main()
