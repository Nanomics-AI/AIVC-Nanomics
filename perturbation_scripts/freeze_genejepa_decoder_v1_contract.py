#!/usr/bin/env python3
"""Freeze and finalize the GeneJEPA Decoder v1 scientific contract."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tahoe_decoder_v1_data import (
    FORMAL_EMBEDDING_MANIFEST,
    FORMAL_EMBEDDINGS,
    FORMAL_PLANS,
    GENE_METADATA,
    PROJECT_ROOT,
    RAW_COLUMNS,
    RESULTS,
    TahoeDecoderV1PairedDataset,
    atomic_write_json,
    display_path,
    load_gene_universe,
    load_panel,
    log1p_cp10k_target,
    plan_descriptors,
    sha256_file,
    utc_now,
)


EXPECTED_SPLIT_CELLS = {
    "train": 22_941_936,
    "val": 2_841_724,
    "test": 2_856_299,
}
EXPECTED_GENE_UNIVERSE = 62_710
EXPECTED_PANEL_SIZE = 5_000
TASK = PROJECT_ROOT.parent / "当前任务.txt"
READINESS_AUDIT = RESULTS / "genejepa_decoder_v1_readiness_audit.json"
SMOKE_PANEL = RESULTS / "genejepa_decoder_v1_gene_panel_smoke_only_current.csv"
LOOKUP_SMOKE = RESULTS / "genejepa_decoder_v1_panel_lookup_equivalence_smoke.json"
CONTRACT = RESULTS / "genejepa_decoder_v1_contract.json"
FORMAL_PANEL = RESULTS / "genejepa_decoder_v1_gene_panel.csv"
FORMAL_PANEL_SUMMARY = RESULTS / "genejepa_decoder_v1_gene_panel.json"
FORMAL_PANEL_STATE = RESULTS / "genejepa_decoder_v1_gene_panel_builder_state.npz"
FORMAL_PANEL_AUDIT = RESULTS / "genejepa_decoder_v1_gene_panel_audit.json"
CONDITION_SPLIT = RESULTS / "tahoe_experiment1_condition_split_manifest.csv"
EDGE_SPLIT = RESULTS / "tahoe_experiment1_edge_split_manifest.csv"
CONDITION_INDEX = RESULTS / "tahoe_experiment1_cache_cap512_all_dmso_condition_index.csv"
DATASET_CODE = PROJECT_ROOT / "perturbation_scripts/tahoe_decoder_v1_data.py"
DECLARED_V21_DOCUMENT = PROJECT_ROOT.parent / "虚拟细胞_GeneJEPA_项目总资料_统一版_v2.1_20260911.md"
CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints/genejepa_quarter_d12_h6_700k_e30_seed42_run1/"
    "scjepa-epoch=25-val_loss=0.179.ckpt"
)
FORMAL_PANEL_COMMAND = """cd /mnt/c/SH/AIVC/GeneJEPA-main
set -o pipefail

.venv/bin/python \\
perturbation_scripts/tahoe_decoder_v1_data.py build-panel \\
  --top-k 5000 \\
  --output results/genejepa_decoder_v1_gene_panel.csv \\
  --summary results/genejepa_decoder_v1_gene_panel.json \\
  --state results/genejepa_decoder_v1_gene_panel_builder_state.npz \\
  --progress results/genejepa_decoder_v1_gene_panel_builder_progress.json \\
2>&1 | tee results/genejepa_decoder_v1_gene_panel_builder_console.log"""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_files(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required files are missing: {missing}")


def run_lookup_smoke(cells: int) -> dict[str, Any]:
    if cells < 1:
        raise ValueError("--cells must be positive")
    require_files(
        SMOKE_PANEL,
        FORMAL_EMBEDDINGS,
        FORMAL_EMBEDDING_MANIFEST,
        CONDITION_INDEX,
        GENE_METADATA,
        *FORMAL_PLANS,
    )
    dataset = TahoeDecoderV1PairedDataset(
        split="train",
        panel_path=SMOKE_PANEL,
        shuffle_shards=False,
        max_cells=cells,
    )
    samples = list(dataset)
    if len(samples) != cells:
        raise AssertionError(f"Expected {cells} smoke cells, observed {len(samples)}")

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[(sample["shard_path"], sample["row_group_index"])].append(sample)
    raw_rows: dict[int, dict[str, Any]] = {}
    for (shard_path, row_group), group in grouped.items():
        table = pq.ParquetFile(PROJECT_ROOT / shard_path).read_row_group(
            row_group, columns=RAW_COLUMNS
        )
        for sample in group:
            row = sample["row_index_in_row_group"]
            raw_rows[sample["global_embedding_index"]] = {
                column: table[column][row].as_py() for column in RAW_COLUMNS
            }

    cache = np.load(FORMAL_EMBEDDINGS, mmap_mode="r")
    max_abs_error = 0.0
    old_new_exact = True
    dataset_new_exact = True
    latent_unchanged = True
    metadata_unchanged = True
    split_unchanged = True
    target_finite = True
    target_nonnegative = True
    condition_splits = dataset.conditions.set_index("pair_id")["split"]
    metadata_fields = (
        "drug",
        "sample",
        "BARCODE_SUB_LIB_ID",
        "cell_line_id",
        "plate",
    )
    for sample in samples:
        raw = raw_rows[sample["global_embedding_index"]]
        old_target, _ = log1p_cp10k_target(
            raw["genes"], raw["expressions"], dataset.token_lookup, dataset.panel_indices
        )
        new_target, _ = log1p_cp10k_target(
            raw["genes"],
            raw["expressions"],
            dataset.token_lookup,
            dataset.panel_indices,
            panel_lookup=dataset.panel_lookup,
        )
        max_abs_error = max(
            max_abs_error, float(np.max(np.abs(old_target - new_target)))
        )
        old_new_exact &= np.array_equal(old_target, new_target)
        dataset_new_exact &= np.array_equal(
            sample["target_expression"].numpy(), new_target
        )
        latent_unchanged &= np.array_equal(
            sample["latent"].numpy(), np.asarray(cache[sample["global_embedding_index"]])
        )
        metadata_unchanged &= all(str(sample[field]) == str(raw[field]) for field in metadata_fields)
        split_unchanged &= condition_splits.loc[sample["pair_id"]] == "train"
        target_finite &= bool(np.isfinite(new_target).all())
        target_nonnegative &= bool(np.all(new_target >= 0))

    expected_positions = np.arange(len(dataset.panel_indices), dtype=np.int32)
    lookup_valid = (
        dataset.panel_lookup.shape == (EXPECTED_GENE_UNIVERSE,)
        and dataset.panel_lookup.dtype == np.int32
        and np.array_equal(dataset.panel_lookup[dataset.panel_indices], expected_positions)
        and np.count_nonzero(dataset.panel_lookup >= 0) == len(dataset.panel_indices)
    )
    checks = {
        "old_per_cell_lookup_target_equals_new_reused_lookup_target": bool(old_new_exact),
        "max_abs_error": max_abs_error,
        "max_abs_error_tolerance": 1e-7,
        "dataset_target_equals_new_target": bool(dataset_new_exact),
        "latent_unchanged": bool(latent_unchanged),
        "metadata_unchanged": bool(metadata_unchanged),
        "split_unchanged": bool(split_unchanged),
        "target_finite": bool(target_finite),
        "target_nonnegative": bool(target_nonnegative),
        "panel_lookup_valid": bool(lookup_valid),
    }
    status = "pass" if (
        checks["old_per_cell_lookup_target_equals_new_reused_lookup_target"]
        and checks["max_abs_error"] <= checks["max_abs_error_tolerance"]
        and checks["dataset_target_equals_new_target"]
        and checks["latent_unchanged"]
        and checks["metadata_unchanged"]
        and checks["split_unchanged"]
        and checks["target_finite"]
        and checks["target_nonnegative"]
        and checks["panel_lookup_valid"]
    ) else "fail"
    result = {
        "schema": "genejepa_decoder_v1_panel_lookup_equivalence_v1",
        "created_at_utc": utc_now(),
        "status": status,
        "cells": cells,
        "split": "train",
        "smoke_panel": {
            "path": display_path(SMOKE_PANEL),
            "sha256": sha256_file(SMOKE_PANEL),
            "genes": len(dataset.panel_indices),
        },
        "lookup": {
            "construction": "once per TahoeDecoderV1PairedDataset instance/worker",
            "shape": list(dataset.panel_lookup.shape),
            "dtype": str(dataset.panel_lookup.dtype),
            "default": -1,
        },
        "checks": checks,
        "dataset_code": {
            "path": display_path(DATASET_CODE),
            "sha256": sha256_file(DATASET_CODE),
        },
        "model_training_run": False,
    }
    if status != "pass":
        raise AssertionError(f"Panel lookup equivalence smoke failed: {checks}")
    atomic_write_json(LOOKUP_SMOKE, result)
    return result


def base_contract() -> dict[str, Any]:
    require_files(
        TASK,
        READINESS_AUDIT,
        FORMAL_EMBEDDING_MANIFEST,
        CONDITION_SPLIT,
        EDGE_SPLIT,
        CONDITION_INDEX,
        GENE_METADATA,
        CHECKPOINT,
        DATASET_CODE,
        LOOKUP_SMOKE,
    )
    readiness = read_json(READINESS_AUDIT)
    cache_manifest = read_json(FORMAL_EMBEDDING_MANIFEST)
    lookup_smoke = read_json(LOOKUP_SMOKE)
    if readiness.get("status") != "pass" or lookup_smoke.get("status") != "pass":
        raise AssertionError("Readiness or lookup-equivalence evidence is not PASS")
    if lookup_smoke["dataset_code"]["sha256"] != sha256_file(DATASET_CODE):
        raise AssertionError("Dataset code changed after lookup-equivalence smoke")
    if readiness["split_leakage"]["treated_unique_cells_by_split"] != EXPECTED_SPLIT_CELLS:
        raise AssertionError("Frozen treated split counts changed")
    checkpoint = readiness["latent_contract"]["checkpoint"]
    if checkpoint["path"] != display_path(CHECKPOINT):
        raise AssertionError("Readiness checkpoint path changed")
    output = cache_manifest["output"]
    integrity = cache_manifest["integrity"]
    if (
        cache_manifest.get("status") != "pass"
        or output["shape"] != [30_839_089, 768]
        or output["dtype"] != "float32"
        or output.get("row_equals_global_embedding_index") is not True
        or integrity != {
            "missing": 0,
            "duplicate": 0,
            "finite": True,
            "cells": 30_839_089,
            "worker_overlap": 0,
        }
    ):
        raise AssertionError("Formal latent cache manifest no longer satisfies its contract")

    return {
        "schema": "genejepa_decoder_v1_scientific_contract_v1",
        "status": "frozen_scientific_contract_panel_pending",
        "created_at_utc": utc_now(),
        "scientific_decisions_frozen": True,
        "encoder": {
            "model": "GeneJEPA",
            "checkpoint": checkpoint,
            "branch": "Epoch25 EMA Teacher get_embedding(use_teacher=True)",
            "frozen": True,
        },
        "decoder_input": {
            "unit": "one physical Tahoe cell",
            "shape": [768],
            "dtype": "float32",
            "space": "raw signed GeneJEPA latent",
            "transforms": {
                "centering": False,
                "whitening": False,
                "l2_normalization": False,
                "clipping": False,
                "relu": False,
                "latent_standardization": False,
            },
        },
        "gene_universe": {
            "genes": EXPECTED_GENE_UNIVERSE,
            "genejepa_index_range": [0, EXPECTED_GENE_UNIVERSE - 1],
            "ordering": "ascending token_id",
            "metadata_path": display_path(GENE_METADATA),
            "metadata_sha256": sha256_file(GENE_METADATA),
        },
        "expression_target": {
            "name": "log1p(CP10000)",
            "construction_order": [
                "remove leading negative sentinel",
                "map all usable genes to GeneJEPA vocabulary",
                "sum counts over all mapped GeneJEPA genes for L_i",
                "CP10000 = 10000 * count / L_i",
                "log1p",
                "select Decoder gene panel",
            ],
            "denominator": "all mapped GeneJEPA genes after sentinel removal",
            "output_dtype": "float32",
            "output_range": "finite and non-negative",
        },
        "panel": {
            "status": "pending_exact_materialization",
            "path": display_path(FORMAL_PANEL),
            "sha256": None,
            "rows": None,
            "candidate_universe": EXPECTED_GENE_UNIVERSE,
            "statistics_scope": "Experiment 1 train treated physical cells only",
            "DMSO_included": False,
            "validation_split_included": False,
            "test_split_included": False,
            "expression_space": "log1p(CP10000)",
            "implicit_zeros_included": True,
            "variance": "population, ddof=0",
            "ranking": "variance descending",
            "tie_breaker": "genejepa_index ascending",
            "top_k": EXPECTED_PANEL_SIZE,
            "exact_materialization_command": FORMAL_PANEL_COMMAND,
        },
        "split": {
            "ownership": "frozen Experiment 1 physical treated-cell edge split",
            "train_treated_cells": EXPECTED_SPLIT_CELLS["train"],
            "val_treated_cells": EXPECTED_SPLIT_CELLS["val"],
            "test_treated_cells": EXPECTED_SPLIT_CELLS["test"],
            "DMSO_used_for_decoder_fitting": False,
            "condition_split_path": display_path(CONDITION_SPLIT),
            "condition_split_sha256": sha256_file(CONDITION_SPLIT),
            "edge_split_path": display_path(EDGE_SPLIT),
            "edge_split_sha256": sha256_file(EDGE_SPLIT),
            "train_val_test_physical_cell_overlap": {
                "train_val": 0,
                "train_test": 0,
                "val_test": 0,
            },
        },
        "architecture": {
            "dimensions": [768, 1024, 1024, 512, EXPECTED_PANEL_SIZE],
            "hidden_block": "Linear -> LayerNorm -> GELU -> Dropout(0.1)",
            "residual_decoder": False,
            "output_activation": "Softplus",
            "parameter_count": 4_931_976,
        },
        "training_loss": {
            "name": "MSE",
            "prediction_dtype": "float32",
            "target_dtype": "float32",
            "reduction_dtype": "float32",
            "expression_space": "log1p(CP10000)",
        },
        "validation_and_checkpoint": {
            "criterion": "exact full-validation global cell-by-gene MSE",
            "definition": "global SSE over all validation cells and 5000 genes divided by global element count",
            "DDP_reduction": "sum global SSE numerator and element-count denominator; do not average per-rank means",
            "best_checkpoint": "strict minimum full-validation MSE",
            "exact_tie": "keep earlier epoch",
        },
        "test_policy": {
            "timing": "only after training is complete",
            "may_influence_panel_or_model_selection": False,
            "prohibited_influences": [
                "panel",
                "target",
                "architecture",
                "optimizer/hyperparameters",
                "early stopping",
                "checkpoint selection",
            ],
        },
        "dataset": {
            "class": "TahoeDecoderV1PairedDataset",
            "implementation": display_path(DATASET_CODE),
            "implementation_sha256": sha256_file(DATASET_CODE),
            "panel_lookup": "[62710] int32, default -1, constructed once per Dataset/worker",
            "latent_lookup": "mmap[global_embedding_index]",
            "rerun_genejepa": False,
        },
        "provenance": {
            "task": {"path": str(TASK), "sha256": sha256_file(TASK)},
            "readiness_audit": {
                "path": display_path(READINESS_AUDIT),
                "sha256": sha256_file(READINESS_AUDIT),
                "status": "pass",
            },
            "lookup_equivalence_smoke": {
                "path": display_path(LOOKUP_SMOKE),
                "sha256": sha256_file(LOOKUP_SMOKE),
                "status": "pass",
            },
            "latent_cache_manifest": {
                "path": display_path(FORMAL_EMBEDDING_MANIFEST),
                "sha256": sha256_file(FORMAL_EMBEDDING_MANIFEST),
                "cache_path": output["path"],
                "cache_sha256": output["sha256"],
                "cache_shape": output["shape"],
                "cache_dtype": output["dtype"],
            },
            "documentation_waiver": {
                "path": str(DECLARED_V21_DOCUMENT),
                "present_in_local_workspace": DECLARED_V21_DOCUMENT.is_file(),
                "waived": True,
                "is_blocker": False,
                "reason": "documentation-location issue; current checkout code and frozen PASS manifests/results are the implementation evidence",
            },
        },
        "expected_train_treated_cells": EXPECTED_SPLIT_CELLS["train"],
        "expected_val_treated_cells": EXPECTED_SPLIT_CELLS["val"],
        "expected_test_treated_cells": EXPECTED_SPLIT_CELLS["test"],
        "expected_panel_size": EXPECTED_PANEL_SIZE,
        "blockers": [],
        "pending_artifacts": ["exact train-only top-5000 panel and its integrity audit"],
        "ready_for_contract_freeze": False,
        "contract_frozen": False,
        "ready_for_exact_config_smoke": False,
        "ready_for_formal_training": False,
    }


def prepare_contract(cells: int) -> dict[str, Any]:
    smoke = run_lookup_smoke(cells)
    contract = base_contract()
    atomic_write_json(CONTRACT, contract)
    return {
        "status": "pass",
        "lookup_smoke": display_path(LOOKUP_SMOKE),
        "lookup_smoke_sha256": sha256_file(LOOKUP_SMOKE),
        "lookup_cells": smoke["cells"],
        "lookup_max_abs_error": smoke["checks"]["max_abs_error"],
        "contract": display_path(CONTRACT),
        "contract_status": contract["status"],
        "formal_panel_pending": True,
        "model_training_run": False,
    }


def audit_formal_panel() -> dict[str, Any]:
    require_files(FORMAL_PANEL, FORMAL_PANEL_SUMMARY, FORMAL_PANEL_STATE, GENE_METADATA)
    summary = read_json(FORMAL_PANEL_SUMMARY)
    expected_summary = {
        "status": "formal_complete",
        "formal_panel": True,
        "processed_train_treated_cells": EXPECTED_SPLIT_CELLS["train"],
        "expected_train_treated_cells": EXPECTED_SPLIT_CELLS["train"],
        "candidate_universe": EXPECTED_GENE_UNIVERSE,
        "top_k": EXPECTED_PANEL_SIZE,
        "statistics_scope": "Decoder train treated physical cells only",
        "expression_space": "full mapped counts -> CP10000 -> log1p",
        "zeros_included": True,
        "variance": "population, ddof=0",
        "ranking": "population variance descending",
        "tie_breaker": "genejepa_index ascending",
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise AssertionError(
                f"Formal panel summary mismatch for {key}: {summary.get(key)!r} != {expected!r}"
            )
    audit_counts = summary.get("audit", {})
    required_audit_keys = {
        "sentinel_cells",
        "unmapped_entries",
        "noninteger_count_entries",
        "duplicate_gene_entries_collapsed",
        "invalid_or_zero_library_cells",
    }
    if set(audit_counts) != required_audit_keys:
        raise AssertionError("Formal panel summary audit fields are incomplete")
    if any(not isinstance(audit_counts[key], int) or audit_counts[key] < 0 for key in audit_counts):
        raise AssertionError("Formal target integrity counts must be non-negative integers")
    if audit_counts["invalid_or_zero_library_cells"] != 0:
        raise AssertionError("Formal panel scan encountered invalid/zero-library cells")

    genes, _ = load_gene_universe(GENE_METADATA)
    indices = load_panel(FORMAL_PANEL, genes)
    panel = pd.read_csv(FORMAL_PANEL, keep_default_na=False, encoding="utf-8-sig")
    if len(panel) != EXPECTED_PANEL_SIZE:
        raise AssertionError("Formal panel does not contain exactly 5000 rows")
    required_columns = {
        "panel_rank",
        "genejepa_index",
        "token_id",
        "gene_symbol",
        "ensembl_id",
        "mean_log1p_cp10k",
        "population_variance_log1p_cp10k",
    }
    if set(panel.columns) != required_columns:
        raise AssertionError(f"Unexpected formal panel columns: {list(panel.columns)}")
    means = panel["mean_log1p_cp10k"].to_numpy(np.float64)
    variances = panel["population_variance_log1p_cp10k"].to_numpy(np.float64)
    if not np.isfinite(means).all() or not np.isfinite(variances).all() or np.any(variances < 0):
        raise AssertionError("Formal panel mean/variance values are invalid")

    with np.load(FORMAL_PANEL_STATE, allow_pickle=False) as state:
        fingerprint = str(state["fingerprint"].item())
        cells = int(state["cells"])
        next_descriptor = int(state["next_descriptor"])
        value_sum = state["value_sum"].astype(np.float64, copy=False)
        value_sumsq = state["value_sumsq"].astype(np.float64, copy=False)
        state_audit = state["audit_counts"].astype(np.int64, copy=False)
    if cells != EXPECTED_SPLIT_CELLS["train"]:
        raise AssertionError("Panel state cell count is not the exact train-treated count")
    if next_descriptor != len(plan_descriptors(FORMAL_PLANS)):
        raise AssertionError("Panel state does not cover every frozen plan descriptor")
    if value_sum.shape != (EXPECTED_GENE_UNIVERSE,) or value_sumsq.shape != (
        EXPECTED_GENE_UNIVERSE,
    ):
        raise AssertionError("Panel state accumulator shapes are invalid")
    state_audit_dict = dict(
        zip(
            (
                "sentinel_cells",
                "unmapped_entries",
                "noninteger_count_entries",
                "duplicate_gene_entries_collapsed",
                "invalid_or_zero_library_cells",
            ),
            (int(value) for value in state_audit),
            strict=True,
        )
    )
    if state_audit_dict != audit_counts:
        raise AssertionError("Panel summary target-integrity counts disagree with resume state")

    all_means = value_sum / cells
    all_variances = np.maximum(value_sumsq / cells - all_means * all_means, 0.0)
    expected_indices = np.lexsort(
        (np.arange(EXPECTED_GENE_UNIVERSE, dtype=np.int64), -all_variances)
    )[:EXPECTED_PANEL_SIZE]
    ranking_exact = np.array_equal(indices, expected_indices)
    means_exact = np.array_equal(means, all_means[expected_indices])
    variances_exact = np.array_equal(variances, all_variances[expected_indices])
    mean_max_abs_error = float(np.max(np.abs(means - all_means[expected_indices])))
    variance_max_abs_error = float(
        np.max(np.abs(variances - all_variances[expected_indices]))
    )
    csv_float64_roundtrip_atol = float(4 * np.finfo(np.float64).eps)
    means_within_roundtrip_tolerance = mean_max_abs_error <= csv_float64_roundtrip_atol
    variances_within_roundtrip_tolerance = (
        variance_max_abs_error <= csv_float64_roundtrip_atol
    )
    if (
        not ranking_exact
        or not means_within_roundtrip_tolerance
        or not variances_within_roundtrip_tolerance
    ):
        raise AssertionError("Formal panel does not exactly reproduce its frozen accumulators/ranking")

    panel_sha = sha256_file(FORMAL_PANEL)
    builder_sha = sha256_file(DATASET_CODE)
    if summary["output"] != {
        "path": display_path(FORMAL_PANEL),
        "sha256": panel_sha,
        "encoding": "utf-8-sig",
    }:
        raise AssertionError("Formal panel file provenance disagrees with its summary")
    if summary["builder"]["sha256"] != builder_sha:
        raise AssertionError("Formal panel builder code changed after the scan")
    if summary["builder"]["fingerprint"] != fingerprint:
        raise AssertionError("Formal panel summary and resume-state fingerprints disagree")

    result = {
        "schema": "genejepa_decoder_v1_formal_gene_panel_audit_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "panel": {
            "path": display_path(FORMAL_PANEL),
            "sha256": panel_sha,
            "rows": len(panel),
            "panel_rank_range": [int(panel["panel_rank"].min()), int(panel["panel_rank"].max())],
            "genejepa_index_unique": int(panel["genejepa_index"].nunique()),
            "genejepa_index_range": [int(indices.min()), int(indices.max())],
            "metadata_exact": True,
            "variance_finite_nonnegative": True,
            "ranking_exact_and_deterministic": ranking_exact,
            "means_exact_to_state": means_exact,
            "variances_exact_to_state": variances_exact,
            "csv_float64_roundtrip_atol": csv_float64_roundtrip_atol,
            "mean_max_abs_error_to_state": mean_max_abs_error,
            "variance_max_abs_error_to_state": variance_max_abs_error,
            "means_within_roundtrip_tolerance": means_within_roundtrip_tolerance,
            "variances_within_roundtrip_tolerance": variances_within_roundtrip_tolerance,
            "maximum_population_variance": float(variances[0]),
            "minimum_selected_population_variance": float(variances[-1]),
        },
        "statistics": {key: summary[key] for key in expected_summary if key not in {"status"}},
        "target_integrity_counts": audit_counts,
        "builder": {
            "path": display_path(DATASET_CODE),
            "sha256": builder_sha,
            "fingerprint": fingerprint,
            "state": display_path(FORMAL_PANEL_STATE),
            "state_sha256": sha256_file(FORMAL_PANEL_STATE),
            "completed_plan_descriptors": next_descriptor,
        },
    }
    atomic_write_json(FORMAL_PANEL_AUDIT, result)
    return result


def finalize_contract() -> dict[str, Any]:
    panel_audit = audit_formal_panel()
    contract = base_contract()
    prior = read_json(CONTRACT) if CONTRACT.is_file() else None
    if prior is not None:
        contract["created_at_utc"] = prior.get("created_at_utc", contract["created_at_utc"])
    contract["updated_at_utc"] = utc_now()
    contract["status"] = "frozen"
    contract["panel"] = {
        "status": "formal_materialized_and_audited",
        "path": panel_audit["panel"]["path"],
        "sha256": panel_audit["panel"]["sha256"],
        "rows": panel_audit["panel"]["rows"],
        "builder_path": panel_audit["builder"]["path"],
        "builder_sha256": panel_audit["builder"]["sha256"],
        "builder_fingerprint": panel_audit["builder"]["fingerprint"],
        "statistics_scope": "Experiment 1 train treated physical cells only",
        "processed_train_treated_cells": EXPECTED_SPLIT_CELLS["train"],
        "candidate_universe": EXPECTED_GENE_UNIVERSE,
        "expression_space": "log1p(CP10000)",
        "implicit_zeros_included": True,
        "variance": "population, ddof=0",
        "ranking": "variance descending",
        "tie_breaker": "genejepa_index ascending",
        "top_k": EXPECTED_PANEL_SIZE,
        "integrity_audit_path": display_path(FORMAL_PANEL_AUDIT),
        "integrity_audit_sha256": sha256_file(FORMAL_PANEL_AUDIT),
    }
    contract["target_integrity_counts"] = panel_audit["target_integrity_counts"]
    contract["pending_artifacts"] = []
    contract["ready_for_contract_freeze"] = True
    contract["contract_frozen"] = True
    contract["ready_for_exact_config_smoke"] = True
    contract["ready_for_formal_training"] = False
    atomic_write_json(CONTRACT, contract)
    return {
        "status": "pass",
        "contract": display_path(CONTRACT),
        "contract_sha256": sha256_file(CONTRACT),
        "contract_status": contract["status"],
        "panel": panel_audit["panel"],
        "target_integrity_counts": panel_audit["target_integrity_counts"],
        "panel_audit": display_path(FORMAL_PANEL_AUDIT),
        "ready_for_exact_config_smoke": True,
        "ready_for_formal_training": False,
        "model_training_run": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--cells", type=int, default=16)
    subparsers.add_parser("finalize")
    return result


def main() -> None:
    args = parser().parse_args()
    result = prepare_contract(args.cells) if args.command == "prepare" else finalize_contract()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
