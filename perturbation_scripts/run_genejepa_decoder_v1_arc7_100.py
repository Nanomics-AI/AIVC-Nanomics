#!/usr/bin/env python3
"""Build and evaluate one-context ST-A -> Decoder predictions with cell-eval 0.8.2."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from evaluate_tahoe_experiment1_st_raw import load_checkpoint as load_st_checkpoint
from run_genejepa_decoder_v1_demo import (
    EXPECTED_DECODER_EPOCH,
    EXPECTED_PANEL_SHA256,
    EXPECTED_ST_EPOCH,
    OWNER_CONTROL,
    artifact,
    atomic_write_json,
    load_decoder,
    locate_cells,
    one_side_index_sha256,
)
from run_tahoe_experiment1_st_formal import (
    ensure_protocol as ensure_st_protocol,
    model_fingerprint as st_fingerprint,
)
from tahoe_decoder_v1_data import (
    GENE_METADATA,
    OWNER_TREATED,
    PROJECT_ROOT,
    RAW_COLUMNS,
    RESULTS,
    build_panel_lookup,
    display_path,
    load_conditions,
    load_gene_universe,
    load_panel,
    log1p_cp10k_target,
    sha256_file,
    utc_now,
)
from tahoe_experiment1_latent_data import (
    LATENT_DIM,
    PERT_DIM,
    SET_SIZE,
    TahoeExperiment1LatentSetDataset,
)


SCRIPT_PATH = Path(__file__).resolve()
TASK_PATH = PROJECT_ROOT.parent / "当前任务.txt"
PANEL_PATH = RESULTS / "genejepa_decoder_v1_gene_panel.csv"
CONDITIONS_PATH = RESULTS / "genejepa_decoder_v1_arc7_100_conditions.csv"
PLAN_PATH = RESULTS / "genejepa_decoder_v1_arc7_100_plan.json"
REAL_PATH = RESULTS / "genejepa_decoder_v1_arc7_100_real.h5ad"
PRED_PATH = RESULTS / "genejepa_decoder_v1_arc7_100_pred.h5ad"
BUILD_MANIFEST_PATH = RESULTS / "genejepa_decoder_v1_arc7_100_build_manifest.json"
CELL_EVAL_OUTDIR = RESULTS / "genejepa_decoder_v1_arc7_100_cell_eval"
SUMMARY_PATH = RESULTS / "genejepa_decoder_v1_arc7_100_summary.csv"
PER_CONDITION_PATH = RESULTS / "genejepa_decoder_v1_arc7_100_per_condition.csv"
RESULT_PATH = RESULTS / "genejepa_decoder_v1_arc7_100_result.json"
REPORT_PATH = RESULTS / "genejepa_decoder_v1_arc7_100_report.md"
SMOKE_PATH = RESULTS / "genejepa_decoder_v1_arc7_100_smoke.json"

CONTROL_PERT = "__CONTROL__"
PERT_COL = "perturbation"
TARGET_CONDITIONS = 100
EXPECTED_CELL_EVAL_VERSION = "0.8.2"
EXPECTED_CELL_EVAL_COMMIT = "6928cf8bd7a706040ccfd13119e4085726dee64a"
EVALUATION_LABEL = (
    "ARC VCC 2025 Generalist metrics adapted to the frozen GeneJEPA "
    "5000-gene expression space"
)
DISCLAIMER = (
    "This is an ARC-style adapted evaluation on Tahoe drug perturbations and the "
    "frozen GeneJEPA 5000-gene panel. It is not an official ARC leaderboard evaluation."
)

METRICS = (
    {
        "label": "DES",
        "cell_eval_name": "overlap_at_N",
        "column": "DES",
        "direction": "↑",
        "scope": "per_condition_mean",
    },
    {
        "label": "PDS",
        "cell_eval_name": "discrimination_score_l1",
        "column": "PDS",
        "direction": "↑",
        "scope": "per_condition_mean",
    },
    {
        "label": "MAE",
        "cell_eval_name": "mae",
        "column": "MAE",
        "direction": "↓",
        "scope": "per_condition_mean",
    },
    {
        "label": "Pearson delta",
        "cell_eval_name": "pearson_delta",
        "column": "Pearson_delta",
        "direction": "↑",
        "scope": "per_condition_mean",
    },
    {
        "label": "Spearman logFC",
        "cell_eval_name": "de_spearman_lfc_sig",
        "column": "Spearman_logFC",
        "direction": "↑",
        "scope": "per_condition_mean",
    },
    {
        "label": "AUPRC",
        "cell_eval_name": "pr_auc",
        "column": "AUPRC",
        "direction": "↑",
        "scope": "per_condition_mean",
    },
    {
        "label": "Spearman effect size",
        "cell_eval_name": "de_spearman_sig",
        "column": "Spearman_effect_size",
        "direction": "↑",
        "scope": "global_across_conditions",
    },
)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
        float_format="%.10g",
        na_rep="NaN",
    )
    os.replace(temporary, path)


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def perturbation_key(row: pd.Series) -> str:
    dose = format(float(row["dose_uM"]), ".15g")
    return f"drug={row['drug']}|dose_uM={dose}|pair_id={row['pair_id']}"


def selection_fingerprint(
    selected: pd.DataFrame, control_indices: np.ndarray, treated_indices: dict[str, np.ndarray]
) -> str:
    digest = hashlib.sha256()
    for row in selected.itertuples(index=False):
        pair_id = str(row.pair_id)
        digest.update(pair_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(treated_indices[pair_id], dtype="<i8").tobytes())
    digest.update(b"\0shared-control\0")
    digest.update(np.asarray(control_indices, dtype="<i8").tobytes())
    return digest.hexdigest()


def select_context(
    limit: int = TARGET_CONDITIONS,
) -> tuple[TahoeExperiment1LatentSetDataset, pd.DataFrame, dict[str, Any]]:
    dataset = TahoeExperiment1LatentSetDataset(split="test", seed=42, epoch=0)
    eligible = dataset.conditions.loc[
        dataset.conditions["split"].eq("test")
        & (dataset.conditions["treated_cached_cell_count"] >= SET_SIZE)
        & (dataset.conditions["control_cached_cell_count"] >= SET_SIZE)
    ].copy()
    groups = (
        eligible.groupby(["cell_line_id", "control_pool_id"], sort=False)
        .agg(
            eligible_conditions=("pair_id", "size"),
            min_cache_condition_index=("cache_condition_index", "min"),
            plate_count=("plate", "nunique"),
            drug_count=("drug", "nunique"),
        )
        .reset_index()
    )
    qualifying = groups.loc[groups["eligible_conditions"] >= TARGET_CONDITIONS].sort_values(
        ["min_cache_condition_index", "cell_line_id", "control_pool_id"], kind="stable"
    )
    if len(qualifying):
        chosen = qualifying.iloc[0]
        mode = "first context with >=100 conditions by minimum cache_condition_index"
    else:
        chosen = groups.sort_values(
            ["eligible_conditions", "min_cache_condition_index", "cell_line_id", "control_pool_id"],
            ascending=[False, True, True, True],
            kind="stable",
        ).iloc[0]
        mode = "fallback: largest single context; ties by minimum cache_condition_index then IDs"

    selected = eligible.loc[
        eligible["cell_line_id"].eq(chosen["cell_line_id"])
        & eligible["control_pool_id"].eq(chosen["control_pool_id"])
    ].sort_values("cache_condition_index", kind="stable")
    selected = selected.head(min(limit, TARGET_CONDITIONS)).reset_index(drop=True)
    full = load_conditions()[["pair_id", "cache_condition_index", "treated_samples"]]
    selected = selected.merge(
        full,
        on=["pair_id", "cache_condition_index"],
        how="left",
        validate="one_to_one",
    )
    selected.insert(0, "perturbation", selected.apply(perturbation_key, axis=1))

    if selected.empty or selected["perturbation"].duplicated().any():
        raise AssertionError("Selected perturbation identities are empty or non-unique")
    if selected["cell_line_id"].nunique() != 1 or selected["control_pool_id"].nunique() != 1:
        raise AssertionError("Selected conditions cross biological contexts")
    if selected["plate"].nunique() != 1:
        raise AssertionError("A single control_pool_id unexpectedly crosses plates")
    if not selected["split"].eq("test").all():
        raise AssertionError("A non-test condition entered the ARC-style selection")

    context = {
        "selection_rule": mode,
        "requested_conditions": TARGET_CONDITIONS,
        "actual_available_in_context": int(chosen["eligible_conditions"]),
        "selected_conditions": len(selected),
        "contexts_with_at_least_100": int(len(qualifying)),
        "eligible_test_conditions": len(eligible),
        "eligible_contexts": len(groups),
        "cell_line_id": str(chosen["cell_line_id"]),
        "control_pool_id": str(chosen["control_pool_id"]),
        "plate": str(selected.iloc[0]["plate"]),
        "control_drug": str(selected.iloc[0]["control_drug"]),
        "control_samples": str(selected.iloc[0]["control_samples"]),
        "control_cached_cell_count": int(selected.iloc[0]["control_cached_cell_count"]),
        "drug_count": int(selected["drug"].nunique()),
        "dose_values_uM": sorted(selected["dose_uM"].astype(float).unique().tolist()),
        "cache_condition_index_min": int(selected["cache_condition_index"].min()),
        "cache_condition_index_max": int(selected["cache_condition_index"].max()),
    }
    return dataset, selected, context


def make_sampling_plan(
    dataset: TahoeExperiment1LatentSetDataset, selected: pd.DataFrame
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    first = selected.iloc[0]
    control_indices = dataset._sample_range(
        int(first["control_embedding_start"]),
        int(first["control_embedding_stop_exclusive"]),
        str(first["pair_id"]),
        "control",
    )
    treated_indices = {
        str(row["pair_id"]): dataset._sample_range(
            int(row["treated_embedding_start"]),
            int(row["treated_embedding_stop_exclusive"]),
            str(row["pair_id"]),
            "treated",
        )
        for _, row in selected.iterrows()
    }
    if len(np.unique(control_indices)) != SET_SIZE:
        raise AssertionError("Shared control set contains a repeated cell")
    all_treated = np.concatenate(list(treated_indices.values()))
    if len(np.unique(all_treated)) != len(all_treated):
        raise AssertionError("Treated conditions unexpectedly share a global embedding index")
    if np.intersect1d(control_indices, all_treated).size:
        raise AssertionError("Shared control and treated sets overlap")

    independent = TahoeExperiment1LatentSetDataset(split="test", seed=42, epoch=0)
    check_control = independent._sample_range(
        int(first["control_embedding_start"]),
        int(first["control_embedding_stop_exclusive"]),
        str(first["pair_id"]),
        "control",
    )
    if not np.array_equal(control_indices, check_control):
        raise AssertionError("Independent Dataset reconstruction changed shared control indices")
    for _, row in selected.iterrows():
        pair_id = str(row["pair_id"])
        check = independent._sample_range(
            int(row["treated_embedding_start"]),
            int(row["treated_embedding_stop_exclusive"]),
            pair_id,
            "treated",
        )
        if not np.array_equal(treated_indices[pair_id], check):
            raise AssertionError(f"Independent treated sampling changed for {pair_id}")
    del independent

    audit = {
        "S": SET_SIZE,
        "base_seed": 42,
        "repeat_epoch": 0,
        "replace": False,
        "dataset_sampling_version": "tahoe_experiment1_set_v1",
        "shared_control_sampling_identity_pair_id": str(first["pair_id"]),
        "shared_control_embedding_indices": control_indices.tolist(),
        "shared_control_embedding_indices_sha256": one_side_index_sha256(control_indices),
        "treated_embedding_indices_sha256": {
            pair_id: one_side_index_sha256(indices)
            for pair_id, indices in treated_indices.items()
        },
        "independent_reconstruction_exact": True,
        "per_side_sha256_encoding": "little-endian int64 bytes in sampled order",
    }
    return control_indices, treated_indices, audit


def plan_payload(
    selected: pd.DataFrame,
    context: dict[str, Any],
    control_indices: np.ndarray,
    treated_indices: dict[str, np.ndarray],
    sampling: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = selected.copy()
    output["treated_embedding_indices_sha256"] = output["pair_id"].map(
        sampling["treated_embedding_indices_sha256"]
    )
    output["treated_sampled_cell_count"] = SET_SIZE
    output["shared_control_sampled_cell_count"] = SET_SIZE
    output["shared_control_embedding_indices_sha256"] = sampling[
        "shared_control_embedding_indices_sha256"
    ]
    keep = [
        "perturbation",
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
        "treated_sampled_cell_count",
        "treated_embedding_indices_sha256",
        "control_pool_id",
        "control_drug",
        "control_samples",
        "control_cached_cell_count",
        "shared_control_sampled_cell_count",
        "shared_control_embedding_indices_sha256",
    ]
    output = output[keep]
    payload = {
        "schema": "genejepa_decoder_v1_arc7_100_plan_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "evaluation_label": EVALUATION_LABEL,
        "disclaimer": DISCLAIMER,
        "context": context,
        "sampling": sampling,
        "selection_fingerprint": selection_fingerprint(
            selected, control_indices, treated_indices
        ),
        "conditions": {
            "path": display_path(CONDITIONS_PATH),
            "rows": len(output),
        },
        "formal_inference_started": False,
        "cell_eval_started": False,
    }
    return output, payload


def write_plan() -> tuple[
    TahoeExperiment1LatentSetDataset,
    pd.DataFrame,
    dict[str, Any],
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, Any],
]:
    dataset, selected, context = select_context()
    control_indices, treated_indices, sampling = make_sampling_plan(dataset, selected)
    conditions_output, payload = plan_payload(
        selected, context, control_indices, treated_indices, sampling
    )
    atomic_write_csv(CONDITIONS_PATH, conditions_output)
    payload["conditions"]["sha256"] = sha256_file(CONDITIONS_PATH)
    payload["conditions"]["size_bytes"] = CONDITIONS_PATH.stat().st_size
    atomic_write_json(PLAN_PATH, payload)
    return dataset, selected, context, control_indices, treated_indices, payload


def load_panel_contract() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    if sha256_file(PANEL_PATH) != EXPECTED_PANEL_SHA256:
        raise AssertionError("Frozen 5000-gene panel SHA-256 changed")
    genes, token_lookup = load_gene_universe(GENE_METADATA)
    panel_indices = load_panel(PANEL_PATH, genes)
    panel_lookup = build_panel_lookup(panel_indices, vocabulary_size=len(genes))
    panel = pd.read_csv(PANEL_PATH, encoding="utf-8-sig", keep_default_na=False).sort_values(
        "panel_rank", kind="stable"
    ).reset_index(drop=True)
    if (
        len(panel) != 5000
        or panel["ensembl_id"].eq("").any()
        or panel["ensembl_id"].duplicated().any()
    ):
        raise AssertionError("Frozen panel Ensembl IDs are not 5000 stable unique var_names")
    return panel, panel_indices, token_lookup, panel_lookup


def build_requests(
    selected: pd.DataFrame,
    control_indices: np.ndarray,
    treated_indices: dict[str, np.ndarray],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    first = selected.iloc[0]
    for offset, embedding_index in enumerate(control_indices):
        records.append(
            {
                "embedding_index": int(embedding_index),
                "matrix_row": offset,
                "side": "control",
                "owner_type_expected": OWNER_CONTROL,
                "owner_index_expected": int(first["control_pool_index"]),
                "plate_expected": str(first["plate"]),
                "cell_line_id_expected": str(first["cell_line_id"]),
                "drug_expected": str(first["control_drug"]),
                "samples_expected": str(first["control_samples"]),
            }
        )
    for condition_offset, row in selected.iterrows():
        pair_id = str(row["pair_id"])
        matrix_start = SET_SIZE + condition_offset * SET_SIZE
        for cell_offset, embedding_index in enumerate(treated_indices[pair_id]):
            records.append(
                {
                    "embedding_index": int(embedding_index),
                    "matrix_row": matrix_start + cell_offset,
                    "side": "treated",
                    "owner_type_expected": OWNER_TREATED,
                    "owner_index_expected": int(row["cache_condition_index"]),
                    "plate_expected": str(row["plate"]),
                    "cell_line_id_expected": str(row["cell_line_id"]),
                    "drug_expected": str(row["drug"]),
                    "samples_expected": str(row["treated_samples"]),
                }
            )
    requests = pd.DataFrame.from_records(records)
    if requests["embedding_index"].duplicated().any() or requests["matrix_row"].duplicated().any():
        raise AssertionError("Expression request plan contains duplicate cells or output rows")
    return requests


def load_true_expression_matrix(
    selected: pd.DataFrame,
    control_indices: np.ndarray,
    treated_indices: dict[str, np.ndarray],
    panel_indices: np.ndarray,
    token_lookup: np.ndarray,
    panel_lookup: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    requests = build_requests(selected, control_indices, treated_indices)
    locators = locate_cells(requests["embedding_index"].to_numpy(np.int64))
    joined = locators.merge(requests, on="embedding_index", how="inner", validate="one_to_one")
    if len(joined) != len(requests):
        raise AssertionError("Stable locator join skipped a requested cell")
    if not joined["owner_type"].astype(int).eq(joined["owner_type_expected"]).all() or not joined[
        "owner_index"
    ].astype(int).eq(joined["owner_index_expected"]).all():
        raise AssertionError("Stable locator ownership differs from condition/control ownership")

    matrix = np.empty((len(requests), len(panel_indices)), dtype=np.float32)
    filled = np.zeros(len(requests), dtype=bool)
    sources: dict[str, pq.ParquetFile] = {}
    row_group_starts: dict[str, np.ndarray] = {}
    audits: list[dict[str, Any]] = []
    groups = list(joined.groupby(["shard_path", "row_group_index"], sort=False))
    started = time.perf_counter()

    for group_number, ((shard_path, row_group_index), group) in enumerate(groups, start=1):
        shard_path = str(shard_path)
        source = sources.get(shard_path)
        if source is None:
            source = pq.ParquetFile(PROJECT_ROOT / shard_path)
            sources[shard_path] = source
            counts = np.asarray(
                [source.metadata.row_group(index).num_rows for index in range(source.num_row_groups)],
                dtype=np.int64,
            )
            row_group_starts[shard_path] = np.concatenate(
                (np.zeros(1, dtype=np.int64), np.cumsum(counts[:-1]))
            )
        row_group_index = int(row_group_index)
        raw = source.read_row_group(row_group_index, columns=RAW_COLUMNS)
        for locator in group.itertuples(index=False):
            row = int(locator.row_index_in_row_group)
            matrix_row = int(locator.matrix_row)
            if row < 0 or row >= raw.num_rows:
                raise AssertionError("row_index_in_row_group is out of range")
            if int(row_group_starts[shard_path][row_group_index]) + row != int(
                locator.row_index_in_shard
            ):
                raise AssertionError("Stable row locator components disagree")
            metadata = {name: raw[name][row].as_py() for name in RAW_COLUMNS[2:]}
            expected_samples = set(str(locator.samples_expected).split("|"))
            if (
                str(metadata["plate"]) != str(locator.plate_expected)
                or str(metadata["cell_line_id"]) != str(locator.cell_line_id_expected)
                or str(metadata["drug"]) != str(locator.drug_expected)
                or str(metadata["sample"]) not in expected_samples
            ):
                raise AssertionError("Raw Tahoe metadata disagrees with stable locator ownership")
            target, audit = log1p_cp10k_target(
                raw["genes"][row].as_py(),
                raw["expressions"][row].as_py(),
                token_lookup,
                panel_indices,
                panel_lookup=panel_lookup,
            )
            matrix[matrix_row] = target
            filled[matrix_row] = True
            audits.append(audit)
        if group_number % 25 == 0 or group_number == len(groups):
            print(
                f"true-expression row-groups={group_number}/{len(groups)} "
                f"cells={int(filled.sum())}/{len(filled)}",
                flush=True,
            )

    if not filled.all() or not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise AssertionError("True expression matrix is incomplete, non-finite, or negative")
    return matrix, {
        "cells": len(matrix),
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "finite": True,
        "nonnegative": True,
        "source_shards": len(sources),
        "source_row_groups": len(groups),
        "sentinel_removed_cells": sum(bool(row["sentinel_removed"]) for row in audits),
        "unmapped_entries": sum(int(row["unmapped_entries"]) for row in audits),
        "duplicate_gene_entries_collapsed": sum(
            int(row["duplicate_gene_entries_collapsed"]) for row in audits
        ),
        "target_transform": "full mapped counts -> CP10000 -> log1p -> frozen panel",
        "elapsed_seconds": time.perf_counter() - started,
    }


def predict_expression_matrix(
    dataset: TahoeExperiment1LatentSetDataset,
    selected: pd.DataFrame,
    control_indices: np.ndarray,
    shared_control_expression: np.ndarray,
    condition_batch_size: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("ST-A -> Decoder inference requires a visible CUDA GPU")
    if condition_batch_size < 1:
        raise ValueError("condition_batch_size must be positive")
    device = torch.device("cuda:0")
    control_latent = np.ascontiguousarray(dataset.embeddings[control_indices], dtype=np.float32)
    if control_latent.shape != (SET_SIZE, LATENT_DIM) or not np.isfinite(control_latent).all():
        raise AssertionError("Shared control latent is invalid")

    st_protocol, st_protocol_sha = ensure_st_protocol(create=False)
    st_model, st_checkpoint = load_st_checkpoint("st-a", device, st_protocol_sha)
    decoder, decoder_checkpoint = load_decoder(device)
    if (
        st_checkpoint["epoch"] != EXPECTED_ST_EPOCH
        or st_checkpoint["best_epoch"] != EXPECTED_ST_EPOCH
        or st_model.predict_residual
        or st_model.final_activation_name != "identity"
        or st_model.apply_output_relu
        or decoder_checkpoint["epoch"] != EXPECTED_DECODER_EPOCH
    ):
        raise AssertionError("Frozen ST-A/Decoder contract changed")

    predicted = np.empty((SET_SIZE * (len(selected) + 1), 5000), dtype=np.float32)
    predicted[:SET_SIZE] = shared_control_expression
    torch.cuda.reset_peak_memory_stats(device)
    st_min = float("inf")
    st_max = float("-inf")
    st_negative = 0
    st_coordinates = 0
    started = time.perf_counter()

    with torch.inference_mode():
        for start in range(0, len(selected), condition_batch_size):
            batch_rows = selected.iloc[start : start + condition_batch_size]
            batch_size = len(batch_rows)
            control = torch.from_numpy(control_latent).to(device).unsqueeze(0).expand(
                batch_size, -1, -1
            )
            vectors = np.stack(
                [
                    dataset.featurizer.encode(str(row["drug"]), float(row["dose_uM"]))
                    for _, row in batch_rows.iterrows()
                ]
            ).astype(np.float32, copy=False)
            perturbation = (
                torch.from_numpy(vectors).to(device).unsqueeze(1).expand(-1, SET_SIZE, -1)
            )
            if tuple(perturbation.shape) != (batch_size, SET_SIZE, PERT_DIM):
                raise AssertionError("Perturbation tensor shape changed")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                latent = st_model(
                    {"ctrl_cell_emb": control, "pert_emb": perturbation}
                ).reshape(batch_size, SET_SIZE, LATENT_DIM)
                expression = decoder(latent.reshape(-1, LATENT_DIM)).reshape(
                    batch_size, SET_SIZE, 5000
                )
            latent32 = latent.float()
            expression32 = expression.float()
            if not torch.isfinite(latent32).all() or not torch.isfinite(expression32).all():
                raise AssertionError("Non-finite ST-A or Decoder output")
            if torch.any(expression32 < 0):
                raise AssertionError("Decoder Softplus output became negative")
            st_min = min(st_min, float(latent32.min()))
            st_max = max(st_max, float(latent32.max()))
            st_negative += int((latent32 < 0).sum())
            st_coordinates += latent32.numel()
            for local in range(batch_size):
                condition_offset = start + local
                output_start = SET_SIZE + condition_offset * SET_SIZE
                predicted[output_start : output_start + SET_SIZE] = (
                    expression32[local].cpu().numpy()
                )
            print(
                f"inference conditions={start + batch_size}/{len(selected)}",
                flush=True,
            )
    torch.cuda.synchronize(device)
    if not st_min < 0 < st_max:
        raise AssertionError("ST-A output did not preserve signed latent coordinates")
    if not np.isfinite(predicted).all() or np.any(predicted < 0):
        raise AssertionError("Predicted expression matrix is invalid")
    if st_fingerprint(st_model) != st_checkpoint["model_state_sha256_before_evaluation"]:
        raise AssertionError("ST-A parameters changed during inference")

    inference = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "visible_cuda_devices": torch.cuda.device_count(),
        "condition_batch_size": condition_batch_size,
        "eval_mode": not st_model.training and not decoder.training,
        "torch_inference_mode": True,
        "autocast": "bfloat16",
        "prediction_shape": list(predicted.shape),
        "prediction_dtype": str(predicted.dtype),
        "finite": True,
        "nonnegative": True,
        "st_output_per_condition_shape": [1, SET_SIZE, LATENT_DIM],
        "decoder_output_per_condition_shape": [SET_SIZE, 5000],
        "st_latent_min": st_min,
        "st_latent_max": st_max,
        "st_latent_negative_ratio": st_negative / st_coordinates,
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    checkpoints = {
        "st_a": st_checkpoint,
        "decoder": decoder_checkpoint,
        "st_training_protocol_sha256": st_protocol_sha,
        "st_training_protocol_schema": st_protocol.get("schema"),
    }
    del st_model, decoder
    torch.cuda.empty_cache()
    return predicted, inference, checkpoints


def make_obs(selected: pd.DataFrame, *, predicted: bool) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    first = selected.iloc[0]
    prefix = "pred" if predicted else "real"
    for cell_index in range(SET_SIZE):
        records.append(
            {
                "obs_name": f"{prefix}_control_{cell_index:03d}",
                "perturbation": CONTROL_PERT,
                "pair_id": "",
                "cache_condition_index": -1,
                "drug": str(first["control_drug"]),
                "dose_uM": np.nan,
                "cell_line_id": str(first["cell_line_id"]),
                "plate": str(first["plate"]),
                "control_pool_id": str(first["control_pool_id"]),
                "set_cell_index": cell_index,
            }
        )
    for _, row in selected.iterrows():
        for cell_index in range(SET_SIZE):
            records.append(
                {
                    "obs_name": f"{prefix}_{row['pair_id']}_{cell_index:03d}",
                    "perturbation": str(row["perturbation"]),
                    "pair_id": str(row["pair_id"]),
                    "cache_condition_index": int(row["cache_condition_index"]),
                    "drug": str(row["drug"]),
                    "dose_uM": float(row["dose_uM"]),
                    "cell_line_id": str(row["cell_line_id"]),
                    "plate": str(row["plate"]),
                    "control_pool_id": str(row["control_pool_id"]),
                    "set_cell_index": cell_index,
                }
            )
    obs = pd.DataFrame.from_records(records).set_index("obs_name", verify_integrity=True)
    return obs


def write_h5ad(
    path: Path,
    matrix: np.ndarray,
    obs: pd.DataFrame,
    panel: pd.DataFrame,
    selected_count: int,
) -> None:
    if matrix.dtype != np.float32 or matrix.shape != (SET_SIZE * (selected_count + 1), 5000):
        raise AssertionError("AnnData matrix shape/dtype is invalid")
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise AssertionError("AnnData matrix is non-finite or negative")
    var = panel[["gene_symbol", "genejepa_index", "panel_rank"]].copy()
    var.index = pd.Index(panel["ensembl_id"].astype(str), name="ensembl_id")
    if var.index.duplicated().any() or len(var) != 5000:
        raise AssertionError("AnnData var_names are not 5000 unique Ensembl IDs")
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.uns["evaluation_label"] = EVALUATION_LABEL
    adata.uns["disclaimer"] = DISCLAIMER
    adata.uns["expression_space"] = "log1p(CP10000)"
    adata.uns["panel_sha256"] = EXPECTED_PANEL_SHA256
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    adata.write_h5ad(temporary, compression="lzf")
    del adata
    os.replace(temporary, path)


def verify_h5ad_pair(real_path: Path, pred_path: Path, selected: pd.DataFrame) -> dict[str, Any]:
    expected_shape = (SET_SIZE * (len(selected) + 1), 5000)
    real = ad.read_h5ad(real_path, backed="r")
    pred = ad.read_h5ad(pred_path, backed="r")
    try:
        if real.shape != expected_shape or pred.shape != expected_shape:
            raise AssertionError("Written AnnData shape changed")
        if real.X.dtype != np.float32 or pred.X.dtype != np.float32:
            raise AssertionError("Written AnnData X is not float32")
        if not np.array_equal(real.var_names.to_numpy(), pred.var_names.to_numpy()):
            raise AssertionError("Real/pred gene order differs")
        expected_perts = {CONTROL_PERT, *selected["perturbation"].astype(str).tolist()}
        real_perts = set(real.obs[PERT_COL].astype(str))
        pred_perts = set(pred.obs[PERT_COL].astype(str))
        if real_perts != expected_perts or pred_perts != expected_perts:
            raise AssertionError("Real/pred perturbation identities differ from the frozen plan")
        if int((real.obs[PERT_COL] == CONTROL_PERT).sum()) != SET_SIZE or int(
            (pred.obs[PERT_COL] == CONTROL_PERT).sum()
        ) != SET_SIZE:
            raise AssertionError("Real/pred AnnData does not contain the shared 256-cell control")
    finally:
        real.file.close()
        pred.file.close()
    return {
        "real_shape": list(expected_shape),
        "pred_shape": list(expected_shape),
        "X_dtype": "float32",
        "gene_order_identical": True,
        "perturbation_identities_identical": True,
        "shared_control_cells_each": SET_SIZE,
        "var_names": "unique ensembl_id",
    }


def build_pair(
    selected: pd.DataFrame,
    dataset: TahoeExperiment1LatentSetDataset,
    control_indices: np.ndarray,
    treated_indices: dict[str, np.ndarray],
    real_path: Path,
    pred_path: Path,
    condition_batch_size: int,
) -> dict[str, Any]:
    panel, panel_indices, token_lookup, panel_lookup = load_panel_contract()
    total_started = time.perf_counter()
    true_expression, target_audit = load_true_expression_matrix(
        selected,
        control_indices,
        treated_indices,
        panel_indices,
        token_lookup,
        panel_lookup,
    )
    shared_control_expression = true_expression[:SET_SIZE].copy()
    real_obs = make_obs(selected, predicted=False)
    write_h5ad(real_path, true_expression, real_obs, panel, len(selected))
    del true_expression, real_obs
    gc.collect()

    predicted_expression, inference, checkpoints = predict_expression_matrix(
        dataset,
        selected,
        control_indices,
        shared_control_expression,
        condition_batch_size,
    )
    if not np.array_equal(predicted_expression[:SET_SIZE], shared_control_expression):
        raise AssertionError("Predicted AnnData control differs from shared true control")
    pred_obs = make_obs(selected, predicted=True)
    write_h5ad(pred_path, predicted_expression, pred_obs, panel, len(selected))
    del predicted_expression, pred_obs, shared_control_expression
    gc.collect()
    pair_audit = verify_h5ad_pair(real_path, pred_path, selected)
    return {
        "status": "pass",
        "conditions": len(selected),
        "cells_per_condition": SET_SIZE,
        "total_cells_per_anndata": SET_SIZE * (len(selected) + 1),
        "genes": 5000,
        "expression_space": "log1p(CP10000)",
        "true_expression": target_audit,
        "inference": inference,
        "checkpoints": checkpoints,
        "anndata_pair_audit": pair_audit,
        "elapsed_seconds": time.perf_counter() - total_started,
    }


def cell_eval_provenance() -> dict[str, Any]:
    import cell_eval

    version = importlib.metadata.version("cell-eval")
    if version != EXPECTED_CELL_EVAL_VERSION:
        raise AssertionError(
            f"Formal evaluation requires cell-eval {EXPECTED_CELL_EVAL_VERSION}, found {version}"
        )
    root = Path(cell_eval.__file__).resolve().parent
    source_files = [
        root / "_evaluator.py",
        root / "_pipeline/_runner.py",
        root / "metrics/_anndata.py",
        root / "metrics/_de.py",
        root / "metrics/_impl.py",
    ]
    return {
        "version": version,
        "official_release_commit": EXPECTED_CELL_EVAL_COMMIT,
        "module": str(Path(cell_eval.__file__).resolve()),
        "installed_distribution_direct_url": None,
        "source_sha256": {path.name: sha256_file(path) for path in source_files},
        "pdex_version": importlib.metadata.version("pdex"),
        "anndata_version": importlib.metadata.version("anndata"),
        "polars_version": importlib.metadata.version("polars"),
    }


def available_threads() -> int:
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return min(16, available)


def run_cell_eval(
    real_path: Path,
    pred_path: Path,
    selected: pd.DataFrame,
    outdir: Path,
    threads: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    from cell_eval import MetricPipeline, MetricsEvaluator

    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty cell-eval output directory: {outdir}")
    provenance = cell_eval_provenance()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    started = time.perf_counter()
    evaluator = MetricsEvaluator(
        adata_pred=str(pred_path),
        adata_real=str(real_path),
        control_pert=CONTROL_PERT,
        pert_col=PERT_COL,
        num_threads=threads,
        outdir=str(outdir),
        allow_discrete=False,
        pdex_kwargs={"is_log1p": True, "epsilon": 0.0},
    )
    if evaluator.de_comparison is None:
        raise AssertionError("cell-eval did not construct its official DE comparison")

    base = selected[["perturbation", "pair_id", "drug", "dose_uM"]].copy()
    failures: dict[str, str] = {}
    official_frames: list[pd.DataFrame] = []
    per_condition = base.copy()

    for spec in METRICS:
        name = str(spec["cell_eval_name"])
        configs = (
            {name: {"exclude_target_gene": False}}
            if name == "discrimination_score_l1"
            else None
        )
        pipeline = MetricPipeline(profile=None, metric_configs=configs, break_on_error=True)
        pipeline.add_metrics([name])
        try:
            pipeline.compute_de_metrics(evaluator.de_comparison)
            pipeline.compute_anndata_metrics(evaluator.anndata_pair)
            frame = pipeline.get_results().to_pandas()
            if frame.empty or name not in frame.columns:
                raise RuntimeError("official metric returned no result column")
            frame = frame[["perturbation", name]].copy()
            official_frames.append(frame.rename(columns={name: str(spec["column"])}))
            per_condition = per_condition.merge(
                frame.rename(columns={name: str(spec["column"])}),
                on="perturbation",
                how="left",
                validate="one_to_one",
            )
        except Exception as error:  # preserve other metrics and record the exact failure
            failures[name] = f"{type(error).__name__}: {error}"
            per_condition[str(spec["column"])] = np.nan
            logging.exception("Official cell-eval metric failed: %s", name)

    real_de = evaluator.de_comparison.real
    pred_de = evaluator.de_comparison.pred
    real_infinite_lfc = int(real_de.data[real_de.log2_fold_change_col].is_infinite().sum())
    pred_infinite_lfc = int(pred_de.data[pred_de.log2_fold_change_col].is_infinite().sum())
    per_condition["true_DEG_count"] = [
        int(real_de.get_significant_genes(key, 0.05).size)
        for key in per_condition["perturbation"].astype(str)
    ]
    per_condition["pred_DEG_count"] = [
        int(pred_de.get_significant_genes(key, 0.05).size)
        for key in per_condition["perturbation"].astype(str)
    ]

    summary_rows: list[dict[str, Any]] = []
    for spec in METRICS:
        column = str(spec["column"])
        values = pd.to_numeric(per_condition[column], errors="coerce").to_numpy(np.float64)
        finite = values[np.isfinite(values)]
        if spec["scope"] == "global_across_conditions":
            unique = np.unique(finite)
            if len(unique) > 1:
                raise AssertionError("cell-eval global metric was not constant across broadcast rows")
            value = float(unique[0]) if len(unique) else np.nan
            valid_n = int(len(unique) == 1)
            nan_n = int(len(unique) == 0)
            mean = value
            median = value
        else:
            value = float(finite.mean()) if len(finite) else np.nan
            mean = value
            median = float(np.median(finite)) if len(finite) else np.nan
            valid_n = int(len(finite))
            nan_n = int(len(values) - len(finite))
        summary_rows.append(
            {
                "metric": spec["label"],
                "cell_eval_name": spec["cell_eval_name"],
                "value": value,
                "mean": mean,
                "median": median,
                "direction": spec["direction"],
                "valid_n": valid_n,
                "nan_n": nan_n,
                "scope": spec["scope"],
            }
        )
    summary = pd.DataFrame.from_records(summary_rows)
    official = base.copy()
    for frame in official_frames:
        official = official.merge(frame, on="perturbation", how="left", validate="one_to_one")
    runtime = {
        "threads": threads,
        "elapsed_seconds": time.perf_counter() - started,
        "metric_failures": failures,
        "de": {
            "fdr_threshold": 0.05,
            "real_rows": int(real_de.data.height),
            "pred_rows": int(pred_de.data.height),
            "real_infinite_log2_fold_change_rows": real_infinite_lfc,
            "pred_infinite_log2_fold_change_rows": pred_infinite_lfc,
            "epsilon": 0.0,
            "infinite_lfc_note": (
                "Retained exactly as produced by official cell-eval/pdex; not clipped or replaced"
            ),
            "real_path": display_path(outdir / "real_de.csv"),
            "pred_path": display_path(outdir / "pred_de.csv"),
        },
        "cell_eval": provenance,
        "metric_orchestration": (
            "MetricsEvaluator builds official AnnData pair and DE; MetricPipeline/registry runs "
            "only the seven requested metrics"
        ),
        "discrimination_score_l1_exclude_target_gene": False,
    }
    return summary, per_condition, official, runtime


def command_plan() -> dict[str, Any]:
    _, selected, context, _, _, payload = write_plan()
    print(
        json.dumps(
            {
                "status": "pass",
                "context": context,
                "conditions": len(selected),
                "conditions_csv": display_path(CONDITIONS_PATH),
                "plan": display_path(PLAN_PATH),
                "formal_inference_started": False,
                "cell_eval_started": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def command_build(condition_batch_size: int = 4) -> dict[str, Any]:
    if REAL_PATH.exists() or PRED_PATH.exists() or BUILD_MANIFEST_PATH.exists():
        raise FileExistsError(
            "Formal build output already exists; refusing to overwrite it. Audit or move it first."
        )
    dataset, selected, context, control_indices, treated_indices, plan = write_plan()
    build = build_pair(
        selected,
        dataset,
        control_indices,
        treated_indices,
        REAL_PATH,
        PRED_PATH,
        condition_batch_size,
    )
    manifest = {
        "schema": "genejepa_decoder_v1_arc7_100_build_manifest_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "evaluation_label": EVALUATION_LABEL,
        "disclaimer": DISCLAIMER,
        "context": context,
        "selection_fingerprint": plan["selection_fingerprint"],
        "sampling": plan["sampling"],
        "build": build,
        "outputs": {"real_h5ad": artifact(REAL_PATH), "pred_h5ad": artifact(PRED_PATH)},
        "provenance": {
            "script": artifact(SCRIPT_PATH),
            "task": artifact(TASK_PATH),
            "plan": artifact(PLAN_PATH),
            "conditions": artifact(CONDITIONS_PATH),
            "panel": artifact(PANEL_PATH),
        },
        "cell_eval_started": False,
        "training_performed": False,
        "genejepa_rerun": False,
    }
    atomic_write_json(BUILD_MANIFEST_PATH, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def summary_for_json(summary: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in summary.to_dict("records"):
        records.append(
            {
                **row,
                "value": finite_or_none(row["value"]),
                "mean": finite_or_none(row["mean"]),
                "median": finite_or_none(row["median"]),
                "valid_n": int(row["valid_n"]),
                "nan_n": int(row["nan_n"]),
            }
        )
    return records


def write_report(summary: pd.DataFrame, context: dict[str, Any], runtime_seconds: float) -> None:
    lines = [
        "# GeneJEPA ST-A → Decoder: ARC-style 7-metric quick evaluation",
        "",
        f"**{DISCLAIMER}**",
        "",
        f"- Frozen test conditions: {context['selected_conditions']}",
        f"- Biological context: `{context['cell_line_id']}` / `{context['plate']}` / `{context['control_pool_id']}`",
        f"- Shared control: {SET_SIZE} cells",
        f"- Treated cells per condition: {SET_SIZE}",
        "- Genes: 5000",
        "- Expression space: `log1p(CP10000)`",
        f"- Runtime: {runtime_seconds:.1f} seconds",
        "",
        "| Metric | cell-eval name | Value | Median | Direction | Valid N | NaN N | Scope |",
        "|---|---|---:|---:|:---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        value = "NaN" if not math.isfinite(float(row.value)) else f"{float(row.value):.6g}"
        median = "NaN" if not math.isfinite(float(row.median)) else f"{float(row.median):.6g}"
        lines.append(
            f"| {row.metric} | `{row.cell_eval_name}` | {value} | {median} | {row.direction} | "
            f"{row.valid_n} | {row.nan_n} | {row.scope} |"
        )
    lines.extend(
        [
            "",
            "No overall ARC score is reported: a Generalist score is a rank across competing models,",
            "which is not defined for this single-model adapted diagnostic.",
            "",
        ]
    )
    temporary = REPORT_PATH.with_name(REPORT_PATH.name + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    os.replace(temporary, REPORT_PATH)


def command_evaluate(threads: int | None = None) -> dict[str, Any]:
    if not BUILD_MANIFEST_PATH.is_file() or not REAL_PATH.is_file() or not PRED_PATH.is_file():
        raise FileNotFoundError("Run the formal build command before evaluation")
    if any(path.exists() for path in (SUMMARY_PATH, PER_CONDITION_PATH, RESULT_PATH, REPORT_PATH)):
        raise FileExistsError("Formal evaluation outputs already exist; refusing to overwrite them")
    build_manifest = json.loads(BUILD_MANIFEST_PATH.read_text(encoding="utf-8"))
    if build_manifest.get("status") != "pass":
        raise AssertionError("Formal build manifest is not PASS")
    if sha256_file(REAL_PATH) != build_manifest["outputs"]["real_h5ad"]["sha256"] or sha256_file(
        PRED_PATH
    ) != build_manifest["outputs"]["pred_h5ad"]["sha256"]:
        raise AssertionError("Formal AnnData SHA-256 differs from the build manifest")

    _, selected, context = select_context()
    if len(selected) != int(build_manifest["build"]["conditions"]):
        raise AssertionError("Current deterministic selection differs from the built AnnData")
    verify_h5ad_pair(REAL_PATH, PRED_PATH, selected)
    threads = available_threads() if threads is None else min(threads, available_threads())
    started = time.perf_counter()
    summary, per_condition, official, runtime = run_cell_eval(
        REAL_PATH, PRED_PATH, selected, CELL_EVAL_OUTDIR, threads
    )
    atomic_write_csv(SUMMARY_PATH, summary)
    atomic_write_csv(PER_CONDITION_PATH, per_condition)
    official_path = CELL_EVAL_OUTDIR / "requested_seven_metrics.csv"
    atomic_write_csv(official_path, official)
    elapsed = time.perf_counter() - started
    write_report(summary, context, elapsed)
    failures = runtime["metric_failures"]
    result = {
        "schema": "genejepa_decoder_v1_arc7_100_result_v1",
        "created_at_utc": utc_now(),
        "status": "pass" if not failures else "fail",
        "evaluation_label": EVALUATION_LABEL,
        "disclaimer": DISCLAIMER,
        "condition_selection": context,
        "cell_count_per_anndata": SET_SIZE * (len(selected) + 1),
        "condition_count": len(selected),
        "cells_per_condition": SET_SIZE,
        "shared_control_cells": SET_SIZE,
        "gene_count": 5000,
        "expression_space": "log1p(CP10000)",
        "provenance": {
            "GeneJEPA": "Epoch25 EMA Teacher frozen embeddings; no rerun",
            "st_a": build_manifest["build"]["checkpoints"]["st_a"],
            "decoder": build_manifest["build"]["checkpoints"]["decoder"],
            "panel": artifact(PANEL_PATH),
            "cell_eval": runtime["cell_eval"],
            "build_manifest": artifact(BUILD_MANIFEST_PATH),
            "script": artifact(SCRIPT_PATH),
            "task": artifact(TASK_PATH),
        },
        "seven_metrics": summary_for_json(summary),
        "undefined_or_failed": {
            "metric_failures": failures,
            "nan_counts": {
                str(row.metric): int(row.nan_n) for row in summary.itertuples(index=False)
            },
        },
        "runtime": {**runtime, "total_evaluation_seconds": elapsed},
        "outputs": {
            "summary": artifact(SUMMARY_PATH),
            "per_condition": artifact(PER_CONDITION_PATH),
            "report": artifact(REPORT_PATH),
            "conditions": artifact(CONDITIONS_PATH),
            "real_h5ad": build_manifest["outputs"]["real_h5ad"],
            "pred_h5ad": build_manifest["outputs"]["pred_h5ad"],
            "official_requested_metrics": artifact(official_path),
            "real_de": artifact(CELL_EVAL_OUTDIR / "real_de.csv"),
            "pred_de": artifact(CELL_EVAL_OUTDIR / "pred_de.csv"),
            "result_json": {"path": display_path(RESULT_PATH)},
        },
        "overall_arc_score_computed": False,
        "training_performed": False,
        "model_optimization_performed": False,
    }
    atomic_write_json(RESULT_PATH, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def command_smoke(conditions: int = 2, condition_batch_size: int = 2) -> dict[str, Any]:
    if conditions < 2:
        raise ValueError("Smoke requires at least two perturbations for the PDS path")
    dataset, selected_all, context = select_context()
    selected = selected_all.head(conditions).reset_index(drop=True)
    context = {**context, "selected_conditions": len(selected)}
    control_indices, treated_indices, sampling = make_sampling_plan(dataset, selected)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="genejepa_arc7_smoke_") as temporary:
        root = Path(temporary)
        real_path = root / "real.h5ad"
        pred_path = root / "pred.h5ad"
        build = build_pair(
            selected,
            dataset,
            control_indices,
            treated_indices,
            real_path,
            pred_path,
            condition_batch_size,
        )
        summary, per_condition, _, runtime = run_cell_eval(
            real_path,
            pred_path,
            selected,
            root / "cell_eval",
            min(4, available_threads()),
        )
        if len(summary) != 7 or len(per_condition) != len(selected):
            raise AssertionError("Smoke did not return seven summary rows and all conditions")
        result = {
            "schema": "genejepa_decoder_v1_arc7_100_smoke_v1",
            "created_at_utc": utc_now(),
            "status": "pass" if not runtime["metric_failures"] else "fail",
            "scope": "reduced real-data plumbing smoke only; not a scientific result",
            "conditions": len(selected),
            "pair_ids": selected["pair_id"].astype(str).tolist(),
            "context": context,
            "sampling": sampling,
            "build": build,
            "cell_eval": {
                "provenance": runtime["cell_eval"],
                "metric_failures": runtime["metric_failures"],
                "seven_rows_returned": len(summary),
                "per_condition_rows": len(per_condition),
                "summary": summary_for_json(summary),
            },
            "temporary_h5ad_deleted_after_validation": True,
            "formal_build_started": False,
            "runtime_seconds": time.perf_counter() - started,
        }
    atomic_write_json(SMOKE_PATH, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="Freeze deterministic context/condition/index plan only")
    smoke = subparsers.add_parser("smoke", help="Run a reduced real-data model + cell-eval smoke")
    smoke.add_argument("--conditions", type=int, default=2)
    smoke.add_argument("--condition-batch-size", type=int, default=2)
    build = subparsers.add_parser("build", help="Run formal inference and write the two AnnData files")
    build.add_argument("--condition-batch-size", type=int, default=4)
    evaluate = subparsers.add_parser("evaluate", help="Run official DE and the seven requested metrics")
    evaluate.add_argument("--threads", type=int, default=None)
    run = subparsers.add_parser("run", help="Run formal build followed by formal evaluation")
    run.add_argument("--condition-batch-size", type=int, default=4)
    run.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    match args.command:
        case "plan":
            command_plan()
        case "smoke":
            command_smoke(args.conditions, args.condition_batch_size)
        case "build":
            command_build(args.condition_batch_size)
        case "evaluate":
            command_evaluate(args.threads)
        case "run":
            command_build(args.condition_batch_size)
            command_evaluate(args.threads)


if __name__ == "__main__":
    main()
