#!/usr/bin/env python3
"""Build and compare ARC7 variants A/B/C on the frozen 23-condition cell sets.

Variant A is read from the existing formal result. Variant B reuses A's treated
predictions byte-for-byte and replaces only the predicted control with the
Decoder output. Variant C decodes the true cached control/treated latents.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch

import run_genejepa_decoder_v1_arc7_100 as arc7
from run_genejepa_decoder_v1 import model_fingerprint as decoder_fingerprint
from run_genejepa_decoder_v1_demo import (
    EXPECTED_DECODER_EPOCH,
    artifact,
    atomic_write_json,
    load_decoder,
    one_side_index_sha256,
)
from tahoe_decoder_v1_data import PROJECT_ROOT, RESULTS, display_path, sha256_file, utc_now
from tahoe_experiment1_latent_data import LATENT_DIM, SET_SIZE


SCRIPT_PATH = Path(__file__).resolve()
TASK_PATH = PROJECT_ROOT.parent / "当前任务.txt"

BUILD_RESULT_PATH = RESULTS / "genejepa_arc7_decoded_control_build_result.json"
B_RESULT_PATH = RESULTS / "genejepa_arc7_variant_b_result.json"
C_RESULT_PATH = RESULTS / "genejepa_arc7_variant_c_result.json"
ABC_SUMMARY_PATH = RESULTS / "genejepa_arc7_abc_summary.csv"
ABC_PER_CONDITION_PATH = RESULTS / "genejepa_arc7_abc_per_condition.csv"
ABC_REPORT_PATH = RESULTS / "genejepa_arc7_abc_comparison.md"
B_PRED_PATH = RESULTS / "genejepa_arc7_variant_b_pred.h5ad"
C_PRED_PATH = RESULTS / "genejepa_arc7_variant_c_pred.h5ad"
B_CELL_EVAL_DIR = RESULTS / "genejepa_arc7_variant_b_cell_eval"
C_CELL_EVAL_DIR = RESULTS / "genejepa_arc7_variant_c_cell_eval"
SMOKE_PATH = RESULTS / "genejepa_arc7_decoded_control_smoke.json"

VARIANTS = {
    "A": "A_real_control",
    "B": "B_decoded_control",
    "C": "C_decoder_ceiling",
}
CONDITION_COLUMNS = (
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
    "control_pool_id",
    "control_drug",
    "control_samples",
)
PER_CONDITION_METRICS = (
    "DES",
    "PDS",
    "MAE",
    "Pearson_delta",
    "Spearman_logFC",
    "AUPRC",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_artifact(path: Path, record: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen {label}: {path}")
    if record.get("path") != display_path(path):
        raise AssertionError(f"Frozen {label} path changed")
    if record.get("sha256") != sha256_file(path):
        raise AssertionError(f"Frozen {label} SHA-256 changed")
    if int(record.get("size_bytes", -1)) != path.stat().st_size:
        raise AssertionError(f"Frozen {label} size changed")


def assert_condition_table(selected: pd.DataFrame, frozen: pd.DataFrame) -> None:
    if len(selected) != 23 or len(frozen) != 23:
        raise AssertionError("The frozen evaluation no longer contains exactly 23 conditions")
    for column in CONDITION_COLUMNS:
        if column == "dose_uM":
            left = pd.to_numeric(selected[column]).to_numpy(np.float64)
            right = pd.to_numeric(frozen[column]).to_numpy(np.float64)
            equal = np.array_equal(left, right)
        elif column == "cache_condition_index":
            left = pd.to_numeric(selected[column]).to_numpy(np.int64)
            right = pd.to_numeric(frozen[column]).to_numpy(np.int64)
            equal = np.array_equal(left, right)
        else:
            equal = selected[column].astype(str).tolist() == frozen[column].astype(str).tolist()
        if not equal:
            raise AssertionError(f"Frozen condition order/content changed in column {column}")


def assert_obs_equal(actual: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    if actual.index.astype(str).tolist() != expected.index.astype(str).tolist():
        raise AssertionError(f"{label} obs_names/order changed")
    if actual.columns.tolist() != expected.columns.tolist():
        raise AssertionError(f"{label} obs schema changed")
    for column in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[column]):
            left = pd.to_numeric(actual[column], errors="coerce").to_numpy(np.float64)
            right = pd.to_numeric(expected[column], errors="coerce").to_numpy(np.float64)
            equal = np.array_equal(left, right, equal_nan=True)
        else:
            equal = actual[column].astype(str).tolist() == expected[column].astype(str).tolist()
        if not equal:
            raise AssertionError(f"{label} obs column changed: {column}")


def audit_original_h5ad(selected: pd.DataFrame, panel: pd.DataFrame) -> dict[str, Any]:
    expected_shape = (SET_SIZE * (len(selected) + 1), len(panel))
    expected_genes = panel["ensembl_id"].astype(str).to_numpy()
    expected_real_obs = arc7.make_obs(selected, predicted=False)
    expected_pred_obs = arc7.make_obs(selected, predicted=True)
    observed: dict[str, Any] = {}
    for label, path, expected_obs in (
        ("real", arc7.REAL_PATH, expected_real_obs),
        ("A_pred", arc7.PRED_PATH, expected_pred_obs),
    ):
        data = ad.read_h5ad(path, backed="r")
        try:
            if data.shape != expected_shape or data.X.dtype != np.float32:
                raise AssertionError(f"Frozen {label} AnnData shape/dtype changed")
            if not np.array_equal(data.var_names.astype(str).to_numpy(), expected_genes):
                raise AssertionError(f"Frozen {label} gene order changed")
            assert_obs_equal(data.obs, expected_obs, label)
            observed[label] = {
                "shape": list(data.shape),
                "dtype": str(data.X.dtype),
                "obs_schema_exact": True,
                "obs_order_exact": True,
                "gene_order_exact": True,
            }
        finally:
            data.file.close()
    return observed


def audit_locator_alignment(
    selected: pd.DataFrame,
    control_indices: np.ndarray,
    treated_indices: dict[str, np.ndarray],
    old_build: dict[str, Any],
) -> dict[str, Any]:
    requests = arc7.build_requests(selected, control_indices, treated_indices)
    locators = arc7.locate_cells(requests["embedding_index"].to_numpy(np.int64))
    joined = locators.merge(requests, on="embedding_index", how="inner", validate="one_to_one")
    if len(joined) != len(requests):
        raise AssertionError("Stable locator plan does not cover every frozen physical cell")
    if not joined["owner_type"].astype(int).eq(joined["owner_type_expected"]).all():
        raise AssertionError("Stable locator owner type differs from the frozen sampling plan")
    if not joined["owner_index"].astype(int).eq(joined["owner_index_expected"]).all():
        raise AssertionError("Stable locator owner index differs from the frozen sampling plan")
    stable = ["shard_path", "row_group_index", "row_index_in_row_group", "row_index_in_shard"]
    if joined[stable].duplicated().any():
        raise AssertionError("Two embedding indices resolve to the same physical cell locator")
    if sorted(joined["matrix_row"].astype(int).tolist()) != list(range(len(requests))):
        raise AssertionError("Physical-cell locator order does not cover every AnnData row once")
    old_true = old_build.get("build", {}).get("true_expression", {})
    if (
        int(old_true.get("cells", -1)) != len(requests)
        or old_true.get("shape") != [len(requests), 5000]
        or not old_true.get("finite")
        or not old_true.get("nonnegative")
    ):
        raise AssertionError("Original real-expression locator audit is absent or incomplete")
    return {
        "cells": len(requests),
        "control_cells": int(requests["side"].eq("control").sum()),
        "treated_cells": int(requests["side"].eq("treated").sum()),
        "embedding_index_unique": not requests["embedding_index"].duplicated().any(),
        "stable_locator_unique": True,
        "owner_type_and_index_exact": True,
        "anndata_matrix_row_coverage_exact": True,
        "real_expression_source": "original formal A build; not reconstructed",
        "original_raw_metadata_and_locator_audit_reused": True,
    }


def load_frozen_contract() -> tuple[
    Any,
    pd.DataFrame,
    dict[str, Any],
    np.ndarray,
    dict[str, np.ndarray],
    pd.DataFrame,
    dict[str, Any],
]:
    required = (
        arc7.PLAN_PATH,
        arc7.CONDITIONS_PATH,
        arc7.BUILD_MANIFEST_PATH,
        arc7.RESULT_PATH,
        arc7.SUMMARY_PATH,
        arc7.PER_CONDITION_PATH,
        arc7.REAL_PATH,
        arc7.PRED_PATH,
    )
    missing = [display_path(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing original formal A artifacts: {missing}")

    plan = read_json(arc7.PLAN_PATH)
    old_build = read_json(arc7.BUILD_MANIFEST_PATH)
    old_result = read_json(arc7.RESULT_PATH)
    if plan.get("status") != "pass" or old_build.get("status") != "pass" or old_result.get(
        "status"
    ) != "pass":
        raise AssertionError("Original formal A plan/build/result is not PASS")
    if int(old_result.get("condition_count", -1)) != 23:
        raise AssertionError("Original formal A result is not the frozen 23-condition evaluation")

    assert_artifact(arc7.REAL_PATH, old_build["outputs"]["real_h5ad"], "A real H5AD")
    assert_artifact(arc7.PRED_PATH, old_build["outputs"]["pred_h5ad"], "A predicted H5AD")
    assert_artifact(arc7.REAL_PATH, old_result["outputs"]["real_h5ad"], "A result real H5AD")
    assert_artifact(arc7.PRED_PATH, old_result["outputs"]["pred_h5ad"], "A result predicted H5AD")
    assert_artifact(arc7.SUMMARY_PATH, old_result["outputs"]["summary"], "A summary")
    assert_artifact(
        arc7.PER_CONDITION_PATH, old_result["outputs"]["per_condition"], "A per-condition"
    )
    assert_artifact(arc7.CONDITIONS_PATH, old_result["outputs"]["conditions"], "A conditions")
    assert_artifact(arc7.PANEL_PATH, old_result["provenance"]["panel"], "frozen gene panel")
    assert_artifact(
        arc7.SCRIPT_PATH, old_result["provenance"]["script"], "original A evaluation script"
    )
    if sha256_file(arc7.CONDITIONS_PATH) != plan["conditions"]["sha256"]:
        raise AssertionError("Frozen conditions CSV differs from the original plan")
    if sha256_file(arc7.PLAN_PATH) != old_build["provenance"]["plan"]["sha256"]:
        raise AssertionError("Frozen plan differs from the original build manifest")

    frozen_conditions = pd.read_csv(
        arc7.CONDITIONS_PATH, encoding="utf-8-sig", keep_default_na=False
    )
    dataset = arc7.TahoeExperiment1LatentSetDataset(split="test", seed=42, epoch=0)
    condition_source = dataset.conditions.copy()
    treated_samples = arc7.load_conditions()[
        ["pair_id", "cache_condition_index", "treated_samples"]
    ]
    condition_source = condition_source.merge(
        treated_samples,
        on=["pair_id", "cache_condition_index"],
        how="left",
        validate="one_to_one",
    )
    selected = frozen_conditions[["pair_id"]].merge(
        condition_source,
        on="pair_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    selected.insert(0, "perturbation", frozen_conditions["perturbation"].astype(str))
    context = dict(plan["context"])
    assert_condition_table(selected, frozen_conditions)
    control_indices, treated_indices, sampling = arc7.make_sampling_plan(dataset, selected)
    frozen_sampling = plan["sampling"]
    if not np.array_equal(
        control_indices,
        np.asarray(frozen_sampling["shared_control_embedding_indices"], dtype=np.int64),
    ):
        raise AssertionError("Shared control indices differ from the original plan")
    if one_side_index_sha256(control_indices) != frozen_sampling[
        "shared_control_embedding_indices_sha256"
    ]:
        raise AssertionError("Shared control index SHA-256 differs from the original plan")
    for pair_id, indices in treated_indices.items():
        if one_side_index_sha256(indices) != frozen_sampling["treated_embedding_indices_sha256"].get(
            pair_id
        ):
            raise AssertionError(f"Treated indices differ from the original plan for {pair_id}")
    fingerprint = arc7.selection_fingerprint(selected, control_indices, treated_indices)
    if fingerprint != plan["selection_fingerprint"] or fingerprint != old_build[
        "selection_fingerprint"
    ]:
        raise AssertionError("Frozen selection fingerprint changed")
    if sampling["base_seed"] != 42 or sampling["repeat_epoch"] != 0 or sampling["replace"]:
        raise AssertionError("Frozen deterministic sampling rule changed")

    panel, _, _, _ = arc7.load_panel_contract()
    h5ad_audit = audit_original_h5ad(selected, panel)
    locator_audit = audit_locator_alignment(
        selected, control_indices, treated_indices, old_build
    )
    audit = {
        "status": "pass",
        "conditions": len(selected),
        "selection_fingerprint": fingerprint,
        "condition_order_exact": True,
        "sampling": {
            "S": SET_SIZE,
            "base_seed": 42,
            "repeat_epoch": 0,
            "replace": False,
            "shared_control_indices_exact": True,
            "all_treated_index_hashes_exact": True,
        },
        "context": context,
        "h5ad": h5ad_audit,
        "locator_alignment": locator_audit,
        "A_result": {
            "path": display_path(arc7.RESULT_PATH),
            "sha256": sha256_file(arc7.RESULT_PATH),
        },
    }
    return dataset, selected, context, control_indices, treated_indices, panel, audit


def ordered_embedding_indices(
    selected: pd.DataFrame,
    control_indices: np.ndarray,
    treated_indices: dict[str, np.ndarray],
) -> np.ndarray:
    pieces = [np.asarray(control_indices, dtype=np.int64)]
    pieces.extend(
        np.asarray(treated_indices[str(row.pair_id)], dtype=np.int64)
        for row in selected.itertuples(index=False)
    )
    indices = np.concatenate(pieces)
    expected = SET_SIZE * (len(selected) + 1)
    if len(indices) != expected or len(np.unique(indices)) != expected:
        raise AssertionError("Ordered latent rows are incomplete or contain duplicate physical cells")
    return indices


def read_h5ad_matrix(path: Path, rows: int) -> np.ndarray:
    data = ad.read_h5ad(path, backed="r")
    try:
        if data.n_obs < rows:
            raise AssertionError(f"{path.name} has fewer rows than requested")
        matrix = data.X[:rows]
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        output = np.asarray(matrix, dtype=np.float32)
    finally:
        data.file.close()
    if output.shape != (rows, 5000) or not np.isfinite(output).all() or np.any(output < 0):
        raise AssertionError(f"{path.name} expression matrix is invalid")
    return output


def decode_true_latents(
    dataset: Any,
    indices: np.ndarray,
    decoder_batch_size: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("Decoder inference requires one visible CUDA GPU")
    if decoder_batch_size < 1:
        raise ValueError("decoder_batch_size must be positive")
    latents = np.ascontiguousarray(dataset.embeddings[indices], dtype=np.float32)
    if latents.shape != (len(indices), LATENT_DIM) or not np.isfinite(latents).all():
        raise AssertionError("Frozen true latent rows are invalid")
    if not (float(latents.min()) < 0 < float(latents.max())):
        raise AssertionError("Frozen true latents unexpectedly lost signed coordinates")

    device = torch.device("cuda:0")
    decoder, checkpoint = load_decoder(device)
    if checkpoint["epoch"] != EXPECTED_DECODER_EPOCH or decoder.training:
        raise AssertionError("Frozen Decoder checkpoint/eval contract changed")
    output = np.empty((len(indices), 5000), dtype=np.float32)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(indices), decoder_batch_size):
            stop = min(start + decoder_batch_size, len(indices))
            latent = torch.from_numpy(latents[start:stop]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                expression = decoder(latent)
            expression32 = expression.float()
            if not torch.isfinite(expression32).all() or torch.any(expression32 < 0):
                raise AssertionError("Decoder returned non-finite or negative expression")
            output[start:stop] = expression32.cpu().numpy()
    torch.cuda.synchronize(device)
    if decoder_fingerprint(decoder) != checkpoint["model_fingerprint"]:
        raise AssertionError("Decoder parameters changed during inference")
    inference = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "decoder_batch_size": decoder_batch_size,
        "autocast": "bfloat16",
        "input_shape": list(latents.shape),
        "input_dtype": str(latents.dtype),
        "input_finite": True,
        "input_signed": True,
        "input_min": float(latents.min()),
        "input_max": float(latents.max()),
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "output_finite": True,
        "output_nonnegative": True,
        "output_min": float(output.min()),
        "output_max": float(output.max()),
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    del decoder, latents
    torch.cuda.empty_cache()
    return output, inference, checkpoint


def build_variant_matrices(
    dataset: Any,
    selected: pd.DataFrame,
    control_indices: np.ndarray,
    treated_indices: dict[str, np.ndarray],
    decoder_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    rows = SET_SIZE * (len(selected) + 1)
    a_prediction = read_h5ad_matrix(arc7.PRED_PATH, rows)
    indices = ordered_embedding_indices(selected, control_indices, treated_indices)
    decoded, inference, decoder_checkpoint = decode_true_latents(
        dataset, indices, decoder_batch_size
    )
    variant_b = a_prediction.copy()
    variant_b[:SET_SIZE] = decoded[:SET_SIZE]
    variant_c = decoded
    if not np.array_equal(variant_b[SET_SIZE:], a_prediction[SET_SIZE:]):
        raise AssertionError("Variant B treated rows are not identical to frozen A")
    if not np.array_equal(variant_b[:SET_SIZE], variant_c[:SET_SIZE]):
        raise AssertionError("Variants B/C do not share identical decoded-control rows")
    return variant_b, variant_c, inference, decoder_checkpoint


def write_variant_pair(
    selected: pd.DataFrame,
    panel: pd.DataFrame,
    variant_b: np.ndarray,
    variant_c: np.ndarray,
    b_path: Path,
    c_path: Path,
) -> dict[str, Any]:
    obs = arc7.make_obs(selected, predicted=True)
    arc7.write_h5ad(b_path, variant_b, obs, panel, len(selected))
    arc7.write_h5ad(c_path, variant_c, obs.copy(), panel, len(selected))
    b_audit = arc7.verify_h5ad_pair(arc7.REAL_PATH, b_path, selected)
    c_audit = arc7.verify_h5ad_pair(arc7.REAL_PATH, c_path, selected)
    b_data = ad.read_h5ad(b_path, backed="r")
    c_data = ad.read_h5ad(c_path, backed="r")
    a_data = ad.read_h5ad(arc7.PRED_PATH, backed="r")
    try:
        assert_obs_equal(b_data.obs, a_data.obs.iloc[: b_data.n_obs], "B vs A")
        assert_obs_equal(c_data.obs, a_data.obs.iloc[: c_data.n_obs], "C vs A")
        if not np.array_equal(b_data.var_names.to_numpy(), a_data.var_names.to_numpy()):
            raise AssertionError("Variant B gene order differs from A")
        if not np.array_equal(c_data.var_names.to_numpy(), a_data.var_names.to_numpy()):
            raise AssertionError("Variant C gene order differs from A")
    finally:
        b_data.file.close()
        c_data.file.close()
        a_data.file.close()
    return {"B": b_audit, "C": c_audit}


def command_audit() -> dict[str, Any]:
    _, _, _, _, _, _, audit = load_frozen_contract()
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return audit


def command_smoke(conditions: int, decoder_batch_size: int) -> dict[str, Any]:
    if conditions < 2 or conditions > 2:
        raise ValueError("The plumbing smoke is intentionally fixed to exactly two conditions")
    dataset, selected_all, _, control_indices, treated_all, panel, frozen_audit = (
        load_frozen_contract()
    )
    selected = selected_all.head(conditions).reset_index(drop=True)
    treated = {str(row.pair_id): treated_all[str(row.pair_id)] for row in selected.itertuples()}
    started = time.perf_counter()
    variant_b, variant_c, inference, decoder_checkpoint = build_variant_matrices(
        dataset, selected, control_indices, treated, decoder_batch_size
    )
    rows = SET_SIZE * (conditions + 1)
    real = read_h5ad_matrix(arc7.REAL_PATH, rows)
    with tempfile.TemporaryDirectory(prefix="genejepa_arc7_abc_smoke_") as temporary:
        root = Path(temporary)
        real_path = root / "real.h5ad"
        b_path = root / "b.h5ad"
        c_path = root / "c.h5ad"
        arc7.write_h5ad(
            real_path, real, arc7.make_obs(selected, predicted=False), panel, len(selected)
        )
        obs = arc7.make_obs(selected, predicted=True)
        arc7.write_h5ad(b_path, variant_b, obs, panel, len(selected))
        arc7.write_h5ad(c_path, variant_c, obs.copy(), panel, len(selected))
        b_summary, b_per, _, b_runtime = arc7.run_cell_eval(
            real_path, b_path, selected, root / "b_cell_eval", min(4, arc7.available_threads())
        )
        c_summary, c_per, _, c_runtime = arc7.run_cell_eval(
            real_path, c_path, selected, root / "c_cell_eval", min(4, arc7.available_threads())
        )
        if len(b_summary) != 7 or len(c_summary) != 7 or len(b_per) != 2 or len(c_per) != 2:
            raise AssertionError("Smoke did not return all seven metrics for both conditions")
        failures = {"B": b_runtime["metric_failures"], "C": c_runtime["metric_failures"]}
        result = {
            "schema": "genejepa_arc7_decoded_control_smoke_v1",
            "created_at_utc": utc_now(),
            "status": "pass" if not failures["B"] and not failures["C"] else "fail",
            "scope": "two-condition plumbing smoke only; not a scientific result",
            "formal_inference_started": False,
            "formal_DE_or_seven_metric_evaluation_started": False,
            "conditions": conditions,
            "pair_ids": selected["pair_id"].astype(str).tolist(),
            "frozen_full_contract_audit": frozen_audit,
            "variant_contract": {
                "B_treated_reused_exactly_from_A": True,
                "B_and_C_decoded_control_identical": True,
                "C_uses_true_control_and_treated_latents": True,
                "C_uses_ST_A": False,
            },
            "decoder_inference": inference,
            "decoder_checkpoint": decoder_checkpoint,
            "cell_eval": {
                "B_metric_failures": failures["B"],
                "C_metric_failures": failures["C"],
                "B_summary": arc7.summary_for_json(b_summary),
                "C_summary": arc7.summary_for_json(c_summary),
            },
            "temporary_h5ad_deleted_after_validation": True,
            "runtime_seconds": time.perf_counter() - started,
            "provenance": {"script": artifact(SCRIPT_PATH), "task": artifact(TASK_PATH)},
        }
    atomic_write_json(SMOKE_PATH, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def command_build(decoder_batch_size: int) -> dict[str, Any]:
    outputs = (BUILD_RESULT_PATH, B_PRED_PATH, C_PRED_PATH)
    existing = [display_path(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite formal decoded-control outputs: {existing}")
    dataset, selected, context, control_indices, treated, panel, frozen_audit = (
        load_frozen_contract()
    )
    started = time.perf_counter()
    variant_b, variant_c, inference, decoder_checkpoint = build_variant_matrices(
        dataset, selected, control_indices, treated, decoder_batch_size
    )
    with tempfile.TemporaryDirectory(prefix="genejepa_arc7_abc_build_", dir=RESULTS) as temporary:
        root = Path(temporary)
        staged_b = root / B_PRED_PATH.name
        staged_c = root / C_PRED_PATH.name
        pair_audit = write_variant_pair(
            selected, panel, variant_b, variant_c, staged_b, staged_c
        )
        os.replace(staged_b, B_PRED_PATH)
        os.replace(staged_c, C_PRED_PATH)
    del variant_b, variant_c
    gc.collect()
    build = {
        "schema": "genejepa_arc7_decoded_control_build_result_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "conditions": len(selected),
        "cells_per_set": SET_SIZE,
        "cells_per_anndata": SET_SIZE * (len(selected) + 1),
        "genes": 5000,
        "context": context,
        "selection_fingerprint": frozen_audit["selection_fingerprint"],
        "frozen_A_contract_audit": frozen_audit,
        "variants": {
            "B": {
                "pred_control": "Decoder(true control latent)",
                "pred_treated": "exact rows reused from formal A: Decoder(ST-A(...))",
                "treated_rows_exactly_equal_A": True,
            },
            "C": {
                "pred_control": "Decoder(true control latent)",
                "pred_treated": "Decoder(true treated latent)",
                "ST_A_used": False,
            },
        },
        "inference": inference,
        "decoder_checkpoint": decoder_checkpoint,
        "ST_A_provenance_reused_from_A_without_rerun": read_json(arc7.RESULT_PATH)[
            "provenance"
        ]["st_a"],
        "anndata_audit": pair_audit,
        "outputs": {"B_pred_h5ad": artifact(B_PRED_PATH), "C_pred_h5ad": artifact(C_PRED_PATH)},
        "inputs": {
            "A_real_h5ad": artifact(arc7.REAL_PATH),
            "A_pred_h5ad": artifact(arc7.PRED_PATH),
            "A_result": artifact(arc7.RESULT_PATH),
            "A_plan": artifact(arc7.PLAN_PATH),
            "A_conditions": artifact(arc7.CONDITIONS_PATH),
            "panel": artifact(arc7.PANEL_PATH),
        },
        "provenance": {"script": artifact(SCRIPT_PATH), "task": artifact(TASK_PATH)},
        "real_h5ad_reused_without_reconstruction": True,
        "training_performed": False,
        "formal_DE_or_seven_metric_evaluation_started": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_write_json(BUILD_RESULT_PATH, build)
    print(json.dumps(build, ensure_ascii=False, indent=2), flush=True)
    return build


def describe_counts(values: pd.Series) -> dict[str, float]:
    array = pd.to_numeric(values).to_numpy(np.float64)
    if not np.isfinite(array).all():
        raise AssertionError("DEG counts contain non-finite values")
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def diagnostic_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "conditions": len(frame),
        "DES_equal_0": int(pd.to_numeric(frame["DES"]).eq(0).sum()),
        "AUPRC_lt_0_01": int(pd.to_numeric(frame["AUPRC"]).lt(0.01).sum()),
        "Pearson_delta_lt_0": int(pd.to_numeric(frame["Pearson_delta"]).lt(0).sum()),
        "Pearson_delta_gt_0_3": int(pd.to_numeric(frame["Pearson_delta"]).gt(0.3).sum()),
    }


def variant_result(
    letter: str,
    summary: pd.DataFrame,
    per_condition: pd.DataFrame,
    runtime: dict[str, Any],
    pred_path: Path,
    cell_eval_dir: Path,
) -> dict[str, Any]:
    failures = runtime["metric_failures"]
    return {
        "schema": f"genejepa_arc7_variant_{letter.lower()}_result_v1",
        "created_at_utc": utc_now(),
        "status": "pass" if not failures else "fail",
        "variant": VARIANTS[letter],
        "condition_count": len(per_condition),
        "cells_per_set": SET_SIZE,
        "gene_count": 5000,
        "seven_metrics": arc7.summary_for_json(summary),
        "predicted_DEG_count": describe_counts(per_condition["pred_DEG_count"]),
        "true_DEG_count": describe_counts(per_condition["true_DEG_count"]),
        "diagnostic_condition_counts": diagnostic_counts(per_condition),
        "undefined_or_failed": {"metric_failures": failures},
        "runtime": runtime,
        "inputs": {
            "build_result": artifact(BUILD_RESULT_PATH),
            "real_h5ad": artifact(arc7.REAL_PATH),
            "pred_h5ad": artifact(pred_path),
        },
        "outputs": {
            "requested_seven_metrics": artifact(cell_eval_dir / "requested_seven_metrics.csv"),
            "real_de": artifact(cell_eval_dir / "real_de.csv"),
            "pred_de": artifact(cell_eval_dir / "pred_de.csv"),
        },
        "cell_eval_contract_changed": False,
        "training_performed": False,
    }


def combined_summary(
    a_summary: pd.DataFrame, b_summary: pd.DataFrame, c_summary: pd.DataFrame
) -> pd.DataFrame:
    frames = []
    for variant, frame in zip(VARIANTS.values(), (a_summary, b_summary, c_summary), strict=True):
        output = frame.copy()
        output.insert(0, "variant", variant)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)[
        ["variant", "metric", "mean", "median", "valid_n", "direction", "nan_n", "scope"]
    ]


def combined_per_condition(
    selected: pd.DataFrame,
    a_per: pd.DataFrame,
    b_per: pd.DataFrame,
    c_per: pd.DataFrame,
) -> pd.DataFrame:
    expected_pairs = selected["pair_id"].astype(str).tolist()
    for label, frame in (("A", a_per), ("B", b_per), ("C", c_per)):
        if frame["pair_id"].astype(str).tolist() != expected_pairs:
            raise AssertionError(f"{label} per-condition order differs from the frozen plan")
    true_counts = pd.to_numeric(a_per["true_DEG_count"]).to_numpy(np.int64)
    if not np.array_equal(true_counts, pd.to_numeric(b_per["true_DEG_count"]).to_numpy(np.int64)):
        raise AssertionError("Variant B true DEG counts differ from A")
    if not np.array_equal(true_counts, pd.to_numeric(c_per["true_DEG_count"]).to_numpy(np.int64)):
        raise AssertionError("Variant C true DEG counts differ from A")

    output = pd.DataFrame(
        {
            "condition": selected["perturbation"].astype(str),
            "pair_id": expected_pairs,
            "drug": selected["drug"].astype(str),
            "dose_uM": selected["dose_uM"].astype(float),
            "true_DEG_count": true_counts,
            "A_pred_DEG_count": pd.to_numeric(a_per["pred_DEG_count"]).astype(int),
            "B_pred_DEG_count": pd.to_numeric(b_per["pred_DEG_count"]).astype(int),
            "C_pred_DEG_count": pd.to_numeric(c_per["pred_DEG_count"]).astype(int),
        }
    )
    for metric in PER_CONDITION_METRICS:
        output[f"A_{metric}"] = pd.to_numeric(a_per[metric])
        output[f"B_{metric}"] = pd.to_numeric(b_per[metric])
        output[f"C_{metric}"] = pd.to_numeric(c_per[metric])
    return output


def metric_mean_map(summary: pd.DataFrame) -> dict[str, float | None]:
    return {
        str(row.metric): arc7.finite_or_none(row.mean)
        for row in summary.itertuples(index=False)
    }


def markdown_table(frame: pd.DataFrame, *, include_index: bool = False) -> str:
    table = frame.reset_index() if include_index else frame.reset_index(drop=True)

    def format_cell(value: Any) -> str:
        if value is None or (isinstance(value, (float, np.floating)) and math.isnan(value)):
            return "NaN"
        if isinstance(value, (float, np.floating)):
            return format(float(value), ".10g")
        return str(value).replace("|", "\\|").replace("\n", " ")

    columns = [str(column) for column in table.columns]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    rows.extend(
        "| " + " | ".join(format_cell(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def difference_table(
    summaries: dict[str, pd.DataFrame], per_frames: dict[str, pd.DataFrame]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comparisons = (("B", "A"), ("C", "A"), ("C", "B"))
    metrics = ("DES", "Pearson delta", "Spearman logFC", "AUPRC")
    summary_maps = {letter: metric_mean_map(frame) for letter, frame in summaries.items()}
    for left, right in comparisons:
        for metric in metrics:
            left_value = summary_maps[left][metric]
            right_value = summary_maps[right][metric]
            rows.append(
                {
                    "comparison": f"{left}_minus_{right}",
                    "quantity": metric,
                    "statistic": "mean",
                    "difference": (
                        None
                        if left_value is None or right_value is None
                        else left_value - right_value
                    ),
                }
            )
        left_counts = pd.to_numeric(per_frames[left]["pred_DEG_count"]).to_numpy(np.float64)
        right_counts = pd.to_numeric(per_frames[right]["pred_DEG_count"]).to_numpy(np.float64)
        for statistic, function in (("mean", np.mean), ("median", np.median)):
            rows.append(
                {
                    "comparison": f"{left}_minus_{right}",
                    "quantity": "predicted DEG count",
                    "statistic": statistic,
                    "difference": float(function(left_counts) - function(right_counts)),
                }
            )
    return rows


def write_comparison_report(
    summary: pd.DataFrame,
    per_frames: dict[str, pd.DataFrame],
    differences: list[dict[str, Any]],
) -> None:
    metric_table = summary.pivot(index="metric", columns="variant", values="mean").reindex(
        [spec["label"] for spec in arc7.METRICS]
    )
    deg_rows = []
    diagnostic_rows = []
    for letter, frame in per_frames.items():
        deg_rows.append({"variant": VARIANTS[letter], **describe_counts(frame["pred_DEG_count"])})
        diagnostic_rows.append(
            {"variant": VARIANTS[letter], **diagnostic_counts(frame)}
        )
    delta_frame = pd.DataFrame.from_records(differences)
    text = [
        "# GeneJEPA ARC7 A/B/C decoded-control diagnostic",
        "",
        arc7.DISCLAIMER,
        "",
        "A is read from the existing formal result and was not recomputed. B reuses A treated "
        "predictions exactly and changes only predicted control. C uses Decoder(true latent) on "
        "both control and treated cells and never invokes ST-A.",
        "",
        "## Seven metrics (mean)",
        "",
        markdown_table(metric_table, include_index=True),
        "",
        "## Predicted DEG counts",
        "",
        markdown_table(pd.DataFrame.from_records(deg_rows)),
        "",
        "## Condition diagnostics",
        "",
        markdown_table(pd.DataFrame.from_records(diagnostic_rows)),
        "",
        "## Numeric differences only",
        "",
        markdown_table(delta_frame),
        "",
        "No composite ARC score, success/failure verdict, or new interpretation threshold was added. "
        "The A/B/C pattern should be interpreted against the four scenarios frozen in the task.",
        "",
    ]
    temporary = ABC_REPORT_PATH.with_name(ABC_REPORT_PATH.name + ".tmp")
    temporary.write_text("\n".join(text), encoding="utf-8", newline="\n")
    os.replace(temporary, ABC_REPORT_PATH)


def command_evaluate(threads: int | None) -> dict[str, Any]:
    if not BUILD_RESULT_PATH.is_file() or not B_PRED_PATH.is_file() or not C_PRED_PATH.is_file():
        raise FileNotFoundError("Run the formal build command before formal evaluation")
    formal_outputs = (
        B_RESULT_PATH,
        C_RESULT_PATH,
        ABC_SUMMARY_PATH,
        ABC_PER_CONDITION_PATH,
        ABC_REPORT_PATH,
    )
    existing = [display_path(path) for path in formal_outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite formal A/B/C evaluation outputs: {existing}")
    for directory in (B_CELL_EVAL_DIR, C_CELL_EVAL_DIR):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty directory: {directory}")

    _, selected, _, _, _, _, frozen_audit = load_frozen_contract()
    build = read_json(BUILD_RESULT_PATH)
    if build.get("status") != "pass" or build.get("selection_fingerprint") != frozen_audit[
        "selection_fingerprint"
    ]:
        raise AssertionError("Decoded-control formal build result is not PASS/current")
    assert_artifact(B_PRED_PATH, build["outputs"]["B_pred_h5ad"], "Variant B H5AD")
    assert_artifact(C_PRED_PATH, build["outputs"]["C_pred_h5ad"], "Variant C H5AD")
    arc7.verify_h5ad_pair(arc7.REAL_PATH, B_PRED_PATH, selected)
    arc7.verify_h5ad_pair(arc7.REAL_PATH, C_PRED_PATH, selected)

    threads = arc7.available_threads() if threads is None else min(threads, arc7.available_threads())
    b_summary, b_per, b_official, b_runtime = arc7.run_cell_eval(
        arc7.REAL_PATH, B_PRED_PATH, selected, B_CELL_EVAL_DIR, threads
    )
    b_official_path = B_CELL_EVAL_DIR / "requested_seven_metrics.csv"
    arc7.atomic_write_csv(b_official_path, b_official)
    b_result = variant_result(
        "B", b_summary, b_per, b_runtime, B_PRED_PATH, B_CELL_EVAL_DIR
    )
    atomic_write_json(B_RESULT_PATH, b_result)

    c_summary, c_per, c_official, c_runtime = arc7.run_cell_eval(
        arc7.REAL_PATH, C_PRED_PATH, selected, C_CELL_EVAL_DIR, threads
    )
    c_official_path = C_CELL_EVAL_DIR / "requested_seven_metrics.csv"
    arc7.atomic_write_csv(c_official_path, c_official)
    c_result = variant_result(
        "C", c_summary, c_per, c_runtime, C_PRED_PATH, C_CELL_EVAL_DIR
    )
    atomic_write_json(C_RESULT_PATH, c_result)

    a_summary = pd.read_csv(arc7.SUMMARY_PATH, encoding="utf-8-sig")
    a_per = pd.read_csv(arc7.PER_CONDITION_PATH, encoding="utf-8-sig")
    summary = combined_summary(a_summary, b_summary, c_summary)
    per_condition = combined_per_condition(selected, a_per, b_per, c_per)
    arc7.atomic_write_csv(ABC_SUMMARY_PATH, summary)
    arc7.atomic_write_csv(ABC_PER_CONDITION_PATH, per_condition)
    summaries = {"A": a_summary, "B": b_summary, "C": c_summary}
    per_frames = {"A": a_per, "B": b_per, "C": c_per}
    differences = difference_table(summaries, per_frames)
    write_comparison_report(summary, per_frames, differences)

    final = {
        "status": (
            "pass"
            if b_result["status"] == "pass" and c_result["status"] == "pass"
            else "fail"
        ),
        "A_recomputed": False,
        "B": b_result,
        "C": c_result,
        "differences": differences,
        "outputs": {
            "B_result": artifact(B_RESULT_PATH),
            "C_result": artifact(C_RESULT_PATH),
            "ABC_summary": artifact(ABC_SUMMARY_PATH),
            "ABC_per_condition": artifact(ABC_PER_CONDITION_PATH),
            "ABC_report": artifact(ABC_REPORT_PATH),
        },
    }
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit", help="Audit exact reuse of the frozen A data and indices")
    smoke = subparsers.add_parser("smoke", help="Run a two-condition temporary B/C plumbing smoke")
    smoke.add_argument("--conditions", type=int, default=2)
    smoke.add_argument("--decoder-batch-size", type=int, default=768)
    build = subparsers.add_parser("build", help="Build formal B/C predicted AnnData only")
    build.add_argument("--decoder-batch-size", type=int, default=2048)
    evaluate = subparsers.add_parser("evaluate", help="Run formal B/C DE and seven metrics")
    evaluate.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    match args.command:
        case "audit":
            command_audit()
        case "smoke":
            command_smoke(args.conditions, args.decoder_batch_size)
        case "build":
            command_build(args.decoder_batch_size)
        case "evaluate":
            command_evaluate(args.threads)


if __name__ == "__main__":
    main()
