#!/usr/bin/env python3
"""Build and evaluate the frozen 23-condition OLD5K/HD100 ARC7 diagnostic."""

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

import run_genejepa_decoder_hd100 as hd100
import run_genejepa_decoder_v1 as decoder_v1
import run_genejepa_decoder_v1_arc7_100 as arc7
import run_genejepa_decoder_v1_arc7_abc as old_abc
from evaluate_tahoe_experiment1_st_raw import load_checkpoint as load_st_checkpoint
from run_genejepa_decoder_v1_demo import artifact, atomic_write_json
from run_tahoe_experiment1_st_formal import (
    ensure_protocol as ensure_st_protocol,
    model_fingerprint as st_fingerprint,
)
from tahoe_decoder_v1_data import PROJECT_ROOT, RESULTS, display_path, sha256_file, utc_now
from tahoe_experiment1_latent_data import LATENT_DIM, PERT_DIM, SET_SIZE


SCRIPT_PATH = Path(__file__).resolve()
TASK_PATH = PROJECT_ROOT.parent / "当前任务.txt"
PANEL_PATH = hd100.PANEL_PATH
PREFLIGHT_PATH = RESULTS / "genejepa_decoder_hd100_arc7_preflight.json"
BUILD_RESULT_PATH = RESULTS / "genejepa_decoder_hd100_arc7_build_result.json"

REAL_PATH = RESULTS / "genejepa_decoder_hd100_arc7_real.h5ad"
OLD_PRED_PATHS = {
    "A": RESULTS / "genejepa_decoder_old5k_subset100_variant_a_pred.h5ad",
    "B": RESULTS / "genejepa_decoder_old5k_subset100_variant_b_pred.h5ad",
    "C": RESULTS / "genejepa_decoder_old5k_subset100_variant_c_pred.h5ad",
}
HD_PRED_PATHS = {
    "A": RESULTS / "genejepa_decoder_hd100_variant_a_pred.h5ad",
    "B": RESULTS / "genejepa_decoder_hd100_variant_b_pred.h5ad",
    "C": RESULTS / "genejepa_decoder_hd100_variant_c_pred.h5ad",
}

SUMMARY_PATH = RESULTS / "genejepa_decoder_hd100_arc7_summary.csv"
PER_CONDITION_PATH = RESULTS / "genejepa_decoder_hd100_arc7_per_condition.csv"
RESULT_PATH = RESULTS / "genejepa_decoder_hd100_arc7_result.json"
COMPARISON_PATH = RESULTS / "genejepa_decoder_hd100_comparison.csv"
REPORT_PATH = RESULTS / "genejepa_decoder_hd100_comparison.md"
CELL_EVAL_ROOT = RESULTS / "genejepa_decoder_hd100_arc7_cell_eval"

GROUPS = {
    "OLD5K_SUBSET100_A": ("Old Decoder", "A", OLD_PRED_PATHS["A"]),
    "OLD5K_SUBSET100_B": ("Old Decoder", "B", OLD_PRED_PATHS["B"]),
    "OLD5K_SUBSET100_C": ("Old Decoder", "C", OLD_PRED_PATHS["C"]),
    "HD100_A": ("HD100 Decoder", "A", HD_PRED_PATHS["A"]),
    "HD100_B": ("HD100 Decoder", "B", HD_PRED_PATHS["B"]),
    "HD100_C": ("HD100 Decoder", "C", HD_PRED_PATHS["C"]),
}
OLD_SOURCE_PATHS = {
    "real": arc7.REAL_PATH,
    "A": arc7.PRED_PATH,
    "B": old_abc.B_PRED_PATH,
    "C": old_abc.C_PRED_PATH,
}
METRIC_COLUMNS = (
    "DES",
    "PDS",
    "MAE",
    "Pearson_delta",
    "Spearman_logFC",
    "AUPRC",
    "Spearman_effect_size",
)
METRIC_LABELS = {
    "DES": "DES",
    "PDS": "PDS",
    "MAE": "MAE",
    "Pearson_delta": "Pearson delta",
    "Spearman_logFC": "Spearman logFC",
    "AUPRC": "AUPRC",
    "Spearman_effect_size": "Spearman effect size",
}
EXPECTED_SELECTION_FINGERPRINT = (
    "c3fcb6b16dabdef6d61e85333b06614a200e0421a10f023170d78a22ab5f685e"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def panel_contract() -> pd.DataFrame:
    if not PANEL_PATH.is_file() or not hd100.PANEL_SUMMARY_PATH.is_file():
        raise FileNotFoundError("Run the HD100 prepare command first")
    summary = read_json(hd100.PANEL_SUMMARY_PATH)
    if (
        summary.get("status") != "frozen"
        or summary.get("genes") != 100
        or summary["output"]["sha256"] != sha256_file(PANEL_PATH)
    ):
        raise AssertionError("HD100 panel/summary is not the frozen 100-gene contract")
    panel = pd.read_csv(PANEL_PATH, encoding="utf-8-sig", keep_default_na=False)
    panel = panel.sort_values("panel_rank", kind="stable").reset_index(drop=True)
    ranks = panel["current_panel_rank"].to_numpy(np.int64)
    if (
        len(panel) != 100
        or not np.array_equal(panel["panel_rank"].to_numpy(np.int64), np.arange(100))
        or len(np.unique(ranks)) != 100
        or np.any((ranks < 0) | (ranks >= 5000))
        or panel["ensembl_id"].astype(str).duplicated().any()
    ):
        raise AssertionError("HD100 panel order/subset mapping is invalid")
    return panel


def load_old_contract() -> tuple[
    Any,
    pd.DataFrame,
    dict[str, Any],
    np.ndarray,
    dict[str, np.ndarray],
    pd.DataFrame,
    dict[str, Any],
]:
    contract = old_abc.load_frozen_contract()
    dataset, selected, context, control_indices, treated, old_panel, audit = contract
    if audit["selection_fingerprint"] != EXPECTED_SELECTION_FINGERPRINT:
        raise AssertionError("Frozen 23-condition ARC7 selection fingerprint changed")
    build = read_json(old_abc.BUILD_RESULT_PATH)
    if build.get("status") != "pass" or build.get("selection_fingerprint") != (
        EXPECTED_SELECTION_FINGERPRINT
    ):
        raise AssertionError("Existing formal old-Decoder B/C build is not PASS/current")
    old_abc.assert_artifact(
        old_abc.B_PRED_PATH, build["outputs"]["B_pred_h5ad"], "old variant B H5AD"
    )
    old_abc.assert_artifact(
        old_abc.C_PRED_PATH, build["outputs"]["C_pred_h5ad"], "old variant C H5AD"
    )
    return dataset, selected, context, control_indices, treated, old_panel, audit


def subset_matrix(path: Path, panel: pd.DataFrame, rows: slice | None = None) -> np.ndarray:
    ranks = panel["current_panel_rank"].to_numpy(np.int64)
    order = np.argsort(ranks)
    inverse = np.argsort(order)
    data = ad.read_h5ad(path, backed="r")
    try:
        expected_genes = pd.read_csv(
            hd100.BASE_PANEL_PATH, encoding="utf-8-sig", keep_default_na=False
        ).sort_values("panel_rank", kind="stable")["ensembl_id"].astype(str).to_numpy()
        if data.shape != (6144, 5000) or data.X.dtype != np.float32:
            raise AssertionError(f"Frozen source H5AD shape/dtype changed: {path}")
        if not np.array_equal(data.var_names.astype(str).to_numpy(), expected_genes):
            raise AssertionError(f"Frozen source H5AD gene order changed: {path}")
        selected_rows = slice(None) if rows is None else rows
        matrix = data.X[selected_rows, np.sort(ranks)]
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        output = np.asarray(matrix, dtype=np.float32)[:, inverse]
    finally:
        data.file.close()
    if output.shape[1] != 100 or not np.isfinite(output).all() or np.any(output < 0):
        raise AssertionError(f"Invalid Top100 subset from {path}")
    return output


def source_obs(path: Path) -> pd.DataFrame:
    data = ad.read_h5ad(path, backed="r")
    try:
        output = data.obs.copy()
    finally:
        data.file.close()
    return output


def write_h5ad(path: Path, matrix: np.ndarray, obs: pd.DataFrame, panel: pd.DataFrame) -> None:
    if matrix.shape != (6144, 100) or matrix.dtype != np.float32:
        raise AssertionError(f"HD100 AnnData matrix shape/dtype is invalid: {matrix.shape}")
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise AssertionError("HD100 AnnData matrix is non-finite or negative")
    var_columns = [
        "gene_symbol",
        "genejepa_index",
        "panel_rank",
        "current_panel_rank",
        "detection_rate",
    ]
    var = panel[var_columns].copy()
    var.index = pd.Index(panel["ensembl_id"].astype(str), name="ensembl_id")
    if len(var) != 100 or var.index.duplicated().any():
        raise AssertionError("HD100 AnnData var_names are not 100 unique Ensembl IDs")
    output = ad.AnnData(X=matrix, obs=obs.copy(), var=var)
    output.uns["evaluation_label"] = (
        "ARC7 23-condition high-detection Top100 Decoder diagnostic"
    )
    output.uns["disclaimer"] = (
        "ARC-style adapted evaluation on frozen Tahoe cells; not an official ARC leaderboard run"
    )
    output.uns["expression_space"] = "log1p(CP10000)"
    output.uns["panel_sha256"] = sha256_file(PANEL_PATH)
    output.uns["selection_fingerprint"] = EXPECTED_SELECTION_FINGERPRINT
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    output.write_h5ad(temporary, compression="lzf")
    del output
    os.replace(temporary, path)


def verify_pair(real_path: Path, pred_path: Path, selected: pd.DataFrame) -> dict[str, Any]:
    real = ad.read_h5ad(real_path, backed="r")
    pred = ad.read_h5ad(pred_path, backed="r")
    try:
        if real.shape != (6144, 100) or pred.shape != (6144, 100):
            raise AssertionError("HD100 real/pred H5AD shape changed")
        if real.X.dtype != np.float32 or pred.X.dtype != np.float32:
            raise AssertionError("HD100 real/pred H5AD dtype changed")
        if not np.array_equal(real.var_names.to_numpy(), pred.var_names.to_numpy()):
            raise AssertionError("HD100 real/pred gene order differs")
        expected = {arc7.CONTROL_PERT, *selected["perturbation"].astype(str).tolist()}
        if set(real.obs[arc7.PERT_COL].astype(str)) != expected or set(
            pred.obs[arc7.PERT_COL].astype(str)
        ) != expected:
            raise AssertionError("HD100 real/pred perturbation identities changed")
        if int(real.obs[arc7.PERT_COL].eq(arc7.CONTROL_PERT).sum()) != SET_SIZE or int(
            pred.obs[arc7.PERT_COL].eq(arc7.CONTROL_PERT).sum()
        ) != SET_SIZE:
            raise AssertionError("HD100 H5AD shared-control size changed")
    finally:
        real.file.close()
        pred.file.close()
    return {
        "shape": [6144, 100],
        "dtype": "float32",
        "gene_order_identical": True,
        "perturbation_identities_identical": True,
        "shared_control_cells_each": SET_SIZE,
    }


def load_hd100_decoder(device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    required = (
        hd100.FORMAL_RESULT_PATH,
        hd100.TRAINING_CONFIG_PATH,
        hd100.FORMAL_CHECKPOINT_DIR / "best.pt",
    )
    missing = [display_path(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"HD100 formal training is not complete: {missing}")
    hd100.configure_for_runtime()
    result = read_json(hd100.FORMAL_RESULT_PATH)
    checkpoint_path = hd100.FORMAL_CHECKPOINT_DIR / "best.pt"
    declared = result.get("checkpoints", {}).get("best", {})
    checkpoint_sha = sha256_file(checkpoint_path)
    if (
        result.get("status") != "pass"
        or result.get("schema") != "genejepa_decoder_hd100_training_result_v1"
        or declared.get("path") != display_path(checkpoint_path)
        or declared.get("sha256") != checkpoint_sha
        or int(result.get("best_epoch", -1)) < 0
    ):
        raise AssertionError("HD100 formal result/checkpoint provenance is invalid")
    config_sha = sha256_file(hd100.TRAINING_CONFIG_PATH)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hd100.validate_checkpoint(
        payload,
        mode="formal",
        kind="best",
        config_sha=config_sha,
        batch_size=decoder_v1.PRIMARY_BATCH_SIZE,
        gradient_accumulation=decoder_v1.PRIMARY_GRADIENT_ACCUMULATION,
    )
    if (
        int(payload.get("epoch", -1)) != int(result["best_epoch"])
        or int(payload.get("best_epoch", -1)) != int(result["best_epoch"])
        or not math.isclose(
            float(payload["best_val_mse"]), float(result["best_val_mse"]), rel_tol=0, abs_tol=1e-15
        )
    ):
        raise AssertionError("HD100 best checkpoint disagrees with the training result")
    model = hd100.build_decoder()
    model.load_state_dict(payload["model_state"], strict=True)
    fingerprint = decoder_v1.model_fingerprint(model)
    if fingerprint != payload["model_fingerprint"]:
        raise AssertionError("HD100 model state did not restore exactly")
    model.to(device).eval()
    if model.training or any(module.training for module in model.modules()):
        raise AssertionError("HD100 eval mode did not propagate")
    record = {
        "path": display_path(checkpoint_path),
        "sha256": checkpoint_sha,
        "checkpoint_kind": "best",
        "epoch": int(payload["epoch"]),
        "best_epoch": int(payload["best_epoch"]),
        "best_val_mse": float(payload["best_val_mse"]),
        "model_fingerprint": fingerprint,
        "training_result": artifact(hd100.FORMAL_RESULT_PATH),
        "training_config": artifact(hd100.TRAINING_CONFIG_PATH),
        "architecture": [768, 1024, 1024, 512, 100],
        "final_activation": "Softplus(beta=1, threshold=20)",
    }
    del payload
    return model, record


def decode_latents(
    model: torch.nn.Module,
    latents: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if batch_size < 1 or latents.ndim != 2 or latents.shape[1] != LATENT_DIM:
        raise ValueError("Invalid HD100 decoder inference input")
    output = np.empty((len(latents), 100), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(latents), batch_size):
            stop = min(start + batch_size, len(latents))
            latent = torch.from_numpy(np.ascontiguousarray(latents[start:stop])).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = model(latent)
            prediction = prediction.float()
            if not torch.isfinite(prediction).all() or torch.any(prediction < 0):
                raise AssertionError("HD100 Decoder returned invalid expression")
            output[start:stop] = prediction.cpu().numpy()
    return output


def build_hd_variants(
    dataset: Any,
    selected: pd.DataFrame,
    control_indices: np.ndarray,
    treated_indices: dict[str, np.ndarray],
    real: np.ndarray,
    condition_batch_size: int,
    decoder_batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("HD100 ARC7 inference requires one visible CUDA GPU")
    if condition_batch_size < 1 or decoder_batch_size < 1:
        raise ValueError("Inference batch sizes must be positive")
    device = torch.device("cuda:0")
    ordered_indices = old_abc.ordered_embedding_indices(selected, control_indices, treated_indices)
    true_latents = np.ascontiguousarray(dataset.embeddings[ordered_indices], dtype=np.float32)
    if true_latents.shape != (6144, LATENT_DIM) or not np.isfinite(true_latents).all():
        raise AssertionError("Frozen true latent matrix is invalid")
    if not float(true_latents.min()) < 0 < float(true_latents.max()):
        raise AssertionError("Frozen true latents lost signed coordinates")

    decoder, decoder_checkpoint = load_hd100_decoder(device)
    st_protocol, st_protocol_sha = ensure_st_protocol(create=False)
    st_model, st_checkpoint = load_st_checkpoint("st-a", device, st_protocol_sha)
    if (
        st_checkpoint["epoch"] != 28
        or st_checkpoint["best_epoch"] != 28
        or st_model.predict_residual
        or st_model.final_activation_name != "identity"
        or st_model.apply_output_relu
        or st_model.training
    ):
        raise AssertionError("Frozen ST-A absolute signed-output contract changed")

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    decoded_true = decode_latents(decoder, true_latents, device, decoder_batch_size)
    a = np.empty((6144, 100), dtype=np.float32)
    a[:SET_SIZE] = real[:SET_SIZE]
    control_latent = np.ascontiguousarray(true_latents[:SET_SIZE])
    st_min = float("inf")
    st_max = float("-inf")
    st_negative = 0
    st_coordinates = 0

    with torch.inference_mode():
        for start in range(0, len(selected), condition_batch_size):
            rows = selected.iloc[start : start + condition_batch_size]
            count = len(rows)
            control = torch.from_numpy(control_latent).to(device).unsqueeze(0).expand(
                count, -1, -1
            )
            vectors = np.stack(
                [
                    dataset.featurizer.encode(str(row["drug"]), float(row["dose_uM"]))
                    for _, row in rows.iterrows()
                ]
            ).astype(np.float32, copy=False)
            perturbation = (
                torch.from_numpy(vectors).to(device).unsqueeze(1).expand(-1, SET_SIZE, -1)
            )
            if tuple(perturbation.shape) != (count, SET_SIZE, PERT_DIM):
                raise AssertionError("Frozen perturbation tensor shape changed")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predicted_latent = st_model(
                    {"ctrl_cell_emb": control, "pert_emb": perturbation}
                ).reshape(count, SET_SIZE, LATENT_DIM)
                predicted_expression = decoder(
                    predicted_latent.reshape(-1, LATENT_DIM)
                ).reshape(count, SET_SIZE, 100)
            latent32 = predicted_latent.float()
            expression32 = predicted_expression.float()
            if not torch.isfinite(latent32).all() or not torch.isfinite(expression32).all():
                raise AssertionError("ST-A/HD100 inference returned non-finite values")
            if torch.any(expression32 < 0):
                raise AssertionError("HD100 Softplus output became negative")
            st_min = min(st_min, float(latent32.min()))
            st_max = max(st_max, float(latent32.max()))
            st_negative += int((latent32 < 0).sum())
            st_coordinates += latent32.numel()
            for local in range(count):
                offset = SET_SIZE + (start + local) * SET_SIZE
                a[offset : offset + SET_SIZE] = expression32[local].cpu().numpy()
            print(f"HD100 inference conditions={start + count}/{len(selected)}", flush=True)
    torch.cuda.synchronize(device)
    if not st_min < 0 < st_max or not np.isfinite(a).all() or np.any(a < 0):
        raise AssertionError("HD100 variant A inference contract failed")

    b = a.copy()
    b[:SET_SIZE] = decoded_true[:SET_SIZE]
    c = decoded_true
    if not np.array_equal(a[SET_SIZE:], b[SET_SIZE:]):
        raise AssertionError("HD100 A/B treated rows are not byte-identical")
    if not np.array_equal(b[:SET_SIZE], c[:SET_SIZE]):
        raise AssertionError("HD100 B/C decoded-control rows are not byte-identical")
    if st_fingerprint(st_model) != st_checkpoint["model_state_sha256_before_evaluation"]:
        raise AssertionError("ST-A parameters changed during inference")
    if decoder_v1.model_fingerprint(decoder) != decoder_checkpoint["model_fingerprint"]:
        raise AssertionError("HD100 Decoder parameters changed during inference")

    inference = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "condition_batch_size": condition_batch_size,
        "decoder_batch_size": decoder_batch_size,
        "autocast": "bfloat16",
        "true_latent_shape": list(true_latents.shape),
        "true_latent_signed": True,
        "st_a_output_per_condition_shape": [SET_SIZE, LATENT_DIM],
        "st_a_latent_min": st_min,
        "st_a_latent_max": st_max,
        "st_a_latent_negative_ratio": st_negative / st_coordinates,
        "output_shape": [6144, 100],
        "output_dtype": "float32",
        "output_finite": True,
        "output_nonnegative": True,
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": time.perf_counter() - started,
        "st_a_checkpoint": st_checkpoint,
        "st_training_protocol_sha256": st_protocol_sha,
        "st_training_protocol_schema": st_protocol.get("schema"),
        "hd100_decoder_checkpoint": decoder_checkpoint,
    }
    del decoder, st_model, true_latents
    torch.cuda.empty_cache()
    return {"A": a, "B": b, "C": c}, inference


def command_preflight() -> dict[str, Any]:
    if sha256_file(TASK_PATH) != hd100.TASK_SHA256:
        raise AssertionError("当前任务.txt changed")
    panel = panel_contract()
    dataset, selected, context, control, treated, _, frozen = load_old_contract()
    samples = {name: subset_matrix(path, panel, slice(0, 8)) for name, path in OLD_SOURCE_PATHS.items()}
    if any(value.shape != (8, 100) for value in samples.values()):
        raise AssertionError("OLD5K subset smoke shape changed")
    cell_eval = arc7.cell_eval_provenance()
    ddp_smoke = read_json(hd100.DDP_SMOKE_PATH)
    if ddp_smoke.get("status") != "pass" or not ddp_smoke.get("ready_for_formal_training"):
        raise AssertionError("HD100 two-GPU exact-config smoke is not PASS")
    formal_complete = hd100.FORMAL_RESULT_PATH.is_file() and (
        hd100.FORMAL_CHECKPOINT_DIR / "best.pt"
    ).is_file()
    result = {
        "schema": "genejepa_decoder_hd100_arc7_preflight_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "scope": "static/frozen-artifact audit only; no formal inference, DE, or metrics",
        "panel": {
            **artifact(PANEL_PATH),
            "rows": len(panel),
            "subset_overlap": 100,
            "current_panel_rank_unique": True,
        },
        "frozen_arc7": {
            "conditions": len(selected),
            "cell_line_id": context["cell_line_id"],
            "drugs": int(selected["drug"].nunique()),
            "doses": sorted(selected["dose_uM"].astype(float).unique().tolist()),
            "shared_control_cells": len(control),
            "treated_cells_per_condition": SET_SIZE,
            "selection_fingerprint": frozen["selection_fingerprint"],
            "condition_order_exact": True,
            "physical_indices_exact": True,
            "sampling_reused_without_resampling": True,
        },
        "old5k_subset_smoke": {
            "sources": {name: artifact(path) for name, path in OLD_SOURCE_PATHS.items()},
            "rows_per_source": 8,
            "columns": 100,
            "finite": True,
            "nonnegative": True,
            "column_order": "HD100 panel order via current_panel_rank",
        },
        "cell_eval": cell_eval,
        "hd100_training_gate": {
            "DDP_smoke": artifact(hd100.DDP_SMOKE_PATH),
            "ready_for_formal_training": True,
            "formal_training_complete": formal_complete,
        },
        "formal_inference_started": False,
        "formal_DE_or_seven_metric_evaluation_started": False,
        "provenance": {"script": artifact(SCRIPT_PATH), "task": artifact(TASK_PATH)},
        "next_stage": "formal HD100 training" if not formal_complete else "formal ARC7 build",
    }
    atomic_write_json(PREFLIGHT_PATH, result)
    del dataset
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def command_build(condition_batch_size: int, decoder_batch_size: int) -> dict[str, Any]:
    output_paths = [REAL_PATH, BUILD_RESULT_PATH, *OLD_PRED_PATHS.values(), *HD_PRED_PATHS.values()]
    existing = [display_path(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite formal HD100 ARC7 outputs: {existing}")
    panel = panel_contract()
    dataset, selected, context, control, treated, _, frozen = load_old_contract()
    started = time.perf_counter()

    real = subset_matrix(OLD_SOURCE_PATHS["real"], panel)
    old_predictions = {
        letter: subset_matrix(OLD_SOURCE_PATHS[letter], panel) for letter in ("A", "B", "C")
    }
    if not np.array_equal(old_predictions["A"][SET_SIZE:], old_predictions["B"][SET_SIZE:]):
        raise AssertionError("OLD5K subset A/B treated rows are not byte-identical")
    if not np.array_equal(old_predictions["B"][:SET_SIZE], old_predictions["C"][:SET_SIZE]):
        raise AssertionError("OLD5K subset B/C decoded-control rows are not byte-identical")
    hd_predictions, inference = build_hd_variants(
        dataset,
        selected,
        control,
        treated,
        real,
        condition_batch_size,
        decoder_batch_size,
    )

    real_obs = source_obs(OLD_SOURCE_PATHS["real"])
    pred_obs = source_obs(OLD_SOURCE_PATHS["A"])
    with tempfile.TemporaryDirectory(prefix="genejepa_hd100_arc7_build_", dir=RESULTS) as temporary:
        root = Path(temporary)
        staged_real = root / REAL_PATH.name
        write_h5ad(staged_real, real, real_obs, panel)
        staged: dict[Path, Path] = {REAL_PATH: staged_real}
        audits: dict[str, Any] = {}
        for letter, matrix in old_predictions.items():
            final = OLD_PRED_PATHS[letter]
            path = root / final.name
            write_h5ad(path, matrix, pred_obs, panel)
            audits[f"OLD5K_SUBSET100_{letter}"] = verify_pair(staged_real, path, selected)
            staged[final] = path
        for letter, matrix in hd_predictions.items():
            final = HD_PRED_PATHS[letter]
            path = root / final.name
            write_h5ad(path, matrix, pred_obs, panel)
            audits[f"HD100_{letter}"] = verify_pair(staged_real, path, selected)
            staged[final] = path
        for final, path in staged.items():
            os.replace(path, final)

    del real, old_predictions, hd_predictions
    gc.collect()
    result = {
        "schema": "genejepa_decoder_hd100_arc7_build_result_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "conditions": len(selected),
        "cells_per_set": SET_SIZE,
        "cells_per_anndata": 6144,
        "genes": 100,
        "expression_space": "log1p(CP10000)",
        "context": context,
        "selection_fingerprint": frozen["selection_fingerprint"],
        "same_conditions_cells_order_and_sampling_as_existing_ARC7": True,
        "old5k_subset": {
            "inference_rerun": False,
            "source_columns": "current_panel_rank",
            "A_B_treated_rows_byte_identical": True,
            "B_C_control_rows_byte_identical": True,
        },
        "hd100_variants": {
            "A": {
                "pred_control": "real control expression Top100",
                "pred_treated": "HD100 Decoder(ST-A predicted latent)",
            },
            "B": {
                "pred_control": "HD100 Decoder(true control latent)",
                "pred_treated": "same bytes as HD100-A treated rows",
            },
            "C": {
                "pred_control": "HD100 Decoder(true control latent)",
                "pred_treated": "HD100 Decoder(true treated latent)",
                "ST_A_used": False,
            },
            "A_B_treated_rows_byte_identical": True,
            "B_C_control_rows_byte_identical": True,
        },
        "inference": inference,
        "anndata_audit": audits,
        "inputs": {
            "panel": artifact(PANEL_PATH),
            "old_sources": {name: artifact(path) for name, path in OLD_SOURCE_PATHS.items()},
            "frozen_plan": artifact(arc7.PLAN_PATH),
            "frozen_conditions": artifact(arc7.CONDITIONS_PATH),
        },
        "outputs": {
            "real_h5ad": artifact(REAL_PATH),
            "old5k_subset100": {
                letter: artifact(path) for letter, path in OLD_PRED_PATHS.items()
            },
            "hd100": {letter: artifact(path) for letter, path in HD_PRED_PATHS.items()},
        },
        "provenance": {"script": artifact(SCRIPT_PATH), "task": artifact(TASK_PATH)},
        "genejepa_rerun": False,
        "training_performed": False,
        "formal_DE_or_seven_metric_evaluation_started": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_write_json(BUILD_RESULT_PATH, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def count_stats(values: pd.Series, denominator: int) -> dict[str, float]:
    array = pd.to_numeric(values).to_numpy(np.float64)
    if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > denominator):
        raise AssertionError("DEG counts are outside the panel universe")
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "max": float(array.max()),
        "fraction_min": float(array.min() / denominator),
        "fraction_median": float(np.median(array) / denominator),
        "fraction_mean": float(array.mean() / denominator),
        "fraction_max": float(array.max() / denominator),
    }


def result_row(
    model: str,
    variant: str,
    panel_name: str,
    genes: int,
    summary: pd.DataFrame,
    per_condition: pd.DataFrame,
    reference_only: bool,
) -> dict[str, Any]:
    metric_map = {
        str(row.metric): float(row.mean) for row in summary.itertuples(index=False)
    }
    expected = set(METRIC_LABELS.values())
    if set(metric_map) != expected:
        raise AssertionError(f"Seven-metric summary mismatch: {set(metric_map)}")
    predicted = count_stats(per_condition["pred_DEG_count"], genes)
    true = count_stats(per_condition["true_DEG_count"], genes)
    return {
        "Model": model,
        "Variant": variant,
        "Panel": panel_name,
        "Genes": genes,
        "DES": metric_map["DES"],
        "PDS": metric_map["PDS"],
        "MAE": metric_map["MAE"],
        "Pearson delta": metric_map["Pearson delta"],
        "Spearman logFC": metric_map["Spearman logFC"],
        "AUPRC": metric_map["AUPRC"],
        "Effect-size Spearman": metric_map["Spearman effect size"],
        "Pred DEG min": predicted["min"],
        "Pred DEG median": predicted["median"],
        "Pred DEG mean": predicted["mean"],
        "Pred DEG max": predicted["max"],
        "Median pred DEG %": 100.0 * predicted["fraction_median"],
        "True DEG min": true["min"],
        "True DEG median": true["median"],
        "True DEG mean": true["mean"],
        "True DEG max": true["max"],
        "Median true DEG %": 100.0 * true["fraction_median"],
        "Reference only": reference_only,
    }


def old_5000_reference_rows() -> list[dict[str, Any]]:
    summary = pd.read_csv(old_abc.ABC_SUMMARY_PATH, encoding="utf-8-sig")
    combined = pd.read_csv(old_abc.ABC_PER_CONDITION_PATH, encoding="utf-8-sig")
    rows = []
    variant_names = {value: key for key, value in old_abc.VARIANTS.items()}
    for name, frame in summary.groupby("variant", sort=False):
        letter = variant_names[str(name)]
        per = pd.DataFrame(
            {
                "true_DEG_count": pd.to_numeric(combined["true_DEG_count"]),
                "pred_DEG_count": pd.to_numeric(combined[f"{letter}_pred_DEG_count"]),
            }
        )
        rows.append(
            result_row(
                "Old Decoder",
                letter,
                "VARIANCE_TOP5000_REFERENCE",
                5000,
                frame,
                per,
                True,
            )
        )
    if len(rows) != 3:
        raise AssertionError("Existing 5000-gene A/B/C reference is incomplete")
    return rows


def comparison_differences(core: pd.DataFrame) -> list[dict[str, Any]]:
    indexed = core.set_index(["Panel", "Variant"])
    metrics = [
        "DES",
        "PDS",
        "MAE",
        "Pearson delta",
        "Spearman logFC",
        "AUPRC",
        "Effect-size Spearman",
        "Median pred DEG %",
    ]
    comparisons = []
    for letter in ("A", "B", "C"):
        comparisons.append(
            (
                "Comparison 1: OLD5K_SUBSET100 minus full 5000 reference",
                ("OLD5K_SUBSET100", letter),
                ("VARIANCE_TOP5000_REFERENCE", letter),
                False,
            )
        )
        comparisons.append(
            (
                "Comparison 2: HD100 minus OLD5K_SUBSET100",
                ("HD100", letter),
                ("OLD5K_SUBSET100", letter),
                True,
            )
        )
    comparisons.append(
        (
            "Comparison 3: HD100 C minus HD100 B",
            ("HD100", "C"),
            ("HD100", "B"),
            True,
        )
    )
    output: list[dict[str, Any]] = []
    for label, left, right, comparable in comparisons:
        for metric in metrics:
            output.append(
                {
                    "comparison": label,
                    "left": f"{left[0]}_{left[1]}",
                    "right": f"{right[0]}_{right[1]}",
                    "quantity": metric,
                    "left_minus_right": float(indexed.loc[left, metric] - indexed.loc[right, metric]),
                    "same_gene_universe": comparable,
                }
            )
    return output


def write_report(core: pd.DataFrame, differences: list[dict[str, Any]]) -> None:
    main = core.loc[~core["Reference only"]].copy()
    reference = core.loc[core["Reference only"]].copy()
    del main["Reference only"], reference["Reference only"]
    diff = pd.DataFrame.from_records(differences)
    text = [
        "# High-detection Top100 Decoder + ARC7 A/B/C comparison",
        "",
        "This is a high-detection extreme diagnostic, not a replacement production model.",
        "All six Top100 groups reuse the same frozen 23 conditions, condition order, physical cells, "
        "sampling indices, expression definition, and cell-eval 0.8.2 implementation.",
        "",
        "## Same-Top100 core result",
        "",
        old_abc.markdown_table(main),
        "",
        "## Existing full 5000-gene reference",
        "",
        old_abc.markdown_table(reference),
        "",
        "The full-5000 rows use a different gene universe. AUPRC, DEG counts/fractions, and other "
        "gene-level metrics must not be treated as directly comparable solely from raw differences.",
        "",
        "## Numeric differences",
        "",
        old_abc.markdown_table(diff),
        "",
        "Comparison 2 is the primary controlled comparison: the panel, conditions, cells, and metrics "
        "are identical; only training on 5000 outputs versus dedicated 100 outputs differs. Comparison "
        "3 isolates the remaining ST-A latent-prediction gap within the HD100 readout.",
        "",
        "No automatic success threshold or additional panel-size recommendation is imposed here.",
        "",
    ]
    temporary = REPORT_PATH.with_name(REPORT_PATH.name + ".tmp")
    temporary.write_text("\n".join(text), encoding="utf-8", newline="\n")
    os.replace(temporary, REPORT_PATH)


def command_evaluate(threads: int | None) -> dict[str, Any]:
    required = [BUILD_RESULT_PATH, REAL_PATH, *OLD_PRED_PATHS.values(), *HD_PRED_PATHS.values()]
    missing = [display_path(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Run the formal HD100 ARC7 build first: {missing}")
    formal_outputs = (SUMMARY_PATH, PER_CONDITION_PATH, RESULT_PATH, COMPARISON_PATH, REPORT_PATH)
    existing = [display_path(path) for path in formal_outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite formal HD100 ARC7 evaluation: {existing}")
    if CELL_EVAL_ROOT.exists() and any(CELL_EVAL_ROOT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {CELL_EVAL_ROOT}")

    build = read_json(BUILD_RESULT_PATH)
    if build.get("status") != "pass" or build.get("selection_fingerprint") != (
        EXPECTED_SELECTION_FINGERPRINT
    ):
        raise AssertionError("HD100 ARC7 build result is not PASS/current")
    for path, record in [(REAL_PATH, build["outputs"]["real_h5ad"])]:
        old_abc.assert_artifact(path, record, path.name)
    for letter, path in OLD_PRED_PATHS.items():
        old_abc.assert_artifact(path, build["outputs"]["old5k_subset100"][letter], path.name)
    for letter, path in HD_PRED_PATHS.items():
        old_abc.assert_artifact(path, build["outputs"]["hd100"][letter], path.name)

    _, selected, _, _, _, _, frozen = load_old_contract()
    if frozen["selection_fingerprint"] != build["selection_fingerprint"]:
        raise AssertionError("Frozen ARC7 selection changed after build")
    threads = arc7.available_threads() if threads is None else min(threads, arc7.available_threads())
    if threads < 1:
        raise ValueError("threads must be positive")
    started = time.perf_counter()
    summaries: dict[str, pd.DataFrame] = {}
    per_frames: dict[str, pd.DataFrame] = {}
    runtimes: dict[str, Any] = {}
    summary_outputs: list[pd.DataFrame] = []
    per_outputs: list[pd.DataFrame] = []
    failures: dict[str, Any] = {}

    for group, (model, variant, pred_path) in GROUPS.items():
        verify_pair(REAL_PATH, pred_path, selected)
        outdir = CELL_EVAL_ROOT / group.lower()
        summary, per_condition, official, runtime = arc7.run_cell_eval(
            REAL_PATH, pred_path, selected, outdir, threads
        )
        arc7.atomic_write_csv(outdir / "requested_seven_metrics.csv", official)
        if runtime["metric_failures"]:
            failures[group] = runtime["metric_failures"]
        summary = summary.copy()
        summary.insert(0, "group", group)
        summary.insert(1, "model", model)
        summary.insert(2, "variant", variant)
        summary.insert(3, "genes", 100)
        per_condition = per_condition.copy()
        per_condition.insert(0, "group", group)
        per_condition.insert(1, "model", model)
        per_condition.insert(2, "variant", variant)
        per_condition.insert(3, "genes", 100)
        per_condition["true_DEG_fraction"] = per_condition["true_DEG_count"] / 100.0
        per_condition["pred_DEG_fraction"] = per_condition["pred_DEG_count"] / 100.0
        summaries[group] = summary
        per_frames[group] = per_condition
        runtimes[group] = runtime
        summary_outputs.append(summary)
        per_outputs.append(per_condition)

    expected_pairs = selected["pair_id"].astype(str).tolist()
    true_counts: np.ndarray | None = None
    for group, frame in per_frames.items():
        if frame["pair_id"].astype(str).tolist() != expected_pairs:
            raise AssertionError(f"{group} per-condition order changed")
        current = pd.to_numeric(frame["true_DEG_count"]).to_numpy(np.int64)
        if true_counts is None:
            true_counts = current
        elif not np.array_equal(true_counts, current):
            raise AssertionError("True Top100 DEG counts differ across the six groups")

    combined_summary = pd.concat(summary_outputs, ignore_index=True)
    combined_per = pd.concat(per_outputs, ignore_index=True)
    arc7.atomic_write_csv(SUMMARY_PATH, combined_summary)
    arc7.atomic_write_csv(PER_CONDITION_PATH, combined_per)

    core_rows = old_5000_reference_rows()
    group_records: dict[str, Any] = {}
    for group, (model, variant, _) in GROUPS.items():
        panel_name = "OLD5K_SUBSET100" if model == "Old Decoder" else "HD100"
        summary = summaries[group]
        per = per_frames[group]
        core_rows.append(result_row(model, variant, panel_name, 100, summary, per, False))
        group_records[group] = {
            "status": "pass" if not runtimes[group]["metric_failures"] else "fail",
            "seven_metrics": arc7.summary_for_json(summary),
            "true_DEG": count_stats(per["true_DEG_count"], 100),
            "predicted_DEG": count_stats(per["pred_DEG_count"], 100),
            "runtime": runtimes[group],
            "pred_h5ad": artifact(GROUPS[group][2]),
        }
    core = pd.DataFrame.from_records(core_rows)
    panel_order = pd.Categorical(
        core["Panel"],
        categories=["VARIANCE_TOP5000_REFERENCE", "OLD5K_SUBSET100", "HD100"],
        ordered=True,
    )
    core = core.assign(_panel_order=panel_order).sort_values(
        ["_panel_order", "Variant"], kind="stable"
    ).drop(columns="_panel_order").reset_index(drop=True)
    arc7.atomic_write_csv(COMPARISON_PATH, core)
    differences = comparison_differences(core)
    write_report(core, differences)

    result = {
        "schema": "genejepa_decoder_hd100_arc7_result_v1",
        "created_at_utc": utc_now(),
        "status": "pass" if not failures else "fail",
        "experiment": "high-detection Top100 extreme diagnostic",
        "conditions": 23,
        "cells_per_set": SET_SIZE,
        "cells_per_anndata": 6144,
        "genes": 100,
        "selection_fingerprint": EXPECTED_SELECTION_FINGERPRINT,
        "same_conditions_cells_order_and_sampling_across_all_groups": True,
        "cell_eval": {
            "version": arc7.EXPECTED_CELL_EVAL_VERSION,
            "commit": arc7.EXPECTED_CELL_EVAL_COMMIT,
            "FDR": 0.05,
            "epsilon": 0.0,
            "threads": threads,
        },
        "groups": group_records,
        "true_DEG_identical_across_six_groups": True,
        "comparisons": differences,
        "comparison_guardrails": {
            "primary": "same Top100: HD100 versus OLD5K_SUBSET100",
            "full5000_reference": (
                "different gene universe; raw AUPRC/DEG counts and other gene-level metrics are not "
                "automatically directly comparable"
            ),
            "val_MSE": (
                "HD100 and 5000-gene Decoder validation MSE use different target distributions"
            ),
            "model_status": "HD100 is a diagnostic experiment, not a new formal main model",
        },
        "metric_failures": failures,
        "outputs": {
            "summary": artifact(SUMMARY_PATH),
            "per_condition": artifact(PER_CONDITION_PATH),
            "comparison": artifact(COMPARISON_PATH),
            "report": artifact(REPORT_PATH),
            "build_result": artifact(BUILD_RESULT_PATH),
        },
        "provenance": {
            "panel": artifact(PANEL_PATH),
            "script": artifact(SCRIPT_PATH),
            "task": artifact(TASK_PATH),
            "cell_eval": arc7.cell_eval_provenance(),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "additional_panel_training_started": False,
    }
    atomic_write_json(RESULT_PATH, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="Audit frozen panel/ARC7 reuse without inference or DE")
    build = commands.add_parser("build", help="Run formal HD100 inference and write six predictions")
    build.add_argument("--condition-batch-size", type=int, default=4)
    build.add_argument("--decoder-batch-size", type=int, default=2048)
    evaluate = commands.add_parser("evaluate", help="Run formal DE and seven metrics for six groups")
    evaluate.add_argument("--threads", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        command_preflight()
    elif args.command == "build":
        command_build(args.condition_batch_size, args.decoder_batch_size)
    elif args.command == "evaluate":
        command_evaluate(args.threads)


if __name__ == "__main__":
    main()
