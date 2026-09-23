#!/usr/bin/env python3
"""Run one deterministic real-test ST-A -> Decoder v1 inference demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch

from evaluate_tahoe_experiment1_b0 import atomic_write_csv
from evaluate_tahoe_experiment1_st_raw import load_checkpoint as load_st_checkpoint
from freeze_tahoe_experiment1_evaluation_sampling import index_sha256
from run_genejepa_decoder_v1 import (
    FORMAL_CHECKPOINT_DIR as DECODER_CHECKPOINT_DIR,
    FORMAL_RESULT_PATH as DECODER_TRAINING_RESULT,
    PRIMARY_BATCH_SIZE as DECODER_BATCH_SIZE,
    PRIMARY_GRADIENT_ACCUMULATION as DECODER_GRADIENT_ACCUMULATION,
    TRAINING_CONFIG_PATH as DECODER_TRAINING_CONFIG,
    build_decoder,
    model_fingerprint as decoder_fingerprint,
    validate_checkpoint as validate_decoder_checkpoint,
)
from run_tahoe_experiment1_st_formal import (
    ensure_protocol as ensure_st_protocol,
    model_fingerprint as st_fingerprint,
)
from tahoe_decoder_v1_data import (
    FORMAL_PLANS,
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
    make_dataloader,
)


SCRIPT_PATH = Path(__file__).resolve()
TASK_PATH = PROJECT_ROOT.parent / "当前任务.txt"
PANEL_PATH = RESULTS / "genejepa_decoder_v1_gene_panel.csv"
DECODER_CHECKPOINT = DECODER_CHECKPOINT_DIR / "best.pt"
RESULT_PATH = RESULTS / "genejepa_decoder_v1_demo_result.json"
PREDICTED_MATRIX_PATH = RESULTS / "genejepa_decoder_v1_demo_predicted_treated_256x5000.csv"
MATRICES_PATH = RESULTS / "genejepa_decoder_v1_demo_expression_matrices.npz"
PSEUDOBULK_PATH = RESULTS / "genejepa_decoder_v1_demo_pseudobulk_5000genes.csv"
TOP20_PATH = RESULTS / "genejepa_decoder_v1_demo_top20_delta_genes.csv"
SCATTER_PATH = RESULTS / "genejepa_decoder_v1_demo_pseudobulk_scatter.png"
TOP20_PLOT_PATH = RESULTS / "genejepa_decoder_v1_demo_top20_delta.png"

EXPECTED_PANEL_SHA256 = "c4f79cfb0a37ec278d04c3f0045fa568872961445a9f251fff37dfa3d38d790e"
EXPECTED_DECODER_SHA256 = "29772ddf20bc36dde956fcd0befecbc0b71aaf4f1a8dd07fdf7195c5e6b590d0"
EXPECTED_DECODER_EPOCH = 9
EXPECTED_ST_EPOCH = 28
OWNER_CONTROL = 1


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": display_path(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_npz(path: Path, **matrices: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **matrices)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def one_side_index_sha256(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def first_batch(dataset: TahoeExperiment1LatentSetDataset) -> tuple[dict[str, Any], Any]:
    loader = make_dataloader(
        dataset,
        batch_size=1,
        shuffle=False,
        seed=42,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )
    if loader.drop_last or type(loader.sampler).__name__ != "SequentialSampler":
        raise AssertionError("Demo DataLoader must use shuffle=False and drop_last=False")
    return next(iter(loader)), loader


def select_and_sample_condition() -> tuple[
    TahoeExperiment1LatentSetDataset, dict[str, Any], pd.Series, dict[str, Any]
]:
    dataset = TahoeExperiment1LatentSetDataset(split="test", seed=42, epoch=0)
    eligible = dataset.conditions.loc[
        (dataset.conditions["treated_cached_cell_count"] >= SET_SIZE)
        & (dataset.conditions["control_cached_cell_count"] >= SET_SIZE)
    ].sort_values("cache_condition_index", kind="stable")
    if eligible.empty:
        raise AssertionError("No eligible frozen test condition is available")
    selected = eligible.iloc[0]
    if int(selected["cache_condition_index"]) != int(dataset.conditions.iloc[0]["cache_condition_index"]):
        raise AssertionError("First eligible test condition is not the first frozen test condition")

    batch, loader = first_batch(dataset)
    if batch["condition_id"] != [str(selected["pair_id"])] or batch["split"] != ["test"]:
        raise AssertionError("Sequential demo sampling did not select the expected first test condition")

    second = TahoeExperiment1LatentSetDataset(split="test", seed=42, epoch=0)
    second_batch, _ = first_batch(second)
    reproducible = all(
        torch.equal(batch[key], second_batch[key])
        for key in ("source_embedding_index", "target_embedding_index")
    )
    if not reproducible or second_batch["condition_id"] != batch["condition_id"]:
        raise AssertionError("Two independent Dataset constructions sampled different cell indices")

    source = batch["source_embedding_index"][0].numpy().astype(np.int64, copy=True)
    target = batch["target_embedding_index"][0].numpy().astype(np.int64, copy=True)
    if len(source) != SET_SIZE or len(target) != SET_SIZE:
        raise AssertionError("A sampled set does not contain exactly 256 cells")
    if len(np.unique(source)) != SET_SIZE or len(np.unique(target)) != SET_SIZE:
        raise AssertionError("Within-set replacement was detected")
    if np.intersect1d(source, target).size:
        raise AssertionError("Control and treated embedding indices overlap")
    if dataset.embedding_transforms:
        raise AssertionError("Frozen latent Dataset unexpectedly applies a transform")
    expected_shapes = {
        "ctrl_cell_emb": (1, SET_SIZE, LATENT_DIM),
        "pert_cell_emb": (1, SET_SIZE, LATENT_DIM),
        "pert_emb": (1, SET_SIZE, PERT_DIM),
    }
    for key, expected in expected_shapes.items():
        if tuple(batch[key].shape) != expected or batch[key].dtype != torch.float32:
            raise AssertionError(f"Unexpected {key} shape/dtype: {batch[key].shape}/{batch[key].dtype}")
        if not torch.isfinite(batch[key]).all():
            raise AssertionError(f"Non-finite values in {key}")

    sampling = {
        "S": SET_SIZE,
        "base_seed": 42,
        "repeat_epoch": 0,
        "shuffle": False,
        "drop_last": False,
        "replace": False,
        "dataset_class": type(dataset).__name__,
        "dataset_sampling_version": "tahoe_experiment1_set_v1",
        "independent_reconstruction_exact": True,
        "control_embedding_indices": source.tolist(),
        "control_embedding_indices_sha256": one_side_index_sha256(source),
        "treated_embedding_indices": target.tolist(),
        "treated_embedding_indices_sha256": one_side_index_sha256(target),
        "combined_frozen_style_indices_sha256": index_sha256(
            [str(selected["pair_id"])], source[None, :], target[None, :]
        ),
        "per_side_sha256_encoding": "little-endian int64 bytes in sampled order",
        "dataloader_sampler": type(loader.sampler).__name__,
    }
    del second_batch, second
    return dataset, batch, selected, sampling


def locate_cells(indices: np.ndarray) -> pd.DataFrame:
    plan = ds.dataset([str(path) for path in FORMAL_PLANS], format="parquet")
    columns = [
        "embedding_index",
        "shard_path",
        "row_group_index",
        "row_index_in_row_group",
        "row_index_in_shard",
        "owner_type",
        "owner_index",
    ]
    table = plan.to_table(
        columns=columns,
        filter=ds.field("embedding_index").isin(np.asarray(indices, dtype=np.int64).tolist()),
    )
    locators = table.to_pandas()
    observed = locators["embedding_index"].to_numpy(np.int64)
    if len(observed) != len(indices) or len(np.unique(observed)) != len(indices):
        raise AssertionError("Frozen plans did not return exactly one locator per selected cell")
    if set(observed.tolist()) != set(np.asarray(indices, dtype=np.int64).tolist()):
        raise AssertionError("Frozen plan locator union differs from selected embedding indices")
    return locators


def load_true_expression(
    control_indices: np.ndarray,
    treated_indices: np.ndarray,
    selected: pd.Series,
    panel_indices: np.ndarray,
    token_lookup: np.ndarray,
    panel_lookup: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    all_indices = np.concatenate((control_indices, treated_indices))
    locators = locate_cells(all_indices)
    control_set = set(control_indices.tolist())
    treated_set = set(treated_indices.tolist())
    condition_records = load_conditions()
    condition_index = int(selected["cache_condition_index"])
    formal_condition = condition_records.iloc[condition_index]
    if str(formal_condition["pair_id"]) != str(selected["pair_id"]):
        raise AssertionError("Decoder condition table is not aligned by cache_condition_index")

    expected = {
        "control": {
            "owner_type": OWNER_CONTROL,
            "owner_index": int(selected["control_pool_index"]),
            "drug": str(selected["control_drug"]),
            "samples": set(str(selected["control_samples"]).split("|")),
        },
        "treated": {
            "owner_type": OWNER_TREATED,
            "owner_index": condition_index,
            "drug": str(selected["drug"]),
            "samples": set(str(formal_condition["treated_samples"]).split("|")),
        },
    }
    targets: dict[int, np.ndarray] = {}
    target_audits: list[dict[str, Any]] = []
    source_files: dict[str, pq.ParquetFile] = {}
    source_row_groups = 0

    for (shard_path, row_group_index), group in locators.groupby(
        ["shard_path", "row_group_index"], sort=False
    ):
        shard_path = str(shard_path)
        source = source_files.get(shard_path)
        if source is None:
            source = pq.ParquetFile(PROJECT_ROOT / shard_path)
            source_files[shard_path] = source
        row_group_index = int(row_group_index)
        raw = source.read_row_group(row_group_index, columns=RAW_COLUMNS)
        row_group_start = sum(
            source.metadata.row_group(index).num_rows for index in range(row_group_index)
        )
        source_row_groups += 1
        for locator in group.itertuples(index=False):
            embedding_index = int(locator.embedding_index)
            side = "control" if embedding_index in control_set else "treated"
            if embedding_index not in control_set and embedding_index not in treated_set:
                raise AssertionError("Unexpected embedding index in locator result")
            contract = expected[side]
            if int(locator.owner_type) != contract["owner_type"] or int(locator.owner_index) != contract[
                "owner_index"
            ]:
                raise AssertionError(f"{side} locator owner does not match the selected condition")
            row = int(locator.row_index_in_row_group)
            if row < 0 or row >= raw.num_rows:
                raise AssertionError("row_index_in_row_group is out of range")
            if row_group_start + row != int(locator.row_index_in_shard):
                raise AssertionError("Stable row locator components disagree")
            metadata = {name: raw[name][row].as_py() for name in RAW_COLUMNS[2:]}
            if (
                str(metadata["plate"]) != str(selected["plate"])
                or str(metadata["cell_line_id"]) != str(selected["cell_line_id"])
                or str(metadata["drug"]) != contract["drug"]
                or str(metadata["sample"]) not in contract["samples"]
            ):
                raise AssertionError(f"Raw Tahoe metadata disagrees with {side} locator ownership")
            target, target_audit = log1p_cp10k_target(
                raw["genes"][row].as_py(),
                raw["expressions"][row].as_py(),
                token_lookup,
                panel_indices,
                panel_lookup=panel_lookup,
            )
            if embedding_index in targets:
                raise AssertionError("A selected embedding index was decoded from raw Tahoe twice")
            targets[embedding_index] = target
            target_audits.append(target_audit)

    if set(targets) != set(all_indices.tolist()):
        raise AssertionError("True-expression target retrieval silently skipped a selected cell")
    control = np.stack([targets[int(index)] for index in control_indices]).astype(np.float32, copy=False)
    treated = np.stack([targets[int(index)] for index in treated_indices]).astype(np.float32, copy=False)
    audit = {
        "selected_cells": len(targets),
        "control_cells": len(control),
        "treated_cells": len(treated),
        "source_shards": len(source_files),
        "source_row_groups": source_row_groups,
        "stable_locator_components_verified": True,
        "owner_and_raw_metadata_verified": True,
        "sentinel_removed_cells": sum(bool(row["sentinel_removed"]) for row in target_audits),
        "unmapped_entries": sum(int(row["unmapped_entries"]) for row in target_audits),
        "duplicate_gene_entries_collapsed": sum(
            int(row["duplicate_gene_entries_collapsed"]) for row in target_audits
        ),
        "target_transform": "full mapped counts -> CP10000 -> log1p -> frozen panel",
    }
    return control, treated, audit


def load_decoder(device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    training_result = json.loads(DECODER_TRAINING_RESULT.read_text(encoding="utf-8"))
    checkpoint_sha = sha256_file(DECODER_CHECKPOINT)
    declared = training_result.get("checkpoints", {}).get("best", {})
    if (
        training_result.get("status") != "pass"
        or training_result.get("best_epoch") != EXPECTED_DECODER_EPOCH
        or declared.get("path") != display_path(DECODER_CHECKPOINT)
        or declared.get("sha256") != EXPECTED_DECODER_SHA256
        or checkpoint_sha != EXPECTED_DECODER_SHA256
    ):
        raise AssertionError("Formal Decoder result/checkpoint provenance changed")
    config_sha = sha256_file(DECODER_TRAINING_CONFIG)
    payload = torch.load(DECODER_CHECKPOINT, map_location="cpu", weights_only=False)
    validate_decoder_checkpoint(
        payload,
        mode="formal",
        kind="best",
        config_sha=config_sha,
        batch_size=DECODER_BATCH_SIZE,
        gradient_accumulation=DECODER_GRADIENT_ACCUMULATION,
    )
    if payload.get("epoch") != EXPECTED_DECODER_EPOCH or payload.get("best_epoch") != EXPECTED_DECODER_EPOCH:
        raise AssertionError("Decoder best.pt is not epoch 9")
    model = build_decoder()
    model.load_state_dict(payload["model_state"], strict=True)
    fingerprint = decoder_fingerprint(model)
    if fingerprint != payload["model_fingerprint"]:
        raise AssertionError("Decoder state did not restore exactly")
    model.to(device).eval()
    if model.training or any(module.training for module in model.modules()):
        raise AssertionError("Decoder eval mode did not propagate to all modules")
    record = {
        "path": display_path(DECODER_CHECKPOINT),
        "sha256": checkpoint_sha,
        "checkpoint_kind": "best",
        "epoch": int(payload["epoch"]),
        "best_epoch": int(payload["best_epoch"]),
        "best_val_mse": float(payload["best_val_mse"]),
        "model_fingerprint": fingerprint,
        "training_result": artifact(DECODER_TRAINING_RESULT),
        "training_config": artifact(DECODER_TRAINING_CONFIG),
        "architecture": [768, 1024, 1024, 512, 5000],
        "final_activation": "Softplus(beta=1, threshold=20)",
    }
    del payload
    return model, record


def validate_expression(name: str, value: np.ndarray) -> dict[str, Any]:
    if value.shape != (SET_SIZE, 5000) or value.dtype != np.float32:
        raise AssertionError(f"{name} has unexpected shape/dtype: {value.shape}/{value.dtype}")
    if not np.isfinite(value).all() or np.any(value < 0):
        raise AssertionError(f"{name} is not finite and non-negative")
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": True,
        "nonnegative": True,
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean(dtype=np.float64)),
    }


def mse(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    value = float(np.mean(difference * difference, dtype=np.float64))
    if not math.isfinite(value):
        raise AssertionError("MSE is non-finite")
    return value


def pearson(left: np.ndarray, right: np.ndarray) -> tuple[float | None, dict[str, Any]]:
    left64 = np.asarray(left, dtype=np.float64).reshape(-1)
    right64 = np.asarray(right, dtype=np.float64).reshape(-1)
    if left64.shape != right64.shape or not np.isfinite(left64).all() or not np.isfinite(right64).all():
        raise AssertionError("Pearson inputs are incompatible or non-finite")
    centered_left = left64 - left64.mean()
    centered_right = right64 - right64.mean()
    denominator = float(np.linalg.norm(centered_left) * np.linalg.norm(centered_right))
    if denominator == 0.0:
        return None, {
            "defined": False,
            "reason": "zero variance",
            "left_std": float(left64.std()),
            "right_std": float(right64.std()),
        }
    value = float(np.dot(centered_left, centered_right) / denominator)
    if not math.isfinite(value):
        return None, {
            "defined": False,
            "reason": "non-finite Pearson result",
            "left_std": float(left64.std()),
            "right_std": float(right64.std()),
        }
    return value, {
        "defined": True,
        "reason": None,
        "left_std": float(left64.std()),
        "right_std": float(right64.std()),
    }


def save_plots(
    pseudobulk: pd.DataFrame,
    top20: pd.DataFrame,
    selected: pd.Series,
    predicted_pearson: float | None,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.5, 6.0))
    x = pseudobulk["true_treated_mean_logcp10k"].to_numpy()
    y = pseudobulk["st_a_decoder_predicted_mean_logcp10k"].to_numpy()
    axis.scatter(x, y, s=8, alpha=0.35, linewidths=0)
    low, high = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    axis.plot([low, high], [low, high], linestyle="--", linewidth=1, color="black")
    correlation = "undefined" if predicted_pearson is None else f"{predicted_pearson:.4f}"
    axis.set_xlabel("True treated pseudobulk, log1p(CP10000)")
    axis.set_ylabel("ST-A -> Decoder predicted pseudobulk")
    axis.set_title(
        f"Pearson r = {correlation}\n{selected['cell_line_id']} / {selected['drug']} / "
        f"{float(selected['dose_uM']):g} uM"
    )
    figure.tight_layout()
    temporary = SCATTER_PATH.with_name(SCATTER_PATH.stem + ".tmp" + SCATTER_PATH.suffix)
    figure.savefig(temporary, dpi=160)
    plt.close(figure)
    os.replace(temporary, SCATTER_PATH)

    positions = np.arange(len(top20))
    figure, axis = plt.subplots(figsize=(9.0, 7.0))
    width = 0.38
    axis.barh(positions - width / 2, top20["true_delta_logcp10k"], height=width, label="True")
    axis.barh(
        positions + width / 2,
        top20["predicted_delta_logcp10k"],
        height=width,
        label="ST-A -> Decoder",
    )
    axis.set_yticks(positions, top20["gene_symbol"])
    axis.invert_yaxis()
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Treated - control pseudobulk, log1p(CP10000)")
    axis.set_title("Top 20 genes by |true perturbation delta|")
    axis.legend()
    figure.tight_layout()
    temporary = TOP20_PLOT_PATH.with_name(TOP20_PLOT_PATH.stem + ".tmp" + TOP20_PLOT_PATH.suffix)
    figure.savefig(temporary, dpi=160)
    plt.close(figure)
    os.replace(temporary, TOP20_PLOT_PATH)
    return {
        "matplotlib_available": True,
        "pseudobulk_scatter": artifact(SCATTER_PATH),
        "top20_delta_plot": artifact(TOP20_PLOT_PATH),
    }


def run() -> dict[str, Any]:
    started = time.perf_counter()
    if SET_SIZE != 256 or LATENT_DIM != 768 or PERT_DIM != 380:
        raise AssertionError("Frozen Experiment 1 dimensions changed")
    if not torch.cuda.is_available():
        raise RuntimeError("This demo requires one visible CUDA GPU")
    device = torch.device("cuda:0")

    dataset, batch, selected, sampling = select_and_sample_condition()
    source_indices = np.asarray(sampling["control_embedding_indices"], dtype=np.int64)
    target_indices = np.asarray(sampling["treated_embedding_indices"], dtype=np.int64)

    genes, token_lookup = load_gene_universe(GENE_METADATA)
    if sha256_file(PANEL_PATH) != EXPECTED_PANEL_SHA256:
        raise AssertionError("Frozen 5000-gene panel SHA-256 changed")
    panel_indices = load_panel(PANEL_PATH, genes)
    panel_lookup = build_panel_lookup(panel_indices, vocabulary_size=len(genes))
    panel = pd.read_csv(PANEL_PATH, encoding="utf-8-sig", keep_default_na=False).sort_values(
        "panel_rank", kind="stable"
    ).reset_index(drop=True)
    if len(panel) != 5000:
        raise AssertionError("Frozen panel no longer contains 5000 genes")
    symbols = panel["gene_symbol"].astype(str).tolist()
    if all(symbols) and len(set(symbols)) == len(symbols):
        gene_columns = symbols
        gene_column_scheme = "gene_symbol"
    else:
        gene_columns = [
            f"{symbol}|{ensembl}"
            for symbol, ensembl in zip(panel["gene_symbol"], panel["ensembl_id"], strict=True)
        ]
        gene_column_scheme = "gene_symbol|ensembl_id"
    if len(gene_columns) != 5000 or len(set(gene_columns)) != 5000:
        raise AssertionError("Predicted-expression gene columns are not stable and unique")

    true_control, true_treated, target_audit = load_true_expression(
        source_indices,
        target_indices,
        selected,
        panel_indices,
        token_lookup,
        panel_lookup,
    )

    st_protocol, st_protocol_sha = ensure_st_protocol(create=False)
    st_model, st_checkpoint = load_st_checkpoint("st-a", device, st_protocol_sha)
    if (
        st_checkpoint["epoch"] != EXPECTED_ST_EPOCH
        or st_checkpoint["best_epoch"] != EXPECTED_ST_EPOCH
        or st_model.predict_residual
        or st_model.final_activation_name != "identity"
        or st_model.apply_output_relu
    ):
        raise AssertionError("Loaded ST-A does not satisfy the frozen absolute signed-output contract")
    st_before = st_checkpoint["model_state_sha256_before_evaluation"]
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    inference_started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        predicted_latent = st_model(
            {
                "ctrl_cell_emb": batch["ctrl_cell_emb"].to(device),
                "pert_emb": batch["pert_emb"].to(device),
            }
        ).reshape(1, SET_SIZE, LATENT_DIM)
    torch.cuda.synchronize(device)
    st_seconds = time.perf_counter() - inference_started
    predicted_latent = predicted_latent.float().cpu().numpy()
    if predicted_latent.shape != (1, SET_SIZE, LATENT_DIM) or not np.isfinite(predicted_latent).all():
        raise AssertionError("ST-A prediction shape/finite check failed")
    if not float(predicted_latent.min()) < 0 < float(predicted_latent.max()):
        raise AssertionError("ST-A predicted latent did not retain signed coordinates")
    if st_fingerprint(st_model) != st_before:
        raise AssertionError("ST-A parameters changed during inference")
    st_peak_vram = int(torch.cuda.max_memory_allocated(device))
    del st_model
    torch.cuda.empty_cache()

    decoder, decoder_checkpoint = load_decoder(device)
    decoder_before = decoder_checkpoint["model_fingerprint"]
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    inference_started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        decoder_true_latent = decoder(batch["pert_cell_emb"][0].to(device)).float()
        predicted_expression = decoder(
            torch.from_numpy(predicted_latent[0]).to(device)
        ).float()
    torch.cuda.synchronize(device)
    decoder_seconds = time.perf_counter() - inference_started
    decoder_true_latent = decoder_true_latent.cpu().numpy().astype(np.float32, copy=False)
    predicted_expression = predicted_expression.cpu().numpy().astype(np.float32, copy=False)
    if decoder_fingerprint(decoder) != decoder_before:
        raise AssertionError("Decoder parameters changed during inference")
    decoder_peak_vram = int(torch.cuda.max_memory_allocated(device))
    del decoder
    torch.cuda.empty_cache()

    matrix_audit = {
        "true_control": validate_expression("true_control", true_control),
        "true_treated": validate_expression("true_treated", true_treated),
        "decoder_true_latent": validate_expression("decoder_true_latent", decoder_true_latent),
        "st_a_decoder_predicted_treated": validate_expression(
            "st_a_decoder_predicted_treated", predicted_expression
        ),
    }
    control_mean = true_control.mean(axis=0, dtype=np.float64)
    treated_mean = true_treated.mean(axis=0, dtype=np.float64)
    decoder_mean = decoder_true_latent.mean(axis=0, dtype=np.float64)
    predicted_mean = predicted_expression.mean(axis=0, dtype=np.float64)
    true_delta = treated_mean - control_mean
    predicted_delta = predicted_mean - control_mean

    decoder_r, decoder_r_audit = pearson(decoder_mean, treated_mean)
    predicted_r, predicted_r_audit = pearson(predicted_mean, treated_mean)
    delta_r, delta_r_audit = pearson(predicted_delta, true_delta)
    metrics = {
        "decoder_ceiling_cell_gene_mse": mse(decoder_true_latent, true_treated),
        "decoder_ceiling_pseudobulk_pearson": decoder_r,
        "decoder_ceiling_pseudobulk_mse": mse(decoder_mean, treated_mean),
        "predicted_vs_true_treated_pseudobulk_pearson": predicted_r,
        "predicted_vs_true_treated_pseudobulk_mse": mse(predicted_mean, treated_mean),
        "delta_pearson": delta_r,
        "delta_mse": mse(predicted_delta, true_delta),
    }

    predicted_frame = pd.DataFrame(predicted_expression, columns=gene_columns)
    predicted_frame.insert(0, "predicted_cell_index", np.arange(SET_SIZE, dtype=np.int32))
    atomic_write_csv(PREDICTED_MATRIX_PATH, predicted_frame)
    atomic_write_npz(
        MATRICES_PATH,
        true_control=true_control,
        true_treated=true_treated,
        decoder_true_latent=decoder_true_latent,
        st_a_decoder_predicted_treated=predicted_expression,
    )

    pseudobulk = panel[["panel_rank", "genejepa_index", "gene_symbol", "ensembl_id"]].copy()
    pseudobulk["control_mean_logcp10k"] = control_mean
    pseudobulk["true_treated_mean_logcp10k"] = treated_mean
    pseudobulk["decoder_true_latent_mean_logcp10k"] = decoder_mean
    pseudobulk["st_a_decoder_predicted_mean_logcp10k"] = predicted_mean
    pseudobulk["true_delta_logcp10k"] = true_delta
    pseudobulk["predicted_delta_logcp10k"] = predicted_delta
    atomic_write_csv(PSEUDOBULK_PATH, pseudobulk)

    top_positions = np.argsort(-np.abs(true_delta), kind="stable")[:20]
    top20 = pseudobulk.iloc[top_positions][
        [
            "panel_rank",
            "genejepa_index",
            "gene_symbol",
            "ensembl_id",
            "control_mean_logcp10k",
            "true_treated_mean_logcp10k",
            "st_a_decoder_predicted_mean_logcp10k",
            "true_delta_logcp10k",
            "predicted_delta_logcp10k",
        ]
    ].copy()
    top20.insert(0, "true_abs_delta_rank", np.arange(1, 21, dtype=np.int32))
    atomic_write_csv(TOP20_PATH, top20)
    plot_outputs = save_plots(pseudobulk, top20, selected, predicted_r)

    with np.load(MATRICES_PATH) as saved:
        if set(saved.files) != {
            "true_control",
            "true_treated",
            "decoder_true_latent",
            "st_a_decoder_predicted_treated",
        } or any(saved[name].shape != (SET_SIZE, 5000) for name in saved.files):
            raise AssertionError("Written NPZ did not round-trip with the four required matrices")
    if predicted_frame.shape != (SET_SIZE, 5001) or pseudobulk.shape[0] != 5000 or len(top20) != 20:
        raise AssertionError("Written report tables do not have the required logical dimensions")

    condition = {
        "pair_id": str(selected["pair_id"]),
        "cache_condition_index": int(selected["cache_condition_index"]),
        "edge_id": str(selected["edge_id"]),
        "split": str(selected["split"]),
        "plate": str(selected["plate"]),
        "cell_line_id": str(selected["cell_line_id"]),
        "drug": str(selected["drug"]),
        "dose_uM": float(selected["dose_uM"]),
        "treated_cached_cell_count": int(selected["treated_cached_cell_count"]),
        "control_pool_id": str(selected["control_pool_id"]),
        "control_cached_cell_count": int(selected["control_cached_cell_count"]),
    }
    result = {
        "schema": "genejepa_decoder_v1_demo_result_v1",
        "status": "pass",
        "created_at_utc": utc_now(),
        "scope": "one deterministic real frozen test condition; inference only",
        "condition": condition,
        "condition_selection": {
            "rule": "ascending cache_condition_index; first test condition with treated/control >=256",
            "frozen_test_conditions": len(dataset),
            "eligible_test_conditions": int(
                (
                    (dataset.conditions["treated_cached_cell_count"] >= SET_SIZE)
                    & (dataset.conditions["control_cached_cell_count"] >= SET_SIZE)
                ).sum()
            ),
            "selected_without_model_metric_inspection": True,
        },
        "sampling": sampling,
        "checkpoints": {
            "st_a": st_checkpoint,
            "decoder": decoder_checkpoint,
        },
        "panel": {
            **artifact(PANEL_PATH),
            "rows": len(panel),
            "gene_column_scheme": gene_column_scheme,
            "gene_columns_unique": True,
        },
        "expression_space": "log1p(CP10000)",
        "latent_contract": {
            "cache_rows_equal_global_embedding_index": True,
            "input_transforms": [],
            "st_a_output_raw_signed": True,
            "centering": False,
            "whitening": False,
            "normalization": False,
            "relu_or_rectification": False,
        },
        "inference": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "visible_cuda_devices": torch.cuda.device_count(),
            "eval_mode": True,
            "torch_inference_mode": True,
            "autocast": "bfloat16",
            "final_output_and_metrics_dtype": "float32/float64 accumulation",
            "st_a_seconds": st_seconds,
            "decoder_seconds": decoder_seconds,
            "st_a_peak_allocated_vram_bytes": st_peak_vram,
            "decoder_peak_allocated_vram_bytes": decoder_peak_vram,
        },
        "st_a_output": {
            "shape": list(predicted_latent.shape),
            "dtype_after_fp32_export": str(predicted_latent.dtype),
            "finite": True,
            "min": float(predicted_latent.min()),
            "max": float(predicted_latent.max()),
            "negative_ratio": float(np.mean(predicted_latent < 0)),
        },
        "shapes": {name: details["shape"] for name, details in matrix_audit.items()},
        "matrix_validation": matrix_audit,
        "true_expression_retrieval": target_audit,
        "metrics": metrics,
        "pearson_diagnostics": {
            "decoder_ceiling_pseudobulk": decoder_r_audit,
            "predicted_vs_true_treated_pseudobulk": predicted_r_audit,
            "delta": delta_r_audit,
        },
        "metric_interpretation": {
            "decoder_ceiling_cell_gene_mse_is_paired": True,
            "st_a_cellwise_paired_mse_reported": False,
            "reason": "ST-A is a set-to-set population predictor; predicted and true cells are not paired",
        },
        "top20_selection": "descending abs(true_delta_logcp10k), stable panel order tie-break",
        "top20_genes": top20["gene_symbol"].astype(str).tolist(),
        "outputs": {
            "predicted_treated_256x5000_csv": artifact(PREDICTED_MATRIX_PATH),
            "expression_matrices_npz": artifact(MATRICES_PATH),
            "pseudobulk_5000genes_csv": artifact(PSEUDOBULK_PATH),
            "top20_delta_genes_csv": artifact(TOP20_PATH),
            **plot_outputs,
            "result_json": {"path": display_path(RESULT_PATH)},
        },
        "provenance": {
            "demo_script": artifact(SCRIPT_PATH),
            "task": artifact(TASK_PATH),
            "st_training_protocol": {
                "sha256": st_protocol_sha,
                "schema": st_protocol.get("schema"),
            },
        },
        "runtime_seconds": time.perf_counter() - started,
        "training_performed": False,
        "genejepa_rerun": False,
        "full_test_evaluation_performed": False,
    }
    atomic_write_json(RESULT_PATH, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "condition": condition,
                "st_a_output_shape": result["st_a_output"]["shape"],
                "predicted_expression_shape": matrix_audit[
                    "st_a_decoder_predicted_treated"
                ]["shape"],
                "metrics": metrics,
                "top20_genes": result["top20_genes"],
                "result": display_path(RESULT_PATH),
                "predicted_matrix": display_path(PREDICTED_MATRIX_PATH),
                "runtime_seconds": result["runtime_seconds"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="run", choices=("run",))
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
