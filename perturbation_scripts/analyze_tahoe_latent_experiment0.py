#!/usr/bin/env python3
"""Run the read-only Tahoe Experiment 0 latent-space audit.

The raw Epoch25 embedding file is opened with mmap_mode="r" and is never
rewritten. Centering and DMSO-baseline residualization exist only as derived
in-memory diagnostics; no centered or whitened embedding matrix is saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient
from scipy.stats import rankdata, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_EMBEDDINGS = RESULTS_DIR / "tahoe_latent_audit_epoch25_embeddings.npy"
DEFAULT_MANIFEST = RESULTS_DIR / "tahoe_latent_audit_epoch25_manifest.json"
DEFAULT_UNIQUE_CELLS = RESULTS_DIR / "tahoe_latent_audit_unique_cells.csv"
DEFAULT_CELL_INDEX_SUMMARY = RESULTS_DIR / "tahoe_latent_audit_cell_index_summary.json"
DEFAULT_COHORT = RESULTS_DIR / "tahoe_latent_audit_cohort.csv"
DEFAULT_SET_INDICES = RESULTS_DIR / "tahoe_latent_audit_set_cell_indices.csv"
DEFAULT_LOCAL_MANIFEST = PROJECT_ROOT / "hf_data_cache" / "local_file_manifest.json"
DEFAULT_GENE_METADATA = (
    PROJECT_ROOT / "hf_data_cache" / "metadata" / "metadata" / "gene_metadata.parquet"
)
DEFAULT_GLOBAL_STATS = PROJECT_ROOT / "hf_data_cache" / "global_stats.json"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "genejepa_quarter_d12_h6_700k_e30_seed42_run1"
    / "scjepa-epoch=25-val_loss=0.179.ckpt"
)

INTEGRITY_OUTPUT = RESULTS_DIR / "tahoe_latent_audit_epoch25_integrity_audit.json"
SUMMARY_OUTPUT = RESULTS_DIR / "tahoe_latent_audit_experiment0_summary.json"
DIMENSION_OUTPUT = RESULTS_DIR / "tahoe_latent_audit_experiment0_dimensions.csv"
PCA_OUTPUT = RESULTS_DIR / "tahoe_latent_audit_experiment0_pca_spectrum.csv"
PAIR_HIST_OUTPUT = RESULTS_DIR / "tahoe_latent_audit_experiment0_pairwise_cosine_hist.csv"
SET_OUTPUT = RESULTS_DIR / "tahoe_latent_audit_experiment0_set_shifts.csv"
STABILITY_OUTPUT = RESULTS_DIR / "tahoe_latent_audit_experiment0_shift_stability.csv"
REPLICATE_OUTPUT = RESULTS_DIR / "tahoe_latent_audit_experiment0_plate_replicates.csv"
DOSE_OUTPUT = RESULTS_DIR / "tahoe_latent_audit_experiment0_dose_geometry.csv"
RESIDUAL_PAIR_OUTPUT = RESULTS_DIR / "tahoe_latent_audit_experiment0_residual_pairs.csv"
NOTEBOOK_OUTPUT = RESULTS_DIR / "tahoe_latent_audit_experiment0.ipynb"

EXPECTED_CELLS = 129_061
EXPECTED_DIM = 768
EXPECTED_PAIR_ROWS = 360
EXPECTED_SET_ROWS = 184_320
EXPECTED_SET_SIZE = 256
EXPECTED_REPEATS = 5
EXPECTED_PERTURBATION_CONDITIONS = 60
EXPECTED_NULL_CONDITIONS = 12
EXPECTED_REPLICATE_GROUPS = 15

QUANTILES = (
    (0.0, "min"),
    (0.01, "p01"),
    (0.05, "p05"),
    (0.25, "p25"),
    (0.50, "median"),
    (0.75, "p75"),
    (0.95, "p95"),
    (0.99, "p99"),
    (1.0, "max"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def display_path(path: Path) -> str:
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


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def describe(values: np.ndarray | list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0}
    result: dict[str, float | int] = {"count": int(array.size)}
    for quantile, name in QUANTILES:
        result[name] = float(np.quantile(array, quantile))
    result["mean"] = float(array.mean())
    result["std"] = float(array.std())
    return result


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else float("nan")


def row_cosines(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.einsum("ij,ij->i", left, right)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def upper_triangle_cosines(vectors: np.ndarray) -> np.ndarray:
    pairs = list(combinations(range(len(vectors)), 2))
    left = np.asarray([vectors[i] for i, _ in pairs], dtype=np.float64)
    right = np.asarray([vectors[j] for _, j in pairs], dtype=np.float64)
    return row_cosines(left, right)


def effective_rank(eigenvalues: np.ndarray) -> dict[str, float | int]:
    values = np.clip(np.asarray(eigenvalues, dtype=np.float64), 0.0, None)
    total = float(values.sum())
    if total == 0:
        return {
            "entropy_effective_rank": 0.0,
            "participation_ratio": 0.0,
            "positive_eigenvalues": 0,
        }
    probabilities = values / total
    positive = probabilities > 0
    entropy_rank = math.exp(
        -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    )
    participation = float(total * total / np.square(values).sum())
    return {
        "entropy_effective_rank": entropy_rank,
        "participation_ratio": participation,
        "positive_eigenvalues": int(np.count_nonzero(values > 0)),
    }


def pca_summary(eigenvalues_descending: np.ndarray) -> dict[str, object]:
    values = np.clip(eigenvalues_descending, 0.0, None)
    ratios = values / values.sum()
    cumulative = np.cumsum(ratios)

    def components_for(fraction: float) -> int:
        return int(np.searchsorted(cumulative, fraction) + 1)

    return {
        **effective_rank(values),
        "top1_explained_variance_ratio": float(ratios[0]),
        "top5_explained_variance_ratio": float(ratios[:5].sum()),
        "top10_explained_variance_ratio": float(ratios[:10].sum()),
        "components_for_50_percent": components_for(0.50),
        "components_for_80_percent": components_for(0.80),
        "components_for_90_percent": components_for(0.90),
        "components_for_95_percent": components_for(0.95),
        "components_for_99_percent": components_for(0.99),
        "total_variance": float(values.sum()),
    }


def verify_hash_chain(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
    with args.manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with args.cell_index_summary.open("r", encoding="utf-8") as handle:
        cell_index_summary = json.load(handle)

    expected = {
        args.embeddings: manifest["output"]["embedding_sha256"],
        args.unique_cells: manifest["inputs"]["unique_cells_sha256"],
        args.cell_index_summary: manifest["inputs"]["cell_index_summary_sha256"],
        args.local_manifest: manifest["inputs"]["local_manifest_sha256"],
        args.gene_metadata: manifest["inputs"]["gene_metadata_sha256"],
        args.global_stats: manifest["inputs"]["global_stats_sha256"],
        args.checkpoint: manifest["checkpoint"]["sha256"],
        args.cohort: cell_index_summary["inputs"]["cohort_sha256"],
        args.set_indices: cell_index_summary["outputs"]["set_indices_sha256"],
    }
    checks = []
    for path, expected_hash in expected.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_hash = sha256_file(path)
        match = observed_hash == expected_hash
        checks.append(
            {
                "path": display_path(path),
                "size_bytes": path.stat().st_size,
                "expected_sha256": expected_hash,
                "observed_sha256": observed_hash,
                "match": match,
            }
        )
        if not match:
            raise AssertionError(f"SHA-256 mismatch: {path}")

    required_manifest = {
        "status": "pass",
        "mode": "full",
        "success_count": EXPECTED_CELLS,
        "failure_count": 0,
    }
    for key, expected_value in required_manifest.items():
        if manifest[key] != expected_value:
            raise AssertionError(f"Manifest {key}={manifest[key]!r}, expected {expected_value!r}")
    if manifest["output"]["shape"] != [EXPECTED_CELLS, EXPECTED_DIM]:
        raise AssertionError("Unexpected manifest embedding shape")
    if manifest["output"]["dtype"] != "float32":
        raise AssertionError("Unexpected manifest embedding dtype")
    contract = manifest["embedding_index_contract"]
    if not (
        contract["output_row_equals_embedding_index"]
        and contract["all_indices_written_once"]
        and contract["first_index"] == 0
        and contract["last_index"] == EXPECTED_CELLS - 1
    ):
        raise AssertionError("Manifest embedding-index contract failed")

    return manifest, {
        "status": "pass",
        "hash_checks": checks,
        "manifest_contract": required_manifest,
        "embedding_index_contract": contract,
    }


def load_and_verify_tables(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    unique_cells = pd.read_csv(
        args.unique_cells,
        usecols=["embedding_index", "cell_id"],
        dtype={"embedding_index": np.int64, "cell_id": str},
    ).sort_values("embedding_index")
    cohort = pd.read_csv(args.cohort, keep_default_na=False)
    memberships = pd.read_csv(
        args.set_indices,
        dtype={"embedding_index": np.int64, "cell_id": str},
    )

    expected_indices = np.arange(EXPECTED_CELLS, dtype=np.int64)
    if len(unique_cells) != EXPECTED_CELLS:
        raise AssertionError("Unexpected unique-cell row count")
    if not np.array_equal(unique_cells["embedding_index"].to_numpy(), expected_indices):
        raise AssertionError("unique_cells embedding_index is not contiguous")
    if unique_cells["cell_id"].duplicated().any():
        raise AssertionError("Duplicate unique-cell cell_id")
    if len(cohort) != EXPECTED_PAIR_ROWS or cohort["audit_pair_id"].duplicated().any():
        raise AssertionError("Unexpected cohort grain")
    if len(memberships) != EXPECTED_SET_ROWS:
        raise AssertionError("Unexpected set-membership row count")
    if memberships["embedding_index"].min() < 0 or memberships["embedding_index"].max() >= EXPECTED_CELLS:
        raise AssertionError("Set membership has out-of-range embedding_index")

    unique_ids = unique_cells["cell_id"].to_numpy()
    membership_indices = memberships["embedding_index"].to_numpy()
    if not np.array_equal(unique_ids[membership_indices], memberships["cell_id"].to_numpy()):
        raise AssertionError("set cell_id does not match unique_cells embedding_index")

    grouped = memberships.groupby(["audit_pair_id", "side"], sort=False)
    sizes = grouped.size()
    unique_sizes = grouped["embedding_index"].nunique()
    position_sizes = grouped["set_position"].nunique()
    if (
        len(sizes) != EXPECTED_PAIR_ROWS * 2
        or not (sizes == EXPECTED_SET_SIZE).all()
        or not (unique_sizes == EXPECTED_SET_SIZE).all()
        or not (position_sizes == EXPECTED_SET_SIZE).all()
    ):
        raise AssertionError("A set side is not exactly 256 unique cells")
    if set(memberships["audit_pair_id"]) != set(cohort["audit_pair_id"]):
        raise AssertionError("Cohort and membership pair IDs differ")

    maximum_source_target_overlap = 0
    for _, group in memberships.groupby("audit_pair_id", sort=False):
        source = set(group.loc[group["side"] == "source", "embedding_index"])
        target = set(group.loc[group["side"] == "target", "embedding_index"])
        maximum_source_target_overlap = max(
            maximum_source_target_overlap, len(source & target)
        )
    if maximum_source_target_overlap:
        raise AssertionError("Source and target overlap within an audit pair")

    condition_counts = cohort.groupby("comparison_type")["condition_id"].nunique()
    if condition_counts.to_dict() != {
        "dmso_null": EXPECTED_NULL_CONDITIONS,
        "perturbation": EXPECTED_PERTURBATION_CONDITIONS,
    }:
        raise AssertionError(f"Unexpected condition counts: {condition_counts.to_dict()}")
    repeats = cohort.groupby(["comparison_type", "condition_id"])["repeat_index"].nunique()
    if not (repeats == EXPECTED_REPEATS).all():
        raise AssertionError("Every condition must contain five repeat indices")

    audit = {
        "unique_cells": len(unique_cells),
        "cohort_pairs": len(cohort),
        "perturbation_pair_repeats": int((cohort["comparison_type"] == "perturbation").sum()),
        "dmso_null_pair_repeats": int((cohort["comparison_type"] == "dmso_null").sum()),
        "perturbation_conditions": int(condition_counts["perturbation"]),
        "dmso_null_conditions": int(condition_counts["dmso_null"]),
        "set_sides": int(len(sizes)),
        "set_membership_rows": len(memberships),
        "set_size": EXPECTED_SET_SIZE,
        "all_sets_unique_without_replacement": True,
        "maximum_source_target_overlap": maximum_source_target_overlap,
        "cell_id_embedding_index_matches": True,
    }
    return unique_cells, cohort, memberships, audit


def profile_embeddings(
    embeddings: np.memmap,
    manifest: dict[str, object],
    chunk_size: int,
    pair_samples: int,
    seed: int,
) -> tuple[
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
]:
    rows, dimensions = embeddings.shape
    dimension_sum = np.zeros(dimensions, dtype=np.float64)
    dimension_sumsq = np.zeros(dimensions, dtype=np.float64)
    dimension_min = np.full(dimensions, np.inf, dtype=np.float64)
    dimension_max = np.full(dimensions, -np.inf, dtype=np.float64)
    dimension_negative = np.zeros(dimensions, dtype=np.int64)
    dimension_positive = np.zeros(dimensions, dtype=np.int64)
    raw_norms = np.empty(rows, dtype=np.float64)
    global_min = math.inf
    global_max = -math.inf
    negative_count = 0
    positive_count = 0
    zero_count = 0
    finite = True

    print("[1/5] Scanning full signed/per-dimension statistics...", flush=True)
    for start in range(0, rows, chunk_size):
        end = min(start + chunk_size, rows)
        values = np.asarray(embeddings[start:end], dtype=np.float64)
        finite &= bool(np.isfinite(values).all())
        global_min = min(global_min, float(values.min()))
        global_max = max(global_max, float(values.max()))
        negative_count += int(np.count_nonzero(values < 0))
        positive_count += int(np.count_nonzero(values > 0))
        zero_count += int(np.count_nonzero(values == 0))
        dimension_sum += values.sum(axis=0)
        dimension_sumsq += np.square(values).sum(axis=0)
        dimension_min = np.minimum(dimension_min, values.min(axis=0))
        dimension_max = np.maximum(dimension_max, values.max(axis=0))
        dimension_negative += np.count_nonzero(values < 0, axis=0)
        dimension_positive += np.count_nonzero(values > 0, axis=0)
        raw_norms[start:end] = np.linalg.norm(values, axis=1)
    if not finite:
        raise AssertionError("Embedding contains non-finite values")

    dimension_mean = dimension_sum / rows
    dimension_variance = np.maximum(
        0.0, dimension_sumsq / rows - np.square(dimension_mean)
    )
    dimension_std = np.sqrt(dimension_variance)
    value_count = rows * dimensions
    global_mean_scalar = float(dimension_sum.sum() / value_count)
    global_sumsq_scalar = float(dimension_sumsq.sum())
    global_std_scalar = math.sqrt(
        max(0.0, global_sumsq_scalar / value_count - global_mean_scalar**2)
    )

    manifest_stats = manifest["embedding_statistics"]
    comparisons = {
        "min": global_min,
        "max": global_max,
        "mean": global_mean_scalar,
        "std": global_std_scalar,
        "negative_ratio": negative_count / value_count,
    }
    for key, observed in comparisons.items():
        if not math.isclose(observed, float(manifest_stats[key]), rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"Recomputed embedding {key} differs from manifest")

    print("[2/5] Building full centered covariance/PCA diagnostics...", flush=True)
    covariance = np.zeros((dimensions, dimensions), dtype=np.float64)
    centered_norms = np.empty(rows, dtype=np.float64)
    for start in range(0, rows, chunk_size):
        end = min(start + chunk_size, rows)
        centered = np.asarray(embeddings[start:end], dtype=np.float64) - dimension_mean
        covariance += centered.T @ centered
        centered_norms[start:end] = np.linalg.norm(centered, axis=1)
    covariance /= rows
    covariance = (covariance + covariance.T) * 0.5
    raw_eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    minimum_raw_eigenvalue = float(raw_eigenvalues.min())
    if minimum_raw_eigenvalue < -1e-9:
        raise AssertionError(f"Centered covariance has negative eigenvalue {minimum_raw_eigenvalue}")
    eigenvalues = np.clip(raw_eigenvalues, 0.0, None)
    explained = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(explained)
    pca_frame = pd.DataFrame(
        {
            "component": np.arange(1, dimensions + 1),
            "eigenvalue": eigenvalues,
            "explained_variance_ratio": explained,
            "cumulative_explained_variance_ratio": cumulative,
        }
    )

    covariance_std = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    if not np.allclose(covariance_std, dimension_std, rtol=1e-10, atol=1e-10):
        raise AssertionError("Per-dimension std disagrees with covariance diagonal")
    median_dimension_std = float(np.median(dimension_std))
    dimension_frame = pd.DataFrame(
        {
            "dimension": np.arange(dimensions),
            "mean": dimension_mean,
            "std": dimension_std,
            "min": dimension_min,
            "max": dimension_max,
            "negative_ratio": dimension_negative / rows,
            "positive_ratio": dimension_positive / rows,
            "near_constant_abs_std_lt_1e_6": dimension_std < 1e-6,
            "near_constant_abs_std_lt_1e_3": dimension_std < 1e-3,
            "near_constant_abs_std_lt_1e_2": dimension_std < 1e-2,
            "near_constant_rel_lt_1pct_median_std": dimension_std
            < 0.01 * median_dimension_std,
            "near_constant_rel_lt_5pct_median_std": dimension_std
            < 0.05 * median_dimension_std,
            "nonnegative": dimension_min >= 0,
            "nonpositive": dimension_max <= 0,
            "both_signs": (dimension_min < 0) & (dimension_max > 0),
        }
    )

    print(f"[3/5] Sampling {pair_samples:,} raw and centered cell pairs...", flush=True)
    rng = np.random.default_rng(seed)
    left_indices = rng.integers(0, rows, size=pair_samples, dtype=np.int64)
    right_indices = rng.integers(0, rows, size=pair_samples, dtype=np.int64)
    same = left_indices == right_indices
    right_indices[same] = (right_indices[same] + 1) % rows
    raw_cosines = np.empty(pair_samples, dtype=np.float64)
    centered_cosines = np.empty(pair_samples, dtype=np.float64)
    pair_chunk_size = min(chunk_size, 4096)
    for start in range(0, pair_samples, pair_chunk_size):
        end = min(start + pair_chunk_size, pair_samples)
        left = np.asarray(embeddings[left_indices[start:end]], dtype=np.float64)
        right = np.asarray(embeddings[right_indices[start:end]], dtype=np.float64)
        raw_cosines[start:end] = row_cosines(left, right)
        centered_cosines[start:end] = row_cosines(
            left - dimension_mean, right - dimension_mean
        )
    raw_cosines = np.clip(raw_cosines, -1.0, 1.0)
    centered_cosines = np.clip(centered_cosines, -1.0, 1.0)
    histogram_edges = np.linspace(-1.0, 1.0, 201)
    raw_counts, _ = np.histogram(raw_cosines, bins=histogram_edges)
    centered_counts, _ = np.histogram(centered_cosines, bins=histogram_edges)
    pair_histogram = pd.DataFrame(
        {
            "bin_left": histogram_edges[:-1],
            "bin_right": histogram_edges[1:],
            "raw_count": raw_counts,
            "centered_count": centered_counts,
        }
    )

    mean_vector_norm = float(np.linalg.norm(dimension_mean))
    mean_raw_squared_norm = float(np.mean(np.square(raw_norms)))
    mean_centered_squared_norm = float(np.mean(np.square(centered_norms)))
    pca = pca_summary(eigenvalues)
    pca["minimum_unclipped_eigenvalue"] = minimum_raw_eigenvalue
    numerical = {
        "shape": [rows, dimensions],
        "dtype": str(embeddings.dtype),
        "finite": finite,
        "global_min": global_min,
        "global_max": global_max,
        "global_mean": global_mean_scalar,
        "global_std": global_std_scalar,
        "negative_count": negative_count,
        "negative_ratio": negative_count / value_count,
        "positive_count": positive_count,
        "positive_ratio": positive_count / value_count,
        "zero_count": zero_count,
        "zero_ratio": zero_count / value_count,
        "signed_representation": bool(negative_count and positive_count),
        "embedding_norm": describe(raw_norms),
        "global_mean_vector_norm": mean_vector_norm,
        "centered_embedding_norm": describe(centered_norms),
        "mean_vector_squared_fraction_of_raw_second_moment": (
            mean_vector_norm**2 / mean_raw_squared_norm
        ),
        "centered_squared_fraction_of_raw_second_moment": (
            mean_centered_squared_norm / mean_raw_squared_norm
        ),
        "per_dimension": {
            "mean": describe(dimension_mean),
            "std": describe(dimension_std),
            "median_std": median_dimension_std,
            "near_constant_abs_std_lt_1e_6": int(np.count_nonzero(dimension_std < 1e-6)),
            "near_constant_abs_std_lt_1e_3": int(np.count_nonzero(dimension_std < 1e-3)),
            "near_constant_abs_std_lt_1e_2": int(np.count_nonzero(dimension_std < 1e-2)),
            "near_constant_rel_lt_1pct_median_std": int(
                np.count_nonzero(dimension_std < 0.01 * median_dimension_std)
            ),
            "near_constant_rel_lt_5pct_median_std": int(
                np.count_nonzero(dimension_std < 0.05 * median_dimension_std)
            ),
            "nonnegative_dimensions": int(np.count_nonzero(dimension_min >= 0)),
            "nonpositive_dimensions": int(np.count_nonzero(dimension_max <= 0)),
            "both_signs_dimensions": int(
                np.count_nonzero((dimension_min < 0) & (dimension_max > 0))
            ),
        },
        "raw_pairwise_cosine": {
            "sampling": "uniform independent cell-index pairs; self-pairs deterministically shifted",
            "seed": seed,
            "sample_pairs": pair_samples,
            "distribution": describe(raw_cosines),
        },
        "global_mean_centered_pairwise_cosine": {
            "same_pairs_as_raw": True,
            "distribution": describe(centered_cosines),
        },
        "centered_pca": pca,
    }
    return numerical, dimension_frame, pca_frame, pair_histogram, dimension_mean


def compute_set_centroids(
    embeddings: np.memmap,
    memberships: pd.DataFrame,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], float]]:
    centroids: dict[tuple[str, str], np.ndarray] = {}
    radii: dict[tuple[str, str], float] = {}
    for key, group in memberships.groupby(["audit_pair_id", "side"], sort=False):
        ordered = group.sort_values("set_position")
        indices = ordered["embedding_index"].to_numpy(dtype=np.int64)
        values = np.asarray(embeddings[indices], dtype=np.float64)
        centroid = values.mean(axis=0)
        centroids[key] = centroid
        radii[key] = float(np.sqrt(np.mean(np.sum(np.square(values - centroid), axis=1))))
    return centroids, radii


def build_set_shifts(
    cohort: pd.DataFrame,
    centroids: dict[tuple[str, str], np.ndarray],
    radii: dict[tuple[str, str], float],
    global_mean_vector: np.ndarray,
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[tuple[str, int], tuple[str, np.ndarray]],
]:
    deltas: dict[str, np.ndarray] = {}
    records: dict[str, dict[str, object]] = {}
    for row in cohort.itertuples(index=False):
        pair_id = row.audit_pair_id
        source = centroids[(pair_id, "source")]
        target = centroids[(pair_id, "target")]
        delta = target - source
        deltas[pair_id] = delta
        source_radius = radii[(pair_id, "source")]
        target_radius = radii[(pair_id, "target")]
        pooled_radius = math.sqrt((source_radius**2 + target_radius**2) / 2)
        records[pair_id] = {
            "audit_pair_id": pair_id,
            "comparison_type": row.comparison_type,
            "condition_id": row.condition_id,
            "null_group_id": row.null_group_id,
            "repeat_index": int(row.repeat_index),
            "plate": row.plate,
            "cell_line_id": row.cell_line_id,
            "target_drug": row.target_drug,
            "target_dose_uM": (
                float(row.target_dose_uM) if row.target_dose_uM != "" else np.nan
            ),
            "dose_series_group_id": row.dose_series_group_id,
            "replicate_group_id": row.replicate_group_id,
            "source_centroid_norm": float(np.linalg.norm(source)),
            "target_centroid_norm": float(np.linalg.norm(target)),
            "source_global_centered_centroid_norm": float(
                np.linalg.norm(source - global_mean_vector)
            ),
            "target_global_centered_centroid_norm": float(
                np.linalg.norm(target - global_mean_vector)
            ),
            "raw_centroid_cosine": cosine(source, target),
            "global_centered_centroid_cosine": cosine(
                source - global_mean_vector, target - global_mean_vector
            ),
            "source_within_set_rms_radius": source_radius,
            "target_within_set_rms_radius": target_radius,
            "pooled_within_set_rms_radius": pooled_radius,
            "delta_norm": float(np.linalg.norm(delta)),
            "delta_over_pooled_within_rms": float(np.linalg.norm(delta) / pooled_radius),
            "matched_null_pair_id": "",
            "matched_null_delta_norm": np.nan,
            "delta_norm_over_matched_null": np.nan,
            "delta_cosine_with_matched_null": np.nan,
            "dmso_baseline_residual_delta_norm": np.nan,
        }

    null_lookup: dict[tuple[str, int], tuple[str, np.ndarray]] = {}
    for row in cohort.loc[cohort["comparison_type"] == "dmso_null"].itertuples(index=False):
        key = (row.null_group_id, int(row.repeat_index))
        if key in null_lookup:
            raise AssertionError(f"Duplicate null lookup key: {key}")
        null_lookup[key] = (row.audit_pair_id, deltas[row.audit_pair_id])

    residual_deltas: dict[str, np.ndarray] = {}
    for row in cohort.loc[cohort["comparison_type"] == "perturbation"].itertuples(index=False):
        pair_id = row.audit_pair_id
        key = (row.null_group_id, int(row.repeat_index))
        if key not in null_lookup:
            raise AssertionError(f"Missing matched DMSO null: {key}")
        null_pair_id, null_delta = null_lookup[key]
        delta = deltas[pair_id]
        residual = delta - null_delta
        residual_deltas[pair_id] = residual
        records[pair_id].update(
            {
                "matched_null_pair_id": null_pair_id,
                "matched_null_delta_norm": float(np.linalg.norm(null_delta)),
                "delta_norm_over_matched_null": float(
                    np.linalg.norm(delta) / np.linalg.norm(null_delta)
                ),
                "delta_cosine_with_matched_null": cosine(delta, null_delta),
                "dmso_baseline_residual_delta_norm": float(np.linalg.norm(residual)),
            }
        )

    frame = pd.DataFrame([records[pair_id] for pair_id in cohort["audit_pair_id"]])
    return frame, deltas, residual_deltas, null_lookup


def stability_fields(prefix: str, vectors: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(vectors, axis=1)
    pair_cosines = upper_triangle_cosines(vectors)
    mean_vector = vectors.mean(axis=0)
    deviations = np.linalg.norm(vectors - mean_vector, axis=1)
    pair_distances = np.asarray(
        [np.linalg.norm(vectors[i] - vectors[j]) for i, j in combinations(range(len(vectors)), 2)],
        dtype=np.float64,
    )
    return {
        f"{prefix}_repeat_norm_mean": float(norms.mean()),
        f"{prefix}_repeat_norm_std": float(norms.std()),
        f"{prefix}_repeat_norm_cv": float(norms.std() / norms.mean()),
        f"{prefix}_mean_vector_norm": float(np.linalg.norm(mean_vector)),
        f"{prefix}_directional_coherence": float(
            np.linalg.norm(mean_vector) / norms.mean()
        ),
        f"{prefix}_repeat_deviation_mean": float(deviations.mean()),
        f"{prefix}_pairwise_distance_mean": float(pair_distances.mean()),
        f"{prefix}_pairwise_cosine_min": float(np.min(pair_cosines)),
        f"{prefix}_pairwise_cosine_median": float(np.median(pair_cosines)),
        f"{prefix}_pairwise_cosine_mean": float(np.mean(pair_cosines)),
        f"{prefix}_pairwise_cosine_max": float(np.max(pair_cosines)),
    }


def build_stability(
    cohort: pd.DataFrame,
    deltas: dict[str, np.ndarray],
    residual_deltas: dict[str, np.ndarray],
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    records = []
    mean_deltas: dict[str, np.ndarray] = {}
    mean_residual_deltas: dict[str, np.ndarray] = {}
    repeat_deltas: dict[str, np.ndarray] = {}
    repeat_residuals: dict[str, np.ndarray] = {}
    for (_, condition_id), group in cohort.groupby(
        ["comparison_type", "condition_id"], sort=False
    ):
        ordered = group.sort_values("repeat_index")
        pair_ids = ordered["audit_pair_id"].tolist()
        vectors = np.asarray([deltas[pair_id] for pair_id in pair_ids])
        if len(vectors) != EXPECTED_REPEATS:
            raise AssertionError("Stability group does not contain five repeats")
        first = ordered.iloc[0]
        record: dict[str, object] = {
            "comparison_type": first["comparison_type"],
            "condition_id": condition_id,
            "null_group_id": first["null_group_id"],
            "plate": first["plate"],
            "cell_line_id": first["cell_line_id"],
            "target_drug": first["target_drug"],
            "target_dose_uM": (
                float(first["target_dose_uM"])
                if first["target_dose_uM"] != ""
                else np.nan
            ),
            "dose_series_group_id": first["dose_series_group_id"],
            "replicate_group_id": first["replicate_group_id"],
            "repeat_count": len(vectors),
            **stability_fields("delta", vectors),
        }
        mean_deltas[condition_id] = vectors.mean(axis=0)
        repeat_deltas[condition_id] = vectors
        if first["comparison_type"] == "perturbation":
            residual_vectors = np.asarray(
                [residual_deltas[pair_id] for pair_id in pair_ids]
            )
            record.update(stability_fields("residual_delta", residual_vectors))
            mean_residual_deltas[condition_id] = residual_vectors.mean(axis=0)
            repeat_residuals[condition_id] = residual_vectors
        records.append(record)
    return (
        pd.DataFrame(records),
        mean_deltas,
        mean_residual_deltas,
        repeat_deltas,
        repeat_residuals,
    )


def add_matched_null_condition_metrics(stability: pd.DataFrame) -> pd.DataFrame:
    result = stability.copy()
    null_norm = result.loc[result["comparison_type"] == "dmso_null"].set_index(
        "condition_id"
    )["delta_repeat_norm_mean"]
    result["matched_null_repeat_norm_mean"] = np.nan
    result["delta_repeat_norm_over_matched_null"] = np.nan
    perturbation = result["comparison_type"] == "perturbation"
    matched = result.loc[perturbation, "null_group_id"].map(null_norm)
    if matched.isna().any():
        raise AssertionError("Condition stability is missing a matched null")
    result.loc[perturbation, "matched_null_repeat_norm_mean"] = matched.to_numpy()
    result.loc[perturbation, "delta_repeat_norm_over_matched_null"] = (
        result.loc[perturbation, "delta_repeat_norm_mean"].to_numpy()
        / matched.to_numpy()
    )
    return result


def build_plate_replicates(
    cohort: pd.DataFrame,
    mean_deltas: dict[str, np.ndarray],
    mean_residuals: dict[str, np.ndarray],
    repeat_deltas: dict[str, np.ndarray],
    repeat_residuals: dict[str, np.ndarray],
) -> pd.DataFrame:
    conditions = cohort.loc[cohort["comparison_type"] == "perturbation"].drop_duplicates(
        "condition_id"
    )
    high_dose = conditions[conditions["target_dose_uM"].astype(float) == 5.0]
    records = []
    for (cell_line, drug), group in high_dose.groupby(
        ["cell_line_id", "target_drug"], sort=True
    ):
        by_plate = group.set_index("plate")
        if set(by_plate.index) != {"plate6", "plate14"}:
            raise AssertionError("High-dose replicate lacks plate6 or plate14")
        condition6 = by_plate.loc["plate6", "condition_id"]
        condition14 = by_plate.loc["plate14", "condition_id"]
        raw6, raw14 = mean_deltas[condition6], mean_deltas[condition14]
        residual6, residual14 = (
            mean_residuals[condition6],
            mean_residuals[condition14],
        )
        matched_raw_cosines = row_cosines(
            repeat_deltas[condition6], repeat_deltas[condition14]
        )
        matched_residual_cosines = row_cosines(
            repeat_residuals[condition6], repeat_residuals[condition14]
        )
        records.append(
            {
                "cell_line_id": cell_line,
                "drug": drug,
                "dose_uM": 5.0,
                "plate6_condition_id": condition6,
                "plate14_condition_id": condition14,
                "mean_delta_cosine": cosine(raw6, raw14),
                "mean_residual_delta_cosine": cosine(residual6, residual14),
                "plate6_mean_delta_norm": float(np.linalg.norm(raw6)),
                "plate14_mean_delta_norm": float(np.linalg.norm(raw14)),
                "plate14_over_plate6_delta_norm": float(
                    np.linalg.norm(raw14) / np.linalg.norm(raw6)
                ),
                "plate6_mean_residual_delta_norm": float(np.linalg.norm(residual6)),
                "plate14_mean_residual_delta_norm": float(np.linalg.norm(residual14)),
                "plate14_over_plate6_residual_norm": float(
                    np.linalg.norm(residual14) / np.linalg.norm(residual6)
                ),
                "matched_repeat_delta_cosine_min": float(matched_raw_cosines.min()),
                "matched_repeat_delta_cosine_median": float(
                    np.median(matched_raw_cosines)
                ),
                "matched_repeat_delta_cosine_mean": float(matched_raw_cosines.mean()),
                "matched_repeat_delta_cosine_max": float(matched_raw_cosines.max()),
                "matched_repeat_residual_cosine_min": float(
                    matched_residual_cosines.min()
                ),
                "matched_repeat_residual_cosine_median": float(
                    np.median(matched_residual_cosines)
                ),
                "matched_repeat_residual_cosine_mean": float(
                    matched_residual_cosines.mean()
                ),
                "matched_repeat_residual_cosine_max": float(
                    matched_residual_cosines.max()
                ),
            }
        )
    frame = pd.DataFrame(records)
    if len(frame) != EXPECTED_REPLICATE_GROUPS:
        raise AssertionError("Unexpected plate6/plate14 replicate-group count")
    return frame


def build_dose_geometry(
    cohort: pd.DataFrame,
    mean_deltas: dict[str, np.ndarray],
    mean_residuals: dict[str, np.ndarray],
    replicate_frame: pd.DataFrame,
) -> pd.DataFrame:
    conditions = cohort.loc[cohort["comparison_type"] == "perturbation"].drop_duplicates(
        "condition_id"
    )
    replicate_lookup = replicate_frame.set_index(["cell_line_id", "drug"])
    records = []
    for (cell_line, drug), group in conditions.groupby(
        ["cell_line_id", "target_drug"], sort=True
    ):
        by_plate = group.set_index("plate")
        if set(by_plate.index) != {"plate4", "plate5", "plate6", "plate14"}:
            raise AssertionError("Dose series lacks an expected plate")
        ids = {plate: by_plate.loc[plate, "condition_id"] for plate in by_plate.index}
        raw = {plate: mean_deltas[condition_id] for plate, condition_id in ids.items()}
        residual = {
            plate: mean_residuals[condition_id] for plate, condition_id in ids.items()
        }
        raw_high = (raw["plate6"] + raw["plate14"]) / 2
        residual_high = (residual["plate6"] + residual["plate14"]) / 2
        raw_magnitudes = np.asarray(
            [np.linalg.norm(raw["plate4"]), np.linalg.norm(raw["plate5"]), np.linalg.norm(raw_high)]
        )
        residual_magnitudes = np.asarray(
            [
                np.linalg.norm(residual["plate4"]),
                np.linalg.norm(residual["plate5"]),
                np.linalg.norm(residual_high),
            ]
        )
        log_doses = np.log10([0.05, 0.5, 5.0])
        raw_rho = float(spearmanr(log_doses, raw_magnitudes).statistic)
        residual_rho = float(spearmanr(log_doses, residual_magnitudes).statistic)
        replicate = replicate_lookup.loc[(cell_line, drug)]
        records.append(
            {
                "cell_line_id": cell_line,
                "drug": drug,
                "dose_plate_confound": True,
                "low_dose_uM": 0.05,
                "mid_dose_uM": 0.5,
                "high_dose_uM": 5.0,
                "low_plate": "plate4",
                "mid_plate": "plate5",
                "high_replicate_plates": "plate6|plate14",
                "low_mean_delta_norm": raw_magnitudes[0],
                "mid_mean_delta_norm": raw_magnitudes[1],
                "high_replicate_average_delta_norm": raw_magnitudes[2],
                "raw_magnitude_spearman_rho": raw_rho,
                "raw_magnitude_nondecreasing": bool(
                    raw_magnitudes[0] <= raw_magnitudes[1] <= raw_magnitudes[2]
                ),
                "low_mean_residual_norm": residual_magnitudes[0],
                "mid_mean_residual_norm": residual_magnitudes[1],
                "high_replicate_average_residual_norm": residual_magnitudes[2],
                "residual_magnitude_spearman_rho": residual_rho,
                "residual_magnitude_nondecreasing": bool(
                    residual_magnitudes[0]
                    <= residual_magnitudes[1]
                    <= residual_magnitudes[2]
                ),
                "residual_low_mid_cosine": cosine(
                    residual["plate4"], residual["plate5"]
                ),
                "residual_mid_high_cosine": cosine(
                    residual["plate5"], residual_high
                ),
                "residual_low_high_cosine": cosine(
                    residual["plate4"], residual_high
                ),
                "high_plate6_plate14_residual_cosine": float(
                    replicate["mean_residual_delta_cosine"]
                ),
            }
        )
    frame = pd.DataFrame(records)
    if len(frame) != EXPECTED_REPLICATE_GROUPS:
        raise AssertionError("Unexpected dose-series count")
    return frame


def auc_probability_greater(positive: np.ndarray, negative: np.ndarray) -> float:
    combined = np.concatenate([positive, negative])
    ranks = rankdata(combined, method="average")
    n_positive = len(positive)
    rank_sum = float(ranks[:n_positive].sum())
    u_statistic = rank_sum - n_positive * (n_positive + 1) / 2
    return u_statistic / (n_positive * len(negative))


def vector_matrix_summary(vectors: np.ndarray) -> dict[str, object]:
    norms = np.linalg.norm(vectors, axis=1)
    mean_vector = vectors.mean(axis=0)
    centered = vectors - mean_vector
    singular_values = np.linalg.svd(centered, compute_uv=False)
    eigenvalues = np.square(singular_values) / len(vectors)
    return {
        "vector_count": len(vectors),
        "vector_norm": describe(norms),
        "mean_vector_norm": float(np.linalg.norm(mean_vector)),
        "directional_coherence": float(np.linalg.norm(mean_vector) / norms.mean()),
        "coordinate_negative_ratio": float(np.mean(vectors < 0)),
        "centered_spectrum": pca_summary(eigenvalues),
    }


def build_residual_geometry(
    cohort: pd.DataFrame,
    mean_deltas: dict[str, np.ndarray],
    mean_residuals: dict[str, np.ndarray],
) -> tuple[dict[str, object], pd.DataFrame]:
    conditions = cohort.loc[cohort["comparison_type"] == "perturbation"].drop_duplicates(
        "condition_id"
    ).reset_index(drop=True)
    raw_vectors = np.asarray([mean_deltas[key] for key in conditions["condition_id"]])
    residual_vectors = np.asarray(
        [mean_residuals[key] for key in conditions["condition_id"]]
    )
    categories: dict[str, dict[str, list[float]]] = {}

    def add(category: str, raw_value: float, residual_value: float) -> None:
        values = categories.setdefault(category, {"raw": [], "residual": []})
        values["raw"].append(raw_value)
        values["residual"].append(residual_value)

    for left_index, right_index in combinations(range(len(conditions)), 2):
        left = conditions.iloc[left_index]
        right = conditions.iloc[right_index]
        same_drug = left["target_drug"] == right["target_drug"]
        same_cell_line = left["cell_line_id"] == right["cell_line_id"]
        same_dose_plate = (
            left["plate"] == right["plate"]
            and float(left["target_dose_uM"]) == float(right["target_dose_uM"])
        )
        category = None
        if same_drug and same_dose_plate and not same_cell_line:
            category = "same_drug_same_dose_plate_cross_cell_line"
        elif not same_drug and same_dose_plate and same_cell_line:
            category = "different_drug_same_cell_line_dose_plate"
        elif same_drug and same_cell_line and not same_dose_plate:
            category = "same_drug_same_cell_line_different_dose_plate"
        if category:
            add(
                category,
                cosine(raw_vectors[left_index], raw_vectors[right_index]),
                cosine(residual_vectors[left_index], residual_vectors[right_index]),
            )

    pair_records = []
    pair_summary = {}
    for category, values in categories.items():
        raw_array = np.asarray(values["raw"])
        residual_array = np.asarray(values["residual"])
        pair_summary[category] = {
            "raw_delta_cosine": describe(raw_array),
            "dmso_baseline_residual_delta_cosine": describe(residual_array),
        }
        for space, array in (("raw_delta", raw_array), ("dmso_baseline_residual_delta", residual_array)):
            record = {"category": category, "space": space, **describe(array)}
            pair_records.append(record)

    same_residual = np.asarray(
        categories["same_drug_same_dose_plate_cross_cell_line"]["residual"]
    )
    different_residual = np.asarray(
        categories["different_drug_same_cell_line_dose_plate"]["residual"]
    )

    def nearest_drug_accuracy(vectors: np.ndarray) -> float:
        correct = 0
        for index, row in conditions.iterrows():
            candidates = conditions.index[
                (conditions["cell_line_id"] != row["cell_line_id"])
                & (conditions["plate"] == row["plate"])
                & (
                    conditions["target_dose_uM"].astype(float)
                    == float(row["target_dose_uM"])
                )
            ].to_numpy()
            similarities = np.asarray(
                [cosine(vectors[index], vectors[candidate]) for candidate in candidates]
            )
            prediction = conditions.loc[candidates[int(np.argmax(similarities))], "target_drug"]
            correct += int(prediction == row["target_drug"])
        return correct / len(conditions)

    summary = {
        "definition": (
            "DeltaZ=mu_perturbed-mu_control. DMSO-baseline residual DeltaZ "
            "subtracts the same-cell-line, same-plate, same-repeat DMSO A-to-B shift."
        ),
        "raw_condition_mean_delta": vector_matrix_summary(raw_vectors),
        "dmso_baseline_residual_condition_mean_delta": vector_matrix_summary(
            residual_vectors
        ),
        "pairwise_cosine_categories": pair_summary,
        "same_drug_vs_different_drug_residual_cosine_auc": auc_probability_greater(
            same_residual, different_residual
        ),
        "cross_cell_line_nearest_drug_accuracy_same_dose_plate": {
            "raw_delta": nearest_drug_accuracy(raw_vectors),
            "dmso_baseline_residual_delta": nearest_drug_accuracy(residual_vectors),
            "chance_if_balanced": 0.2,
            "queries": len(conditions),
            "candidate_rule": "other cell lines, same dose and plate",
        },
    }
    return summary, pd.DataFrame(pair_records)


def summarize_experiment(
    set_frame: pd.DataFrame,
    stability: pd.DataFrame,
    replicate_frame: pd.DataFrame,
    dose_frame: pd.DataFrame,
) -> dict[str, object]:
    perturb_sets = set_frame[set_frame["comparison_type"] == "perturbation"]
    null_sets = set_frame[set_frame["comparison_type"] == "dmso_null"]
    perturb_stability = stability[stability["comparison_type"] == "perturbation"]
    null_stability = stability[stability["comparison_type"] == "dmso_null"]
    return {
        "dmso_null_vs_perturbation": {
            "repeat_level_delta_norm": {
                "dmso_null": describe(null_sets["delta_norm"].to_numpy()),
                "perturbation": describe(perturb_sets["delta_norm"].to_numpy()),
                "perturbation_dmso_baseline_residual": describe(
                    perturb_sets["dmso_baseline_residual_delta_norm"].to_numpy()
                ),
                "fraction_perturbation_gt_exact_matched_null": float(
                    np.mean(
                        perturb_sets["delta_norm"].to_numpy()
                        > perturb_sets["matched_null_delta_norm"].to_numpy()
                    )
                ),
                "perturbation_over_exact_matched_null": describe(
                    perturb_sets["delta_norm_over_matched_null"].to_numpy()
                ),
            },
            "condition_level_mean_repeat_delta_norm": {
                "dmso_null": describe(null_stability["delta_repeat_norm_mean"].to_numpy()),
                "perturbation": describe(
                    perturb_stability["delta_repeat_norm_mean"].to_numpy()
                ),
                "perturbation_over_matched_null": describe(
                    perturb_stability["delta_repeat_norm_over_matched_null"].to_numpy()
                ),
                "fraction_perturbation_condition_gt_matched_null": float(
                    np.mean(
                        perturb_stability["delta_repeat_norm_over_matched_null"].to_numpy()
                        > 1
                    )
                ),
            },
            "delta_relative_to_within_set_cell_variation": {
                "dmso_null": describe(null_sets["delta_over_pooled_within_rms"].to_numpy()),
                "perturbation": describe(
                    perturb_sets["delta_over_pooled_within_rms"].to_numpy()
                ),
            },
        },
        "five_repeat_shift_stability": {
            "dmso_null_pairwise_delta_cosine": describe(
                null_stability["delta_pairwise_cosine_mean"].to_numpy()
            ),
            "perturbation_pairwise_delta_cosine": describe(
                perturb_stability["delta_pairwise_cosine_mean"].to_numpy()
            ),
            "perturbation_pairwise_residual_delta_cosine": describe(
                perturb_stability["residual_delta_pairwise_cosine_mean"].to_numpy()
            ),
            "perturbation_residual_directional_coherence": describe(
                perturb_stability["residual_delta_directional_coherence"].to_numpy()
            ),
            "qualification": (
                "Repeated 256-cell subsamples may overlap and are stability probes, "
                "not five independent biological replicates."
            ),
        },
        "plate6_plate14_replicates": {
            "groups": len(replicate_frame),
            "mean_delta_cosine": describe(replicate_frame["mean_delta_cosine"].to_numpy()),
            "mean_dmso_baseline_residual_delta_cosine": describe(
                replicate_frame["mean_residual_delta_cosine"].to_numpy()
            ),
            "matched_repeat_residual_cosine_mean": describe(
                replicate_frame["matched_repeat_residual_cosine_mean"].to_numpy()
            ),
            "residual_magnitude_plate14_over_plate6": describe(
                replicate_frame["plate14_over_plate6_residual_norm"].to_numpy()
            ),
        },
        "dose_associated_geometry": {
            "series": len(dose_frame),
            "raw_magnitude_spearman_rho": describe(
                dose_frame["raw_magnitude_spearman_rho"].to_numpy()
            ),
            "residual_magnitude_spearman_rho": describe(
                dose_frame["residual_magnitude_spearman_rho"].to_numpy()
            ),
            "raw_nondecreasing_fraction": float(
                dose_frame["raw_magnitude_nondecreasing"].mean()
            ),
            "residual_nondecreasing_fraction": float(
                dose_frame["residual_magnitude_nondecreasing"].mean()
            ),
            "residual_low_mid_cosine": describe(
                dose_frame["residual_low_mid_cosine"].to_numpy()
            ),
            "residual_mid_high_cosine": describe(
                dose_frame["residual_mid_high_cosine"].to_numpy()
            ),
            "qualification": (
                "0.05, 0.5, and 5.0 uM are assigned to different plates; "
                "magnitudes are dose-associated, not identifiable pure dose effects."
            ),
        },
    }


def create_notebook(summary: dict[str, object]) -> None:
    numerical = summary["latent_numerics"]
    signal = summary["experiment0"]["dmso_null_vs_perturbation"]
    stability = summary["experiment0"]["five_repeat_shift_stability"]
    replicate = summary["experiment0"]["plate6_plate14_replicates"]
    dose = summary["experiment0"]["dose_associated_geometry"]
    residual = summary["residual_geometry"]
    raw_cos = numerical["raw_pairwise_cosine"]["distribution"]
    centered_cos = numerical["global_mean_centered_pairwise_cosine"]["distribution"]
    perturb_ratio = signal["condition_level_mean_repeat_delta_norm"][
        "perturbation_over_matched_null"
    ]

    cells = [
        nbformat.v4.new_markdown_cell(
            f"""# Tahoe Epoch25 Experiment 0 latent audit

## tl;dr

- Integrity passed for **129,061 x 768** raw float32 embeddings; the saved array was not modified.
- The latent is signed: **{100 * numerical['negative_ratio']:.2f}%** of coordinates are negative.
- Raw random-pair cosine has median **{raw_cos['median']:.4f}**; after subtracting only the global mean vector, the same-pair median is **{centered_cos['median']:.4f}**.
- The global mean vector norm is **{numerical['global_mean_vector_norm']:.3f}**, versus median centered-cell norm **{numerical['centered_embedding_norm']['median']:.3f}**.
- Median condition-level perturbation/null shift-norm ratio is **{perturb_ratio['median']:.3f}**.
- Median plate6/plate14 DMSO-baseline-residual shift cosine is **{replicate['mean_dmso_baseline_residual_delta_cosine']['median']:.3f}**.
- Dose trends remain descriptive because dose and plate are confounded.
"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Context & Methods

The authoritative inputs are the frozen Epoch25 EMA Teacher `.npy`, its extraction manifest, `unique_cells.csv`, the audited cohort, and set-membership CSV.

### Key Assumptions

- `Z_centered = Z - mean(Z)` is computed only in memory for geometry diagnostics.
- `DeltaZ = mean(perturbed set) - mean(control set)`.
- `DeltaZ_residual = DeltaZ_perturbation - DeltaZ_DMSO-null`, matching cell line, plate, and repeat index.
- Five subsamples can overlap; they assess sampling stability and are not independent biological replicates.
- Dose is confounded with plate (`0.05=plate4`, `0.5=plate5`, `5.0=plate6/plate14`).
- No centered/whitened array is persisted, and no STATE/ST training is performed.
"""
        ),
        nbformat.v4.new_markdown_cell("## Data"),
        nbformat.v4.new_code_cell(
            """from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / 'results').is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
RESULTS = PROJECT_ROOT / 'results'

summary = json.loads((RESULTS / 'tahoe_latent_audit_experiment0_summary.json').read_text())
dimensions = pd.read_csv(RESULTS / 'tahoe_latent_audit_experiment0_dimensions.csv')
pca = pd.read_csv(RESULTS / 'tahoe_latent_audit_experiment0_pca_spectrum.csv')
pair_hist = pd.read_csv(RESULTS / 'tahoe_latent_audit_experiment0_pairwise_cosine_hist.csv')
set_shifts = pd.read_csv(RESULTS / 'tahoe_latent_audit_experiment0_set_shifts.csv')
stability = pd.read_csv(RESULTS / 'tahoe_latent_audit_experiment0_shift_stability.csv')
replicates = pd.read_csv(RESULTS / 'tahoe_latent_audit_experiment0_plate_replicates.csv')
dose = pd.read_csv(RESULTS / 'tahoe_latent_audit_experiment0_dose_geometry.csv')

pd.DataFrame({
    'artifact': ['embeddings', 'cohort pairs', 'set membership rows'],
    'count': [summary['integrity']['npy']['rows'], summary['integrity']['tables']['cohort_pairs'], summary['integrity']['tables']['set_membership_rows']],
})"""
        ),
        nbformat.v4.new_markdown_cell("## Results\n\n### 1. Common component and centered PCA"),
        nbformat.v4.new_code_cell(
            """centers = (pair_hist['bin_left'] + pair_hist['bin_right']) / 2
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(centers, pair_hist['raw_count'] / pair_hist['raw_count'].sum())
axes[0].set(title='Raw cell-pair cosine', xlabel='cosine', ylabel='fraction', xlim=(0.7, 1.0))
axes[1].plot(centers, pair_hist['centered_count'] / pair_hist['centered_count'].sum(), color='#d97706')
axes[1].set(title='Global-mean-centered cosine', xlabel='cosine', ylabel='fraction', xlim=(-1.0, 1.0))
axes[2].plot(pca['component'][:100], pca['cumulative_explained_variance_ratio'][:100])
axes[2].set(title='Centered PCA cumulative variance', xlabel='components', ylabel='cumulative ratio', ylim=(0, 1))
plt.tight_layout()
plt.show()

pd.DataFrame({
    'metric': ['raw cosine median', 'centered cosine median', 'mean-vector norm', 'centered norm median', 'PCA entropy effective rank'],
    'value': [
        summary['latent_numerics']['raw_pairwise_cosine']['distribution']['median'],
        summary['latent_numerics']['global_mean_centered_pairwise_cosine']['distribution']['median'],
        summary['latent_numerics']['global_mean_vector_norm'],
        summary['latent_numerics']['centered_embedding_norm']['median'],
        summary['latent_numerics']['centered_pca']['entropy_effective_rank'],
    ],
})"""
        ),
        nbformat.v4.new_markdown_cell("### 2. DMSO null versus perturbation"),
        nbformat.v4.new_code_cell(
            """null_norms = set_shifts.loc[set_shifts.comparison_type == 'dmso_null', 'delta_norm']
pert_norms = set_shifts.loc[set_shifts.comparison_type == 'perturbation', 'delta_norm']
residual_norms = set_shifts.loc[set_shifts.comparison_type == 'perturbation', 'dmso_baseline_residual_delta_norm']
plt.figure(figsize=(7, 4))
plt.boxplot(
    [null_norms, pert_norms, residual_norms],
    tick_labels=['||ΔZ DMSO A→B||', '||ΔZ drug||', '||ΔZ drug − ΔZ DMSO||'],
    showfliers=False,
)
plt.ylabel('L2 shift magnitude')
plt.title('Repeat-level 256-cell centroid shifts (60 null; 300 drug)')
plt.tight_layout()
plt.show()

pd.DataFrame(summary['experiment0']['dmso_null_vs_perturbation']['condition_level_mean_repeat_delta_norm'])"""
        ),
        nbformat.v4.new_markdown_cell("### 3. Repeat and cross-plate stability"),
        nbformat.v4.new_code_cell(
            """fig, axes = plt.subplots(1, 2, figsize=(12, 4))
pert_stability = stability[stability.comparison_type == 'perturbation']
axes[0].hist(pert_stability['residual_delta_pairwise_cosine_mean'], bins=15)
axes[0].set(title='Five-subsample residual shift stability', xlabel='mean pairwise cosine', ylabel='conditions')
axes[1].hist(replicates['mean_residual_delta_cosine'], bins=12)
axes[1].set(title='plate6 vs plate14 residual shifts', xlabel='cosine', ylabel='replicate groups')
plt.tight_layout()
plt.show()

replicates[['cell_line_id', 'drug', 'mean_delta_cosine', 'mean_residual_delta_cosine', 'plate14_over_plate6_residual_norm']].sort_values('mean_residual_delta_cosine')"""
        ),
        nbformat.v4.new_markdown_cell("### 4. Dose-associated magnitude (plate-confounded)"),
        nbformat.v4.new_code_cell(
            """dose[['cell_line_id', 'drug', 'low_mean_residual_norm', 'mid_mean_residual_norm', 'high_replicate_average_residual_norm', 'residual_magnitude_spearman_rho', 'residual_magnitude_nondecreasing']].sort_values(['cell_line_id', 'drug'])"""
        ),
        nbformat.v4.new_markdown_cell(
            f"""## Takeaways

1. The raw GeneJEPA latent is signed; a non-negative final output activation would be incompatible with the observed target space.
2. Raw cell cosine is dominated by a common component. Centered-space and shift-space diagnostics are required; raw cosine alone is not a perturbation metric.
3. Perturbation-vs-null strength, repeated-subsample stability, and plate6/plate14 reproducibility must be interpreted together. The median residual cross-plate cosine is **{replicate['mean_dmso_baseline_residual_delta_cosine']['median']:.3f}**.
4. The same-dose/plate cross-cell-line nearest-drug diagnostic is **{100 * residual['cross_cell_line_nearest_drug_accuracy_same_dose_plate']['dmso_baseline_residual_delta']:.1f}%** after DMSO-baseline residualization (balanced chance 20%).
5. Only **{100 * dose['residual_nondecreasing_fraction']:.1f}%** of series have nondecreasing residual magnitude; because dose and plate are confounded, this is not a causal dose-response estimate.
6. These results remain Experiment 0 diagnostics and do not authorize STATE/ST training.
"""
        ),
    ]
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
    )
    NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    ).execute()
    nbformat.write(notebook, NOTEBOOK_OUTPUT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--unique-cells", type=Path, default=DEFAULT_UNIQUE_CELLS)
    parser.add_argument("--cell-index-summary", type=Path, default=DEFAULT_CELL_INDEX_SUMMARY)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--set-indices", type=Path, default=DEFAULT_SET_INDICES)
    parser.add_argument("--local-manifest", type=Path, default=DEFAULT_LOCAL_MANIFEST)
    parser.add_argument("--gene-metadata", type=Path, default=DEFAULT_GENE_METADATA)
    parser.add_argument("--global-stats", type=Path, default=DEFAULT_GLOBAL_STATS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--pair-samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    print("Verifying complete SHA-256 provenance chain...", flush=True)
    manifest, integrity = verify_hash_chain(args)
    embeddings = np.load(args.embeddings, mmap_mode="r")
    if embeddings.shape != (EXPECTED_CELLS, EXPECTED_DIM) or embeddings.dtype != np.float32:
        raise AssertionError(
            f"Unexpected npy structure: shape={embeddings.shape}, dtype={embeddings.dtype}"
        )
    _, cohort, memberships, table_audit = load_and_verify_tables(args)
    integrity["npy"] = {
        "path": display_path(args.embeddings),
        "mmap_mode": "r",
        "rows": embeddings.shape[0],
        "dimensions": embeddings.shape[1],
        "dtype": str(embeddings.dtype),
        "raw_file_modified": False,
    }
    integrity["tables"] = table_audit

    numerical, dimensions, pca, pair_histogram, global_mean_vector = profile_embeddings(
        embeddings,
        manifest,
        args.chunk_size,
        args.pair_samples,
        args.seed,
    )
    integrity["independent_numeric_reconciliation"] = {
        "status": "pass",
        "manifest_statistics_match": True,
        "finite": numerical["finite"],
        "min": numerical["global_min"],
        "max": numerical["global_max"],
        "mean": numerical["global_mean"],
        "std": numerical["global_std"],
        "negative_ratio": numerical["negative_ratio"],
    }
    integrity["created_at_utc"] = utc_now()
    write_json(INTEGRITY_OUTPUT, integrity)

    print("[4/5] Aggregating 720 fixed 256-cell sets and condition shifts...", flush=True)
    centroids, radii = compute_set_centroids(embeddings, memberships)
    set_frame, deltas, residual_deltas, _ = build_set_shifts(
        cohort, centroids, radii, global_mean_vector
    )
    (
        stability,
        mean_deltas,
        mean_residuals,
        repeat_deltas,
        repeat_residuals,
    ) = build_stability(cohort, deltas, residual_deltas)
    stability = add_matched_null_condition_metrics(stability)
    replicate_frame = build_plate_replicates(
        cohort,
        mean_deltas,
        mean_residuals,
        repeat_deltas,
        repeat_residuals,
    )
    dose_frame = build_dose_geometry(
        cohort, mean_deltas, mean_residuals, replicate_frame
    )
    residual_summary, residual_pairs = build_residual_geometry(
        cohort, mean_deltas, mean_residuals
    )
    experiment_summary = summarize_experiment(
        set_frame, stability, replicate_frame, dose_frame
    )

    print("[5/5] Writing bounded audit tables and companion notebook...", flush=True)
    write_csv(DIMENSION_OUTPUT, dimensions)
    write_csv(PCA_OUTPUT, pca)
    write_csv(PAIR_HIST_OUTPUT, pair_histogram)
    write_csv(SET_OUTPUT, set_frame)
    write_csv(STABILITY_OUTPUT, stability)
    write_csv(REPLICATE_OUTPUT, replicate_frame)
    write_csv(DOSE_OUTPUT, dose_frame)
    write_csv(RESIDUAL_PAIR_OUTPUT, residual_pairs)

    summary = {
        "created_at_utc": utc_now(),
        "status": "pass",
        "experiment": "Tahoe GeneJEPA Epoch25 Experiment 0 latent audit",
        "scope": {
            "state_or_st_training_performed": False,
            "genejepa_modified": False,
            "raw_embedding_modified": False,
            "centered_or_whitened_embedding_saved": False,
            "derived_centering_used_only_for_diagnostics": True,
        },
        "definitions": {
            "global_centered_embedding": "Z - mean_all_unique_cells(Z)",
            "delta_z": "mean(target_256) - mean(source_256)",
            "dmso_baseline_residual_delta_z": (
                "DeltaZ_perturbation - DeltaZ_DMSO_A_to_B matched by cell line, plate, repeat"
            ),
        },
        "integrity": integrity,
        "latent_numerics": numerical,
        "experiment0": experiment_summary,
        "residual_geometry": residual_summary,
        "limitations": [
            "Five fixed-seed subsamples can overlap and are not independent biological replicates.",
            "The DMSO residual uses the observed A-to-B sample direction and is a diagnostic baseline correction, not a learned preprocessing transform.",
            "Dose is structurally confounded with plate; plate-matched DMSO does not identify a pure causal dose effect.",
            "The 60 perturbation conditions were deliberately selected for coverage/capacity and are not a random sample of all Tahoe perturbations.",
            "No centered or whitened cell embedding is persisted; all reported transformed geometry is derived in memory from the immutable raw array.",
        ],
        "outputs": {},
        "elapsed_seconds": time.time() - started,
    }
    write_json(SUMMARY_OUTPUT, summary)
    create_notebook(summary)

    output_paths = [
        INTEGRITY_OUTPUT,
        DIMENSION_OUTPUT,
        PCA_OUTPUT,
        PAIR_HIST_OUTPUT,
        SET_OUTPUT,
        STABILITY_OUTPUT,
        REPLICATE_OUTPUT,
        DOSE_OUTPUT,
        RESIDUAL_PAIR_OUTPUT,
        NOTEBOOK_OUTPUT,
    ]
    summary["outputs"] = {
        display_path(path): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in output_paths
    }
    summary["elapsed_seconds"] = time.time() - started
    write_json(SUMMARY_OUTPUT, summary)

    print(json.dumps({
        "status": summary["status"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "negative_ratio": numerical["negative_ratio"],
        "raw_pairwise_cosine_median": numerical["raw_pairwise_cosine"]["distribution"]["median"],
        "centered_pairwise_cosine_median": numerical["global_mean_centered_pairwise_cosine"]["distribution"]["median"],
        "pca_entropy_effective_rank": numerical["centered_pca"]["entropy_effective_rank"],
        "output": display_path(SUMMARY_OUTPUT),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
