#!/usr/bin/env python3
"""Freeze Experiment 1 edge splits and estimate bounded latent caches.

This script reads the existing full-Tahoe condition audit only. It does not
open expression parquet files, run GeneJEPA, or train STATE/ST.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
CONDITIONS_INPUT = RESULTS / "tahoe_set_to_set_pairs.csv"
METADATA_AUDIT_INPUT = RESULTS / "tahoe_metadata_audit_full.json"
EXTRACTION_MANIFEST_INPUT = RESULTS / "tahoe_latent_audit_epoch25_manifest.json"

EDGE_OUTPUT = RESULTS / "tahoe_experiment1_edge_split_manifest.csv"
CONDITION_OUTPUT = RESULTS / "tahoe_experiment1_condition_split_manifest.csv"
CACHE_OUTPUT = RESULTS / "tahoe_experiment1_cache_estimates.csv"
CAPACITY_OUTPUT = RESULTS / "tahoe_experiment1_condition_cache_capacity.csv"
DRUG_VOCAB_OUTPUT = RESULTS / "tahoe_experiment1_drug_vocabulary.csv"
PERTURBATION_FEATURE_OUTPUT = RESULTS / "tahoe_experiment1_perturbation_featurization.json"
SUMMARY_OUTPUT = RESULTS / "tahoe_experiment1_preparation_summary.json"

SEED = 42
CACHE_CAPS = (256, 512, 1024)
SET_SIZE = 256
EMBEDDING_DIM = 768
FLOAT32_BYTES = 4
SPLIT_ALGORITHM = "sha256_greedy_train_vertex_coverage_v1"
EXPECTED_DRUGS = 379
PERTURBATION_DIM = EXPECTED_DRUGS + 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(kind: str, *values: object, seed: int | None = None) -> str:
    parts = [kind]
    if seed is not None:
        parts.append(f"seed={seed}")
    text = "|".join([*parts, *(str(value) for value in values)])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n", encoding="utf-8-sig")
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def numeric_summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
    }


def split_samples(values: pd.Series) -> list[str]:
    return sorted(
        {
            sample
            for joined in values.astype(str)
            for sample in joined.split("|")
            if sample
        }
    )


def plate_sort_key(plate: str) -> int:
    return int(plate.removeprefix("plate"))


def build_edges(conditions: pd.DataFrame) -> pd.DataFrame:
    records = []
    grouped = conditions.groupby(["cell_line_id", "drug"], sort=True)
    for edge_index, ((cell_line, drug), group) in enumerate(grouped, start=1):
        plates = sorted(group["plate"].unique(), key=plate_sort_key)
        doses = sorted(group["dose_uM"].astype(float).unique())
        samples = split_samples(group["treated_samples"])
        plate6_doses = set(group.loc[group["plate"] == "plate6", "dose_uM"].astype(float))
        plate14_doses = set(group.loc[group["plate"] == "plate14", "dose_uM"].astype(float))
        records.append(
            {
                "edge_id": f"edge_{edge_index:05d}",
                "edge_key_sha256": stable_hash("tahoe_experiment1_edge_v1", cell_line, drug),
                "cell_line_id": cell_line,
                "drug": drug,
                "split_priority_sha256": stable_hash(
                    "tahoe_experiment1_split_v1", cell_line, drug, seed=SEED
                ),
                "condition_count": int(len(group)),
                "dose_values_uM": "|".join(f"{dose:g}" for dose in doses),
                "plate_values": "|".join(plates),
                "treated_samples": "|".join(samples),
                "treated_sample_count": len(samples),
                "treated_cell_count": int(group["treated_cell_count"].sum()),
                "minimum_condition_treated_cells": int(group["treated_cell_count"].min()),
                "maximum_condition_treated_cells": int(group["treated_cell_count"].max()),
                "has_plate6_plate14_same_dose_replicate": bool(plate6_doses & plate14_doses),
            }
        )
    return pd.DataFrame(records)


def assign_splits(edges: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    result = edges.copy()
    edge_count = len(result)
    targets = {
        "val": int(math.floor(edge_count * 0.10 + 0.5)),
        "test": int(math.floor(edge_count * 0.10 + 0.5)),
    }
    result["split"] = "train"

    cell_degree = Counter(result["cell_line_id"])
    drug_degree = Counter(result["drug"])
    assigned = {"val": 0, "test": 0}
    next_split = "val"
    for row in result.sort_values("split_priority_sha256").itertuples():
        if assigned == targets:
            break
        if cell_degree[row.cell_line_id] <= 1 or drug_degree[row.drug] <= 1:
            continue
        if assigned[next_split] >= targets[next_split]:
            next_split = "test" if next_split == "val" else "val"
        if assigned[next_split] >= targets[next_split]:
            continue
        result.at[row.Index, "split"] = next_split
        assigned[next_split] += 1
        cell_degree[row.cell_line_id] -= 1
        drug_degree[row.drug] -= 1
        next_split = "test" if next_split == "val" else "val"

    if assigned != targets:
        raise AssertionError(f"Could not satisfy split quotas: {assigned} != {targets}")
    return result, {
        "train": edge_count - targets["val"] - targets["test"],
        **targets,
    }


def add_edge_and_pool_ids(conditions: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    merged = conditions.merge(
        edges[["edge_id", "edge_key_sha256", "cell_line_id", "drug", "split"]],
        on=["cell_line_id", "drug"],
        how="left",
        validate="many_to_one",
    )
    if merged[["edge_id", "split"]].isna().any().any():
        raise AssertionError("Condition-to-edge join lost rows")
    merged["control_pool_id"] = [
        stable_hash("tahoe_dmso_pool_v1", plate, cell_line, samples)
        for plate, cell_line, samples in zip(
            merged["plate"], merged["cell_line_id"], merged["control_samples"]
        )
    ]
    merged["split_seed"] = SEED
    return merged.sort_values("pair_id").reset_index(drop=True)


def audit_split(
    conditions: pd.DataFrame, edges: pd.DataFrame, expected_counts: dict[str, int]
) -> dict[str, object]:
    edge_sets = {
        split: set(map(tuple, group[["cell_line_id", "drug"]].to_numpy()))
        for split, group in edges.groupby("split")
    }
    pair_sets = {
        split: set(group["pair_id"])
        for split, group in conditions.groupby("split")
    }
    overlap = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap[f"{left}_{right}_edge_overlap"] = len(edge_sets[left] & edge_sets[right])
        overlap[f"{left}_{right}_condition_overlap"] = len(pair_sets[left] & pair_sets[right])

    train_edges = edge_sets["train"]
    train_cells_by_drug: dict[str, set[str]] = {}
    train_drugs_by_cell: dict[str, set[str]] = {}
    for cell_line, drug in train_edges:
        train_cells_by_drug.setdefault(drug, set()).add(cell_line)
        train_drugs_by_cell.setdefault(cell_line, set()).add(drug)
    coverage_failures = {"val": [], "test": []}
    for split in ("val", "test"):
        for cell_line, drug in sorted(edge_sets[split]):
            drug_seen_elsewhere = bool(train_cells_by_drug.get(drug, set()) - {cell_line})
            cell_seen_elsewhere = bool(train_drugs_by_cell.get(cell_line, set()) - {drug})
            if not (drug_seen_elsewhere and cell_seen_elsewhere):
                coverage_failures[split].append(
                    {"cell_line_id": cell_line, "drug": drug}
                )

    replicate_groups = 0
    replicate_leakage = 0
    for _, group in conditions.groupby(["cell_line_id", "drug", "dose_uM"]):
        if {"plate6", "plate14"}.issubset(set(group["plate"])):
            replicate_groups += 1
            replicate_leakage += int(group["split"].nunique() != 1)

    by_split = {}
    individual_samples = {}
    for split in ("train", "val", "test"):
        group = conditions[conditions["split"] == split]
        individual_samples[split] = set(split_samples(group["treated_samples"]))
        by_split[split] = {
            "edges": int(edges["split"].eq(split).sum()),
            "edge_fraction": float(edges["split"].eq(split).mean()),
            "conditions": int(len(group)),
            "condition_fraction": float(len(group) / len(conditions)),
            "drugs": int(group["drug"].nunique()),
            "cell_lines": int(group["cell_line_id"].nunique()),
            "treated_samples": len(individual_samples[split]),
            "dose_condition_counts": {
                f"{dose:g}": int(count)
                for dose, count in group["dose_uM"].astype(float).value_counts().sort_index().items()
            },
            "plate_condition_counts": {
                plate: int(count)
                for plate, count in sorted(
                    group["plate"].value_counts().items(), key=lambda item: plate_sort_key(item[0])
                )
            },
        }

    sample_overlap = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = individual_samples[left] & individual_samples[right]
        sample_overlap[f"{left}_{right}"] = {
            "count": len(shared),
            "examples": sorted(shared)[:10],
            "expected_under_edge_split": True,
        }

    if any(overlap.values()) or any(coverage_failures.values()) or replicate_leakage:
        raise AssertionError("Split leakage or train-coverage audit failed")
    observed_counts = edges["split"].value_counts().to_dict()
    if observed_counts != expected_counts:
        raise AssertionError(f"Split counts differ: {observed_counts} != {expected_counts}")

    return {
        "status": "pass",
        "split_key": ["cell_line_id", "drug"],
        "algorithm": SPLIT_ALGORITHM,
        "seed": SEED,
        "target_edge_fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
        "total": {
            "edges": int(len(edges)),
            "conditions": int(len(conditions)),
            "drugs": int(conditions["drug"].nunique()),
            "cell_lines": int(conditions["cell_line_id"].nunique()),
            "doses_uM": sorted(float(value) for value in conditions["dose_uM"].unique()),
            "plates": sorted(conditions["plate"].unique(), key=plate_sort_key),
        },
        "by_split": by_split,
        "overlap_checks": overlap,
        "train_seen_constraint_failures": {
            split: len(failures) for split, failures in coverage_failures.items()
        },
        "plate6_plate14_replicate_groups": replicate_groups,
        "plate6_plate14_split_leakage_groups": replicate_leakage,
        "treated_sample_id_overlap": sample_overlap,
        "sample_overlap_qualification": (
            "Tahoe sample IDs span many cell lines. Exact treated cells and (cell_line, drug) "
            "edges do not cross splits, but the same experimental sample label may appear in "
            "multiple splits. Shared DMSO controls are audited separately."
        ),
    }


def build_perturbation_featurization(
    conditions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    drugs = sorted(conditions["drug"].unique())
    if len(drugs) != EXPECTED_DRUGS:
        raise AssertionError(f"Expected {EXPECTED_DRUGS} drugs, found {len(drugs)}")
    if any(not drug or drug != drug.strip() for drug in drugs):
        raise AssertionError("Drug vocabulary contains an empty or untrimmed value")

    condition_counts = (
        conditions.groupby(["drug", "split"]).size().unstack(fill_value=0)
    )
    edge_counts = (
        conditions[["edge_id", "drug", "split"]]
        .drop_duplicates()
        .groupby(["drug", "split"])
        .size()
        .unstack(fill_value=0)
    )
    records = []
    for drug_id, drug in enumerate(drugs):
        records.append(
            {
                "drug_id": drug_id,
                "drug": drug,
                "drug_key_sha256": stable_hash(
                    "tahoe_experiment1_drug_v1", drug
                ),
                "condition_count": int(condition_counts.loc[drug].sum()),
                "edge_count": int(edge_counts.loc[drug].sum()),
                **{
                    f"{split}_condition_count": int(
                        condition_counts.loc[drug].get(split, 0)
                    )
                    for split in ("train", "val", "test")
                },
                **{
                    f"{split}_edge_count": int(edge_counts.loc[drug].get(split, 0))
                    for split in ("train", "val", "test")
                },
            }
        )
    vocabulary = pd.DataFrame(records)
    if vocabulary["drug_id"].tolist() != list(range(EXPECTED_DRUGS)):
        raise AssertionError("Drug IDs are not contiguous from zero")

    train_drugs = set(conditions.loc[conditions["split"] == "train", "drug"])
    seen_failures = {
        split: sorted(
            set(conditions.loc[conditions["split"] == split, "drug"])
            - train_drugs
        )
        for split in ("val", "test")
    }
    if any(seen_failures.values()) or train_drugs != set(drugs):
        raise AssertionError("Drug vocabulary is not fully covered by train")

    doses = conditions["dose_uM"].astype(np.float64).to_numpy()
    if not np.isfinite(doses).all() or (doses <= 0).any():
        raise AssertionError("dose_uM must be finite and positive")
    train_log_dose = np.log10(
        conditions.loc[conditions["split"] == "train", "dose_uM"]
        .astype(np.float64)
        .to_numpy()
    )
    dose_mean = float(train_log_dose.mean())
    dose_std = float(train_log_dose.std(ddof=0))
    if not np.isfinite(dose_std) or dose_std <= 0:
        raise AssertionError("Train log10 dose standard deviation is invalid")

    dose_values = sorted(float(value) for value in np.unique(doses))
    contract = {
        "version": "tahoe_experiment1_drug_onehot_logdose_v1",
        "status": "pass",
        "vector_dtype": "float32",
        "pert_dim": PERTURBATION_DIM,
        "set_replication": "The same perturbation vector is repeated for all 256 cells in a set.",
        "drug_feature": {
            "encoding": "one_hot",
            "dimensions": EXPECTED_DRUGS,
            "index_range": [0, EXPECTED_DRUGS - 1],
            "vocabulary_order": "Python lexicographic order of exact drug strings",
            "vocabulary_scope": "all Experiment 1 eligible_S256 conditions",
            "shared_across_splits": True,
            "train_covers_all_vocabulary_drugs": True,
            "val_unseen_in_train": len(seen_failures["val"]),
            "test_unseen_in_train": len(seen_failures["test"]),
        },
        "dose_feature": {
            "dimension_index": EXPECTED_DRUGS,
            "input_unit": "uM",
            "transform": "(log10(dose_uM) - mean) / std",
            "fit_scope": "train eligible condition rows, one vote per condition",
            "standard_deviation": "population, ddof=0",
            "log10_mean": dose_mean,
            "log10_std": dose_std,
            "allowed_dose_uM": dose_values,
            "encoded_values": {
                f"{dose:g}": float((math.log10(dose) - dose_mean) / dose_std)
                for dose in dose_values
            },
        },
        "excluded_features": [
            "cell_line_id",
            "plate",
            "sample",
            "moa",
            "smiles",
            "pubchem",
            "esm",
        ],
    }
    return vocabulary, contract


def log10_choose(n: int, k: int) -> float:
    if n < k:
        return float("-inf")
    return (
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    ) / math.log(10)


def build_cache_estimates(
    conditions: pd.DataFrame, extraction_rate: float
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    pool_columns = [
        "control_pool_id",
        "plate",
        "cell_line_id",
        "control_samples",
        "control_cell_count",
    ]
    control_pools = conditions[pool_columns].drop_duplicates()
    if control_pools["control_pool_id"].duplicated().any():
        raise AssertionError("A control_pool_id maps to conflicting metadata")

    pool_split_count = conditions.groupby("control_pool_id")["split"].nunique()
    pool_reuse = {
        str(number_of_splits): int(count)
        for number_of_splits, count in pool_split_count.value_counts().sort_index().items()
    }
    estimate_records = []
    capacity_records = []
    cache_summary = {}

    for cap in CACHE_CAPS:
        treated_cached = np.minimum(conditions["treated_cell_count"].to_numpy(np.int64), cap)
        control_pools_for_cap = control_pools.assign(
            cached_cells=np.minimum(control_pools["control_cell_count"].to_numpy(np.int64), cap)
        )
        control_cache_lookup = control_pools_for_cap.set_index("control_pool_id")["cached_cells"]
        condition_control_cached = conditions["control_pool_id"].map(control_cache_lookup).to_numpy(np.int64)

        treated_log_sets = np.asarray([log10_choose(int(n), SET_SIZE) for n in treated_cached])
        control_log_sets = np.asarray([log10_choose(int(n), SET_SIZE) for n in condition_control_cached])
        condition_capacity = pd.DataFrame(
            {
                "cache_cap_per_condition_or_pool": cap,
                "pair_id": conditions["pair_id"],
                "edge_id": conditions["edge_id"],
                "split": conditions["split"],
                "control_pool_id": conditions["control_pool_id"],
                "treated_available_cells": conditions["treated_cell_count"].astype(np.int64),
                "control_available_cells": conditions["control_cell_count"].astype(np.int64),
                "treated_cached_cells": treated_cached,
                "control_cached_cells": condition_control_cached,
                "treated_disjoint_S256_sets": treated_cached // SET_SIZE,
                "control_disjoint_S256_sets": condition_control_cached // SET_SIZE,
                "treated_log10_possible_S256_sets": treated_log_sets,
                "control_log10_possible_S256_sets": control_log_sets,
                "paired_log10_possible_source_target_sets": treated_log_sets + control_log_sets,
            }
        )
        capacity_records.append(condition_capacity)

        scopes = ["global", "train", "val", "test"]
        for scope in scopes:
            if scope == "global":
                condition_mask = np.ones(len(conditions), dtype=bool)
                referenced_pool_ids = set(control_pools["control_pool_id"])
            else:
                condition_mask = conditions["split"].eq(scope).to_numpy()
                referenced_pool_ids = set(
                    conditions.loc[condition_mask, "control_pool_id"]
                )
            pool_mask = control_pools_for_cap["control_pool_id"].isin(referenced_pool_ids)
            treated_cells = int(treated_cached[condition_mask].sum())
            dmso_cells = int(control_pools_for_cap.loc[pool_mask, "cached_cells"].sum())
            total_cells = treated_cells + dmso_cells
            byte_count = total_cells * EMBEDDING_DIM * FLOAT32_BYTES
            estimate_records.append(
                {
                    "cache_cap": cap,
                    "scope": scope,
                    "edges": int(
                        conditions.loc[condition_mask, "edge_id"].nunique()
                    ),
                    "conditions": int(condition_mask.sum()),
                    "control_pools": len(referenced_pool_ids),
                    "unique_treated_cells": treated_cells,
                    "unique_dmso_cells": dmso_cells,
                    "unique_total_cells": total_cells,
                    "embedding_bytes": byte_count,
                    "embedding_GB_decimal": byte_count / 1e9,
                    "embedding_GiB": byte_count / 2**30,
                    "extraction_hours_at_measured_rate": total_cells / extraction_rate / 3600,
                    "extraction_hours_low_if_1_25x_rate": total_cells / (1.25 * extraction_rate) / 3600,
                    "extraction_hours_high_if_0_75x_rate": total_cells / (0.75 * extraction_rate) / 3600,
                }
            )

        per_cap_capacity = {}
        for scope in ("global", "train", "val", "test"):
            mask = (
                np.ones(len(conditions), dtype=bool)
                if scope == "global"
                else conditions["split"].eq(scope).to_numpy()
            )
            cached = treated_cached[mask]
            log_sets = treated_log_sets[mask]
            paired_log_sets = (treated_log_sets + control_log_sets)[mask]
            per_cap_capacity[scope] = {
                "treated_cached_cells_per_condition": numeric_summary(cached),
                "treated_disjoint_S256_sets_per_condition": numeric_summary(cached // SET_SIZE),
                "conditions_with_exactly_one_possible_treated_set": int(np.count_nonzero(cached == SET_SIZE)),
                "conditions_with_at_least_two_disjoint_treated_sets": int(np.count_nonzero(cached >= 2 * SET_SIZE)),
                "conditions_with_four_disjoint_treated_sets": int(np.count_nonzero(cached >= 4 * SET_SIZE)),
                "treated_log10_possible_S256_sets": numeric_summary(log_sets),
                "paired_log10_possible_source_target_sets": numeric_summary(paired_log_sets),
            }
        cache_summary[str(cap)] = per_cap_capacity

    estimates = pd.DataFrame(estimate_records).sort_values(
        ["cache_cap", "scope"],
        key=lambda series: series.map({"global": 0, "train": 1, "val": 2, "test": 3})
        if series.name == "scope"
        else series,
    )
    capacities = pd.concat(capacity_records, ignore_index=True)
    return estimates, capacities, {
        "status": "pass",
        "control_pool_key": ["plate", "cell_line_id", "control_samples"],
        "global_control_pools": int(len(control_pools)),
        "control_pool_split_reuse_histogram": pool_reuse,
        "deduplication_rule": (
            "Treated cells are capped once per eligible condition; DMSO cells are capped once "
            "per unique plate + cell line + control-sample pool and reused across all edges."
        ),
        "control_reuse_qualification": (
            "A DMSO pool can be referenced by multiple edge splits. Global cache deduplication "
            "therefore reuses exact basal cells across those splits; treated cells remain split-exclusive."
        ),
        "dynamic_sampling_rule": (
            "Extract every selected unique cell once, then sample 256 unique indices without "
            "replacement inside each training set; sets across steps may overlap."
        ),
        "capacity": cache_summary,
        "extraction_rate_source": {
            "cells_per_second": extraction_rate,
            "device": "NVIDIA RTX A6000",
            "source": "Experiment 0 full Epoch25 extraction manifest",
            "estimate_range": "0.75x to 1.25x the measured throughput",
        },
        "size_scope": "float32 embedding payload only; index tables and array headers excluded",
    }


def main() -> None:
    source = pd.read_csv(
        CONDITIONS_INPUT, keep_default_na=False, encoding="utf-8-sig"
    )
    required = {
        "pair_id",
        "plate",
        "cell_line_id",
        "drug",
        "dose_uM",
        "treated_samples",
        "treated_cell_count",
        "control_samples",
        "control_cell_count",
        "matched_capacity",
        "eligible_S256",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")

    conditions = source[source["eligible_S256"].astype(int) == 1].copy()
    if conditions["pair_id"].duplicated().any():
        raise AssertionError("Eligible pair_id is not unique")
    if conditions.duplicated(["plate", "cell_line_id", "drug", "dose_uM"]).any():
        raise AssertionError("Eligible condition grain is duplicated")
    if (conditions[["treated_cell_count", "control_cell_count", "matched_capacity"]] < SET_SIZE).any().any():
        raise AssertionError("eligible_S256 contains a pool below 256 cells")
    if conditions[list(required - {"eligible_S256"})].astype(str).eq("").any().any():
        raise AssertionError("Required eligible condition field is empty")

    edges, expected_split_counts = assign_splits(build_edges(conditions))
    conditions = add_edge_and_pool_ids(conditions, edges)
    split_audit = audit_split(conditions, edges, expected_split_counts)

    with EXTRACTION_MANIFEST_INPUT.open("r", encoding="utf-8") as handle:
        extraction_manifest = json.load(handle)
    extraction_rate = float(extraction_manifest["inference"]["cells_per_second"])
    cache_estimates, condition_capacity, cache_audit = build_cache_estimates(
        conditions, extraction_rate
    )
    cap512_single_hours = float(
        cache_estimates.loc[
            (cache_estimates["cache_cap"] == 512)
            & (cache_estimates["scope"] == "global"),
            "extraction_hours_at_measured_rate",
        ].item()
    )
    cache_audit["cap512_two_a6000_projection"] = {
        "parallelism": "two independent inference workers; no DDP or NCCL",
        "single_a6000_reference_hours": cap512_single_hours,
        "ideal_2x_wall_hours": cap512_single_hours / 2,
        "shared_parquet_io_scenarios": {
            f"{speedup:g}x_effective_speedup": cap512_single_hours / speedup
            for speedup in (1.3, 1.5, 1.7)
        },
        "qualification": (
            "Actual speedup must be measured because both workers share the same parquet storage; "
            "disk and WSL filesystem I/O can prevent ideal 2x scaling."
        ),
    }
    drug_vocabulary, perturbation_featurization = build_perturbation_featurization(
        conditions
    )

    write_csv(EDGE_OUTPUT, edges.sort_values("edge_id"))
    write_csv(CONDITION_OUTPUT, conditions)
    write_csv(CACHE_OUTPUT, cache_estimates)
    write_csv(CAPACITY_OUTPUT, condition_capacity)
    write_csv(DRUG_VOCAB_OUTPUT, drug_vocabulary)
    perturbation_featurization["drug_vocabulary"] = str(
        DRUG_VOCAB_OUTPUT.relative_to(PROJECT_ROOT)
    )
    perturbation_featurization["drug_vocabulary_sha256"] = sha256_file(
        DRUG_VOCAB_OUTPUT
    )
    write_json(PERTURBATION_FEATURE_OUTPUT, perturbation_featurization)

    csv_outputs = [
        EDGE_OUTPUT,
        CONDITION_OUTPUT,
        CACHE_OUTPUT,
        CAPACITY_OUTPUT,
        DRUG_VOCAB_OUTPUT,
    ]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "scope": {
            "expression_parquet_opened": False,
            "genejepa_extraction_run": False,
            "state_or_st_training_run": False,
            "cache_is_estimate_not_materialized": True,
        },
        "csv_encoding": "utf-8-sig (UTF-8 with BOM for Windows/Excel)",
        "inputs": {
            "conditions_csv": str(CONDITIONS_INPUT.relative_to(PROJECT_ROOT)),
            "conditions_csv_sha256": sha256_file(CONDITIONS_INPUT),
            "metadata_audit_json": str(METADATA_AUDIT_INPUT.relative_to(PROJECT_ROOT)),
            "metadata_audit_json_sha256": sha256_file(METADATA_AUDIT_INPUT),
            "extraction_manifest": str(EXTRACTION_MANIFEST_INPUT.relative_to(PROJECT_ROOT)),
            "extraction_manifest_sha256": sha256_file(EXTRACTION_MANIFEST_INPUT),
            "source_rows": int(len(source)),
            "eligible_S256_rows": int(len(conditions)),
        },
        "split": split_audit,
        "cache": cache_audit,
        "perturbation_featurization": perturbation_featurization,
        "outputs": {
            str(path.relative_to(PROJECT_ROOT)): {
                "rows": int(pd.read_csv(path, encoding="utf-8-sig").shape[0]),
                "encoding": "utf-8-sig",
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in csv_outputs
        },
    }
    summary["outputs"][str(PERTURBATION_FEATURE_OUTPUT.relative_to(PROJECT_ROOT))] = {
        "size_bytes": PERTURBATION_FEATURE_OUTPUT.stat().st_size,
        "sha256": sha256_file(PERTURBATION_FEATURE_OUTPUT),
    }
    write_json(SUMMARY_OUTPUT, summary)
    print(
        json.dumps(
            {
                "status": "pass",
                "edges": split_audit["total"]["edges"],
                "conditions": split_audit["total"]["conditions"],
                "split_edges": {
                    split: values["edges"]
                    for split, values in split_audit["by_split"].items()
                },
                "cache_global": cache_estimates[
                    cache_estimates["scope"] == "global"
                ][
                    [
                        "cache_cap",
                        "unique_treated_cells",
                        "unique_dmso_cells",
                        "unique_total_cells",
                        "embedding_GiB",
                        "extraction_hours_at_measured_rate",
                    ]
                ].to_dict("records"),
                "summary": str(SUMMARY_OUTPUT.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
