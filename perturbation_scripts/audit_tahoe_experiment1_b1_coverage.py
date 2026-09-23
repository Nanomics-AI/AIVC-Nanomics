#!/usr/bin/env python3
"""Audit exact train-to-test drug-dose coverage for Experiment 1 B1."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from tahoe_experiment1_latent_data import (
    CONDITION_SPLIT_MANIFEST,
    EDGE_SPLIT_MANIFEST,
    PREPARATION_SUMMARY,
    PROJECT_ROOT,
    RESULTS,
    sha256_file,
)


SCRIPT_PATH = Path(__file__).resolve()
GOALS_PATH = PROJECT_ROOT.parent / "一、第二阶段总体目标.md"
CONDITION_COVERAGE_PATH = RESULTS / "tahoe_experiment1_b1_train_test_drug_dose_coverage.csv"
AUDIT_JSON_PATH = RESULTS / "tahoe_experiment1_b1_train_test_coverage_audit.json"
AUDIT_MD_PATH = RESULTS / "tahoe_experiment1_b1_train_test_coverage_audit.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", lineterminator="\n")
    temporary.replace(path)


def verified_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    preparation = json.loads(PREPARATION_SUMMARY.read_text(encoding="utf-8"))
    if preparation.get("status") != "pass" or preparation["split"].get("status") != "pass":
        raise AssertionError("Experiment 1 preparation manifest is not recorded as pass")
    outputs = preparation["outputs"]
    for path in (CONDITION_SPLIT_MANIFEST, EDGE_SPLIT_MANIFEST):
        key = path.relative_to(PROJECT_ROOT).as_posix()
        if sha256_file(path) != outputs[key]["sha256"]:
            raise AssertionError(f"Frozen manifest SHA-256 changed: {key}")

    condition_columns = [
        "pair_id",
        "edge_id",
        "split",
        "plate",
        "cell_line_id",
        "drug",
        "dose_uM",
        "eligible_S256",
    ]
    conditions = pd.read_csv(
        CONDITION_SPLIT_MANIFEST,
        usecols=condition_columns,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    edges = pd.read_csv(
        EDGE_SPLIT_MANIFEST,
        usecols=["edge_id", "cell_line_id", "drug", "split"],
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    if conditions["pair_id"].duplicated().any() or edges["edge_id"].duplicated().any():
        raise AssertionError("Frozen condition or edge IDs are not unique")
    if set(conditions["split"]) != {"train", "val", "test"}:
        raise AssertionError("Unexpected condition split labels")
    if not conditions["eligible_S256"].astype(bool).all():
        raise AssertionError("Condition manifest contains an ineligible row")

    matched_edges = edges.set_index("edge_id").loc[conditions["edge_id"]].reset_index()
    for column in ("cell_line_id", "drug", "split"):
        if not conditions[column].eq(matched_edges[column]).all():
            raise AssertionError(f"Condition-to-edge mapping mismatch: {column}")
    if conditions.groupby("edge_id")["split"].nunique().max() != 1:
        raise AssertionError("An edge crosses train/val/test")

    expected = preparation["split"]["by_split"]
    for split in ("train", "test"):
        selected = conditions.loc[conditions["split"].eq(split)]
        selected_edges = edges.loc[edges["split"].eq(split)]
        if len(selected) != int(expected[split]["conditions"]):
            raise AssertionError(f"{split} condition count changed")
        if len(selected_edges) != int(expected[split]["edges"]):
            raise AssertionError(f"{split} edge count changed")
    return conditions, edges, preparation


def main() -> None:
    conditions, edges, preparation = verified_inputs()
    train = conditions.loc[conditions["split"].eq("train")].copy()
    test = conditions.loc[conditions["split"].eq("test")].copy()

    train_pairs = train[["drug", "dose_uM"]].drop_duplicates()
    test_pairs = test[["drug", "dose_uM"]].drop_duplicates()
    covered_pairs = train_pairs.assign(exact_train_drug_dose_available=True)
    condition_coverage = test.merge(
        covered_pairs,
        on=["drug", "dose_uM"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    condition_coverage["exact_train_drug_dose_available"] = condition_coverage[
        "exact_train_drug_dose_available"
    ].fillna(False).astype(bool)

    edge_coverage = condition_coverage.groupby("edge_id", sort=False).agg(
        all_test_doses_have_exact_train_coverage=(
            "exact_train_drug_dose_available",
            "all",
        ),
        test_condition_count=("pair_id", "size"),
        test_dose_count=("dose_uM", "nunique"),
    )
    test_edges = edges.loc[edges["split"].eq("test"), "edge_id"]
    if set(edge_coverage.index) != set(test_edges):
        raise AssertionError("Test condition and edge manifests disagree")

    missing = (
        condition_coverage.loc[
            ~condition_coverage["exact_train_drug_dose_available"]
        ]
        .groupby(["drug", "dose_uM"], as_index=False, sort=True)
        .agg(test_condition_count=("pair_id", "size"), test_edge_count=("edge_id", "nunique"))
    )
    missing_records = [
        {
            "drug": str(row.drug),
            "dose_uM": float(row.dose_uM),
            "test_condition_count": int(row.test_condition_count),
            "test_edge_count": int(row.test_edge_count),
        }
        for row in missing.itertuples(index=False)
    ]

    covered_conditions = int(condition_coverage["exact_train_drug_dose_available"].sum())
    covered_edges = int(edge_coverage["all_test_doses_have_exact_train_coverage"].sum())
    status = "pass" if not missing_records and covered_edges == len(test_edges) else "fail"

    output_columns = [
        "pair_id",
        "edge_id",
        "cell_line_id",
        "drug",
        "dose_uM",
        "plate",
        "exact_train_drug_dose_available",
    ]
    condition_output = condition_coverage[output_columns].rename(
        columns={"pair_id": "condition_id"}
    )
    atomic_write_csv(CONDITION_COVERAGE_PATH, condition_output)

    goals_text = GOALS_PATH.read_text(encoding="utf-8")
    if "ΔZ(cell_context, drug, dose)" not in goals_text or "μperturbed - μcontrol" not in goals_text:
        raise AssertionError("Existing project goal no longer contains the condition-level shift definition")

    audit = {
        "schema": "tahoe_experiment1_b1_train_test_coverage_audit_v1",
        "created_at_utc": utc_now(),
        "status": status,
        "scope": {
            "audit_only": True,
            "embedding_cache_read": False,
            "condition_centroids_computed": False,
            "mean_delta_table_generated": False,
            "energy_distance_computed": False,
            "prediction_generated": False,
            "b2_or_state_or_st_run": False,
            "fallback_applied": False,
        },
        "inputs": {
            "condition_manifest": {
                "path": CONDITION_SPLIT_MANIFEST.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(CONDITION_SPLIT_MANIFEST),
            },
            "edge_manifest": {
                "path": EDGE_SPLIT_MANIFEST.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(EDGE_SPLIT_MANIFEST),
            },
            "preparation_summary": {
                "path": PREPARATION_SUMMARY.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(PREPARATION_SUMMARY),
            },
            "project_goal": {
                "path": str(GOALS_PATH),
                "sha256": sha256_file(GOALS_PATH),
            },
        },
        "counts": {
            "train_conditions": int(len(train)),
            "test_conditions": int(len(test)),
            "train_unique_drugs": int(train["drug"].nunique()),
            "train_unique_drug_dose": int(len(train_pairs)),
            "test_unique_drugs": int(test["drug"].nunique()),
            "test_unique_drug_dose": int(len(test_pairs)),
            "test_conditions_with_exact_train_drug_dose": covered_conditions,
            "test_condition_exact_coverage_rate": covered_conditions / len(test),
            "test_edges": int(len(test_edges)),
            "test_edges_all_doses_exactly_covered": covered_edges,
            "test_edge_exact_coverage_rate": covered_edges / len(test_edges),
            "missing_unique_drug_dose": int(len(missing_records)),
            "missing_test_conditions": int(len(test) - covered_conditions),
            "missing_test_edges": int(len(test_edges) - covered_edges),
        },
        "missing_drug_dose": missing_records,
        "b1_fitting_unit": {
            "definition": "macro-average condition shift; no cell-count weighting",
            "fit_split": "train only",
            "condition_shift": "Delta_condition = treated centroid - matched control centroid",
            "condition_shift_dimension": 768,
            "group_key": ["drug", "dose_uM"],
            "mean_delta": "equal arithmetic mean of Delta_condition over train conditions sharing exact (drug, dose_uM)",
            "condition_weight": 1.0,
            "cell_count_weighting": False,
            "val_or_test_used_for_fit": False,
            "fallbacks": [],
        },
        "consistency_with_frozen_project_materials": {
            "status": "consistent",
            "existing_definition": "condition-level DeltaZ = mu_perturbed - mu_control",
            "newly_frozen_detail": "equal train-condition macro-average within exact (drug, dose_uM)",
            "conflicting_existing_b1_implementation_found": False,
        },
        "checks": {
            "formal_manifest_sha256": "pass",
            "train_only_coverage_reference": "pass",
            "condition_level_grain": "pass",
            "edge_split_integrity": "pass",
            "no_fallback": "pass",
        },
        "outputs": {
            "condition_coverage_csv": {
                "path": CONDITION_COVERAGE_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(CONDITION_COVERAGE_PATH),
                "rows": len(condition_output),
                "encoding": "utf-8-sig",
            },
            "audit_json": AUDIT_JSON_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "audit_md": AUDIT_MD_PATH.relative_to(PROJECT_ROOT).as_posix(),
        },
        "implementation": {
            "script": SCRIPT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(SCRIPT_PATH),
        },
    }
    atomic_write_text(
        AUDIT_JSON_PATH,
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )

    missing_lines = (
        ["None. Every test `(drug, dose_uM)` has an exact train match."]
        if not missing_records
        else [
            "| drug | dose_uM | test conditions | test edges |",
            "|---|---:|---:|---:|",
            *[
                f"| {row['drug']} | {row['dose_uM']:g} | {row['test_condition_count']} | {row['test_edge_count']} |"
                for row in missing_records
            ],
        ]
    )
    md = "\n".join(
        [
            "# Tahoe Experiment 1 B1 train-to-test drug-dose coverage audit",
            "",
            f"- Status: **{status.upper()}**",
            "- Scope: coverage and fitting-unit audit only; no embedding read, mean-delta table, prediction, Energy, B2, STATE, or ST.",
            f"- Train: **{len(train):,} conditions**, **{train['drug'].nunique():,} drugs**, **{len(train_pairs):,} exact drug-dose pairs**.",
            f"- Test: **{len(test):,} conditions**, **{test['drug'].nunique():,} drugs**, **{len(test_pairs):,} exact drug-dose pairs**, **{len(test_edges):,} edges**.",
            f"- Exact condition coverage: **{covered_conditions:,}/{len(test):,} ({covered_conditions / len(test):.6%})**.",
            f"- All-dose edge coverage: **{covered_edges:,}/{len(test_edges):,} ({covered_edges / len(test_edges):.6%})**.",
            "",
            "## Frozen B1 fitting unit",
            "",
            "Each train condition first contributes one 768-d `Delta_condition = treated centroid - matched control centroid`. "
            "Within each exact `(drug, dose_uM)`, these condition vectors receive equal weight. No cell-count weighting or fallback is allowed.",
            "",
            "## Missing exact train drug-dose coverage",
            "",
            *missing_lines,
            "",
            "## Outputs",
            "",
            f"- `{CONDITION_COVERAGE_PATH.relative_to(PROJECT_ROOT).as_posix()}`",
            f"- `{AUDIT_JSON_PATH.relative_to(PROJECT_ROOT).as_posix()}`",
            f"- `{AUDIT_MD_PATH.relative_to(PROJECT_ROOT).as_posix()}`",
            "",
        ]
    )
    atomic_write_text(AUDIT_MD_PATH, md)
    print(
        json.dumps(
            {
                "status": status,
                "condition_coverage_rate": audit["counts"]["test_condition_exact_coverage_rate"],
                "missing_unique_drug_dose": len(missing_records),
                "missing_test_conditions": audit["counts"]["missing_test_conditions"],
                "missing_test_edges": audit["counts"]["missing_test_edges"],
                "audit": AUDIT_JSON_PATH.relative_to(PROJECT_ROOT).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
