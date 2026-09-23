#!/usr/bin/env python3
"""Finalize Experiment 1 from frozen edge results and ST raw test outputs."""

from __future__ import annotations

import inspect
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_tahoe_experiment1_b0 import (
    aggregate,
    atomic_write_csv,
    atomic_write_text,
    describe,
)
from tahoe_experiment1_latent_data import PROJECT_ROOT, RESULTS, sha256_file


SCRIPT_PATH = Path(__file__).resolve()
TASK_PATH = PROJECT_ROOT.parent / "当前任务.txt"

B0_EDGE = RESULTS / "tahoe_experiment1_b0_identity_edge.csv"
B0_RESULT = RESULTS / "tahoe_experiment1_b0_identity_result.json"
B1_EDGE = RESULTS / "tahoe_experiment1_b1_mean_shift_edge_comparison.csv"
B1_RESULT = RESULTS / "tahoe_experiment1_b1_mean_shift_result.json"
B2_EDGE = RESULTS / "tahoe_experiment1_b2_v2_test_edge_comparison.csv"
B2_RESULT = RESULTS / "tahoe_experiment1_b2_v2_test_result.json"
ST_A_RAW = RESULTS / "tahoe_experiment1_st_a_test_condition_repeat.csv"
ST_R_RAW = RESULTS / "tahoe_experiment1_st_r_test_condition_repeat.csv"
ST_RAW_AUDIT = RESULTS / "tahoe_experiment1_st_test_raw_audit.json"
EVALUATION_PROTOCOL = RESULTS / "tahoe_experiment1_evaluation_sampling_protocol.json"
ST_TRAINING_PROTOCOL = RESULTS / "tahoe_experiment1_st_training_protocol_v2.json"
AGGREGATION_SCRIPT = PROJECT_ROOT / "perturbation_scripts" / "evaluate_tahoe_experiment1_b0.py"
DATASET_SCRIPT = PROJECT_ROOT / "perturbation_scripts" / "tahoe_experiment1_latent_data.py"
ST_A_CHECKPOINT = RESULTS / "tahoe_experiment1_st_formal_checkpoints_v2" / "st-a" / "best.pt"
ST_R_CHECKPOINT = RESULTS / "tahoe_experiment1_st_formal_checkpoints_v2" / "st-r" / "best.pt"

ST_A_EDGE = RESULTS / "tahoe_experiment1_st_a_test_edge.csv"
ST_R_EDGE = RESULTS / "tahoe_experiment1_st_r_test_edge.csv"
FINAL_EDGE = RESULTS / "tahoe_experiment1_final_edge_comparison.csv"
MODEL_SUMMARY = RESULTS / "tahoe_experiment1_final_model_summary.csv"
PAIRWISE_SUMMARY = RESULTS / "tahoe_experiment1_final_pairwise_summary.csv"
REPORT_TABLE = RESULTS / "tahoe_experiment1_final_report_table.csv"
FIGURE_MODEL_ENERGY = RESULTS / "tahoe_experiment1_final_model_energy.png"
FIGURE_GAIN_VS_B2 = RESULTS / "tahoe_experiment1_final_gain_vs_b2.png"
FIGURE_ST_PAIR = RESULTS / "tahoe_experiment1_final_st_a_vs_st_r.png"
FINAL_RESULT = RESULTS / "tahoe_experiment1_final_comparison_result.json"
FINAL_HANDOFF = RESULTS / "tahoe_experiment1_final_comparison_handoff.md"

EXPECTED = {
    B0_EDGE: "70f4a388a54dc900662c6d83796e71411aaf4151625851580a827c3cea665059",
    B0_RESULT: "918b6241f02e95185f74a227344e17f371832b7ffc79dd7e97b7dc463220547d",
    B1_EDGE: "76893c25a97341552dd21e5f4c763536639c381154a5deeb4f1d3c67a20bc8fe",
    B1_RESULT: "a2f49dd3edc7cc50a37700a3a5da4e61e5d3ceb683c3e3939aea379658eb3c6d",
    B2_EDGE: "148e76668dc921600bd597d3d3044ee781f14aa888e60126cc0a39162de37f31",
    B2_RESULT: "fa16e6f83f7056858004d3016cd8f2b69de9c0ca810898ec86fd364c8b397e4e",
    ST_A_RAW: "a62f0d5e85f5c5a91951f47e87cfc325a52075de7862b60641ed090ea96cec63",
    ST_R_RAW: "d0b30ec5eb1ed1aa8663175c01e10e03a730559174ed3889d029894e61c0cb44",
    ST_RAW_AUDIT: "81159c66ecd99b64076b30a711da6772817712cbda13458564c1003a28fa254f",
    EVALUATION_PROTOCOL: "66ddf436acb54a9c662553f48df3af382d57b947a3a796173ee5cc6fb30555a1",
    ST_TRAINING_PROTOCOL: "3630a8e527aac812b1bb182c35d310b71122dca6257b63f109ac0ae0ad5b581b",
    AGGREGATION_SCRIPT: "e35dc32ace9eb941175f57cef4e31836f25d76f459631ea6f3c45874bb7af900",
    DATASET_SCRIPT: "e0cb346797a3517e7d4aeddc4a40d0560f11f483520da8a43f0bc808f14e641d",
    ST_A_CHECKPOINT: "9bd0f2719f42dac4fa5a9aabc4fd2bb242a442fb652ae90a8ee52fa54ca03652",
    ST_R_CHECKPOINT: "357fd6ec00473f5a6f2b82f5693c58365ea08f0af30500d092d354d0fbefd3ab",
}

EXPECTED_CONDITIONS = 5_684
EXPECTED_REPEATS = 5
EXPECTED_RAW_ROWS = 28_420
EXPECTED_EDGES = 1_717
EXPECTED_PAIRED_GROUPS = 397
EXPECTED_DOSE_UNITS = 4_980
EPS = np.finfo(np.float64).eps
KEYS = ["edge_id", "cell_line_id", "drug"]
RAW_KEYS = ["condition_id", *KEYS, "dose_uM", "plate", "repeat_epoch"]
MODELS = {
    "B0": "b0_energy",
    "B1": "b1_energy",
    "B2-v2": "b2_energy",
    "ST-R": "st_r_energy",
    "ST-A": "st_a_energy",
}
PAIR_SPECS = [
    ("B1", "B0"),
    ("B2-v2", "B0"),
    ("B2-v2", "B1"),
    ("ST-R", "B0"),
    ("ST-R", "B1"),
    ("ST-R", "B2-v2"),
    ("ST-A", "B0"),
    ("ST-A", "B1"),
    ("ST-A", "B2-v2"),
    ("ST-A", "ST-R"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n",
    )


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def verify_frozen_sources() -> dict[Path, str]:
    observed: dict[Path, str] = {}
    for path, expected in EXPECTED.items():
        if not path.is_file():
            raise AssertionError(f"Missing frozen source: {path}")
        observed[path] = sha256_file(path)
        if observed[path] != expected:
            raise AssertionError(f"Frozen source SHA mismatch: {relative(path)}")
    return observed


def verify_protocols_and_results() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    evaluation = read_json(EVALUATION_PROTOCOL)
    expected_evaluation = {
        "status": "frozen",
        "S": 256,
        "evaluation_repeat_count": EXPECTED_REPEATS,
        "evaluation_base_seed": 42,
        "evaluation_repeat_epochs": [0, 1, 2, 3, 4],
        "shuffle": False,
        "drop_last": False,
        "all-models-share-identical-evaluation-indices": True,
    }
    for key, expected in expected_evaluation.items():
        if evaluation.get(key) != expected:
            raise AssertionError(f"Evaluation protocol mismatch: {key}")

    st_protocol = read_json(ST_TRAINING_PROTOCOL)
    if st_protocol.get("status") != "frozen":
        raise AssertionError("ST training protocol is not frozen")
    variants = st_protocol.get("variants", {})
    if variants.get("st-a", {}).get("predict_residual") is not False:
        raise AssertionError("ST-A contract mismatch")
    if (
        variants.get("st-r", {}).get("predict_residual") is not True
        or variants.get("st-r", {}).get("residual_mode") != "output"
    ):
        raise AssertionError("ST-R output-space residual contract mismatch")

    results = {
        "b0": read_json(B0_RESULT),
        "b1": read_json(B1_RESULT),
        "b2": read_json(B2_RESULT),
        "st_raw": read_json(ST_RAW_AUDIT),
    }
    for name, value in results.items():
        if value.get("status") != "pass":
            raise AssertionError(f"Source result is not PASS: {name}")

    declared = {
        B0_EDGE: results["b0"]["outputs"]["edge_csv"]["sha256"],
        B1_EDGE: results["b1"]["outputs"]["edge_comparison_csv"]["sha256"],
        B2_EDGE: results["b2"]["outputs"]["edge_comparison_csv"]["sha256"],
        ST_A_RAW: results["st_raw"]["outputs"]["st_a_condition_repeat_csv"]["sha256"],
        ST_R_RAW: results["st_raw"]["outputs"]["st_r_condition_repeat_csv"]["sha256"],
    }
    for path, declared_sha in declared.items():
        if declared_sha != EXPECTED[path]:
            raise AssertionError(f"Source result declares wrong SHA: {relative(path)}")

    checkpoints = results["st_raw"].get("checkpoints", {})
    for variant, epoch, path in (
        ("st-a", 28, ST_A_CHECKPOINT),
        ("st-r", 26, ST_R_CHECKPOINT),
    ):
        item = checkpoints.get(variant, {})
        if (
            item.get("epoch") != epoch
            or item.get("best_epoch") != epoch
            or item.get("sha256") != EXPECTED[path]
            or item.get("selected_using_test") is not False
            or item.get("parameters_unchanged") is not True
        ):
            raise AssertionError(f"Checkpoint audit mismatch: {variant}")

    execution = results["st_raw"].get("execution", {})
    forbidden = [
        "ddp",
        "optimizer_constructed",
        "backward_called",
        "parameter_update",
        "test_time_fitting_or_calibration",
        "test_selected_epoch",
        "edge_aggregation_run",
        "b0_b1_b2_energy_recomputed",
    ]
    if any(execution.get(key) is not False for key in forbidden):
        raise AssertionError("ST raw audit execution contract mismatch")
    dataset_audit = results["st_raw"].get("dataset", {})
    if (
        dataset_audit.get("current_sha256") != EXPECTED[DATASET_SCRIPT]
        or dataset_audit.get("historical_evaluation_protocol_sha256")
        != evaluation.get("tahoe_experiment1_latent_data.py_sha256")
    ):
        raise AssertionError("ST raw audit does not document the Dataset SHA history")
    return evaluation, results


def verify_aggregation_implementation() -> None:
    source = inspect.getsource(aggregate)
    required = [
        "float(key[-1]) == 5.0",
        '{"plate6", "plate14"} <= plates',
        'energy_distance=("energy_distance", "mean")',
    ]
    if any(fragment not in source for fragment in required):
        raise AssertionError("Existing aggregate() is not the frozen high-dose-only implementation")


def load_and_validate_raw(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    expected_columns = [*RAW_KEYS, "energy_distance"]
    if list(raw.columns) != expected_columns:
        raise AssertionError(f"Unexpected raw columns: {relative(path)}")
    if len(raw) != EXPECTED_RAW_ROWS:
        raise AssertionError(f"Raw row count mismatch: {relative(path)}")
    if raw["condition_id"].nunique() != EXPECTED_CONDITIONS:
        raise AssertionError(f"Condition count mismatch: {relative(path)}")
    if raw["edge_id"].nunique() != EXPECTED_EDGES:
        raise AssertionError(f"Unaggregated edge count mismatch: {relative(path)}")
    if raw.duplicated(["condition_id", "repeat_epoch"]).any():
        raise AssertionError(f"Duplicate condition/repeat rows: {relative(path)}")
    counts = raw.groupby("condition_id")["repeat_epoch"].agg(["size", "nunique"])
    if not ((counts["size"] == EXPECTED_REPEATS) & (counts["nunique"] == EXPECTED_REPEATS)).all():
        raise AssertionError(f"Each condition must have five repeats: {relative(path)}")
    epochs = raw.groupby("condition_id")["repeat_epoch"].agg(lambda x: set(x))
    if not epochs.map(lambda x: x == {0, 1, 2, 3, 4}).all():
        raise AssertionError(f"Repeat epochs mismatch: {relative(path)}")
    if not np.isfinite(raw["energy_distance"].to_numpy(np.float64)).all():
        raise AssertionError(f"Non-finite Energy: {relative(path)}")
    metadata_counts = raw.groupby("condition_id")[[*KEYS, "dose_uM", "plate"]].nunique()
    if not (metadata_counts == 1).all().all():
        raise AssertionError(f"Condition metadata changed across repeats: {relative(path)}")
    return raw


def aggregate_st(raw: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    edge, details = aggregate(raw)
    audit = details["replicate_audit"].iloc[0]
    counts = {
        "plate conditions": len(details["plate_condition"]),
        "high-dose paired groups": int(audit["plate6_plate14_groups"]),
        "dose units": len(details["dose"]),
        "edges": len(edge),
    }
    expected = {
        "plate conditions": EXPECTED_CONDITIONS,
        "high-dose paired groups": EXPECTED_PAIRED_GROUPS,
        "dose units": EXPECTED_DOSE_UNITS,
        "edges": EXPECTED_EDGES,
    }
    if counts != expected:
        raise AssertionError(f"{label} aggregation counts mismatch: {counts}")
    if edge.duplicated(KEYS).any() or not np.isfinite(edge["energy_distance"]).all():
        raise AssertionError(f"{label} edge output invalid")
    return edge, details


def load_baseline_edges() -> tuple[pd.DataFrame, dict[str, float]]:
    b0 = pd.read_csv(B0_EDGE, encoding="utf-8-sig")
    b1 = pd.read_csv(B1_EDGE, encoding="utf-8-sig")
    b2 = pd.read_csv(B2_EDGE, encoding="utf-8-sig")
    for name, frame in (("B0", b0), ("B1", b1), ("B2-v2", b2)):
        if len(frame) != EXPECTED_EDGES or frame.duplicated(KEYS).any():
            raise AssertionError(f"{name} does not contain one row per expected edge")
    canonical = b0[KEYS].sort_values(KEYS, kind="stable").reset_index(drop=True)
    for name, frame in (("B1", b1), ("B2-v2", b2)):
        observed = frame[KEYS].sort_values(KEYS, kind="stable").reset_index(drop=True)
        pd.testing.assert_frame_equal(canonical, observed, check_dtype=False)

    joined = b0[KEYS + ["energy_distance"]].rename(
        columns={"energy_distance": "b0_energy"}
    )
    joined = joined.merge(b1[KEYS + ["b0_energy", "b1_energy"]], on=KEYS, validate="one_to_one")
    joined = joined.rename(columns={"b0_energy_x": "b0_energy", "b0_energy_y": "b0_from_b1"})
    joined = joined.merge(
        b2[KEYS + ["b0_energy", "b1_energy", "b2_energy"]],
        on=KEYS,
        validate="one_to_one",
        suffixes=("", "_from_b2"),
    )
    differences = {
        "b0_vs_b1_max_abs": float(np.max(np.abs(joined["b0_energy"] - joined["b0_from_b1"]))),
        "b0_vs_b2_max_abs": float(np.max(np.abs(joined["b0_energy"] - joined["b0_energy_from_b2"]))),
        "b1_vs_b2_max_abs": float(np.max(np.abs(joined["b1_energy"] - joined["b1_energy_from_b2"]))),
    }
    if max(differences.values()) > 1e-9:
        raise AssertionError(f"Baseline references disagree: {differences}")
    final = joined[KEYS + ["b0_energy", "b1_energy", "b2_energy"]].copy()
    if not np.isfinite(final[list(MODELS.values())[:3]].to_numpy(np.float64)).all():
        raise AssertionError("A baseline edge Energy is non-finite")
    return final, differences


def add_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    pairs = {
        "st_a_minus_b0": ("b0_energy", "st_a_energy"),
        "st_a_minus_b1": ("b1_energy", "st_a_energy"),
        "st_a_minus_b2": ("b2_energy", "st_a_energy"),
        "st_a_minus_st_r": ("st_r_energy", "st_a_energy"),
        "st_r_minus_b0": ("b0_energy", "st_r_energy"),
        "st_r_minus_b1": ("b1_energy", "st_r_energy"),
        "st_r_minus_b2": ("b2_energy", "st_r_energy"),
    }
    for name, (reference, candidate) in pairs.items():
        frame[name] = frame[reference] - frame[candidate]
        frame[f"gain_{name.removesuffix('_minus_' + name.split('_minus_')[-1])}_vs_{name.split('_minus_')[-1]}"] = (
            frame[name] / (frame[reference] + EPS)
        )
    expected_gain_names = {
        "gain_st_a_vs_b0",
        "gain_st_a_vs_b1",
        "gain_st_a_vs_b2",
        "gain_st_a_vs_st_r",
        "gain_st_r_vs_b0",
        "gain_st_r_vs_b1",
        "gain_st_r_vs_b2",
    }
    if not expected_gain_names <= set(frame.columns):
        raise AssertionError("Paired gain columns were not constructed as requested")

    values = frame[list(MODELS.values())].to_numpy(np.float64)
    minima = values.min(axis=1)
    winner_labels: list[str] = []
    for row, minimum in zip(values, minima, strict=True):
        winners = [label for label, value in zip(MODELS, row, strict=True) if value == minimum]
        winner_labels.append(winners[0] if len(winners) == 1 else "TIE[" + "|".join(winners) + "]")
    frame["best_model"] = winner_labels
    return frame


def build_model_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float | int]]]:
    rows = []
    full: dict[str, dict[str, float | int]] = {}
    for model, column in MODELS.items():
        stats = describe(frame[column])
        full[model] = stats
        rows.append({"model": model, **stats})
    summary = pd.DataFrame(rows).sort_values("mean", kind="stable").reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    summary = summary.rename(columns={"mean": "mean_energy", "median": "median_energy"})
    return summary, full


def ratio_of_means(stats: dict[str, dict[str, float | int]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "formula": "(reference_mean - candidate_mean) / reference_mean",
        "by_reference": {},
    }
    for reference in ("B0", "B1", "B2-v2", "ST-R"):
        reference_mean = float(stats[reference]["mean"])
        result["by_reference"][reference] = {
            candidate: (reference_mean - float(values["mean"])) / reference_mean
            for candidate, values in stats.items()
        }
    return result


def pairwise_row(frame: pd.DataFrame, candidate: str, reference: str) -> dict[str, Any]:
    candidate_values = frame[MODELS[candidate]].to_numpy(np.float64)
    reference_values = frame[MODELS[reference]].to_numpy(np.float64)
    delta = reference_values - candidate_values
    gain = delta / (reference_values + EPS)
    beats = int((delta > 0).sum())
    ties = int((delta == 0).sum())
    worse = int((delta < 0).sum())
    if beats + ties + worse != EXPECTED_EDGES:
        raise AssertionError("Pairwise counts do not partition all edges")
    return {
        "comparison": f"{candidate} vs {reference}",
        "candidate": candidate,
        "reference": reference,
        "edge_count": EXPECTED_EDGES,
        "mean_delta_ed": float(delta.mean()),
        "median_delta_ed": float(np.quantile(delta, 0.5)),
        "mean_energy_gain": float(gain.mean()),
        "median_energy_gain": float(np.quantile(gain, 0.5)),
        "p05_energy_gain": float(np.quantile(gain, 0.05)),
        "p95_energy_gain": float(np.quantile(gain, 0.95)),
        "candidate_beats_reference_count": beats,
        "candidate_beats_reference_fraction": beats / EXPECTED_EDGES,
        "ties": ties,
        "worse_edges": worse,
    }


def build_winners(frame: pd.DataFrame) -> dict[str, Any]:
    counts = frame["best_model"].value_counts().to_dict()
    tie_counts = {key: int(value) for key, value in counts.items() if key.startswith("TIE[")}
    return {
        "rule": "exact float64 minimum; exact ties are not assigned to a single model",
        "models": {
            model: {
                "count": int(counts.get(model, 0)),
                "fraction": int(counts.get(model, 0)) / EXPECTED_EDGES,
            }
            for model in MODELS
        },
        "tie_edges": int(sum(tie_counts.values())),
        "tie_groups": tie_counts,
    }


def build_cell_line_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    aggregated = frame.groupby("cell_line_id", sort=True).agg(
        edge_count=("edge_id", "size"),
        **{f"{column}_mean": (column, "mean") for column in MODELS.values()},
    )
    return aggregated.reset_index().to_dict(orient="records")


def by_dose_lookup(rows: Iterable[dict[str, Any]], field: str) -> dict[float, tuple[int, float]]:
    return {float(row["dose_uM"]): (int(row["count"]), float(row[field])) for row in rows}


def build_dose_summary(
    source_results: dict[str, dict[str, Any]],
    st_a_details: dict[str, pd.DataFrame],
    st_r_details: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    baseline = {
        "b0_energy": by_dose_lookup(source_results["b0"]["stratified_summary"]["by_dose"], "mean"),
        "b1_energy": by_dose_lookup(source_results["b1"]["stratified_summary"]["by_dose"], "b1_mean_energy"),
        "b2_energy": by_dose_lookup(source_results["b2"]["stratified_summary"]["by_dose"], "b2_mean_energy"),
    }
    st_lookup: dict[str, dict[float, tuple[int, float]]] = {}
    for column, details in (("st_a_energy", st_a_details), ("st_r_energy", st_r_details)):
        grouped = details["dose"].groupby("dose_uM")["energy_distance"].agg(["size", "mean"])
        st_lookup[column] = {
            float(dose): (int(row["size"]), float(row["mean"])) for dose, row in grouped.iterrows()
        }

    expected_counts = {0.05: 1667, 0.5: 1665, 5.0: 1648}
    rows: list[dict[str, Any]] = []
    for dose, expected_count in expected_counts.items():
        all_values = {**baseline, **st_lookup}
        if any(lookup[dose][0] != expected_count for lookup in all_values.values()):
            raise AssertionError(f"Dose-level count mismatch at {dose} uM")
        rows.append(
            {
                "dose_uM": dose,
                "dose_level_units": expected_count,
                **{column: lookup[dose][1] for column, lookup in all_values.items()},
            }
        )
    return rows


def save_figures(summary: pd.DataFrame, frame: pd.DataFrame, ratios: dict[str, Any]) -> None:
    order = ["B0", "B1", "B2-v2", "ST-R", "ST-A"]
    colors = ["#7A7A7A", "#4C78A8", "#59A14F", "#F28E2B", "#E15759"]
    means = [float(summary.loc[summary["model"] == model, "mean_energy"].iloc[0]) for model in order]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(order, means, color=colors, width=0.68)
    ax.set_ylabel("Mean edge-level Energy (lower is better)")
    fig.suptitle("Mean edge-level Energy by model", fontsize=14, fontweight="bold", y=0.98)
    ax.set_title(
        "Held-out unseen (cell line, drug) combinations; N = 1,717 edges",
        fontsize=9,
        color="#444444",
        pad=10,
    )
    ax.set_ylim(0, max(means) * 1.20)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    for bar, value in zip(bars, means, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(means) * 0.025, f"{value:.5f}", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURE_MODEL_ENERGY, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    gain_values = [
        float(ratios["by_reference"]["B2-v2"]["ST-R"]),
        float(ratios["by_reference"]["B2-v2"]["ST-A"]),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars = ax.bar(["ST-R", "ST-A"], np.asarray(gain_values) * 100, color=colors[-2:], width=0.55)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Ratio-of-means improvement vs B2-v2 (%)")
    fig.suptitle(
        "ST improvement over pooled MLP baseline", fontsize=14, fontweight="bold", y=0.98
    )
    ax.set_title(
        "Held-out unseen (cell line, drug) combinations; N = 1,717 edges",
        fontsize=9,
        color="#444444",
        pad=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    top = max(gain_values) * 100
    ax.set_ylim(min(0, min(gain_values) * 115), top * 1.22)
    for bar, value in zip(bars, gain_values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 100 + top * 0.04, f"{value:.1%}", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURE_GAIN_VS_B2, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    x = frame["st_r_energy"].to_numpy(np.float64)
    y = frame["st_a_energy"].to_numpy(np.float64)
    upper = max(float(x.max()), float(y.max())) * 1.03
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.scatter(x, y, s=11, alpha=0.42, color="#4C78A8", edgecolors="none")
    ax.plot([0, upper], [0, upper], linestyle="--", color="#555555", linewidth=1.0, label="y = x")
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("ST-R edge Energy")
    ax.set_ylabel("ST-A edge Energy")
    fig.suptitle("ST-A vs ST-R paired edge comparison", fontsize=14, fontweight="bold", y=0.98)
    ax.set_title(
        "Held-out unseen (cell line, drug) combinations; N = 1,717 edges",
        fontsize=9,
        color="#444444",
        pad=10,
    )
    ax.text(0.03, 0.96, "Below line: ST-A better", transform=ax.transAxes, va="top", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURE_ST_PAIR, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_handoff(
    summary: pd.DataFrame,
    ratios: dict[str, Any],
    pairwise: dict[str, dict[str, Any]],
    winners: dict[str, Any],
    dose_summary: list[dict[str, Any]],
) -> str:
    means = {row["model"]: float(row["mean_energy"]) for row in summary.to_dict(orient="records")}
    st_a_vs_st_r = ratios["by_reference"]["ST-R"]["ST-A"]
    pair_a_r = pairwise["ST-A vs ST-R"]
    pair_a_b2 = pairwise["ST-A vs B2-v2"]
    pair_r_b2 = pairwise["ST-R vs B2-v2"]
    ranking_rows = "\n".join(
        f"| {int(row['rank'])} | {row['model']} | {float(row['mean_energy']):.8f} |"
        for row in summary.to_dict(orient="records")
    )
    winner_rows = "\n".join(
        f"| {model} | {value['count']} | {percent(value['fraction'])} |"
        for model, value in winners["models"].items()
    )
    dose_rows = "\n".join(
        "| {dose:g} | {b0:.6f} | {b1:.6f} | {b2:.6f} | {str_:.6f} | {sta:.6f} |".format(
            dose=row["dose_uM"],
            b0=row["b0_energy"],
            b1=row["b1_energy"],
            b2=row["b2_energy"],
            str_=row["st_r_energy"],
            sta=row["st_a_energy"],
        )
        for row in dose_summary
    )
    return f"""# Experiment 1 最终结果

## 一句话结论

在 1,717 个 held-out 的 `(cell line, drug)` 未见组合上，ST-A 的 mean edge-level Energy 为 **{means['ST-A']:.8f}**，在五种方法中最低；它相对 B2-v2 的 ratio-of-means improvement 为 **{percent(ratios['by_reference']['B2-v2']['ST-A'])}**，并在 **{pair_a_b2['candidate_beats_reference_count']} / 1,717** 个 edge 上更优。

## 实验在测什么

实验用冻结的 GeneJEPA Epoch25 latent 表示细胞状态。给定同一 context 的 control cell population 与 drug/dose，预测 treated cell population；最终 test 只包含训练时未见过的 `(cell_line_id, drug)` **组合**。主指标是 edge-level Energy distance，越低越好。

## 五种方法分别是什么

- **B0**：no-change，直接用 control population 作为预测。
- **B1**：使用 train context 估计的 drug-dose 平均 condition shift。
- **B2-v2**：context-conditioned pooled MLP，以 condition-centroid shift MSE 训练。
- **ST-R**：set-to-set 模型预测 output-space residual，再加回 raw control latent。
- **ST-A**：set-to-set 模型直接预测 absolute treated latent population。

## 最终 test 结果

| 排名 | 模型 | Mean edge Energy |
|---:|---|---:|
{ranking_rows}

![五模型 mean edge Energy]({FIGURE_MODEL_ENERGY.name})

按 dose 的 mean Energy（每个 dose-level unit 等权）：

| Dose (uM) | B0 | B1 | B2-v2 | ST-R | ST-A |
|---:|---:|---:|---:|---:|---:|
{dose_rows}

## ST-A / ST-R 对 baseline 的提升

这里的百分比是 **ratio of means**，即 `(reference mean - candidate mean) / reference mean`；它不等于逐 edge EnergyGain 的平均值。

- ST-A 相对 B0 / B1 / B2-v2：**{percent(ratios['by_reference']['B0']['ST-A'])} / {percent(ratios['by_reference']['B1']['ST-A'])} / {percent(ratios['by_reference']['B2-v2']['ST-A'])}**。
- ST-R 相对 B0 / B1 / B2-v2：**{percent(ratios['by_reference']['B0']['ST-R'])} / {percent(ratios['by_reference']['B1']['ST-R'])} / {percent(ratios['by_reference']['B2-v2']['ST-R'])}**。
- ST-A 相对 ST-R：**{percent(st_a_vs_st_r)}**；ST-A 在 **{pair_a_r['candidate_beats_reference_count']} / 1,717** 个 edge 上优于 ST-R。
- 相对 B2-v2，ST-A / ST-R 分别在 **{pair_a_b2['candidate_beats_reference_count']} / 1,717** 和 **{pair_r_b2['candidate_beats_reference_count']} / 1,717** 个 edge 上更优。

![ST 相对 B2-v2 的提升]({FIGURE_GAIN_VS_B2.name})

![ST-A 与 ST-R 的 paired edge 比较]({FIGURE_ST_PAIR.name})

Edge winner（严格取最小 Energy；精确 ties 单独计数）：

| 模型 | Unique winner count | Fraction of all edges |
|---|---:|---:|
{winner_rows}

精确 tie edges：**{winners['tie_edges']}**。

## 这说明什么

1. 在 held-out 的 `(cell line, drug)` 未见组合上，STATE-style ST 能结合 control cell population 与 drug/dose，生成更接近真实 treated population 的 GeneJEPA latent distribution。
2. 完整 set-to-set 模型在主指标上进一步超过 no-change、drug-dose average shift 和 context-conditioned pooled MLP；本实验中 ST-A 表现最佳。
3. 冻结的 GeneJEPA latent 中包含了足以支持下游 perturbation prediction 与组合泛化的可利用信息。

## 不能说明什么

- 这不是 unseen-drug 或 unseen-cell-line 泛化：test 中 drug 与 cell line 个体都在 train 出现，只是二者组合未出现。
- 不能把 ST 相对 B2-v2 的全部提升只归因于 architecture；两者同时存在架构与训练目标差异（B2-v2 使用 centroid-shift MSE，ST 使用 set-level Energy）。
- 不能说 GeneJEPA 本身预测了药物扰动；GeneJEPA 是冻结的 cell-state encoder，ST/B1/B2 才是 perturbation predictor。

## 技术配置摘要

- 正式 test：5,684 conditions，1,717 `(cell_line_id, drug)` edges。
- 每个 condition：5 个固定 S=256 repeats（epochs 0–4，base seed 42）。
- 聚合：repeat → plate-level condition → biological replicate → `(cell line, drug, dose)` → equal dose average → edge。
- 只有 `dose_uM == 5.0` 且 plate6/plate14 同时存在时合并，得到 397 个 high-dose replicate groups；不使用 cell-count weighting。
- ST-A：best epoch 28；ST-R：best epoch 26。所有输入 latent 保持 frozen raw signed 768-d 表示。
- 本轮只聚合与比较已有正式结果；无训练、无推理、无 test-time fitting。
- 冻结 evaluation protocol 记录的是历史 Dataset whole-file SHA；后续 ST v2 Dataset whole-file SHA 不同，但 sampler 规则、固定 spotcheck 指纹与五模型 raw row keys 已在 ST raw audit 中确认一致。

## 主要产物

- ST edge：`{relative(ST_A_EDGE)}`、`{relative(ST_R_EDGE)}`
- 五模型 paired edge 表：`{relative(FINAL_EDGE)}`
- 模型与 paired summaries：`{relative(MODEL_SUMMARY)}`、`{relative(PAIRWISE_SUMMARY)}`
- 汇报核心表：`{relative(REPORT_TABLE)}`
- 机器可读结果：`{relative(FINAL_RESULT)}`
- 本报告：`{relative(FINAL_HANDOFF)}`
"""


def main() -> None:
    source_hashes_before = verify_frozen_sources()
    evaluation_protocol, source_results = verify_protocols_and_results()
    verify_aggregation_implementation()

    st_a_raw = load_and_validate_raw(ST_A_RAW)
    st_r_raw = load_and_validate_raw(ST_R_RAW)
    pd.testing.assert_frame_equal(
        st_a_raw[RAW_KEYS], st_r_raw[RAW_KEYS], check_dtype=False, check_exact=True
    )
    st_a_edge, st_a_details = aggregate_st(st_a_raw, "ST-A")
    st_r_edge, st_r_details = aggregate_st(st_r_raw, "ST-R")

    st_a_mean = float(st_a_edge["energy_distance"].mean())
    st_r_mean = float(st_r_edge["energy_distance"].mean())
    if not np.isclose(st_a_mean, 0.02110, atol=0.001) or not np.isclose(st_r_mean, 0.02796, atol=0.001):
        raise AssertionError(f"ST aggregation failed sanity check: ST-A={st_a_mean}, ST-R={st_r_mean}")

    baseline, baseline_reference_differences = load_baseline_edges()
    final = baseline.merge(
        st_r_edge[KEYS + ["energy_distance"]].rename(columns={"energy_distance": "st_r_energy"}),
        on=KEYS,
        validate="one_to_one",
    ).merge(
        st_a_edge[KEYS + ["energy_distance"]].rename(columns={"energy_distance": "st_a_energy"}),
        on=KEYS,
        validate="one_to_one",
    )
    if len(final) != EXPECTED_EDGES or final.duplicated(KEYS).any():
        raise AssertionError("Five-model merge is not one-to-one over all 1,717 edges")
    if not np.isfinite(final[list(MODELS.values())].to_numpy(np.float64)).all():
        raise AssertionError("Five-model table contains non-finite Energy")
    final = add_comparisons(final).sort_values("edge_id", kind="stable").reset_index(drop=True)

    summary, descriptive = build_model_summary(final)
    ratios = ratio_of_means(descriptive)
    pairwise_rows = [pairwise_row(final, candidate, reference) for candidate, reference in PAIR_SPECS]
    pairwise_frame = pd.DataFrame(pairwise_rows)
    pairwise = {row["comparison"]: row for row in pairwise_rows}
    winners = build_winners(final)
    cell_line_summary = build_cell_line_summary(final)
    dose_summary = build_dose_summary(source_results, st_a_details, st_r_details)

    top_columns = [*KEYS, "b2_energy", "st_a_energy", "st_a_minus_b2", "gain_st_a_vs_b2"]
    top_st_a_vs_b2 = final.nlargest(10, "st_a_minus_b2")[top_columns].to_dict(orient="records")
    bottom_st_a_vs_b2 = final.nsmallest(10, "st_a_minus_b2")[top_columns].to_dict(orient="records")

    report_rows = []
    for model, column in MODELS.items():
        values = final[column].to_numpy(np.float64)
        b2 = final["b2_energy"].to_numpy(np.float64)
        report_rows.append(
            {
                "Model": model,
                "Mean Energy": float(values.mean()),
                "Relative improvement vs B0": ratios["by_reference"]["B0"][model],
                "Relative improvement vs B2": ratios["by_reference"]["B2-v2"][model],
                "Edge win rate vs B2": float((values < b2).mean()),
            }
        )
    report_frame = pd.DataFrame(report_rows)

    atomic_write_csv(ST_A_EDGE, st_a_edge)
    atomic_write_csv(ST_R_EDGE, st_r_edge)
    atomic_write_csv(FINAL_EDGE, final)
    atomic_write_csv(MODEL_SUMMARY, summary)
    atomic_write_csv(PAIRWISE_SUMMARY, pairwise_frame)
    atomic_write_csv(REPORT_TABLE, report_frame)
    save_figures(summary, final, ratios)
    handoff = build_handoff(summary, ratios, pairwise, winners, dose_summary)
    atomic_write_text(FINAL_HANDOFF, handoff)

    source_hashes_after = {path: sha256_file(path) for path in EXPECTED}
    if source_hashes_after != source_hashes_before:
        raise AssertionError("A frozen source changed during finalization")

    artifact_paths = [
        ST_A_EDGE,
        ST_R_EDGE,
        FINAL_EDGE,
        MODEL_SUMMARY,
        PAIRWISE_SUMMARY,
        REPORT_TABLE,
        FIGURE_MODEL_ENERGY,
        FIGURE_GAIN_VS_B2,
        FIGURE_ST_PAIR,
        FINAL_HANDOFF,
    ]
    outputs = {
        path.stem: {"path": relative(path), "sha256": sha256_file(path)} for path in artifact_paths
    }
    outputs[FINAL_RESULT.stem] = {
        "path": relative(FINAL_RESULT),
        "sha256": None,
        "note": "A JSON file cannot embed its own SHA-256 without changing that SHA; reported after write.",
    }

    result = {
        "schema": "tahoe_experiment1_final_comparison_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "experiment1_status": "final comparison complete",
        "scope": "held-out unseen (cell_line_id, drug) combinations; not unseen drug or unseen cell line",
        "source_provenance": {
            "evaluation_protocol": {"path": relative(EVALUATION_PROTOCOL), "sha256": EXPECTED[EVALUATION_PROTOCOL]},
            "st_raw_audit": {"path": relative(ST_RAW_AUDIT), "sha256": EXPECTED[ST_RAW_AUDIT]},
            "b0_result": {"path": relative(B0_RESULT), "sha256": EXPECTED[B0_RESULT]},
            "b1_result": {"path": relative(B1_RESULT), "sha256": EXPECTED[B1_RESULT]},
            "b2_v2_result": {"path": relative(B2_RESULT), "sha256": EXPECTED[B2_RESULT]},
            "st_a_raw": {"path": relative(ST_A_RAW), "sha256": EXPECTED[ST_A_RAW]},
            "st_r_raw": {"path": relative(ST_R_RAW), "sha256": EXPECTED[ST_R_RAW]},
            "st_a_checkpoint": {"path": relative(ST_A_CHECKPOINT), "sha256": EXPECTED[ST_A_CHECKPOINT], "epoch": 28},
            "st_r_checkpoint": {"path": relative(ST_R_CHECKPOINT), "sha256": EXPECTED[ST_R_CHECKPOINT], "epoch": 26},
            "st_training_protocol": {"path": relative(ST_TRAINING_PROTOCOL), "sha256": EXPECTED[ST_TRAINING_PROTOCOL]},
            "aggregation_implementation": {"path": relative(AGGREGATION_SCRIPT), "sha256": EXPECTED[AGGREGATION_SCRIPT], "function": "aggregate"},
            "current_dataset": {
                "path": relative(DATASET_SCRIPT),
                "sha256": EXPECTED[DATASET_SCRIPT],
                "historical_evaluation_protocol_sha256": evaluation_protocol["tahoe_experiment1_latent_data.py_sha256"],
                "sampling_equivalence_evidence": "ST raw audit: sampler rule and frozen spotcheck fingerprints match",
            },
            "finalization_script": {"path": relative(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
            "task": {"path": TASK_PATH.as_posix(), "sha256": sha256_file(TASK_PATH)},
        },
        "counts": {
            "test_conditions": EXPECTED_CONDITIONS,
            "repeat_count": EXPECTED_REPEATS,
            "raw_rows_per_st_model": EXPECTED_RAW_ROWS,
            "plate_level_conditions": EXPECTED_CONDITIONS,
            "biological_replicate_units": EXPECTED_CONDITIONS - EXPECTED_PAIRED_GROUPS,
            "high_dose_replicate_groups": EXPECTED_PAIRED_GROUPS,
            "dose_level_units": EXPECTED_DOSE_UNITS,
            "final_edges_per_model": EXPECTED_EDGES,
        },
        "aggregation": {
            "hierarchy": [
                "repeat",
                "plate-level condition",
                "biological replicate",
                "(cell_line_id, drug, dose_uM)",
                "equal dose average",
                "(cell_line_id, drug) edge",
            ],
            "plate6_plate14_rule": "only dose_uM == 5.0 groups with both plate6 and plate14 are paired",
            "weighting": "equal within every hierarchy level; no cell-count weighting",
            "implementation_verified_before_reuse": True,
        },
        "comparison_formulas": {
            "delta_ed": "reference_energy - candidate_energy; positive means candidate is better",
            "energy_gain": "(reference_energy - candidate_energy) / (reference_energy + eps)",
            "eps": float(EPS),
            "ratio_of_means": "(reference_mean - candidate_mean) / reference_mean",
            "ratio_of_means_is_distinct_from_mean_per_edge_energy_gain": True,
        },
        "five_model_descriptive_statistics": descriptive,
        "ranking_by_mean_edge_energy": summary[["rank", "model", "mean_energy"]].to_dict(orient="records"),
        "ratio_of_means_improvements": ratios,
        "paired_edge_comparisons": pairwise_rows,
        "edge_winners": winners,
        "stratified_summary": {
            "by_cell_line": cell_line_summary,
            "by_dose": dose_summary,
            "st_a_vs_b2_top_10_edges": top_st_a_vs_b2,
            "st_a_vs_b2_bottom_10_edges": bottom_st_a_vs_b2,
        },
        "scientific_conclusions": [
            "STATE-style ST predicts GeneJEPA latent treated populations more accurately on held-out unseen cell-line x drug combinations.",
            "ST-A has the lowest mean edge-level Energy among the five evaluated methods.",
            "Frozen GeneJEPA latent contains information usable by downstream perturbation predictors and combination generalization.",
        ],
        "scientific_caveats": [
            "This is unseen cell-line x drug combination generalization, not unseen-drug or unseen-cell-line generalization.",
            "ST versus B2-v2 differs in both architecture and objective: set-level Energy versus centroid-shift MSE; the gain cannot be attributed solely to architecture.",
            "GeneJEPA is the frozen cell-state encoder; ST, B1, and B2 are the perturbation predictors, so GeneJEPA itself is not claimed to predict drug perturbation.",
        ],
        "engineering_audit": {
            "all_models_edges_1717": True,
            "edge_ids_cell_line_drug_exact_match": True,
            "all_energy_finite": True,
            "st_a_st_r_raw_row_keys_exact_match": True,
            "high_dose_only_replicate_rule_verified": True,
            "baseline_reference_max_absolute_differences": baseline_reference_differences,
            "frozen_sources_unchanged": True,
            "training_run": False,
            "inference_run": False,
            "b0_b1_b2_prediction_or_raw_energy_recomputed": False,
            "test_time_fitting": False,
            "evaluation_protocol_unchanged_by_this_run": source_hashes_before[EVALUATION_PROTOCOL] == source_hashes_after[EVALUATION_PROTOCOL],
            "dataset_unchanged_by_this_run": source_hashes_before[DATASET_SCRIPT] == source_hashes_after[DATASET_SCRIPT],
            "historical_dataset_sha_difference_documented_in_st_raw_audit": True,
        },
        "outputs": outputs,
        "warnings": source_results["st_raw"].get("warnings", []),
        "blockers": [],
    }
    write_json(FINAL_RESULT, result)

    # Minimal runnable self-check: written tables must retain the frozen cardinality.
    for path in (ST_A_EDGE, ST_R_EDGE, FINAL_EDGE):
        written = pd.read_csv(path, encoding="utf-8-sig")
        if len(written) != EXPECTED_EDGES or written.duplicated(KEYS).any():
            raise AssertionError(f"Written edge artifact failed round-trip validation: {relative(path)}")
    if any(not path.is_file() or path.stat().st_size == 0 for path in artifact_paths):
        raise AssertionError("An output artifact is missing or empty")

    result_sha = sha256_file(FINAL_RESULT)
    print(
        json.dumps(
            {
                "status": "pass",
                "ranking": summary[["rank", "model", "mean_energy"]].to_dict(orient="records"),
                "st_a_beats_st_r": pairwise["ST-A vs ST-R"]["candidate_beats_reference_count"],
                "st_a_beats_b2": pairwise["ST-A vs B2-v2"]["candidate_beats_reference_count"],
                "st_r_beats_b2": pairwise["ST-R vs B2-v2"]["candidate_beats_reference_count"],
                "result_json": relative(FINAL_RESULT),
                "result_json_sha256": result_sha,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
