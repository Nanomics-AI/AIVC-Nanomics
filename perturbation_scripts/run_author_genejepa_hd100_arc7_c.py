#!/usr/bin/env python3
"""Build and evaluate Author-GeneJEPA HD100 Decoder-only C on frozen ARC7 cells."""

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

import numpy as np
import pandas as pd
import torch

import author_genejepa_epoch49_hd100_cache as author_cache
import run_author_genejepa_hd100 as trainer
import run_genejepa_decoder_hd100_arc7 as ours_arc7
import run_genejepa_decoder_v1 as engine
import run_genejepa_decoder_v1_arc7_100 as arc7
import run_genejepa_decoder_v1_arc7_abc as old_abc
from tahoe_decoder_v1_data import PROJECT_ROOT, RESULTS, display_path, sha256_file, utc_now


SCRIPT_PATH = Path(__file__).resolve()
TASK_PATH = PROJECT_ROOT.parent / "当前任务.txt"
PANEL_PATH = trainer.PANEL_PATH
REAL_PATH = ours_arc7.REAL_PATH
REFERENCE_RESULT_PATH = ours_arc7.RESULT_PATH

PREFLIGHT_PATH = RESULTS / "author_genejepa_hd100_arc7_c_preflight.json"
BUILD_RESULT_PATH = RESULTS / "author_genejepa_hd100_arc7_c_build_result.json"
PRED_PATH = RESULTS / "author_genejepa_hd100_arc7_c_pred.h5ad"
RESULT_PATH = RESULTS / "author_genejepa_hd100_arc7_c_result.json"
PER_CONDITION_PATH = RESULTS / "author_genejepa_hd100_arc7_c_per_condition.csv"
COMPARISON_PATH = RESULTS / "author_genejepa_hd100_vs_ours_comparison.csv"
REPORT_PATH = RESULTS / "author_genejepa_hd100_vs_ours_comparison.md"
CELL_EVAL_OUTDIR = RESULTS / "author_genejepa_hd100_arc7_c_cell_eval"

EXPECTED_SELECTION_FINGERPRINT = ours_arc7.EXPECTED_SELECTION_FINGERPRINT
METRICS = (
    ("DES", "DES", "higher"),
    ("PDS", "PDS", "higher"),
    ("MAE", "MAE", "lower"),
    ("Pearson delta", "Pearson delta", "higher"),
    ("Spearman logFC", "Spearman logFC", "higher"),
    ("AUPRC", "AUPRC", "higher"),
    ("Spearman effect size", "Spearman effect size", "higher"),
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact(path: Path) -> dict[str, Any]:
    return engine.artifact_record(path)


def frozen_arc7() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    dataset, selected, context, control, treated, _, frozen = ours_arc7.load_old_contract()
    del dataset
    if (
        frozen["selection_fingerprint"] != EXPECTED_SELECTION_FINGERPRINT
        or len(selected) != 23
        or len(control) != 256
        or any(len(values) != 256 for values in treated.values())
    ):
        raise AssertionError("Frozen 23-condition ARC7 C protocol changed")
    summary = read_json(author_cache.PLAN_SUMMARY)
    if (
        summary.get("status") != "pass"
        or summary["selection"]["arc7_selection_fingerprint"]
        != EXPECTED_SELECTION_FINGERPRINT
        or summary["selection"]["arc7_sampling"] != read_json(arc7.PLAN_PATH)["sampling"]
    ):
        raise AssertionError("Author cache ARC7 indices/order differ from the frozen protocol")
    return selected, context, frozen


def validate_real_reference(selected: pd.DataFrame) -> dict[str, Any]:
    required = [REAL_PATH, ours_arc7.BUILD_RESULT_PATH, REFERENCE_RESULT_PATH]
    missing = [display_path(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Existing Our-HD100-C reference is incomplete: {missing}")
    build = read_json(ours_arc7.BUILD_RESULT_PATH)
    result = read_json(REFERENCE_RESULT_PATH)
    if (
        build.get("status") != "pass"
        or result.get("status") != "pass"
        or build.get("selection_fingerprint") != EXPECTED_SELECTION_FINGERPRINT
        or result.get("selection_fingerprint") != EXPECTED_SELECTION_FINGERPRINT
        or int(result.get("conditions", -1)) != 23
        or int(result.get("genes", -1)) != 100
    ):
        raise AssertionError("Existing Our-HD100-C reference is not PASS/current")
    old_abc.assert_artifact(REAL_PATH, build["outputs"]["real_h5ad"], "frozen real H5AD")
    old_abc.assert_artifact(
        ours_arc7.HD_PRED_PATHS["C"],
        result["groups"]["HD100_C"]["pred_h5ad"],
        "Our-HD100-C prediction",
    )
    ours_arc7.verify_pair(REAL_PATH, ours_arc7.HD_PRED_PATHS["C"], selected)
    return result


def load_author_decoder(device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    trainer.configure_engine()
    trainer.load_runtime_contract()
    checkpoint_path = trainer.FORMAL_CHECKPOINT_DIR / "best.pt"
    required = [trainer.FORMAL_RESULT_PATH, trainer.TRAINING_CONFIG_PATH, checkpoint_path]
    missing = [display_path(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Author-HD100 formal training is incomplete: {missing}")
    result = read_json(trainer.FORMAL_RESULT_PATH)
    checkpoint_sha = sha256_file(checkpoint_path)
    declared = result.get("checkpoints", {}).get("best", {})
    if (
        result.get("status") != "pass"
        or result.get("schema") != "author_genejepa_hd100_training_result_v1"
        or declared.get("path") != display_path(checkpoint_path)
        or declared.get("sha256") != checkpoint_sha
        or int(result.get("best_epoch", -1)) < 0
    ):
        raise AssertionError("Author-HD100 training result/checkpoint provenance is invalid")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    trainer.validate_checkpoint(
        payload,
        mode="formal",
        kind="best",
        config_sha=sha256_file(trainer.TRAINING_CONFIG_PATH),
        batch_size=engine.PRIMARY_BATCH_SIZE,
        gradient_accumulation=engine.PRIMARY_GRADIENT_ACCUMULATION,
    )
    if (
        int(payload.get("epoch", -1)) != int(result["best_epoch"])
        or int(payload.get("best_epoch", -1)) != int(result["best_epoch"])
        or not math.isclose(
            float(payload["best_val_mse"]),
            float(result["best_val_mse"]),
            rel_tol=0,
            abs_tol=1e-15,
        )
    ):
        raise AssertionError("Author-HD100 best checkpoint disagrees with training result")
    model = trainer.build_decoder()
    model.load_state_dict(payload["model_state"], strict=True)
    fingerprint = engine.model_fingerprint(model)
    if fingerprint != payload["model_fingerprint"]:
        raise AssertionError("Author-HD100 model state did not restore exactly")
    model.to(device).eval()
    if model.training or any(module.training for module in model.modules()):
        raise AssertionError("Author-HD100 eval mode did not propagate")
    record = {
        **artifact(checkpoint_path),
        "checkpoint_kind": "best",
        "epoch": int(payload["epoch"]),
        "best_epoch": int(payload["best_epoch"]),
        "best_val_mse": float(payload["best_val_mse"]),
        "model_fingerprint": fingerprint,
        "training_result": artifact(trainer.FORMAL_RESULT_PATH),
        "training_config": artifact(trainer.TRAINING_CONFIG_PATH),
        "architecture": [768, 1024, 1024, 512, 100],
        "final_activation": "Softplus(beta=1, threshold=20)",
    }
    del payload
    return model, record


def author_arc7_latents() -> np.ndarray:
    manifest = trainer.validate_cache()["manifest"]
    sections = manifest["sections"]
    control = sections["arc7_control"]
    treated = sections["arc7_treated"]
    if (
        int(control["cells"]) != 256
        or int(treated["cells"]) != 23 * 256
        or int(control["stop_exclusive"]) != int(treated["start"])
    ):
        raise AssertionError("Author cache ARC7 section layout changed")
    embeddings = np.load(trainer.CACHE_EMBEDDINGS, mmap_mode="r")
    start = int(control["start"])
    stop = int(treated["stop_exclusive"])
    output = np.ascontiguousarray(embeddings[start:stop], dtype=np.float32)
    if (
        output.shape != (6144, 768)
        or not np.isfinite(output).all()
        or not float(output.min()) < 0 < float(output.max())
    ):
        raise AssertionError("Author ARC7 latent matrix is invalid")
    return output


def decode(model: torch.nn.Module, latents: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    if batch_size < 1 or latents.shape != (6144, 768):
        raise ValueError("Invalid Author-HD100 decoder input")
    output = np.empty((6144, 100), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(latents), batch_size):
            stop = min(start + batch_size, len(latents))
            batch = torch.from_numpy(latents[start:stop]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = model(batch).float()
            if not torch.isfinite(prediction).all() or torch.any(prediction < 0):
                raise AssertionError("Author-HD100 Decoder returned invalid expression")
            output[start:stop] = prediction.cpu().numpy()
    return output


def command_preflight() -> dict[str, Any]:
    panel = ours_arc7.panel_contract()
    selected, context, frozen = frozen_arc7()
    reference = validate_real_reference(selected)
    model, checkpoint = load_author_decoder(torch.device("cpu"))
    del model
    cell_eval = arc7.cell_eval_provenance()
    result = {
        "schema": "author_genejepa_hd100_arc7_c_preflight_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "scope": "static/frozen-artifact audit only; no formal inference, DE, or metrics",
        "panel": {**artifact(PANEL_PATH), "rows": len(panel)},
        "frozen_arc7": {
            "conditions": len(selected),
            "cell_line_id": context["cell_line_id"],
            "shared_control_cells": 256,
            "treated_cells_per_condition": 256,
            "selection_fingerprint": frozen["selection_fingerprint"],
            "same_physical_cells_order_and_sampling": True,
        },
        "author_cache": artifact(trainer.CACHE_MANIFEST),
        "author_decoder_checkpoint": checkpoint,
        "our_hd100_c_reference": {
            "result": artifact(REFERENCE_RESULT_PATH),
            "status": reference["groups"]["HD100_C"]["status"],
        },
        "cell_eval": cell_eval,
        "variant": "C only",
        "ST_A_used": False,
        "variant_A_or_B_constructed": False,
        "formal_inference_started": False,
        "formal_DE_or_seven_metric_evaluation_started": False,
        "provenance": {"script": artifact(SCRIPT_PATH), "task": artifact(TASK_PATH)},
    }
    data_path = PREFLIGHT_PATH
    from tahoe_decoder_v1_data import atomic_write_json

    atomic_write_json(data_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def command_build(batch_size: int) -> dict[str, Any]:
    if PRED_PATH.exists() or BUILD_RESULT_PATH.exists():
        raise FileExistsError("Refusing to overwrite Author-HD100-C formal build outputs")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or torch.cuda.device_count() != 1:
        raise RuntimeError("Author-HD100-C build requires CUDA_VISIBLE_DEVICES=0")
    panel = ours_arc7.panel_contract()
    selected, context, frozen = frozen_arc7()
    validate_real_reference(selected)
    latents = author_arc7_latents()
    device = torch.device("cuda:0")
    model, checkpoint = load_author_decoder(device)
    fingerprint_before = engine.model_fingerprint(model)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    prediction = decode(model, latents, device, batch_size)
    torch.cuda.synchronize(device)
    if engine.model_fingerprint(model) != fingerprint_before:
        raise AssertionError("Author-HD100 parameters changed during inference")

    obs = ours_arc7.source_obs(ours_arc7.HD_PRED_PATHS["C"])
    with tempfile.TemporaryDirectory(prefix="author_hd100_c_", dir=RESULTS) as temporary:
        staged = Path(temporary) / PRED_PATH.name
        ours_arc7.write_h5ad(staged, prediction, obs, panel)
        audit = ours_arc7.verify_pair(REAL_PATH, staged, selected)
        os.replace(staged, PRED_PATH)
    result = {
        "schema": "author_genejepa_hd100_arc7_c_build_result_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "variant": "C",
        "definition": {
            "predicted_control": "Author-HD100 Decoder(true Author control latent)",
            "predicted_treated": "Author-HD100 Decoder(true Author treated latent)",
            "predicted_delta": "decoded true treated - decoded true control",
            "ST_A_used": False,
        },
        "conditions": 23,
        "cells_per_set": 256,
        "cells": 6144,
        "genes": 100,
        "expression_space": "log1p(CP10000)",
        "context": context,
        "selection_fingerprint": frozen["selection_fingerprint"],
        "same_conditions_cells_order_and_sampling_as_existing_ARC7": True,
        "inference": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "batch_size": batch_size,
            "autocast": "bfloat16",
            "input_shape": list(latents.shape),
            "input_dtype": str(latents.dtype),
            "input_signed": True,
            "output_shape": list(prediction.shape),
            "output_dtype": str(prediction.dtype),
            "output_finite_nonnegative": True,
            "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "decoder_checkpoint": checkpoint,
        "anndata_audit": audit,
        "inputs": {
            "author_cache": artifact(trainer.CACHE_MANIFEST),
            "panel": artifact(PANEL_PATH),
            "real_h5ad": artifact(REAL_PATH),
            "frozen_plan": artifact(arc7.PLAN_PATH),
            "frozen_conditions": artifact(arc7.CONDITIONS_PATH),
        },
        "outputs": {"pred_h5ad": artifact(PRED_PATH)},
        "formal_DE_or_seven_metric_evaluation_started": False,
        "provenance": {"script": artifact(SCRIPT_PATH), "task": artifact(TASK_PATH)},
    }
    from tahoe_decoder_v1_data import atomic_write_json

    atomic_write_json(BUILD_RESULT_PATH, result)
    del model, latents, prediction
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def metric_map(rows: list[dict[str, Any]]) -> dict[str, float]:
    output = {str(row["metric"]): float(row["mean"]) for row in rows}
    expected = {label for _, label, _ in METRICS}
    if set(output) != expected or not all(math.isfinite(value) for value in output.values()):
        raise AssertionError("Seven-metric result is incomplete or non-finite")
    return output


def write_comparison(
    ours: dict[str, float],
    author: dict[str, float],
    ours_deg: dict[str, Any],
    author_deg: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    rows: list[dict[str, Any]] = []
    improved: list[str] = []
    degraded: list[str] = []
    for display, key, direction in METRICS:
        difference = author[key] - ours[key]
        rows.append(
            {
                "Metric": display + (" ↓" if direction == "lower" else " ↑"),
                "Our Epoch25 half-size + HD100": ours[key],
                "Author epoch49 full-size + HD100": author[key],
                "Author - Ours": difference,
            }
        )
        better = difference < 0 if direction == "lower" else difference > 0
        worse = difference > 0 if direction == "lower" else difference < 0
        if better:
            improved.append(display)
        elif worse:
            degraded.append(display)
    for label, field in (("Median pred DEG %", "fraction_median"), ("Mean pred DEG %", "fraction_mean")):
        ours_value = 100.0 * float(ours_deg[field])
        author_value = 100.0 * float(author_deg[field])
        rows.append(
            {
                "Metric": label,
                "Our Epoch25 half-size + HD100": ours_value,
                "Author epoch49 full-size + HD100": author_value,
                "Author - Ours": author_value - ours_value,
            }
        )
    frame = pd.DataFrame.from_records(rows)
    arc7.atomic_write_csv(COMPARISON_PATH, frame)
    lines = [
        "# Author GeneJEPA epoch49 + HD100 versus our Epoch25 + HD100",
        "",
        old_abc.markdown_table(frame),
        "",
        "The comparison reuses the same 23 conditions, physical cells and order, Top100 panel, "
        "Decoder architecture, target definition, and Cell-Eval implementation.",
        "",
        "This is a backbone-level comparison, not a checkpoint-weight-only ablation: our model is "
        "12 blocks / 6 heads / Epoch25, while the author model is 24 blocks / 12 heads / final Epoch49.",
        "",
        "Predicted-DEG percentages are descriptive and are not assigned an improve/degrade direction.",
        "",
    ]
    temporary = REPORT_PATH.with_name(REPORT_PATH.name + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    os.replace(temporary, REPORT_PATH)
    return frame, improved, degraded


def command_evaluate(threads: int | None) -> dict[str, Any]:
    required = [BUILD_RESULT_PATH, PRED_PATH, REAL_PATH]
    missing = [display_path(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Run Author-HD100-C build first: {missing}")
    outputs = [RESULT_PATH, PER_CONDITION_PATH, COMPARISON_PATH, REPORT_PATH]
    existing = [display_path(path) for path in outputs if path.exists()]
    if existing or (CELL_EVAL_OUTDIR.exists() and any(CELL_EVAL_OUTDIR.iterdir())):
        raise FileExistsError(f"Refusing to overwrite formal evaluation outputs: {existing}")
    build = read_json(BUILD_RESULT_PATH)
    if (
        build.get("status") != "pass"
        or build.get("selection_fingerprint") != EXPECTED_SELECTION_FINGERPRINT
    ):
        raise AssertionError("Author-HD100-C build result is not PASS/current")
    old_abc.assert_artifact(PRED_PATH, build["outputs"]["pred_h5ad"], "Author-HD100-C H5AD")
    selected, _, frozen = frozen_arc7()
    reference = validate_real_reference(selected)
    ours_arc7.verify_pair(REAL_PATH, PRED_PATH, selected)
    threads = arc7.available_threads() if threads is None else min(threads, arc7.available_threads())
    if threads < 1:
        raise ValueError("threads must be positive")

    started = time.perf_counter()
    summary, per_condition, official, runtime = arc7.run_cell_eval(
        REAL_PATH, PRED_PATH, selected, CELL_EVAL_OUTDIR, threads
    )
    arc7.atomic_write_csv(CELL_EVAL_OUTDIR / "requested_seven_metrics.csv", official)
    per_condition = per_condition.copy()
    per_condition.insert(0, "group", "AUTHOR_HD100_C")
    per_condition.insert(1, "model", "Author epoch49 + HD100")
    per_condition.insert(2, "variant", "C")
    per_condition.insert(3, "genes", 100)
    per_condition["true_DEG_fraction"] = per_condition["true_DEG_count"] / 100.0
    per_condition["pred_DEG_fraction"] = per_condition["pred_DEG_count"] / 100.0
    if per_condition["pair_id"].astype(str).tolist() != selected["pair_id"].astype(str).tolist():
        raise AssertionError("Author per-condition ARC7 order changed")
    arc7.atomic_write_csv(PER_CONDITION_PATH, per_condition)

    author_rows = arc7.summary_for_json(summary)
    author_metrics = metric_map(author_rows)
    ours_group = reference["groups"]["HD100_C"]
    ours_metrics = metric_map(ours_group["seven_metrics"])
    true_deg = ours_arc7.count_stats(per_condition["true_DEG_count"], 100)
    predicted_deg = ours_arc7.count_stats(per_condition["pred_DEG_count"], 100)
    if true_deg != ours_group["true_DEG"]:
        raise AssertionError("True DEG statistics differ from the frozen Our-HD100-C reference")
    comparison, improved, degraded = write_comparison(
        ours_metrics, author_metrics, ours_group["predicted_DEG"], predicted_deg
    )
    failures = runtime["metric_failures"]
    result = {
        "schema": "author_genejepa_hd100_arc7_c_result_v1",
        "created_at_utc": utc_now(),
        "status": "pass" if not failures else "fail",
        "experiment": "Author GeneJEPA epoch49 EMA teacher -> HD100 -> Decoder-only C",
        "conditions": 23,
        "cells_per_set": 256,
        "cells": 6144,
        "genes": 100,
        "selection_fingerprint": frozen["selection_fingerprint"],
        "same_conditions_cells_order_sampling_panel_and_metrics_as_ours": True,
        "variant": "C only",
        "ST_A_used": False,
        "seven_metrics": author_rows,
        "true_DEG": true_deg,
        "predicted_DEG": predicted_deg,
        "comparison": {
            "reference": artifact(REFERENCE_RESULT_PATH),
            "our_epoch25_hd100_c": ours_metrics,
            "author_epoch49_hd100_c": author_metrics,
            "improved_metrics": improved,
            "degraded_metrics": degraded,
            "unchanged_metric_count": len(METRICS) - len(improved) - len(degraded),
            "no_composite_score": True,
        },
        "cell_eval": {
            "version": arc7.EXPECTED_CELL_EVAL_VERSION,
            "commit": arc7.EXPECTED_CELL_EVAL_COMMIT,
            "FDR": 0.05,
            "epsilon": 0.0,
            "threads": threads,
        },
        "metric_failures": failures,
        "runtime": runtime,
        "outputs": {
            "pred_h5ad": artifact(PRED_PATH),
            "per_condition": artifact(PER_CONDITION_PATH),
            "comparison": artifact(COMPARISON_PATH),
            "report": artifact(REPORT_PATH),
            "build_result": artifact(BUILD_RESULT_PATH),
        },
        "provenance": {
            "author_cache": artifact(trainer.CACHE_MANIFEST),
            "author_training_result": artifact(trainer.FORMAL_RESULT_PATH),
            "panel": artifact(PANEL_PATH),
            "script": artifact(SCRIPT_PATH),
            "task": artifact(TASK_PATH),
            "cell_eval": arc7.cell_eval_provenance(),
        },
        "interpretation_guardrail": (
            "Backbone-level comparison only: half-size Epoch25 versus full-size final Epoch49; "
            "not a checkpoint-weight-only ablation."
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "comparison_rows": len(comparison),
    }
    from tahoe_decoder_v1_data import atomic_write_json

    atomic_write_json(RESULT_PATH, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="Audit frozen inputs; no inference, DE, or metrics")
    build = commands.add_parser("build", help="Run formal Decoder-only C inference")
    build.add_argument("--decoder-batch-size", type=int, default=2048)
    evaluate = commands.add_parser("evaluate", help="Run formal DE and seven metrics")
    evaluate.add_argument("--threads", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        command_preflight()
    elif args.command == "build":
        command_build(args.decoder_batch_size)
    elif args.command == "evaluate":
        command_evaluate(args.threads)


if __name__ == "__main__":
    main()
