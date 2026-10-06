#!/usr/bin/env python3
"""Same-cell Phase-III evaluation for Old Decoder, D1, D2, and full pipeline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
SCRIPT_PATH = Path(__file__).resolve()
PROTOCOL_PATH = RESULTS / "phase3_set_decoder_training_protocol.json"
SELECTION_PATH = RESULTS / "phase3_decoder_selection.json"
EXPRESSION_CACHE = RESULTS / "phase3_top20_expression_cache.npy"
EXPRESSION_MANIFEST = RESULTS / "phase3_top20_expression_cache_manifest.json"
PANEL_PATH = RESULTS / "phase1_top20_gene_panel.csv"
PANEL_MANIFEST = RESULTS / "phase1_top20_gene_panel.json"
OLD_DECODER_CHECKPOINT = RESULTS / "phase1_author_top20_decoder/best.pt"
OLD_DECODER_TRAINING_RESULT = RESULTS / "phase1_author_top20_decoder/training_result.json"
ST_CHECKPOINT = RESULTS / "phase2_stav2_checkpoints/best.pt"
ST_TRAINING_RESULT = RESULTS / "phase2_stav2_training_result.json"

DECODER_ONLY_CONDITIONS = RESULTS / "phase3_decoder_only_condition_metrics.csv"
DECODER_ONLY_REPEATS = RESULTS / "phase3_decoder_only_repeat_metrics.csv"
DECODER_ONLY_RESULT = RESULTS / "phase3_decoder_only_evaluation.json"
FULL_CONDITIONS = RESULTS / "phase3_full_pipeline_condition_metrics.csv"
FULL_REPEATS = RESULTS / "phase3_full_pipeline_repeat_metrics.csv"
FULL_RESULT = RESULTS / "phase3_full_pipeline_evaluation.json"
COMPARISON = RESULTS / "phase3_comparison.json"
HANDOFF = RESULTS / "phase3_handoff.md"
DE_STATE = RESULTS / "phase3_true_deg_builder_state.npz"
DE_PROGRESS = RESULTS / "phase3_true_deg_builder_progress.json"

REPEAT_EPOCHS = (0, 1, 2, 3, 4)
SEED = 42
SET_SIZE = 256
TEST_CONDITIONS = 500
CONDITION_REPEAT_ROWS = 2500
LATENT_DIM = 768
GENE_DIM = 20
CONTROL_LABEL = "__CONTROL__"
TREATED_LABEL = "__TREATED__"

_DE_EXPRESSION: np.ndarray | None = None
_DE_GENES: tuple[str, ...] = ()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        display = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        display = str(path.resolve())
    return {"path": display, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def variant_paths(variant: str) -> dict[str, Path]:
    root = RESULTS / f"phase3_{variant}_checkpoints"
    return {
        "best": root / "best.pt",
        "last": root / "last.pt",
        "result": RESULTS / f"phase3_{variant}_training_result.json",
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
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


def _panel() -> tuple[pd.DataFrame, tuple[str, ...]]:
    manifest = json.loads(PANEL_MANIFEST.read_text(encoding="utf-8"))
    panel = pd.read_csv(PANEL_PATH, encoding="utf-8-sig", keep_default_na=False).sort_values(
        "panel_rank", kind="stable"
    )
    if (
        manifest.get("status") != "frozen"
        or manifest["output"]["sha256"] != sha256_file(PANEL_PATH)
        or len(panel) != GENE_DIM
        or not np.array_equal(panel["panel_rank"].to_numpy(np.int64), np.arange(GENE_DIM))
    ):
        raise AssertionError("Frozen Top20 panel changed")
    genes = tuple(panel["ensembl_id"].astype(str))
    if len(set(genes)) != GENE_DIM:
        raise AssertionError("Top20 Ensembl IDs are not unique")
    return panel, genes


def _plan_sha256(
    condition_ids: list[str], repeats: np.ndarray, control: np.ndarray, treated: np.ndarray
) -> str:
    digest = hashlib.sha256()
    for condition_id, repeat, left, right in zip(
        condition_ids, repeats, control, treated, strict=True
    ):
        digest.update(condition_id.encode("utf-8"))
        digest.update(np.asarray([repeat], dtype="<i8").tobytes())
        digest.update(left.astype("<i8", copy=False).tobytes())
        digest.update(right.astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def build_evaluation_plan() -> dict[str, Any]:
    from phase3_set_decoder_data import Phase3SetDecoderDataset

    dataset = Phase3SetDecoderDataset(split="test", seed=SEED, epoch=0)
    if len(dataset) != TEST_CONDITIONS:
        raise AssertionError("Frozen Phase-II test split is no longer 500 conditions")
    control = np.empty((CONDITION_REPEAT_ROWS, SET_SIZE), dtype=np.int64)
    treated = np.empty_like(control)
    true_delta = np.empty((CONDITION_REPEAT_ROWS, GENE_DIM), dtype=np.float32)
    condition_ids: list[str] = []
    repeats = np.empty(CONDITION_REPEAT_ROWS, dtype=np.int64)
    metadata_rows: list[dict[str, Any]] = []
    output_row = 0
    for repeat in REPEAT_EPOCHS:
        dataset.set_epoch(repeat)
        for _, row in dataset.conditions.iterrows():
            pair_id = str(row["pair_id"])
            left = np.sort(
                dataset._sample_range(
                    int(row["control_embedding_start"]),
                    int(row["control_embedding_stop_exclusive"]),
                    pair_id,
                    "control",
                ),
                kind="stable",
            )
            right = np.sort(
                dataset._sample_range(
                    int(row["treated_embedding_start"]),
                    int(row["treated_embedding_stop_exclusive"]),
                    pair_id,
                    "treated",
                ),
                kind="stable",
            )
            if (
                len(np.unique(left)) != SET_SIZE
                or len(np.unique(right)) != SET_SIZE
                or np.any(left[1:] < left[:-1])
                or np.any(right[1:] < right[:-1])
            ):
                raise AssertionError("Evaluation membership/order contract failed")
            control[output_row] = left
            treated[output_row] = right
            true_delta[output_row] = (
                np.asarray(dataset.expression[right], dtype=np.float32).mean(axis=0)
                - np.asarray(dataset.expression[left], dtype=np.float32).mean(axis=0)
            )
            condition_ids.append(pair_id)
            repeats[output_row] = repeat
            metadata_rows.append(
                {
                    "row_index": output_row,
                    "condition_id": pair_id,
                    "edge_id": str(row["edge_id"]),
                    "cell_line_id": str(row["cell_line_id"]),
                    "drug": str(row["drug"]),
                    "dose_uM": float(row["dose_uM"]),
                    "plate": str(row["plate"]),
                    "control_pool_id": str(row["control_pool_id"]),
                    "repeat_epoch": repeat,
                }
            )
            output_row += 1
    if output_row != CONDITION_REPEAT_ROWS or not np.isfinite(true_delta).all():
        raise AssertionError("Phase-III evaluation plan is incomplete or non-finite")
    digest = _plan_sha256(condition_ids, repeats, control, treated)
    return {
        "dataset": dataset,
        "metadata": pd.DataFrame.from_records(metadata_rows),
        "condition_ids": condition_ids,
        "repeat_epochs": repeats,
        "control_indices": control,
        "treated_indices": treated,
        "true_delta": true_delta,
        "membership_sha256": digest,
    }


def _de_initializer(cache_path: str, genes: tuple[str, ...]) -> None:
    global _DE_EXPRESSION, _DE_GENES
    _DE_EXPRESSION = np.load(cache_path, mmap_mode="r")
    _DE_GENES = genes


def _de_worker(task: tuple[int, np.ndarray, np.ndarray]) -> tuple[int, np.ndarray]:
    import contextlib
    import warnings

    import anndata as ad
    import polars as pl
    from pdex import pdex

    row_index, control_indices, treated_indices = task
    if _DE_EXPRESSION is None or len(_DE_GENES) != GENE_DIM:
        raise RuntimeError("DE worker was not initialized")
    matrix = np.concatenate(
        (
            np.asarray(_DE_EXPRESSION[control_indices], dtype=np.float32),
            np.asarray(_DE_EXPRESSION[treated_indices], dtype=np.float32),
        )
    )
    obs = pd.DataFrame(
        {"perturbation": [CONTROL_LABEL] * SET_SIZE + [TREATED_LABEL] * SET_SIZE}
    )
    var = pd.DataFrame(index=pd.Index(_DE_GENES, name="ensembl_id"))
    with warnings.catch_warnings(), open(os.devnull, "w", encoding="utf-8") as sink:
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            adata = ad.AnnData(X=matrix, obs=obs, var=var)
            frame = pdex(
                adata=adata,
                mode="ref",
                reference=CONTROL_LABEL,
                groupby="perturbation",
                threads=1,
                is_log1p=True,
                epsilon=0.0,
            )
    target_frame = frame.filter(pl.col("target") == TREATED_LABEL)
    target_features = target_frame["feature"].cast(pl.Utf8).to_list()
    if target_frame.height != GENE_DIM or set(target_features) != set(_DE_GENES):
        raise AssertionError("pdex did not return exactly the frozen Top20 treated-gene universe")
    significant = set(
        target_frame.filter(pl.col("fdr") < 0.05)["feature"].cast(pl.Utf8).to_list()
    )
    labels = np.asarray([gene in significant for gene in _DE_GENES], dtype=np.int8)
    return row_index, labels


def _de_input_fingerprint(plan_sha256: str) -> str:
    expression = json.loads(EXPRESSION_MANIFEST.read_text(encoding="utf-8"))
    panel = json.loads(PANEL_MANIFEST.read_text(encoding="utf-8"))
    payload = {
        "membership_sha256": plan_sha256,
        "expression_cache_sha256": expression["output"]["sha256"],
        "panel_sha256": panel["output"]["sha256"],
        "pdex_version": importlib.metadata.version("pdex"),
        "fdr_threshold": 0.05,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_de_state(
    labels: np.ndarray,
    completed: np.ndarray,
    plan_sha256: str,
    input_fingerprint: str,
) -> None:
    temporary = DE_STATE.with_name(DE_STATE.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            labels=labels,
            completed=completed,
            plan_sha256=np.asarray(plan_sha256),
            input_fingerprint=np.asarray(input_fingerprint),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, DE_STATE)


def build_true_deg_labels(plan: dict[str, Any], workers: int) -> tuple[np.ndarray, dict[str, Any]]:
    if workers < 1:
        raise ValueError("--de-workers must be positive")
    _, genes = _panel()
    plan_sha = str(plan["membership_sha256"])
    input_fingerprint = _de_input_fingerprint(plan_sha)
    if DE_STATE.exists():
        with np.load(DE_STATE, allow_pickle=False) as state:
            labels = state["labels"].astype(np.int8, copy=True)
            completed = state["completed"].astype(np.bool_, copy=True)
            state_plan_sha = str(state["plan_sha256"])
            state_fingerprint = (
                str(state["input_fingerprint"])
                if "input_fingerprint" in state.files
                else ""
            )
        if state_plan_sha != plan_sha or state_fingerprint != input_fingerprint:
            raise AssertionError("Existing Phase-III DE state belongs to different inputs")
    else:
        labels = np.full((CONDITION_REPEAT_ROWS, GENE_DIM), -1, dtype=np.int8)
        completed = np.zeros(CONDITION_REPEAT_ROWS, dtype=np.bool_)
        _atomic_de_state(labels, completed, plan_sha, input_fingerprint)
    if labels.shape != (CONDITION_REPEAT_ROWS, GENE_DIM) or completed.shape != (
        CONDITION_REPEAT_ROWS,
    ):
        raise AssertionError("Phase-III DE resume state shape changed")
    pending = np.flatnonzero(~completed)
    started = time.perf_counter()
    if len(pending):
        tasks = [
            (
                int(index),
                plan["control_indices"][index],
                plan["treated_indices"][index],
            )
            for index in pending
        ]
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_de_initializer,
            initargs=(str(EXPRESSION_CACHE), genes),
        ) as executor:
            for count, (row_index, row_labels) in enumerate(
                executor.map(_de_worker, tasks, chunksize=1), start=1
            ):
                if completed[row_index] or row_labels.shape != (GENE_DIM,):
                    raise AssertionError("DE worker returned a duplicate or invalid row")
                labels[row_index] = row_labels
                completed[row_index] = True
                if count % 25 == 0 or count == len(tasks):
                    _atomic_de_state(labels, completed, plan_sha, input_fingerprint)
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    new_rate = count / elapsed
                    progress = {
                        "schema": "phase3_true_deg_builder_progress_v1",
                        "updated_at_utc": utc_now(),
                        "status": "running" if not completed.all() else "complete",
                        "completed_rows": int(completed.sum()),
                        "total_rows": CONDITION_REPEAT_ROWS,
                        "remaining_rows": int((~completed).sum()),
                        "session_rows_per_second": new_rate,
                        "estimated_remaining_seconds": int((~completed).sum()) / new_rate,
                        "membership_sha256": plan_sha,
                        "workers": workers,
                    }
                    atomic_json(DE_PROGRESS, progress)
                    print(
                        f"phase3 true DE rows={int(completed.sum()):,}/{CONDITION_REPEAT_ROWS:,}",
                        flush=True,
                    )
    if not completed.all() or np.any((labels != 0) & (labels != 1)):
        raise AssertionError("Phase-III true DEG labels are incomplete or non-binary")
    import cell_eval

    cell_eval_source = Path(cell_eval.__file__).resolve().parent / "metrics/_anndata.py"
    provenance = {
        "state": artifact(DE_STATE),
        "membership_sha256": plan_sha,
        "input_fingerprint": input_fingerprint,
        "rows": CONDITION_REPEAT_ROWS,
        "genes": GENE_DIM,
        "implementation": "pdex.pdex, same real-side implementation used by installed Cell-Eval",
        "mode": "ref",
        "reference": CONTROL_LABEL,
        "groupby": "perturbation",
        "is_log1p": True,
        "epsilon": 0.0,
        "fdr_threshold": 0.05,
        "comparison": "exact 256 treated vs exact matched 256 control physical cells per condition-repeat",
        "pdex_version": importlib.metadata.version("pdex"),
        "cell_eval_version": importlib.metadata.version("cell-eval"),
        "cell_eval_discrimination_source": artifact(cell_eval_source),
        "top20_only": True,
    }
    return labels.astype(bool), provenance


def _load_set_decoder(variant: str, device: Any) -> Any:
    import torch

    from phase3_set_decoder_model import build_set_decoder

    checkpoint = torch.load(variant_paths(variant)["best"], map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema") != "phase3_set_decoder_checkpoint_v1"
        or checkpoint.get("variant") != variant
        or checkpoint.get("kind") != "best"
    ):
        raise AssertionError(f"Invalid Phase-III {variant} best checkpoint")
    model = build_set_decoder(variant, SEED)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval().requires_grad_(False).to(device)


def run_inference(plan: dict[str, Any], batch_size: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch

    from phase2_stav2_model import build_stav2
    from phase3_set_decoder_data import Phase3SetDecoderDataset, make_dataloader
    from run_phase1_top20_decoder import build_decoder

    if batch_size < 1 or not torch.cuda.is_available():
        raise RuntimeError("Phase-III formal inference requires a positive batch and one visible GPU")
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if selection.get("status") != "pass" or selection.get("test_used_for_selection") is not False:
        raise AssertionError("Validation-only architecture selection is not frozen")
    selected = str(selection["selected_architecture"])
    if selected not in {"d1", "d2"}:
        raise AssertionError("Unknown selected architecture")
    device = torch.device("cuda:0")
    d1 = _load_set_decoder("d1", device)
    d2 = _load_set_decoder("d2", device)
    selected_model = d1 if selected == "d1" else d2

    old_checkpoint = torch.load(OLD_DECODER_CHECKPOINT, map_location="cpu", weights_only=False)
    if (
        old_checkpoint.get("schema") != "phase1_top20_decoder_checkpoint_v1"
        or old_checkpoint.get("representation") != "author"
        or old_checkpoint.get("checkpoint_kind") != "best"
    ):
        raise AssertionError("Invalid frozen Author GeneJEPA old-decoder checkpoint")
    old_decoder = build_decoder()
    old_decoder.load_state_dict(old_checkpoint["model_state_dict"], strict=True)
    old_decoder.eval().requires_grad_(False).to(device)

    st_checkpoint = torch.load(ST_CHECKPOINT, map_location="cpu", weights_only=False)
    if st_checkpoint.get("schema") != "phase2_stav2_checkpoint_v1" or st_checkpoint.get("kind") != "best":
        raise AssertionError("Invalid frozen Phase-II ST-A v2 checkpoint")
    st, _ = build_stav2(SEED)
    st.load_state_dict(st_checkpoint["model_state_dict"], strict=True)
    st.eval().requires_grad_(False).to(device)

    predictions = {
        branch: np.empty((CONDITION_REPEAT_ROWS, GENE_DIM), dtype=np.float32)
        for branch in ("old_decoder", "d1", "d2", "full_pipeline")
    }
    written = np.zeros(CONDITION_REPEAT_ROWS, dtype=np.bool_)
    old_negative = {
        "control_coordinates": 0,
        "treated_coordinates": 0,
        "control_min": float("inf"),
        "treated_min": float("inf"),
        "total_coordinates_each_side": 0,
    }
    dataset = Phase3SetDecoderDataset(split="test", seed=SEED, epoch=0)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for repeat in REPEAT_EPOCHS:
            dataset.set_epoch(repeat)
            loader = make_dataloader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                seed=SEED,
                drop_last=False,
                num_workers=2,
                pin_memory=True,
                persistent_workers=False,
                prefetch_factor=2,
            )
            offset = 0
            for raw in loader:
                size = len(raw["condition_id"])
                rows = np.arange(repeat * TEST_CONDITIONS + offset, repeat * TEST_CONDITIONS + offset + size)
                source = raw["source_embedding_index"].numpy().astype(np.int64)
                target = raw["target_embedding_index"].numpy().astype(np.int64)
                if not np.array_equal(source, plan["control_indices"][rows]) or not np.array_equal(
                    target, plan["treated_indices"][rows]
                ):
                    raise AssertionError("Inference membership/order differs from frozen evaluation plan")
                if list(raw["condition_id"]) != [plan["condition_ids"][row] for row in rows]:
                    raise AssertionError("Inference condition order differs from frozen evaluation plan")
                if written[rows].any():
                    raise AssertionError("Inference attempted to write a condition-repeat twice")
                control = raw["ctrl_cell_emb"].to(device, non_blocking=True)
                treated = raw["pert_cell_emb"].to(device, non_blocking=True)
                gt = raw["gt_delta"].numpy()
                if not np.array_equal(gt, plan["true_delta"][rows]):
                    raise AssertionError("Inference GT differs from exact expression-cache GT")
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    raw_control = old_decoder(control.reshape(-1, LATENT_DIM)).reshape(
                        size, SET_SIZE, GENE_DIM
                    )
                    raw_treated = old_decoder(treated.reshape(-1, LATENT_DIM)).reshape(
                        size, SET_SIZE, GENE_DIM
                    )
                    prediction_d1 = d1(control, treated)
                    prediction_d2 = d2(control, treated)
                    synthetic = st(
                        {
                            "ctrl_cell_emb": control,
                            "drug_id": raw["drug_id"].to(device, non_blocking=True),
                            "dose_scaled": raw["dose_scaled"].to(device, non_blocking=True),
                        }
                    ).reshape(size, SET_SIZE, LATENT_DIM)
                    prediction_full = selected_model(control, synthetic)
                raw_control32 = raw_control.float()
                raw_treated32 = raw_treated.float()
                old_negative["control_coordinates"] += int((raw_control32 < 0).sum())
                old_negative["treated_coordinates"] += int((raw_treated32 < 0).sum())
                old_negative["control_min"] = min(old_negative["control_min"], float(raw_control32.min()))
                old_negative["treated_min"] = min(old_negative["treated_min"], float(raw_treated32.min()))
                old_negative["total_coordinates_each_side"] += raw_control32.numel()
                prediction_old = raw_treated32.clamp_min(0).mean(dim=1) - raw_control32.clamp_min(0).mean(dim=1)
                batch_predictions = {
                    "old_decoder": prediction_old,
                    "d1": prediction_d1.float(),
                    "d2": prediction_d2.float(),
                    "full_pipeline": prediction_full.float(),
                }
                for branch, value in batch_predictions.items():
                    if value.shape != (size, GENE_DIM) or not torch.isfinite(value).all():
                        raise AssertionError(f"Invalid {branch} prediction")
                    predictions[branch][rows] = value.cpu().numpy()
                written[rows] = True
                offset += size
            if offset != TEST_CONDITIONS:
                raise AssertionError("Evaluation DataLoader dropped test conditions")
            print(f"phase3 inference repeat={repeat} conditions={offset}", flush=True)
    torch.cuda.synchronize(device)
    if not written.all() or not all(np.isfinite(value).all() for value in predictions.values()):
        raise AssertionError("Formal Phase-III inference is incomplete or non-finite")
    old_negative["control_negative_ratio"] = (
        old_negative["control_coordinates"] / old_negative["total_coordinates_each_side"]
    )
    old_negative["treated_negative_ratio"] = (
        old_negative["treated_coordinates"] / old_negative["total_coordinates_each_side"]
    )
    audit = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "batch_size": batch_size,
        "autocast": "bfloat16",
        "rows": CONDITION_REPEAT_ROWS,
        "all_predictions_finite": True,
        "d1_d2_signed_output_unclamped": True,
        "full_pipeline_signed_output_unclamped": True,
        "old_decoder_expression_projection": "max(raw_decoder_output, 0) before set means",
        "old_decoder_raw_negative_audit": old_negative,
        "selected_architecture": selected,
        "checkpoints": {
            "old_decoder": artifact(OLD_DECODER_CHECKPOINT),
            "d1": artifact(variant_paths("d1")["best"]),
            "d2": artifact(variant_paths("d2")["best"]),
            "st_a_v2": artifact(ST_CHECKPOINT),
        },
        "selection": artifact(SELECTION_PATH),
        "synthetic_treated_order": "inherits canonical sorted control token order; no synthetic index or resort",
        "peak_allocated_GiB": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_GiB": torch.cuda.max_memory_reserved(device) / 2**30,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return predictions, audit


def _pearson(prediction: np.ndarray, truth: np.ndarray) -> float:
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise AssertionError("Correlation received non-finite input")
    if float(np.var(prediction)) == 0.0 or float(np.var(truth)) == 0.0:
        return float("nan")
    return float(np.corrcoef(prediction, truth)[0, 1])


def _spearman(prediction: np.ndarray, truth: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if float(np.var(prediction)) == 0.0 or float(np.var(truth)) == 0.0:
        return float("nan")
    return float(spearmanr(prediction, truth).statistic)


def condition_metrics(
    branch: str,
    prediction: np.ndarray,
    truth: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    from sklearn.metrics import average_precision_score

    records: list[dict[str, Any]] = []
    for index in range(CONDITION_REPEAT_ROWS):
        pred = prediction[index]
        true = truth[index]
        significant = labels[index]
        count = int(significant.sum())
        if count == 0:
            des = float("nan")
        else:
            predicted_top = np.argsort(-np.abs(pred), kind="stable")[:count]
            des = float(significant[predicted_top].sum() / count)
        if count in (0, GENE_DIM):
            auprc = float("nan")
            reason = "single_class_true_labels"
        else:
            auprc = float(average_precision_score(significant.astype(np.int8), np.abs(pred)))
            reason = ""
        row = metadata.iloc[index].to_dict()
        row.update(
            {
                "branch": branch,
                "mae": float(np.mean(np.abs(pred - true))),
                "pearson_delta": _pearson(pred, true),
                "spearman_delta": _spearman(pred, true),
                "set_level_des": des,
                "auprc": auprc,
                "auprc_non_informative_reason": reason,
                "true_DEG_count": count,
                "true_delta_norm": float(np.linalg.norm(true)),
                "pred_delta_norm": float(np.linalg.norm(pred)),
            }
        )
        records.append(row)
    return pd.DataFrame.from_records(records)


def context_pds(
    branch: str,
    prediction: np.ndarray,
    truth: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    condition_scores = np.full(CONDITION_REPEAT_ROWS, np.nan, dtype=np.float64)
    context_records: list[dict[str, Any]] = []
    manual_candidates: list[tuple[int, str, dict[str, Any]]] = []
    context_columns = ["repeat_epoch", "cell_line_id", "plate", "control_pool_id"]
    for key, group in metadata.groupby(context_columns, sort=True):
        if len(group) < 2:
            continue
        group = group.sort_values("condition_id", kind="stable")
        rows = group["row_index"].to_numpy(np.int64)
        predicted = prediction[rows]
        actual = truth[rows]
        distances = np.abs(predicted[:, None, :] - actual[None, :, :]).sum(axis=2)
        scores: list[float] = []
        ranks: list[int] = []
        for local in range(len(rows)):
            order = np.argsort(distances[local])
            rank = int(np.flatnonzero(order == local)[0])
            score = 1.0 - rank / len(rows)
            ranks.append(rank)
            scores.append(score)
            condition_scores[rows[local]] = score
        context_id = "|".join(str(value) for value in key[1:])
        record = {
            "branch": branch,
            "repeat_epoch": int(key[0]),
            "context_id": context_id,
            "cell_line_id": str(key[1]),
            "plate": str(key[2]),
            "control_pool_id": str(key[3]),
            "candidate_count": len(rows),
            "conditions": len(rows),
            "context_score": float(np.mean(scores)),
        }
        context_records.append(record)
        manual_candidates.append(
            (
                len(rows),
                f"{key[0]}|{context_id}",
                {
                    **record,
                    "candidate_condition_ids": group["condition_id"].astype(str).tolist(),
                    "predicted_delta_vectors": predicted.tolist(),
                    "true_delta_vectors": actual.tolist(),
                    "pairwise_l1_distance_matrix_pred_rows_true_columns": distances.tolist(),
                    "own_target_rank_zero_based": ranks,
                    "condition_scores": scores,
                },
            )
        )
    contexts = pd.DataFrame.from_records(context_records)
    if contexts.empty:
        raise AssertionError("No valid multi-perturbation context exists for context-level set PDS")
    manual = sorted(manual_candidates, key=lambda value: (value[0], value[1]))[0][2]
    provenance = {
        "metric_name": "context-level set PDS",
        "metric_type": "set-level adaptation",
        "source_semantics": "Cell-Eval discrimination_score_l1",
        "identical_to_original_predicted_cell_PDS": False,
        "formula": "L1 rank of each predicted delta against every true delta in matched-control context; score=1-rank/K",
        "target_gene_exclusion": False,
        "valid_multi_perturbation_contexts": len(contexts),
        "conditions_represented": int(np.isfinite(condition_scores).sum()),
        "candidate_count_distribution": {
            "min": int(contexts["candidate_count"].min()),
            "median": float(contexts["candidate_count"].median()),
            "max": int(contexts["candidate_count"].max()),
        },
        "manual_real_context_audit": manual,
    }
    return contexts, provenance


def summarize_branch(
    branch: str,
    frame: pd.DataFrame,
    contexts: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    repeat_records: list[dict[str, Any]] = []
    metric_columns = (
        "mae",
        "pearson_delta",
        "spearman_delta",
        "set_level_des",
        "auprc",
    )
    for repeat in REPEAT_EPOCHS:
        subset = frame.loc[frame["repeat_epoch"].eq(repeat)]
        context_subset = contexts.loc[contexts["repeat_epoch"].eq(repeat)]
        if len(subset) != TEST_CONDITIONS:
            raise AssertionError(f"{branch}/repeat{repeat} does not contain 500 conditions")
        record: dict[str, Any] = {
            "branch": branch,
            "repeat_epoch": repeat,
            "conditions": len(subset),
        }
        for column in metric_columns:
            values = pd.to_numeric(subset[column], errors="coerce").to_numpy(np.float64)
            finite = np.isfinite(values)
            record[column] = float(values[finite].mean()) if finite.any() else float("nan")
            record[f"{column}_valid_n"] = int(finite.sum())
            record[f"{column}_nan_n"] = int((~finite).sum())
        pds_values = context_subset["context_score"].to_numpy(np.float64)
        record["context_level_set_pds"] = float(pds_values.mean())
        record["context_level_set_pds_valid_context_n"] = len(context_subset)
        record["context_level_set_pds_conditions_represented"] = int(
            context_subset["conditions"].sum()
        )
        record["context_candidate_count_min"] = int(context_subset["candidate_count"].min())
        record["context_candidate_count_median"] = float(context_subset["candidate_count"].median())
        record["context_candidate_count_max"] = int(context_subset["candidate_count"].max())
        magnitude = _spearman(
            subset["pred_delta_norm"].to_numpy(np.float64),
            subset["true_delta_norm"].to_numpy(np.float64),
        )
        record["delta_magnitude_spearman"] = magnitude
        record["delta_magnitude_spearman_valid_n"] = int(math.isfinite(magnitude))
        record["delta_magnitude_spearman_nan_n"] = int(not math.isfinite(magnitude))
        record["auprc_true_DEG_count_0_n"] = int(subset["true_DEG_count"].eq(0).sum())
        record["auprc_true_DEG_count_20_n"] = int(subset["true_DEG_count"].eq(GENE_DIM).sum())
        repeat_records.append(record)
    repeats = pd.DataFrame.from_records(repeat_records)
    summaries: dict[str, Any] = {}
    for column in (*metric_columns, "context_level_set_pds", "delta_magnitude_spearman"):
        values = repeats[column].to_numpy(np.float64)
        finite = np.isfinite(values)
        summaries[column] = {
            "repeat_values": values.tolist(),
            "mean": float(values[finite].mean()) if finite.any() else float("nan"),
            "std": float(values[finite].std(ddof=1)) if finite.sum() > 1 else 0.0,
            "valid_repeats": int(finite.sum()),
            "nan_repeats": int((~finite).sum()),
            "condition_valid_n_by_repeat": repeats.get(f"{column}_valid_n", pd.Series([None] * 5)).tolist(),
            "condition_nan_n_by_repeat": repeats.get(f"{column}_nan_n", pd.Series([None] * 5)).tolist(),
        }
    return repeats, {
        "branch": branch,
        "condition_repeat_rows": len(frame),
        "test_conditions": int(frame["condition_id"].nunique()),
        "repeat_epochs": list(REPEAT_EPOCHS),
        "metrics": summaries,
        "auprc_noninformative": {
            "true_DEG_count_0_by_repeat": repeats["auprc_true_DEG_count_0_n"].tolist(),
            "true_DEG_count_20_by_repeat": repeats["auprc_true_DEG_count_20_n"].tolist(),
            "policy": "single-class true labels (0 or 20 true DEGs) -> NaN",
        },
    }


def _result_payload(
    scope: str,
    branches: dict[str, dict[str, Any]],
    pds: dict[str, Any],
    plan: dict[str, Any],
    de_provenance: dict[str, Any],
    inference: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": f"phase3_{scope}_evaluation_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "scope": scope,
        "branches": branches,
        "evaluation_contract": {
            "condition_level_delta": True,
            "predicted_cells_fabricated": False,
            "set_size": SET_SIZE,
            "test_conditions": TEST_CONDITIONS,
            "repeat_epochs": list(REPEAT_EPOCHS),
            "condition_repeat_rows_per_branch": CONDITION_REPEAT_ROWS,
            "same_physical_cells_all_branches": True,
            "canonical_order": "phase2_cell_index ascending",
            "membership_sha256": plan["membership_sha256"],
            "gt": "real treated expression mean - exact matched real control expression mean",
            "undefined_correlations": "NaN, never zero-imputed",
            "aggregation": "condition macro within repeat, then five-repeat mean/std",
        },
        "true_de": de_provenance,
        "context_level_set_pds": pds,
        "inference": inference,
        "protocol": artifact(PROTOCOL_PATH),
        "evaluator": artifact(SCRIPT_PATH),
        "phase3_complete": False,
        "phase4_started": False,
    }


def write_handoff(
    decoder_result: dict[str, Any],
    full_result: dict[str, Any],
    comparison: dict[str, Any],
) -> None:
    selected = comparison["selected_architecture"]
    lines = [
        "# Phase III final handoff",
        "",
        "- Status: PASS",
        "- Phase III tests a paired cell-set to condition-level Top20-delta formulation; it is not only a decoder architecture swap.",
        "- D1 and D2 used identical data, memberships, canonical ordering, GT, MSE loss, and training budget.",
        "- Every formal epoch consumed all 4000/4000 train conditions with no missing or duplicate pair_id.",
        f"- Validation-only selected architecture: `{selected}`; test_used_for_selection = false.",
        "- Test: 500 conditions x 5 deterministic repeats, with identical real physical cells across Old/D1/D2/full branches.",
        "- True DEG, DES, and AUPRC are restricted to the frozen Top20 universe, not genome-wide DE.",
        "- context-level set PDS is an L1 ranking adaptation of Cell-Eval discrimination_score_l1; it is not the original predicted-cell PDS.",
        "",
        "## Five-repeat metric means",
        "",
        "| branch | MAE | Pearson delta | Spearman delta | set-level DES | AUPRC | context-level set PDS | delta-magnitude Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    all_branches = {**decoder_result["branches"], **full_result["branches"]}
    for branch in ("old_decoder", "d1", "d2", "full_pipeline"):
        metrics = all_branches[branch]["metrics"]
        values = [
            metrics[name]["mean"]
            for name in (
                "mae", "pearson_delta", "spearman_delta", "set_level_des", "auprc",
                "context_level_set_pds", "delta_magnitude_spearman",
            )
        ]
        lines.append("| " + branch + " | " + " | ".join(f"{value:.6g}" for value in values) + " |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "D1 is the formal CNN route and is order-sensitive; its ascending phase2_cell_index order is only a deterministic convention, not biological spatial structure.",
            "D2 is permutation-invariant by contract.",
        ]
    )
    if selected == "d1":
        lines.append(
            "Because D1 was selected, Decoder-only real-treated ordering and Full Pipeline synthetic-treated inherited-control ordering differ unavoidably; the full-pipeline degradation may therefore include D1 ordering sensitivity in addition to ST-A latent-quality degradation."
        )
    lines.extend(["", "```ini", "phase3_complete = true", "phase4_started = false", "```", ""])
    temporary = HANDOFF.with_name(HANDOFF.name + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    os.replace(temporary, HANDOFF)


def evaluate(batch_size: int, de_workers: int) -> dict[str, Any]:
    from run_phase3_set_decoder import verify_protocol

    verify_protocol()
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if selection.get("status") != "pass":
        raise AssertionError("Validation-only architecture selection is not PASS")
    for path in (OLD_DECODER_TRAINING_RESULT, ST_TRAINING_RESULT):
        if json.loads(path.read_text(encoding="utf-8")).get("status") != "pass":
            raise AssertionError(f"Frozen upstream training result is not PASS: {path}")
    for variant in ("d1", "d2"):
        result = json.loads(variant_paths(variant)["result"].read_text(encoding="utf-8"))
        if result.get("status") != "pass" or not result.get("coverage_all_epochs_pass"):
            raise AssertionError(f"Phase-III {variant} formal training is not PASS")
    expression = json.loads(EXPRESSION_MANIFEST.read_text(encoding="utf-8"))
    if expression.get("status") != "pass":
        raise AssertionError("Phase-III expression cache is not PASS")

    plan = build_evaluation_plan()
    labels, de_provenance = build_true_deg_labels(plan, de_workers)
    predictions, inference = run_inference(plan, batch_size)
    metadata = plan["metadata"]
    condition_frames: dict[str, pd.DataFrame] = {}
    repeat_frames: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict[str, Any]] = {}
    pds_provenance: dict[str, Any] = {}
    for branch, prediction in predictions.items():
        frame = condition_metrics(branch, prediction, plan["true_delta"], labels, metadata)
        contexts, pds = context_pds(branch, prediction, plan["true_delta"], metadata)
        repeats, summary = summarize_branch(branch, frame, contexts)
        condition_frames[branch] = frame
        repeat_frames[branch] = repeats
        summaries[branch] = summary
        pds_provenance[branch] = pds

    decoder_conditions = pd.concat(
        [condition_frames[name] for name in ("old_decoder", "d1", "d2")], ignore_index=True
    )
    decoder_repeats = pd.concat(
        [repeat_frames[name] for name in ("old_decoder", "d1", "d2")], ignore_index=True
    )
    atomic_csv(DECODER_ONLY_CONDITIONS, decoder_conditions)
    atomic_csv(DECODER_ONLY_REPEATS, decoder_repeats)
    decoder_result = _result_payload(
        "decoder_only",
        {name: summaries[name] for name in ("old_decoder", "d1", "d2")},
        {name: pds_provenance[name] for name in ("old_decoder", "d1", "d2")},
        plan,
        de_provenance,
        inference,
    )
    decoder_result["outputs"] = {
        "condition_metrics": artifact(DECODER_ONLY_CONDITIONS),
        "repeat_metrics": artifact(DECODER_ONLY_REPEATS),
    }
    atomic_json(DECODER_ONLY_RESULT, decoder_result)

    atomic_csv(FULL_CONDITIONS, condition_frames["full_pipeline"])
    atomic_csv(FULL_REPEATS, repeat_frames["full_pipeline"])
    full_result = _result_payload(
        "full_pipeline",
        {"full_pipeline": summaries["full_pipeline"]},
        {"full_pipeline": pds_provenance["full_pipeline"]},
        plan,
        de_provenance,
        inference,
    )
    full_result["selected_architecture"] = selection["selected_architecture"]
    full_result["selection"] = artifact(SELECTION_PATH)
    full_result["outputs"] = {
        "condition_metrics": artifact(FULL_CONDITIONS),
        "repeat_metrics": artifact(FULL_REPEATS),
    }
    atomic_json(FULL_RESULT, full_result)

    selected = str(selection["selected_architecture"])
    metric_names = (
        "mae", "pearson_delta", "spearman_delta", "set_level_des", "auprc",
        "context_level_set_pds", "delta_magnitude_spearman",
    )
    means = {
        branch: {name: summaries[branch]["metrics"][name]["mean"] for name in metric_names}
        for branch in ("old_decoder", "d1", "d2", "full_pipeline")
    }
    comparison = {
        "schema": "phase3_comparison_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "selected_architecture": selected,
        "test_used_for_selection": False,
        "metric_means": means,
        "decoder_improvement": {
            variant: {
                name: means[variant][name] - means["old_decoder"][name]
                for name in metric_names
            }
            for variant in ("d1", "d2")
        },
        "st_a_degradation_selected_decoder_only_to_full": {
            name: means["full_pipeline"][name] - means[selected][name]
            for name in metric_names
        },
        "scientific_boundary": (
            "Phase III changes single-cell absolute-expression decoding to paired cell-set condition-delta "
            "decoding as well as architecture; D1-vs-D2 is the controlled architecture comparison."
        ),
        "outputs": {
            "decoder_only": artifact(DECODER_ONLY_RESULT),
            "full_pipeline": artifact(FULL_RESULT),
            "selection": artifact(SELECTION_PATH),
        },
        "phase3_complete": True,
        "phase4_started": False,
    }
    write_handoff(decoder_result, full_result, comparison)
    comparison["handoff"] = artifact(HANDOFF)
    atomic_json(COMPARISON, comparison)
    print(json.dumps(_json_safe(comparison), ensure_ascii=False, indent=2), flush=True)
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("evaluate")
    command.add_argument("--batch-size", type=int, default=20)
    command.add_argument("--de-workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args.batch_size, args.de_workers)


if __name__ == "__main__":
    main()
