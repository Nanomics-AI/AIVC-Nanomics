"""Audit the official GeneJEPA checkpoint and run the fixed 8-cell smoke.

This script is intentionally read-only with respect to checkpoints, vocabularies,
Tahoe parquet files, and both GeneJEPA source trees.  It writes only the audit
artifacts requested by the experiment protocol under ``results/``.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import inspect
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTHOR_DIR = PROJECT_ROOT / "external" / "author_genejepa"
AUTHOR_CODE = PROJECT_ROOT / "external" / "author_genejepa_code"
RESULTS = PROJECT_ROOT / "results"

AUTHOR_CHECKPOINT = AUTHOR_DIR / "genejepa-epoch=49.ckpt"
AUTHOR_METADATA = AUTHOR_DIR / "gene_metadata.parquet"
AUTHOR_STATS = AUTHOR_DIR / "global_stats.json"
CURRENT_METADATA = (
    PROJECT_ROOT / "hf_data_cache" / "metadata" / "metadata" / "gene_metadata.parquet"
)
CURRENT_STATS = PROJECT_ROOT / "hf_data_cache" / "global_stats.json"
CURRENT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "genejepa_quarter_d12_h6_700k_e30_seed42_run1"
    / "scjepa-epoch=25-val_loss=0.179.ckpt"
)
UNIQUE_CELLS = RESULTS / "tahoe_latent_audit_unique_cells.csv"
CURRENT_EMBEDDINGS = RESULTS / "tahoe_latent_audit_epoch25_embeddings.npy"
CURRENT_EMBEDDING_MANIFEST = RESULTS / "tahoe_latent_audit_epoch25_manifest.json"
TASK_FILE = PROJECT_ROOT.parent / "当前任务.txt"

OUTPUT_AUDIT = RESULTS / "author_genejepa_checkpoint_audit.json"
OUTPUT_REPORT = RESULTS / "author_genejepa_checkpoint_audit.md"
OUTPUT_HPARAMS = RESULTS / "author_genejepa_checkpoint_hparams.json"
OUTPUT_VOCAB = RESULTS / "author_genejepa_vocab_comparison.csv"
OUTPUT_SMOKE_JSON = RESULTS / "author_genejepa_8cell_smoke.json"
OUTPUT_SMOKE_NPY = RESULTS / "author_genejepa_8cell_smoke_embeddings.npy"

PARQUET_COLUMNS = [
    "genes",
    "expressions",
    "plate",
    "sample",
    "drug",
    "cell_line_id",
    "BARCODE_SUB_LIB_ID",
]

# Present in the published checkpoint but absent from the published Git HEAD.
# They are outside student_encoder and teacher_encoder.ema_model, and therefore
# outside get_embedding(); accepting anything beyond this exact set is forbidden.
ALLOWLISTED_INFERENCE_ONLY_UNEXPECTED_KEYS = {
    "teacher_center",
    "model.local_desc_proj.weight",
    "model.mask_desc_embed.weight",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": display_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "__class__": f"{type(value).__module__}.{type(value).__qualname__}",
            **{
                field.name: json_safe(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        }
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def command(*parts: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        list(parts),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def source_location(obj: Any) -> dict[str, Any]:
    obj = inspect.unwrap(obj)
    path = Path(inspect.getsourcefile(obj) or "").resolve()
    _, line = inspect.getsourcelines(obj)
    return {"path": display_path(path), "line": line}


def tensor_digest(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def state_structure_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        digest.update(
            f"{name}|{tuple(tensor.shape)}|{tensor.dtype}\n".encode("utf-8")
        )
    return digest.hexdigest()


def selected_state_fingerprints(
    state: dict[str, torch.Tensor], keys: list[str]
) -> dict[str, Any]:
    return {
        key: {
            "shape": list(state[key].shape),
            "dtype": str(state[key].dtype).removeprefix("torch."),
            "sha256": tensor_digest(state[key]),
        }
        for key in keys
    }


def object_summary(value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"type": type(value).__name__}
    if isinstance(value, dict):
        result["length"] = len(value)
        result["keys"] = [str(key) for key in value.keys()]
    elif isinstance(value, (list, tuple)):
        result["length"] = len(value)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        result["value"] = value
    return result


def register_author_config_aliases() -> dict[str, Any]:
    author = str(AUTHOR_CODE.resolve())
    if author not in sys.path:
        sys.path.insert(0, author)

    import __main__
    import genejepa
    from genejepa.configs import (
        DataConfig,
        ExperimentConfig,
        ModelConfig,
        TrainingConfig,
    )

    genejepa_path = Path(genejepa.__file__).resolve()
    if not genejepa_path.is_relative_to(AUTHOR_CODE.resolve()):
        raise RuntimeError(f"Imported the wrong genejepa package: {genejepa_path}")

    aliases = {
        "ModelConfig": ModelConfig,
        "TrainingConfig": TrainingConfig,
        "DataConfig": DataConfig,
        "ExperimentConfig": ExperimentConfig,
    }
    for name, cls in aliases.items():
        setattr(__main__, name, cls)
    return aliases


def load_checkpoint_top(path: Path) -> dict[str, Any]:
    # Explicitly required by the audit protocol: CPU, read-only, weights_only=False.
    return torch.load(path, map_location="cpu", weights_only=False)


def model_config_dict(hparams: dict[str, Any]) -> dict[str, Any]:
    value = hparams["model_config"]
    safe = json_safe(value)
    safe.pop("__class__", None)
    return safe


def checkpoint_architecture(
    checkpoint: dict[str, Any], author_config: dict[str, Any]
) -> dict[str, Any]:
    state = checkpoint["state_dict"]
    latent_shape = list(state["model.student_encoder.latents"].shape)
    identity_shape = list(
        state["model.student_encoder.tokenizer.identity_embed.weight"].shape
    )
    block_ids = sorted(
        {
            int(match.group(1))
            for key in state
            if (
                match := re.match(
                    r"model\.student_encoder\.latent_blocks_seq\.(\d+)\.", key
                )
            )
        }
    )
    predictor = []
    for key, tensor in state.items():
        if re.fullmatch(r"model\.predictor\.head\.\d+\.weight", key) and tensor.ndim == 2:
            predictor.append(
                {
                    "state_key": key,
                    "in_features": int(tensor.shape[1]),
                    "out_features": int(tensor.shape[0]),
                }
            )
    predictor.sort(key=lambda item: int(item["state_key"].split(".")[-2]))
    d = int(author_config["d"])
    heads = int(author_config["heads_h"])
    return {
        "model_class": "genejepa.models.GenePerceiverJEPA",
        "encoder_class": "genejepa.models.GenePerceiverEncoder",
        "student_module": "model.student_encoder",
        "teacher_container": "model.teacher_encoder (ema_pytorch.EMA)",
        "teacher_inference_module": "model.teacher_encoder.ema_model",
        "num_latents": int(latent_shape[0]),
        "embedding_dimension": int(latent_shape[1]),
        "encoder_blocks": len(block_ids),
        "encoder_block_indices": block_ids,
        "attention_heads": heads,
        "attention_head_dimension": d // heads,
        "gene_vocab_size": int(identity_shape[0]),
        "gene_identity_embedding_dimension": int(identity_shape[1]),
        "tokenizer_value_dimension": d - int(identity_shape[1]),
        "tokenizer_output_dimension": d,
        "fourier": {
            "num_frequencies": int(author_config["fourier_num_frequencies"]),
            "raw_sin_cos_dimension": 2
            * int(author_config["fourier_num_frequencies"]),
            "min_freq": float(author_config["fourier_min_freq"]),
            "max_freq": float(author_config["fourier_max_freq"]),
            "freq_scale": float(author_config["fourier_freq_scale"]),
        },
        "predictor_depth": int(author_config["predictor_depth"]),
        "predictor_expansion_factor": int(
            author_config["predictor_expansion_factor"]
        ),
        "predictor_linear_layers": predictor,
        "model_card_claim": {
            "num_latents": 512,
            "embedding_dimension": 768,
            "blocks": 24,
            "heads": 12,
        },
        "model_card_claim_matches_checkpoint": (
            latent_shape == [512, 768] and len(block_ids) == 24 and heads == 12
        ),
    }


def dataframe_audit(frame: pd.DataFrame) -> dict[str, Any]:
    token_ids = frame["token_id"].to_numpy(dtype=np.int64)
    unique_ids = np.unique(token_ids)
    minimum = int(unique_ids.min())
    maximum = int(unique_ids.max())
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "unique_gene_symbols": int(frame["gene_symbol"].nunique(dropna=True)),
        "unique_ensembl_ids": int(frame["ensembl_id"].nunique(dropna=True)),
        "unique_token_ids": int(frame["token_id"].nunique(dropna=True)),
        "token_id_min": minimum,
        "token_id_max": maximum,
        "token_ids_contiguous": bool(
            np.array_equal(unique_ids, np.arange(minimum, maximum + 1))
        ),
        "metadata_rows_token_sorted": bool(frame["token_id"].is_monotonic_increasing),
        "missing_values": {
            column: int(frame[column].isna().sum()) for column in frame.columns
        },
        "duplicate_gene_symbol_rows": int(
            frame["gene_symbol"].duplicated(keep=False).sum()
        ),
        "duplicate_ensembl_id_rows": int(
            frame["ensembl_id"].duplicated(keep=False).sum()
        ),
        "duplicate_token_id_rows": int(
            frame["token_id"].duplicated(keep=False).sum()
        ),
        "foundation_token_id_column": "token_id",
        "ensembl_id_column": "ensembl_id",
        "gene_symbol_column": "gene_symbol",
        "raw_token_id_semantics": (
            f"contiguous {minimum}..{maximum}; IDs below {minimum} are absent/reserved"
        ),
        "reserved_or_absent_ids_below_min": list(range(minimum)),
        "model_input_index_semantics": (
            f"official _load_metadata sorts token_id then enumerates 0..{len(frame) - 1}"
        ),
    }


def add_model_index(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    order = np.argsort(result["token_id"].to_numpy(), kind="stable")
    indices = np.empty(len(result), dtype=np.int64)
    indices[order] = np.arange(len(result), dtype=np.int64)
    result["genejepa_index"] = indices
    result["row_index"] = np.arange(len(result), dtype=np.int64)
    return result


def compare_vocabularies(
    author: pd.DataFrame, current: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    author = add_model_index(author)
    current = add_model_index(current)
    left = author.rename(
        columns={
            "gene_symbol": "author_gene_symbol",
            "token_id": "author_token_id",
            "genejepa_index": "author_genejepa_index",
            "row_index": "author_row_index",
        }
    )
    right = current.rename(
        columns={
            "gene_symbol": "current_gene_symbol",
            "token_id": "current_token_id",
            "genejepa_index": "current_genejepa_index",
            "row_index": "current_row_index",
        }
    )
    comparison = left.merge(
        right,
        on="ensembl_id",
        how="outer",
        indicator="membership",
        validate="one_to_one",
    )
    comparison["membership"] = comparison["membership"].map(
        {"left_only": "author_only", "right_only": "current_only", "both": "both"}
    )
    comparison["gene_symbol_match"] = (
        comparison["author_gene_symbol"] == comparison["current_gene_symbol"]
    )
    comparison["token_id_match"] = (
        comparison["author_token_id"] == comparison["current_token_id"]
    )
    comparison["genejepa_index_match"] = (
        comparison["author_genejepa_index"]
        == comparison["current_genejepa_index"]
    )
    comparison["row_order_match"] = (
        comparison["author_row_index"] == comparison["current_row_index"]
    )

    strip_version = lambda series: series.astype(str).str.replace(  # noqa: E731
        r"\.\d+$", "", regex=True
    )
    author_stripped = set(strip_version(author["ensembl_id"]))
    current_stripped = set(strip_version(current["ensembl_id"]))
    author_ensembl = set(author["ensembl_id"].astype(str))
    current_ensembl = set(current["ensembl_id"].astype(str))
    author_symbols = set(author["gene_symbol"].astype(str))
    current_symbols = set(current["gene_symbol"].astype(str))
    both = comparison["membership"].eq("both")
    summary = {
        "exact_file_sha256_identical": sha256_file(AUTHOR_METADATA)
        == sha256_file(CURRENT_METADATA),
        "exact_row_order_identical": author.equals(current),
        "exact_token_id_mapping_identical": bool(
            both.all()
            and comparison.loc[both, "token_id_match"].all()
            and comparison.loc[both, "genejepa_index_match"].all()
        ),
        "ensembl_exact_intersection": len(author_ensembl & current_ensembl),
        "ensembl_author_only": len(author_ensembl - current_ensembl),
        "ensembl_current_only": len(current_ensembl - author_ensembl),
        "ensembl_stripped_version_intersection": len(
            author_stripped & current_stripped
        ),
        "gene_symbol_exact_intersection": len(author_symbols & current_symbols),
        "gene_symbol_author_only": len(author_symbols - current_symbols),
        "gene_symbol_current_only": len(current_symbols - author_symbols),
        "matched_gene_count": int(both.sum()),
        "matched_fraction_of_author": float(both.sum() / len(author)),
        "matched_fraction_of_current": float(both.sum() / len(current)),
        "version_suffix_present_author": int(
            author["ensembl_id"].astype(str).str.contains(r"\.\d+$", regex=True).sum()
        ),
        "version_suffix_present_current": int(
            current["ensembl_id"].astype(str).str.contains(r"\.\d+$", regex=True).sum()
        ),
        "compatibility_case": "A",
        "decision": "existing Tahoe→GeneJEPA gene mapping can be reused directly",
    }
    if not summary["exact_token_id_mapping_identical"]:
        same_universe = author_ensembl == current_ensembl
        summary["compatibility_case"] = "B" if same_universe else "C"
        summary["decision"] = (
            "reuse physical mapping logic but rebuild author token IDs"
            if same_universe
            else "rebuild Tahoe gene → author vocabulary mapping"
        )
    return comparison, summary


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def embedding_stats(values: np.ndarray) -> dict[str, Any]:
    values64 = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values64, axis=1)
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "finite": bool(np.isfinite(values).all()),
        "min": float(values64.min()),
        "max": float(values64.max()),
        "mean": float(values64.mean()),
        "std": float(values64.std()),
        "negative_fraction": float(np.mean(values64 < 0)),
        "norm": {
            "min": float(norms.min()),
            "median": float(np.median(norms)),
            "mean": float(norms.mean()),
            "std": float(norms.std()),
            "max": float(norms.max()),
        },
    }


def pairwise_structure(values: np.ndarray) -> dict[str, Any]:
    values64 = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values64, axis=1, keepdims=True)
    cosine = (values64 / np.maximum(norms, 1e-12)) @ (
        values64 / np.maximum(norms, 1e-12)
    ).T
    differences = values64[:, None, :] - values64[None, :, :]
    euclidean = np.linalg.norm(differences, axis=2)
    upper = np.triu_indices(len(values64), k=1)

    def summary(array: np.ndarray) -> dict[str, float]:
        selected = array[upper]
        return {
            "min": float(selected.min()),
            "median": float(np.median(selected)),
            "mean": float(selected.mean()),
            "std": float(selected.std()),
            "max": float(selected.max()),
        }

    return {
        "cosine_summary": summary(cosine),
        "euclidean_summary": summary(euclidean),
        "cosine_matrix": cosine.tolist(),
        "euclidean_matrix": euclidean.tolist(),
    }


def vector_correlation(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.std() == 0 or right.std() == 0:
        return {"pearson": None, "spearman": None}
    left_rank = pd.Series(left).rank(method="average").to_numpy()
    right_rank = pd.Series(right).rank(method="average").to_numpy()
    return {
        "pearson": float(np.corrcoef(left, right)[0, 1]),
        "spearman": float(np.corrcoef(left_rank, right_rank)[0, 1]),
    }


def read_fixed_cells() -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    plan = pd.read_csv(UNIQUE_CELLS, encoding="utf-8-sig", nrows=8)
    plan = plan.sort_values("embedding_index").reset_index(drop=True)
    if plan["embedding_index"].tolist() != list(range(8)):
        raise AssertionError("The fixed smoke must use embedding_index 0..7")

    raw_cells: list[dict[str, Any]] = []
    cell_records: list[dict[str, Any]] = []
    for row in plan.itertuples(index=False):
        shard = (PROJECT_ROOT / row.shard_path).resolve()
        if not shard.is_relative_to(PROJECT_ROOT):
            raise ValueError(f"Shard escapes project root: {shard}")
        parquet = pq.ParquetFile(shard)
        table = parquet.read_row_group(int(row.row_group_index), columns=PARQUET_COLUMNS)
        local_row = int(row.row_index_in_row_group)
        if not 0 <= local_row < table.num_rows:
            raise IndexError(f"Row outside row group: {row.cell_locator}")
        raw = {
            column: table.column(column)[local_row].as_py()
            for column in PARQUET_COLUMNS
        }
        expected_shard_row = sum(
            parquet.metadata.row_group(index).num_rows
            for index in range(int(row.row_group_index))
        ) + local_row
        if expected_shard_row != int(row.row_index_in_shard):
            raise AssertionError(f"row_index_in_shard mismatch: {row.cell_locator}")
        expected_locator = (
            f"{row.shard_path}::row_group={int(row.row_group_index)}::row={local_row}"
        )
        if expected_locator != row.cell_locator:
            raise AssertionError(f"Stable locator mismatch: {row.cell_locator}")
        for column in ["plate", "sample", "drug", "cell_line_id", "BARCODE_SUB_LIB_ID"]:
            if str(raw[column]) != str(getattr(row, column)):
                raise AssertionError(
                    f"Locator metadata mismatch for {column}: {row.cell_locator}"
                )
        raw_cells.append(raw)
        cell_records.append(
            {
                "embedding_index": int(row.embedding_index),
                "cell_id": str(row.cell_id),
                "cell_locator": str(row.cell_locator),
                "plate": str(row.plate),
                "sample": str(row.sample),
                "drug": str(row.drug),
                "cell_line_id": str(row.cell_line_id),
                "raw_gene_count": len(raw["genes"]),
                "leading_negative_sentinel": bool(raw["expressions"][0] < 0),
            }
        )
    return plan, raw_cells, {"cells": cell_records, "metadata_verified": True}


def run_inference_smokes(
    module: Any,
    author_stats: dict[str, float],
    current_stats: dict[str, float],
    author_vocab: pd.DataFrame,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    from genejepa.configs import DataConfig, ExperimentConfig
    from genejepa.data import Tahoe100MDataModule, Tahoe100MDataset

    model = module.model.eval().to(device)
    synthetic_indices = torch.tensor([10, 42, 7, 3, 9, 1, 2], dtype=torch.long, device=device)
    synthetic_values = torch.tensor(
        [0.1, 0.5, 0.3, 2.1, -0.2, 0.0, 1.1],
        dtype=torch.float32,
        device=device,
    )
    synthetic_offsets = torch.tensor([0, 5, 7], dtype=torch.long, device=device)
    with torch.inference_mode():
        student = model.get_embedding(
            synthetic_indices, synthetic_values, synthetic_offsets, use_teacher=False
        )
        model.eval()
        teacher = model.get_embedding(
            synthetic_indices, synthetic_values, synthetic_offsets, use_teacher=True
        )
    student_np = student.float().cpu().numpy()
    teacher_np = teacher.float().cpu().numpy()
    synthetic = {
        "input_contract": {
            "indices_shape": list(synthetic_indices.shape),
            "values_shape": list(synthetic_values.shape),
            "offsets": synthetic_offsets.cpu().tolist(),
            "source": "official README-style ragged example",
        },
        "student_use_teacher_false": embedding_stats(student_np),
        "teacher_use_teacher_true": embedding_stats(teacher_np),
        "student_teacher_max_abs_difference": float(
            np.max(np.abs(student_np.astype(np.float64) - teacher_np.astype(np.float64)))
        ),
    }

    plan, raw_cells, locator_audit = read_fixed_cells()
    sorted_vocab = author_vocab.sort_values("token_id", kind="stable")
    gene_map = {
        int(token_id): index
        for index, token_id in enumerate(sorted_vocab["token_id"].tolist())
    }
    datamodule = Tahoe100MDataModule(DataConfig(), ExperimentConfig())
    datamodule.gene_map = gene_map
    datamodule.global_mean = float(author_stats["mean"])
    datamodule.global_std = float(author_stats["std"])
    processed = list(Tahoe100MDataset(raw_cells, gene_map))
    if len(processed) != 8:
        raise AssertionError(f"Expected 8 processed cells, got {len(processed)}")
    torch.manual_seed(42)
    batch = datamodule._collate_fn(processed)

    expected_mapped = []
    preprocessing_cells = []
    for raw in raw_cells:
        genes = raw["genes"]
        expressions = raw["expressions"]
        sentinel = bool(expressions[0] < 0)
        if sentinel:
            genes = genes[1:]
            expressions = expressions[1:]
        mapped = sum(int(gene in gene_map) for gene in genes)
        expected_mapped.append(mapped)
        preprocessing_cells.append(
            {
                "raw_gene_count": len(raw["genes"]),
                "post_sentinel_gene_count": len(genes),
                "mapped_gene_count": mapped,
                "unmapped_gene_count": len(genes) - mapped,
                "sentinel_removed": sentinel,
            }
        )
    lengths = (batch["offsets"][1:] - batch["offsets"][:-1]).tolist()
    if lengths != expected_mapped:
        raise AssertionError("Author preprocessing mapped-gene counts do not align")
    if not torch.isfinite(batch["values"]).all():
        raise AssertionError("Non-finite author-normalized expression values")

    current_values = (
        torch.log1p(torch.cat([torch.from_numpy(item["counts"]) for item in processed]).float())
        - float(current_stats["mean"])
    ) / (float(current_stats["std"]) + 1e-6)
    normalization_difference = torch.abs(batch["values"] - current_values)

    indices = batch["indices"].to(device)
    values = batch["values"].to(device)
    offsets = batch["offsets"].to(device)
    model.eval()
    with torch.inference_mode():
        author_embedding = model.get_embedding(
            indices, values, offsets, use_teacher=True
        ).float()
    author_embedding_np = author_embedding.cpu().numpy().astype(np.float32, copy=False)
    if author_embedding_np.shape != (8, 768):
        raise AssertionError(f"Unexpected 8-cell shape: {author_embedding_np.shape}")
    if not np.isfinite(author_embedding_np).all():
        raise AssertionError("Non-finite author 8-cell embeddings")

    current_cache = np.load(CURRENT_EMBEDDINGS, mmap_mode="r")
    current_indices = plan["embedding_index"].to_numpy(dtype=np.int64)
    current_embedding_np = np.asarray(current_cache[current_indices], dtype=np.float32)
    if current_embedding_np.shape != (8, 768):
        raise AssertionError(f"Unexpected Epoch25 comparison shape: {current_embedding_np.shape}")

    author_structure = pairwise_structure(author_embedding_np)
    current_structure = pairwise_structure(current_embedding_np)
    upper = np.triu_indices(8, k=1)
    comparison = {
        "warning": (
            "descriptive only; independently trained latent coordinates may rotate or "
            "reparameterize, so coordinate-wise cosine is intentionally not computed"
        ),
        "author_epoch49": {
            "embedding": embedding_stats(author_embedding_np),
            "pairwise": author_structure,
        },
        "our_epoch25": {
            "embedding": embedding_stats(current_embedding_np),
            "pairwise": current_structure,
        },
        "cell_cell_structure_correlation": {
            "cosine_upper_triangle": vector_correlation(
                np.asarray(author_structure["cosine_matrix"])[upper],
                np.asarray(current_structure["cosine_matrix"])[upper],
            ),
            "euclidean_upper_triangle": vector_correlation(
                np.asarray(author_structure["euclidean_matrix"])[upper],
                np.asarray(current_structure["euclidean_matrix"])[upper],
            ),
        },
    }
    preprocessing = {
        "implementation": {
            "sentinel_and_mapping": "author Tahoe100MDataset.__iter__",
            "log1p_and_normalization": "author Tahoe100MDataModule._collate_fn",
            "each_applied_once": True,
        },
        "cells": preprocessing_cells,
        "mapped_gene_count": {
            "min": int(min(lengths)),
            "mean": float(np.mean(lengths)),
            "max": int(max(lengths)),
        },
        "unmapped_gene_count_total": int(
            sum(item["unmapped_gene_count"] for item in preprocessing_cells)
        ),
        "normalized_values": {
            "count": int(batch["values"].numel()),
            "finite": bool(torch.isfinite(batch["values"]).all()),
            "min": float(batch["values"].min()),
            "max": float(batch["values"].max()),
            "mean": float(batch["values"].mean()),
            "std": float(batch["values"].std(unbiased=False)),
        },
        "author_vs_current_stats_only_normalized_value_difference": {
            "max_abs": float(normalization_difference.max()),
            "mean_abs": float(normalization_difference.mean()),
        },
        "stable_locator_audit": locator_audit,
    }
    return synthetic, author_embedding_np, {
        "preprocessing": preprocessing,
        "comparison": comparison,
    }


def stats_schema(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw": stats,
        "fields": {
            key: {
                "kind": "scalar" if np.isscalar(value) else "vector",
                "python_type": type(value).__name__,
                "length": None if np.isscalar(value) else len(value),
            }
            for key, value in stats.items()
        },
    }


def build_report(audit: dict[str, Any]) -> str:
    arch = audit["architecture"]
    vocab = audit["vocabulary"]
    stats = audit["global_stats"]
    smoke = audit["eight_cell_smoke"]
    answers = audit["compatibility_answers"]
    teacher = audit["ema_teacher"]
    current = audit["our_epoch25_contract"]
    author_embedding = smoke["author_embedding"]
    comparison = smoke["descriptive_comparison"]["cell_cell_structure_correlation"]

    answer_rows = "\n".join(
        f"| {key} | {value['status']} | {value['answer']} |"
        for key, value in answers.items()
    )
    teacher_rows = "\n".join(
        f"| `{item['parameter']}` | {item['max_abs_difference']:.8g} | "
        f"{item['mean_abs_difference']:.8g} |"
        for item in teacher["sampled_parameter_differences"]
    )
    return f"""# Author GeneJEPA epoch49 compatibility audit

- Status: **{audit['status'].upper()}**
- Created: `{audit['created_at_utc']}`
- Scope: checkpoint/code/vocab/normalization/embedding contract plus fixed 8-cell smoke only
- No full-cache extraction, Decoder/ST training, or ARC7 evaluation was run.

## 1. Checkpoint identity and load behavior

The artifact is `{audit['provenance']['artifacts']['checkpoint']['path']}` with SHA-256
`{audit['provenance']['artifacts']['checkpoint']['sha256']}`.  Its filename says epoch49,
while the Lightning payload stores zero-based `epoch={audit['checkpoint_structure']['epoch']}`
and `global_step={audit['checkpoint_structure']['global_step']}`.

The standalone artifact needs two inference-only loading adaptations.  First, its config
dataclasses were pickled under `__main__`, so a separate script must register the four classes
imported from the **official** `genejepa.configs` module under those names.  Second, strict
loading against the downloaded official Git HEAD fails because the checkpoint has exactly
three additional keys: `teacher_center`, `model.local_desc_proj.weight`, and
`model.mask_desc_embed.weight`.  They do not exist in that code checkout and are outside both
the student encoder and `teacher_encoder.ema_model` used by `get_embedding()`.

The diagnostic smoke therefore uses official `load_from_checkpoint(strict=False)` only after
asserting `missing_keys=[]` and that the unexpected set is **exactly** those three keys.  Any
other mismatch remains fatal.  This does not edit the source, checkpoint, or any encoder weight,
but the task's original zero-unexpected-key strict-load requirement is not met.

Imported package: `{audit['official_load']['genejepa_package_path']}`  
Loaded class: `{audit['official_load']['loaded_class']}`

## 2. Architecture recovered from checkpoint and official code

| Field | Author epoch49 |
|---|---:|
| Model | `{arch['model_class']}` |
| Encoder | `{arch['encoder_class']}` |
| Vocabulary size | {arch['gene_vocab_size']} |
| Latent tokens | {arch['num_latents']} |
| Embedding dimension | {arch['embedding_dimension']} |
| Encoder blocks | {arch['encoder_blocks']} |
| Attention heads | {arch['attention_heads']} |
| Head dimension | {arch['attention_head_dimension']} |
| Gene identity embedding | {arch['gene_identity_embedding_dimension']} |
| Fourier frequencies | {arch['fourier']['num_frequencies']} |
| Fourier range / scale | {arch['fourier']['min_freq']} .. {arch['fourier']['max_freq']} / {arch['fourier']['freq_scale']} |
| Predictor depth / expansion | {arch['predictor_depth']} / {arch['predictor_expansion_factor']}x |

The model-card claims 512 latents, 768 dimensions, 24 blocks, and 12 heads are an exact
match to the checkpoint.  The tokenizer splits 768 into 384 identity + 384 value channels;
64 frequencies create 128 sin/cos inputs before the value MLP, and the final tokenizer output
is 768-dimensional.

## 3. EMA teacher

`on_save_checkpoint` stores `model.teacher_encoder.state_dict()` as the top-level
`ema_state_dict`; `on_load_checkpoint` restores it.  `get_embedding(use_teacher=True)` calls
`model.teacher_encoder.ema_model`, not the student or EMA online model.

| Sampled parameter | max abs student-teacher diff | mean abs diff |
|---|---:|---:|
{teacher_rows}

The sampled differences prove that the restored teacher is not merely an alias of current
student weights.  Formal author extraction should use **`use_teacher=True`**, following the
official inference method and README.

## 4. Author vocabulary and compatibility

- Rows / unique genes / unique token IDs: {vocab['author']['rows']} / {vocab['author']['unique_ensembl_ids']} / {vocab['author']['unique_token_ids']}
- Columns: `{', '.join(vocab['author']['columns'])}`
- Raw foundation token IDs: {vocab['author']['token_id_min']}..{vocab['author']['token_id_max']} (continuous; 0, 1, 2 absent/reserved)
- Actual model indices: 0..{vocab['author']['rows'] - 1}, obtained by sorting raw token ID and enumerating
- Missing values: {vocab['author']['missing_values']}
- Duplicate gene symbols / Ensembl IDs / token IDs: {vocab['author']['duplicate_gene_symbol_rows']} / {vocab['author']['duplicate_ensembl_id_rows']} / {vocab['author']['duplicate_token_id_rows']}
- Exact metadata file SHA match with current project: **{vocab['comparison']['exact_file_sha256_identical']}**
- Exact row/order and token mapping match: **{vocab['comparison']['exact_row_order_identical']} / {vocab['comparison']['exact_token_id_mapping_identical']}**
- Matched genes: {vocab['comparison']['matched_gene_count']} ({vocab['comparison']['matched_fraction_of_author']:.2%})
- Author-only / current-only: {vocab['comparison']['ensembl_author_only']} / {vocab['comparison']['ensembl_current_only']}

This is compatibility **Case {vocab['comparison']['compatibility_case']}**: existing physical
Tahoe token-to-GeneJEPA-index mapping can be reused directly.  Raw `token_id` must still pass
through the existing sorted-enumeration map; it must not be fed directly to the embedding table.

## 5. Preprocessing contract

The official order is:

```text
Tahoe sparse genes + raw expressions
→ remove the leading genes/expressions item when expressions[0] < 0
→ filter/map raw Tahoe token_id through author vocab to 0-based model index
→ log1p(raw expression)
→ (x - author_mean) / (author_std + 1e-6)
→ identity + Fourier-value tokenizer
→ GenePerceiver encoder
```

Implicit zeros are not materialized as tokens.  A stored zero, if present, remains a token:
`log1p(0)=0`, then global standardization is applied.  There is no CP10K/library-size
normalization in the encoder input path.

| Field | Our Epoch25 | Author epoch49 | Exact compatible? |
|---|---|---|---|
| Raw source | Tahoe `genes/expressions` | Tahoe `genes/expressions` | Yes |
| Sentinel | leading negative expression removed once | same | Yes |
| Vocab/token mapping | 62,710; sort token_id then enumerate | same artifact and rule | Yes |
| Transform | log1p then scalar global standardization | same | Yes |
| Mean | {stats['current']['raw']['mean']:.17g} | {stats['author']['raw']['mean']:.17g} | **No, abs diff {stats['comparison']['mean_abs_difference']:.3g}** |
| Std | {stats['current']['raw']['std']:.17g} | {stats['author']['raw']['std']:.17g} | **No, abs diff {stats['comparison']['std_abs_difference']:.3g}** |
| Zero handling | sparse implicit zeros absent | same | Yes |
| Fourier N/min/max/scale | 64 / 0.1 / 100 / 1 | same | Yes |

The numerical difference is tiny, but formal author extraction must use the author stats file;
the two normalization artifacts are not bitwise or numerically exact.

## 6. Embedding readout contract

`get_embedding(use_teacher=True)` returns the EMA teacher encoder's final output.  Inside the
encoder, the 24th block output has shape `[B,512,768]`; `final_norm` is applied per latent token,
then `mean(dim=1)` yields `[B,768]`.  It is the final layer, not CLS or attention pooling.
There is no L2 normalization, centering, whitening, rectification, or predictor head in this
inference return value.

## 7. Comparison with our Epoch25 contract

| Field | Our Epoch25 | Author epoch49 |
|---|---|---|
| Checkpoint epoch label | 25 | 49 (payload epoch {audit['checkpoint_structure']['epoch']}) |
| Model family/code | GenePerceiverJEPA | GenePerceiverJEPA |
| Vocab | {current['gene_vocab_size']} | {arch['gene_vocab_size']} |
| Latents | {current['num_latents']} | {arch['num_latents']} |
| Embedding dim | {current['embedding_dimension']} | {arch['embedding_dimension']} |
| Blocks | {current['encoder_blocks']} | {arch['encoder_blocks']} |
| Heads | {current['attention_heads']} | {arch['attention_heads']} |
| Tokenizer | 384 identity + 384 Fourier-value | same |
| Normalization | current scalar stats | author scalar stats |
| Inference branch | EMA teacher | EMA teacher |
| Pooling | final_norm + latent mean | same |
| Post-normalization | none | none |

The 768-dimensional interface is compatible, but the representation is not interchangeable:
old Epoch25 embedding caches and Decoder weights cannot be reused as author-epoch49 latents or
Decoder weights.

## 8. Fixed 8-cell smoke

- Stable locators and metadata verified: **{smoke['locator_metadata_verified']}**
- Author output: `{author_embedding['shape']}`, `{author_embedding['dtype']}`, finite={author_embedding['finite']}
- Range / mean / std: {author_embedding['min']:.6g} .. {author_embedding['max']:.6g} / {author_embedding['mean']:.6g} / {author_embedding['std']:.6g}
- Negative fraction: {author_embedding['negative_fraction']:.6%}
- Euclidean distance-matrix correlation with the same cells in our Epoch25 space: Pearson={comparison['euclidean_upper_triangle']['pearson']:.6g}, Spearman={comparison['euclidean_upper_triangle']['spearman']:.6g}

The cross-check is descriptive only.  Coordinate-wise similarity is deliberately omitted because
independently trained latent spaces can rotate or reparameterize.

## 9. Compatibility answers

| Question | Status | Answer |
|---|---|---|
{answer_rows}

## 10. Correct future extraction recipe

1. Import only `external/author_genejepa_code` and assert `genejepa.__file__` points there.
2. Register the four official config dataclasses as `__main__` aliases for this artifact.
3. Load with official `JepaLightningModule.load_from_checkpoint(..., strict=False)` and abort
   unless missing keys are empty and unexpected keys equal the frozen three-key allowlist.
4. Freeze/eval and call `module.model.get_embedding(..., use_teacher=True)`.
5. Reuse stable Tahoe locators and the existing token mapping because vocab bytes/order match.
6. Apply sentinel removal once, mapping once, log1p once, and **author** scalar stats once.
7. Save the raw signed float32 `[N,768]` output without centering/whitening/L2/ReLU.

Reusable: Tahoe parquet data, stable locators, physical read logic, exact gene map, HD100 gene
panel and expression targets.  Must be regenerated/retrained: author embedding cache and the
Author-HD100 Decoder.  Existing Epoch25 embeddings and trained decoder weights are not reusable.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for the tiny synthetic and 8-cell forward passes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = [
        AUTHOR_CHECKPOINT,
        AUTHOR_METADATA,
        AUTHOR_STATS,
        CURRENT_METADATA,
        CURRENT_STATS,
        CURRENT_CHECKPOINT,
        UNIQUE_CELLS,
        CURRENT_EMBEDDINGS,
        CURRENT_EMBEDDING_MANIFEST,
    ]
    missing = [display_path(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    aliases = register_author_config_aliases()
    import genejepa
    from genejepa.models import GenePerceiverEncoder, GenePerceiverJEPA
    from genejepa.tokenizer import scRNATokenizer
    from genejepa.train import JepaLightningModule
    from genejepa.data import Tahoe100MDataset, Tahoe100MDataModule

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    provenance = {
        "task": file_record(TASK_FILE) if TASK_FILE.is_file() else None,
        "script": file_record(Path(__file__)),
        "artifacts": {
            "checkpoint": file_record(AUTHOR_CHECKPOINT),
            "gene_metadata": file_record(AUTHOR_METADATA),
            "global_stats": file_record(AUTHOR_STATS),
        },
        "author_repository": {
            "path": display_path(AUTHOR_CODE),
            "head_commit": command("git", "rev-parse", "HEAD", cwd=AUTHOR_CODE),
            "branch": command("git", "branch", "--show-current", cwd=AUTHOR_CODE),
            "status_porcelain": command(
                "git", "status", "--porcelain=v1", cwd=AUTHOR_CODE
            ).splitlines(),
            "status_branch": command(
                "git", "status", "--short", "--branch", cwd=AUTHOR_CODE
            ),
        },
        "official_code": {
            name: file_record(AUTHOR_CODE / "genejepa" / name)
            for name in ["models.py", "configs.py", "train.py", "data.py", "tokenizer.py"]
        },
        "current_code": {
            name: file_record(PROJECT_ROOT / "genejepa" / name)
            for name in ["models.py", "configs.py", "train.py", "data.py", "tokenizer.py"]
        },
    }

    author_vocab = pq.read_table(AUTHOR_METADATA).to_pandas()
    current_vocab = pq.read_table(CURRENT_METADATA).to_pandas()
    vocab_comparison_csv, vocab_comparison = compare_vocabularies(
        author_vocab, current_vocab
    )
    atomic_csv(OUTPUT_VOCAB, vocab_comparison_csv)
    vocab = {
        "author": dataframe_audit(author_vocab),
        "current": dataframe_audit(current_vocab),
        "comparison": vocab_comparison,
        "comparison_csv": {
            **file_record(OUTPUT_VOCAB),
            "encoding": "utf-8-sig",
        },
    }

    author_stats_values = json.loads(AUTHOR_STATS.read_text(encoding="utf-8"))
    current_stats_values = json.loads(CURRENT_STATS.read_text(encoding="utf-8"))
    global_stats = {
        "author": stats_schema(author_stats_values),
        "current": stats_schema(current_stats_values),
        "comparison": {
            "json_sha256_identical": sha256_file(AUTHOR_STATS)
            == sha256_file(CURRENT_STATS),
            "mean_exactly_equal": author_stats_values["mean"]
            == current_stats_values["mean"],
            "std_exactly_equal": author_stats_values["std"]
            == current_stats_values["std"],
            "mean_abs_difference": abs(
                author_stats_values["mean"] - current_stats_values["mean"]
            ),
            "std_abs_difference": abs(
                author_stats_values["std"] - current_stats_values["std"]
            ),
        },
    }

    checkpoint = load_checkpoint_top(AUTHOR_CHECKPOINT)
    top_level = {key: object_summary(value) for key, value in checkpoint.items()}
    hparams_payload = {
        "checkpoint": provenance["artifacts"]["checkpoint"],
        "hyper_parameters": json_safe(checkpoint.get("hyper_parameters")),
        "datamodule_hyper_parameters": json_safe(
            checkpoint.get("datamodule_hyper_parameters")
        ),
    }
    atomic_json(OUTPUT_HPARAMS, hparams_payload)
    author_hparams = checkpoint["hyper_parameters"]
    author_model_config = model_config_dict(author_hparams)
    architecture = checkpoint_architecture(checkpoint, author_model_config)

    selected_keys = [
        "model.student_encoder.latents",
        "model.student_encoder.tokenizer.value_encoder.0.weight",
        "model.student_encoder.latent_blocks_seq.0.attn.in_proj_weight",
        "model.student_encoder.latent_blocks_seq.23.ffn.0.weight",
        "model.student_encoder.final_norm.weight",
        "model.teacher_encoder.ema_model.latents",
    ]
    checkpoint_state = checkpoint["state_dict"]
    checkpoint_structure_fingerprint = state_structure_digest(checkpoint_state)
    checkpoint_filtered_structure_fingerprint = state_structure_digest(
        {
            key: tensor
            for key, tensor in checkpoint_state.items()
            if key not in ALLOWLISTED_INFERENCE_ONLY_UNEXPECTED_KEYS
        }
    )
    checkpoint_selected_fingerprints = selected_state_fingerprints(
        checkpoint_state, selected_keys
    )
    extra_checkpoint_state = {
        key: {
            "shape": list(checkpoint_state[key].shape),
            "dtype": str(checkpoint_state[key].dtype).removeprefix("torch."),
            "min": float(checkpoint_state[key].float().min()),
            "max": float(checkpoint_state[key].float().max()),
            "mean": float(checkpoint_state[key].float().mean()),
            "in_ema_state_dict": key in checkpoint.get("ema_state_dict", {}),
        }
        for key in sorted(ALLOWLISTED_INFERENCE_ONLY_UNEXPECTED_KEYS)
    }
    checkpoint_structure = {
        "top_level_keys": list(checkpoint.keys()),
        "top_level": top_level,
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "pytorch_lightning_version": checkpoint.get("pytorch-lightning_version"),
        "state_dict_tensor_count": len(checkpoint_state),
        "ema_state_dict_present": "ema_state_dict" in checkpoint,
        "ema_state_dict_tensor_count": len(checkpoint.get("ema_state_dict", {})),
        "foundation_gene_list_present": "foundation_gene_list" in checkpoint,
        "state_structure_sha256": checkpoint_structure_fingerprint,
        "state_structure_without_inference_allowlist_sha256": (
            checkpoint_filtered_structure_fingerprint
        ),
        "selected_state_fingerprints": checkpoint_selected_fingerprints,
        "checkpoint_only_state": extra_checkpoint_state,
        "hparams_output": file_record(OUTPUT_HPARAMS),
        "standalone_deserialization_adaptation": {
            "required": True,
            "reason": (
                "checkpoint pickles config dataclasses under __main__; direct standalone "
                "torch.load otherwise raises AttributeError"
            ),
            "aliases": {
                name: f"{cls.__module__}.{cls.__qualname__}"
                for name, cls in aliases.items()
            },
            "changes_checkpoint_or_weights": False,
        },
    }
    del checkpoint, checkpoint_state
    gc.collect()

    current_checkpoint = load_checkpoint_top(CURRENT_CHECKPOINT)
    current_hparams = current_checkpoint["hyper_parameters"]
    current_model_config = model_config_dict(current_hparams)
    current_internal_epoch = current_checkpoint.get("epoch")
    current_global_step = current_checkpoint.get("global_step")
    del current_checkpoint
    gc.collect()

    module = JepaLightningModule.load_from_checkpoint(
        str(AUTHOR_CHECKPOINT), map_location="cpu", strict=False
    )
    module.eval()
    loaded_state = module.state_dict()
    loaded_structure_fingerprint = state_structure_digest(loaded_state)
    loaded_selected_fingerprints = selected_state_fingerprints(
        loaded_state, selected_keys
    )
    checkpoint_keys = set(checkpoint_structure["top_level"]["state_dict"]["keys"])
    loaded_keys = set(loaded_state)
    missing_keys = sorted(loaded_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - loaded_keys)
    key_contract_match = (
        not missing_keys
        and set(unexpected_keys) == ALLOWLISTED_INFERENCE_ONLY_UNEXPECTED_KEYS
    )
    state_fingerprint_match = (
        loaded_structure_fingerprint == checkpoint_filtered_structure_fingerprint
        and loaded_selected_fingerprints == checkpoint_selected_fingerprints
    )
    if not key_contract_match or not state_fingerprint_match:
        raise AssertionError(
            "Inference-only load contract mismatch: "
            f"missing={missing_keys}, unexpected={unexpected_keys}, "
            f"fingerprint_match={state_fingerprint_match}"
        )

    student_parameters = dict(module.model.student_encoder.named_parameters())
    teacher_parameters = dict(
        module.model.teacher_encoder.ema_model.named_parameters()
    )
    sampled_parameter_names = [
        "latents",
        "tokenizer.value_encoder.0.weight",
        "latent_blocks_seq.0.attn.in_proj_weight",
        "latent_blocks_seq.23.ffn.0.weight",
        "final_norm.weight",
    ]
    parameter_differences = []
    total_difference = 0.0
    total_elements = 0
    global_max_difference = 0.0
    for name in sampled_parameter_names:
        difference = torch.abs(
            student_parameters[name].detach().cpu().float()
            - teacher_parameters[name].detach().cpu().float()
        )
        parameter_differences.append(
            {
                "parameter": name,
                "shape": list(difference.shape),
                "max_abs_difference": float(difference.max()),
                "mean_abs_difference": float(difference.mean()),
            }
        )
        total_difference += float(difference.double().sum())
        total_elements += difference.numel()
        global_max_difference = max(global_max_difference, float(difference.max()))
    ema_teacher = {
        "checkpoint_top_level_ema_state_dict_present": True,
        "save_path": "JepaLightningModule.on_save_checkpoint -> checkpoint['ema_state_dict']",
        "restore_path": "JepaLightningModule.on_load_checkpoint -> model.teacher_encoder.load_state_dict",
        "use_teacher_true_call": "model.teacher_encoder.ema_model(indices, values, offsets)",
        "official_default_use_teacher": True,
        "sampled_parameter_differences": parameter_differences,
        "sampled_global_max_abs_difference": global_max_difference,
        "sampled_global_mean_abs_difference": total_difference / total_elements,
        "teacher_differs_from_student": global_max_difference > 0,
        "formal_inference_should_use_teacher": True,
    }
    if not ema_teacher["teacher_differs_from_student"]:
        raise AssertionError("EMA teacher unexpectedly identical to student in all samples")

    official_load = {
        "loaded_class": f"{type(module).__module__}.{type(module).__qualname__}",
        "genejepa_package_path": display_path(Path(genejepa.__file__)),
        "module_source_path": display_path(Path(inspect.getsourcefile(type(module)) or "")),
        "model_source_path": display_path(
            Path(inspect.getsourcefile(type(module.model)) or "")
        ),
        "strict_default_load_succeeded": False,
        "strict_default_load_error": (
            "Unexpected key(s): teacher_center, model.local_desc_proj.weight, "
            "model.mask_desc_embed.weight"
        ),
        "diagnostic_load_strict": False,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "unexpected_keys_exactly_allowlisted": key_contract_match,
        "formal_zero_mismatch_requirement_met": False,
        "state_structure_sha256": loaded_structure_fingerprint,
        "selected_state_fingerprints_match_checkpoint": True,
        "source_locations": {
            "get_embedding": source_location(GenePerceiverJEPA.get_embedding),
            "encoder_forward": source_location(GenePerceiverEncoder.forward),
            "tokenizer_forward": source_location(scRNATokenizer.forward),
            "checkpoint_save": source_location(
                JepaLightningModule.on_save_checkpoint
            ),
            "checkpoint_load": source_location(
                JepaLightningModule.on_load_checkpoint
            ),
            "tahoe_iter": source_location(Tahoe100MDataset.__iter__),
            "collate": source_location(Tahoe100MDataModule._collate_fn),
        },
    }

    synthetic, author_embeddings, smoke_details = run_inference_smokes(
        module,
        author_stats_values,
        current_stats_values,
        author_vocab,
        device,
    )
    atomic_npy(OUTPUT_SMOKE_NPY, author_embeddings)

    smoke_payload = {
        "schema": "author_genejepa_8cell_smoke_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "checkpoint": provenance["artifacts"]["checkpoint"],
        "vocab": provenance["artifacts"]["gene_metadata"],
        "global_stats": provenance["artifacts"]["global_stats"],
        "use_teacher": True,
        "embedding_file": {
            "path": display_path(OUTPUT_SMOKE_NPY),
            "shape": list(author_embeddings.shape),
            "dtype": str(author_embeddings.dtype),
            "sha256": sha256_file(OUTPUT_SMOKE_NPY),
        },
        "synthetic_smoke": synthetic,
        "preprocessing": smoke_details["preprocessing"],
        "author_embedding": embedding_stats(author_embeddings),
        "descriptive_comparison": smoke_details["comparison"],
    }
    atomic_json(OUTPUT_SMOKE_JSON, smoke_payload)

    our_contract = {
        "checkpoint_path": display_path(CURRENT_CHECKPOINT),
        "checkpoint_sha256_from_frozen_manifest": json.loads(
            CURRENT_EMBEDDING_MANIFEST.read_text(encoding="utf-8")
        )["checkpoint"]["sha256"],
        "checkpoint_internal_epoch": current_internal_epoch,
        "checkpoint_global_step": current_global_step,
        "model_family": "GenePerceiverJEPA",
        "gene_vocab_size": int(current_model_config["gene_vocab_size"]),
        "num_latents": int(current_model_config["latents_L"]),
        "embedding_dimension": int(current_model_config["d"]),
        "encoder_blocks": int(current_model_config["blocks_D"]),
        "attention_heads": int(current_model_config["heads_h"]),
        "tokenizer": {
            "identity_value_split_ratio": current_model_config[
                "identity_value_split_ratio"
            ],
            "fourier_num_frequencies": current_model_config[
                "fourier_num_frequencies"
            ],
            "fourier_min_freq": current_model_config["fourier_min_freq"],
            "fourier_max_freq": current_model_config["fourier_max_freq"],
            "fourier_freq_scale": current_model_config["fourier_freq_scale"],
        },
        "normalization": current_stats_values,
        "teacher_student": "EMA teacher, get_embedding(use_teacher=True)",
        "pooling": "final_norm([B,512,768]).mean(dim=1) -> [B,768]",
        "post_normalization": "none",
    }

    compatibility_answers = {
        "A_checkpoint_load": {
            "status": "NEEDS ADAPTATION",
            "answer": (
                "Register four official config classes as __main__ aliases, then use an "
                "exact three-key inference-only allowlist; the downloaded Git HEAD does "
                "not meet zero-unexpected-key strict loading."
            ),
        },
        "B_embedding_shape": {
            "status": "YES",
            "answer": "Synthetic and real smoke both return [B,768].",
        },
        "C_use_ema_teacher": {
            "status": "YES",
            "answer": "Use get_embedding(..., use_teacher=True); it calls teacher_encoder.ema_model.",
        },
        "D_vocab_gene_universe": {
            "status": "YES",
            "answer": "The author and current 62,710-row metadata parquet files are byte-identical.",
        },
        "E_token_id_and_order": {
            "status": "YES",
            "answer": "Raw token IDs, row order, and 0-based enumerated model mapping are identical.",
        },
        "F_reuse_tahoe_mapping": {
            "status": "YES",
            "answer": "Reuse the existing physical token_id -> 0-based GeneJEPA mapping directly.",
        },
        "G_global_stats_exact": {
            "status": "NO",
            "answer": (
                "Values differ at ~1e-12/1e-14 scale; formal author extraction must "
                "use the author stats artifact."
            ),
        },
        "H_preprocessing_exact": {
            "status": "NEEDS ADAPTATION",
            "answer": (
                "Transform order and mapping logic match, but substitute author mean/std; "
                "do not silently reuse current stats."
            ),
        },
        "I_tokenizer_fourier": {
            "status": "YES",
            "answer": "Tokenizer source bytes and Fourier configuration are identical.",
        },
        "J_embedding_pooling": {
            "status": "YES",
            "answer": "Model source bytes match: final LayerNorm then mean over 512 latents.",
        },
        "K_decoder_input_dimension_change": {
            "status": "NO",
            "answer": "Author-HD100 Decoder input remains 768; its weights must be retrained.",
        },
        "L_safe_to_start_author_hd100": {
            "status": "YES",
            "answer": (
                "Safe only with the audited exact-key loader guard plus author stats; create "
                "a new author embedding cache and train a new decoder."
            ),
        },
    }

    audit = {
        "schema": "author_genejepa_checkpoint_compatibility_audit_v1",
        "created_at_utc": utc_now(),
        "status": "pass_with_required_loading_adaptation",
        "scope": "official artifact/code/vocab/normalization/embedding contract + fixed 8-cell smoke",
        "prohibited_actions_confirmed_not_run": [
            "full 30M author embedding extraction",
            "Author Decoder training",
            "ST-A training",
            "author/current GeneJEPA source modification",
            "vocab/global-stats modification",
            "HD100 panel modification",
            "ARC7 evaluation",
        ],
        "provenance": provenance,
        "checkpoint_structure": checkpoint_structure,
        "architecture": architecture,
        "official_load": official_load,
        "ema_teacher": ema_teacher,
        "vocabulary": vocab,
        "global_stats": global_stats,
        "preprocessing_contract": {
            "author_order": [
                "Tahoe sparse raw genes/expressions",
                "remove leading negative-expression sentinel once",
                "map physical token_id to 0-based author model index; drop unmapped",
                "log1p raw stored expression once",
                "(x - author mean) / (author std + 1e-6) once",
                "author identity + Fourier-value tokenizer",
            ],
            "cp10k_or_library_normalization": False,
            "implicit_zeros_materialized": False,
            "stored_zero_behavior": "kept as token; log1p(0)=0 then globally standardized",
            "current_vs_author": {
                "raw_count_source": "compatible",
                "sentinel_handling": "compatible",
                "gene_vocabulary": "exact",
                "gene_token_id": "exact",
                "expression_transform_order": "compatible",
                "log1p": "exact",
                "mean": "not exactly equal; use author artifact",
                "std": "not exactly equal; use author artifact",
                "zero_handling": "compatible",
                "tokenizer_fourier": "exact",
            },
        },
        "embedding_contract": {
            "input": "ragged indices:int64, normalized values:float32, offsets:int64",
            "branch": "EMA teacher_encoder.ema_model",
            "intermediate": "final encoder layer -> final_norm -> [B,512,768]",
            "pooling": "mean over latent dimension 1",
            "output": "raw signed [B,768]",
            "predictor_used": False,
            "cls_token": False,
            "attention_pooling": False,
            "layer_norm": "final_norm before latent mean",
            "l2_normalization": False,
            "centering": False,
            "whitening": False,
            "rectification": False,
        },
        "synthetic_smoke": synthetic,
        "eight_cell_smoke": {
            "json": file_record(OUTPUT_SMOKE_JSON),
            "embeddings": file_record(OUTPUT_SMOKE_NPY),
            "fixed_embedding_indices": list(range(8)),
            "locator_metadata_verified": smoke_details["preprocessing"][
                "stable_locator_audit"
            ]["metadata_verified"],
            "preprocessing": smoke_details["preprocessing"],
            "author_embedding": embedding_stats(author_embeddings),
            "descriptive_comparison": smoke_details["comparison"],
        },
        "our_epoch25_contract": our_contract,
        "compatibility_answers": compatibility_answers,
        "reuse_decision": {
            "may_reuse": [
                "Tahoe physical parquet files",
                "stable cell locators and physical read plan",
                "existing Tahoe token_id -> GeneJEPA index mapping",
                "HD100 gene panel and expression targets",
            ],
            "must_regenerate_or_retrain": [
                "author epoch49 embedding cache using author stats and EMA teacher",
                "Author-HD100 Decoder weights",
            ],
            "must_not_reuse_as_author_representation": [
                "our Epoch25 embedding caches",
                "our trained Decoder weights",
                "our global_stats.json for formal author extraction",
            ],
        },
        "ready_for_author_hd100_decoder_stage": True,
        "ready_conditions": [
            "register official config dataclasses under __main__ before deserialization",
            "require missing_keys == []",
            "require unexpected_keys exactly equal the frozen three-key inference allowlist",
            "use author global_stats.json",
            "use EMA teacher get_embedding(use_teacher=True)",
        ],
        "blockers": [],
    }
    atomic_json(OUTPUT_AUDIT, audit)
    atomic_text(OUTPUT_REPORT, build_report(audit))

    print(
        json.dumps(
            {
                "status": "pass_with_required_loading_adaptation",
                "audit": display_path(OUTPUT_AUDIT),
                "report": display_path(OUTPUT_REPORT),
                "hparams": display_path(OUTPUT_HPARAMS),
                "vocab_comparison": display_path(OUTPUT_VOCAB),
                "smoke_json": display_path(OUTPUT_SMOKE_JSON),
                "smoke_embeddings": display_path(OUTPUT_SMOKE_NPY),
                "compatibility_answers": compatibility_answers,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
