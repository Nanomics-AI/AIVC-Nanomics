#!/usr/bin/env python3
"""Run the focused GeneJEPA Decoder v1 D1/D2 readiness audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import default_collate

from tahoe_decoder_v1_data import (
    DEFAULT_PANEL,
    FORMAL_CACHE_SUMMARY,
    FORMAL_CONDITION_INDEX,
    FORMAL_EMBEDDING_MANIFEST,
    FORMAL_EMBEDDINGS,
    FORMAL_PLANS,
    GENE_METADATA,
    LATENT_DIM,
    OWNER_TREATED,
    PLAN_COLUMNS,
    PROJECT_ROOT,
    RAW_COLUMNS,
    RESULTS,
    SPLIT_TO_CODE,
    TahoeDecoderV1PairedDataset,
    atomic_write_json,
    display_path,
    load_conditions,
    load_gene_universe,
    prepare_mapped_counts,
    sha256_file,
)


STATE_ROOT = PROJECT_ROOT.parent / "state-main"
TASK_PATH = PROJECT_ROOT.parent / "当前任务.txt"
DECLARED_SOURCE = PROJECT_ROOT.parent / "虚拟细胞_GeneJEPA_项目总资料_统一版_v2.1_20260911.md"
AVAILABLE_UNIFIED_SOURCE = PROJECT_ROOT.parent / "虚拟细胞_GeneJEPA_项目总资料_统一版_v1.0.md"
SMOKE_PANEL = RESULTS / "genejepa_decoder_v1_gene_panel_smoke_only_current.csv"
SMOKE_PANEL_SUMMARY = RESULTS / "genejepa_decoder_v1_gene_panel_smoke_only_current.json"
SMOKE_PANEL_STATE = RESULTS / "genejepa_decoder_v1_gene_panel_smoke_only_current_state.npz"
AUDIT_JSON = RESULTS / "genejepa_decoder_v1_readiness_audit.json"
AUDIT_MD = RESULTS / "genejepa_decoder_v1_readiness_audit.md"
CONTRACT_JSON = RESULTS / "genejepa_decoder_v1_contract_candidate.json"
CONTROL_INDEX = Path(str(RESULTS / "tahoe_experiment1_cache_cap512_all_dmso") + "_control_pool_index.csv")
FROZEN_CONDITIONS = RESULTS / "tahoe_experiment1_condition_split_manifest.csv"
FROZEN_EDGES = RESULTS / "tahoe_experiment1_edge_split_manifest.csv"
GLOBAL_STATS = PROJECT_ROOT / "hf_data_cache/global_stats.json"
WORKER_MANIFESTS = tuple(
    RESULTS / f"tahoe_experiment1_cache_cap512_all_dmso_worker{worker}_extraction_manifest.json"
    for worker in range(2)
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_contains(path: Path, *needles: str) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise AssertionError(f"{path} no longer contains expected code: {missing}")


def source_record(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": display_path(path) if path.is_relative_to(PROJECT_ROOT) else str(path),
        "exists": path.is_file(),
    }
    if path.is_file():
        record["size_bytes"] = path.stat().st_size
        if hash_file:
            record["sha256"] = sha256_file(path)
    return record


def static_code_audit() -> dict[str, Any]:
    models = PROJECT_ROOT / "genejepa/models.py"
    tokenizer = PROJECT_ROOT / "genejepa/tokenizer.py"
    data = PROJECT_ROOT / "genejepa/data.py"
    configs = PROJECT_ROOT / "genejepa/configs.py"
    extractor = PROJECT_ROOT / "perturbation_scripts/extract_tahoe_latent_audit_embeddings.py"
    st_runner = PROJECT_ROOT / "perturbation_scripts/run_tahoe_experiment1_st_formal.py"
    state_base = STATE_ROOT / "src/state/tx/models/base.py"
    state_transition = STATE_ROOT / "src/state/tx/models/state_transition.py"
    state_decoders = STATE_ROOT / "src/state/tx/models/decoders.py"
    state_finetune = STATE_ROOT / "src/state/emb/finetune_decoder.py"
    state_embedding = STATE_ROOT / "src/state/emb/nn/model.py"

    assert_contains(
        models,
        "self.latents = nn.Parameter(torch.randn(config.latents_L, config.d))",
        "processed_latents.mean(dim=1)",
        "self.teacher_encoder.ema_model.eval()",
        "use_teacher: bool = True",
    )
    assert_contains(
        configs,
        "d: int = 768",
        "latents_L: int = 512",
        "blocks_D: int = 12",
        "heads_h: int = 6",
    )
    assert_contains(
        data,
        'expressions[0] < 0',
        'self.gene_map = {entry["token_id"]: i for i, entry in enumerate(sorted_genes)}',
        "values = torch.log1p(values.float())",
        "values = (values - self.global_mean) / (self.global_std + 1e-6)",
    )
    assert_contains(
        extractor,
        "module.model.get_embedding(",
        "use_teacher=True",
        "return embeddings.float().cpu().numpy()",
    )
    assert_contains(
        state_transition,
        "out_pred = self.project_out(res_pred) + basal",
        "out_pred = self.project_out(res_pred)",
        'final_activation="identity"',
    )
    assert_contains(
        state_base,
        "class LatentToGeneDecoder",
        "nn.LayerNorm(hidden_dim)",
        "nn.GELU()",
        "nn.Dropout(dropout)",
        "layers.append(nn.ReLU())",
        "latent_preds = pred.detach()",
    )
    assert_contains(
        state_decoders,
        "class FinetuneVCICountsDecoder",
        "self.read_depth",
        "self.finetune.get_gene_embedding(self.genes)",
        "ds_emb_dim",
        "return torch.nn.functional.relu(decoded_gene + decoded_x)",
    )
    assert_contains(
        state_finetune,
        "protein_embeds_dict",
        "self.model.binary_decoder",
        "task_counts = self.read_depth.expand",
    )
    assert_contains(
        state_embedding,
        "self.binary_decoder = nn.Sequential(",
        "self.gene_embedding_layer = self.encoder",
        "def resize_batch(cell_embeds, task_embeds, task_counts=None, sampled_rda=None, ds_emb=None)",
    )
    assert_contains(
        st_runner,
        '"st-a": {',
        '"predict_residual": False',
        '"formula": "Zpred = project_out(ST_hidden)"',
    )

    files = (
        models,
        tokenizer,
        data,
        configs,
        extractor,
        st_runner,
        state_base,
        state_transition,
        state_decoders,
        state_finetune,
        state_embedding,
    )
    return {str(path.relative_to(PROJECT_ROOT.parent)).replace("\\", "/"): source_record(path) for path in files}


def live_state_decoder_probe() -> dict[str, Any]:
    python = STATE_ROOT / ".venv/bin/python"
    code = (
        "import json,torch; "
        "from state.tx.models.base import LatentToGeneDecoder; "
        "torch.manual_seed(42); "
        "m=LatentToGeneDecoder(768,5000,[1024,1024,512],0.1,False).eval(); "
        "x=torch.randn(2,3,768); y=m(x); "
        "print(json.dumps({'input_shape':list(x.shape),'output_shape':list(y.shape),"
        "'parameter_count':sum(p.numel() for p in m.parameters()),"
        "'finite':bool(torch.isfinite(y).all()),'minimum':float(y.min()),"
        "'final_activation':type(m.decoder[-1]).__name__}))"
    )
    completed = subprocess.run(
        [str(python), "-c", code],
        cwd=STATE_ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=180,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    if result != {
        "input_shape": [2, 3, 768],
        "output_shape": [2, 3, 5000],
        "parameter_count": 4_931_976,
        "finite": True,
        "minimum": 0.0,
        "final_activation": "ReLU",
    }:
        raise AssertionError(f"Unexpected live STATE decoder probe: {result}")
    result["environment"] = str(python)
    result["training_run"] = False
    return result


def gene_and_preprocessing_audit() -> dict[str, Any]:
    original = pq.read_table(GENE_METADATA).to_pandas()
    genes, _ = load_gene_universe(GENE_METADATA)
    token_ids = genes["token_id"].to_numpy(np.int64)
    stats = read_json(GLOBAL_STATS)
    workers = [read_json(path) for path in WORKER_MANIFESTS]
    preprocessing = [worker["statistics"]["preprocessing"] for worker in workers]
    total_cells = sum(int(worker["completed"]["cells"]) for worker in workers)
    sentinel_cells = sum(int(item["sentinel_cells"]) for item in preprocessing)
    unmapped_max = max(float(item["unmapped_gene_count"]["max"]) for item in preprocessing)
    unmapped_total_mean_weighted = sum(
        float(item["unmapped_gene_count"]["mean"]) * int(worker["completed"]["cells"])
        for item, worker in zip(preprocessing, workers, strict=True)
    ) / total_cells
    normalized_min = min(float(item["normalized_values"]["min"]) for item in preprocessing)
    return {
        "gene_universe": {
            "vocabulary_size": len(genes),
            "genejepa_index_range": [0, len(genes) - 1],
            "token_id_range": [int(token_ids.min()), int(token_ids.max())],
            "token_ids_contiguous": bool(np.array_equal(token_ids, np.arange(3, 62_713))),
            "source_metadata_already_token_sorted": bool(original["token_id"].is_monotonic_increasing),
            "production_ordering": "stable ascending token_id; enumerate to genejepa_index",
            "metadata_columns": ["gene_symbol", "ensembl_id", "token_id"],
            "null_counts": {column: int(original[column].isna().sum()) for column in original.columns},
            "unique_counts": {
                column: int(original[column].nunique(dropna=True)) for column in original.columns
            },
            "metadata": source_record(GENE_METADATA),
        },
        "sentinel_and_mapping_full_cache_evidence": {
            "audited_cells": total_cells,
            "sentinel_cells": sentinel_cells,
            "sentinel_rate": sentinel_cells / total_cells,
            "production_definition": "remove first genes/expressions entry when expressions[0] < 0",
            "unmapped_gene_count_max_per_cell": unmapped_max,
            "unmapped_gene_count_weighted_mean": unmapped_total_mean_weighted,
            "all_post_sentinel_genes_mapped": unmapped_max == 0 and unmapped_total_mean_weighted == 0,
        },
        "encoder_preprocessing": {
            "pipeline": [
                "remove leading negative-expression sentinel once",
                "token_id -> genejepa_index via ascending-token metadata",
                "log1p(raw expression)",
                "(value - global_mean) / (global_std + 1e-6)",
                "tokenizer + GenePerceiver encoder",
            ],
            "cp10k": False,
            "library_size_normalization": False,
            "global_mean": float(stats["mean"]),
            "global_std": float(stats["std"]),
            "global_stats": source_record(GLOBAL_STATS),
            "full_cache_normalized_value_min": normalized_min,
        },
    }


def exact_plan_audit() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = read_json(FORMAL_CACHE_SUMMARY)
    merged = read_json(FORMAL_EMBEDDING_MANIFEST)
    conditions = load_conditions(FORMAL_CONDITION_INDEX)
    condition_records = conditions.to_dict("records")
    controls = pd.read_csv(
        CONTROL_INDEX,
        usecols=[
            "control_pool_index",
            "control_pool_id",
            "plate",
            "cell_line_id",
            "control_drug",
            "control_samples",
            "cached_cell_count",
            "embedding_start",
            "embedding_stop_exclusive",
        ],
        keep_default_na=False,
        encoding="utf-8-sig",
    ).sort_values("control_pool_index").reset_index(drop=True)
    control_records = controls.to_dict("records")

    total = int(merged["output"]["shape"][0])
    treated_total = int(summary["embedding_index"]["treated_range"][1])
    seen = np.zeros(total, dtype=np.bool_)
    observed_condition = np.zeros(len(conditions), dtype=np.int64)
    observed_control = np.zeros(len(controls), dtype=np.int64)
    split_counts = np.zeros(3, dtype=np.int64)
    condition_split = conditions["split"].map(SPLIT_TO_CODE).to_numpy(np.int8)
    condition_start = conditions["treated_embedding_start"].to_numpy(np.int64)
    condition_stop = conditions["treated_embedding_stop_exclusive"].to_numpy(np.int64)
    condition_expected = conditions["treated_cached_cell_count"].to_numpy(np.int64)
    control_start = controls["embedding_start"].to_numpy(np.int64)
    control_stop = controls["embedding_stop_exclusive"].to_numpy(np.int64)
    control_expected = controls["cached_cell_count"].to_numpy(np.int64)
    worker_shards: list[set[int]] = []
    row_count = 0
    samples: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "dmso": []}
    sample_keys: dict[str, set[tuple[str, int]]] = {key: set() for key in samples}
    started = time.perf_counter()

    for worker, plan_path in enumerate(FORMAL_PLANS):
        expected_hash = summary["outputs"][f"worker{worker}_plans"]["sha256"]
        observed_hash = sha256_file(plan_path)
        if observed_hash != expected_hash:
            raise AssertionError(f"Worker {worker} plan SHA-256 changed")
        plan = pq.ParquetFile(plan_path)
        expected_part = 0
        shards: set[int] = set()
        for plan_row_group in range(plan.num_row_groups):
            table = plan.read_row_group(
                plan_row_group, columns=("part_index", "shard_index", *PLAN_COLUMNS)
            )
            part = table["part_index"].to_numpy(zero_copy_only=False).astype(np.int64)
            embedding = table["embedding_index"].to_numpy(zero_copy_only=False).astype(np.int64)
            shard = table["shard_index"].to_numpy(zero_copy_only=False).astype(np.int64)
            shard_path_values = table["shard_path"].unique().to_pylist()
            source_group = table["row_group_index"].to_numpy(zero_copy_only=False).astype(np.int64)
            row_in_group = table["row_index_in_row_group"].to_numpy(zero_copy_only=False).astype(np.int64)
            row_in_shard = table["row_index_in_shard"].to_numpy(zero_copy_only=False).astype(np.int64)
            owner_type = table["owner_type"].to_numpy(zero_copy_only=False).astype(np.int8)
            owner = table["owner_index"].to_numpy(zero_copy_only=False).astype(np.int64)

            if not np.array_equal(part, np.arange(expected_part, expected_part + len(part))):
                raise AssertionError("part_index is not worker-local contiguous")
            expected_part += len(part)
            if len(np.unique(shard)) != 1 or len(shard_path_values) != 1:
                raise AssertionError("Plan row group is not a single shard")
            shard_value = int(shard[0])
            if shard_value in shards or (len(row_in_shard) > 1 and np.any(np.diff(row_in_shard) <= 0)):
                raise AssertionError("A physical locator is duplicated or out of order")
            shards.add(shard_value)
            if np.any((embedding < 0) | (embedding >= total)):
                raise AssertionError("embedding_index out of range")
            seen[embedding] = True
            row_count += len(embedding)

            treated = owner_type == OWNER_TREATED
            dmso = owner_type == 1
            if not np.all(treated | dmso):
                raise AssertionError("Unknown plan owner_type")
            treated_owner = owner[treated]
            treated_embedding = embedding[treated]
            if np.any((treated_owner < 0) | (treated_owner >= len(conditions))):
                raise AssertionError("Treated owner_index out of range")
            if np.any(
                (treated_embedding < condition_start[treated_owner])
                | (treated_embedding >= condition_stop[treated_owner])
            ):
                raise AssertionError("Treated embedding_index is outside its condition range")
            observed_condition += np.bincount(treated_owner, minlength=len(conditions))
            split_counts += np.bincount(condition_split[treated_owner], minlength=3)

            control_owner = owner[dmso]
            control_embedding = embedding[dmso]
            if np.any((control_owner < 0) | (control_owner >= len(controls))):
                raise AssertionError("DMSO owner_index out of range")
            if np.any(
                (control_embedding < control_start[control_owner])
                | (control_embedding >= control_stop[control_owner])
            ):
                raise AssertionError("DMSO embedding_index is outside its pool range")
            observed_control += np.bincount(control_owner, minlength=len(controls))

            for category, category_mask in (
                ("train", treated & (condition_split[np.minimum(owner, len(conditions) - 1)] == 0)),
                ("val", treated & (condition_split[np.minimum(owner, len(conditions) - 1)] == 1)),
                ("dmso", dmso),
            ):
                if len(samples[category]) >= 8:
                    continue
                for position in np.flatnonzero(category_mask):
                    key = (str(shard_path_values[0]), int(source_group[position]))
                    if key in sample_keys[category]:
                        continue
                    sample_keys[category].add(key)
                    samples[category].append(
                        {
                            "category": category,
                            "worker": worker,
                            "embedding_index": int(embedding[position]),
                            "shard_index": shard_value,
                            "shard_path": str(shard_path_values[0]),
                            "row_group_index": int(source_group[position]),
                            "row_index_in_row_group": int(row_in_group[position]),
                            "row_index_in_shard": int(row_in_shard[position]),
                            "owner_index": int(owner[position]),
                        }
                    )
                    if len(samples[category]) >= 8:
                        break
        if expected_part != plan.metadata.num_rows:
            raise AssertionError("Worker plan part_index coverage failed")
        worker_shards.append(shards)

    if row_count != total or int(seen.sum()) != total:
        raise AssertionError("Plan union is not exactly the merged-cache index universe")
    if worker_shards[0] & worker_shards[1]:
        raise AssertionError("Worker shard overlap is non-zero")
    if not np.array_equal(observed_condition, condition_expected):
        raise AssertionError("Per-condition treated locator counts changed")
    if not np.array_equal(observed_control, control_expected):
        raise AssertionError("Per-control-pool locator counts changed")
    if sum(len(values) for values in samples.values()) != 24:
        raise AssertionError("Could not collect the requested locator spotcheck cells")

    frozen = pd.read_csv(
        FROZEN_CONDITIONS,
        usecols=["pair_id", "edge_id", "split", "plate", "cell_line_id", "drug", "dose_uM"],
        keep_default_na=False,
        encoding="utf-8-sig",
    ).set_index("pair_id")
    indexed = conditions.set_index("pair_id")
    if set(frozen.index) != set(indexed.index):
        raise AssertionError("Frozen and cache condition IDs differ")
    frozen = frozen.loc[indexed.index]
    for column in ("edge_id", "split", "plate", "cell_line_id", "drug"):
        if not frozen[column].astype(str).eq(indexed[column].astype(str)).all():
            raise AssertionError(f"Frozen condition metadata changed: {column}")
    if not np.array_equal(
        frozen["dose_uM"].to_numpy(np.float64), indexed["dose_uM"].to_numpy(np.float64)
    ):
        raise AssertionError("Frozen condition dose changed")
    edge_splits = pd.read_csv(
        FROZEN_EDGES,
        usecols=["edge_id", "cell_line_id", "drug", "split"],
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    if edge_splits.groupby(["cell_line_id", "drug"])["split"].nunique().max() != 1:
        raise AssertionError("A frozen edge crosses split")

    names = ("train", "val", "test")
    split_result = {name: int(split_counts[index]) for index, name in enumerate(names)}
    expected_split = summary["selection"]["treated"]["cached_cells_by_split"]
    if split_result != {name: int(expected_split[name]) for name in names}:
        raise AssertionError("Exact plan split counts disagree with frozen summary")
    flat_samples = [item for category in ("train", "val", "dmso") for item in samples[category]]
    return (
        {
            "status": "pass",
            "exact_plan_rows_scanned": row_count,
            "plan_row_groups_scanned": sum(pq.ParquetFile(path).num_row_groups for path in FORMAL_PLANS),
            "worker_plan_sha256_recomputed": True,
            "embedding_index": {
                "first": 0,
                "last": total - 1,
                "missing": total - int(seen.sum()),
                "duplicate": row_count - int(seen.sum()),
                "row_equals_global_embedding_index": merged["output"]["row_equals_global_embedding_index"],
            },
            "treated_unique_cells_by_split": split_result,
            "treated_total": int(split_counts.sum()),
            "dmso_unique_cells": int(observed_control.sum()),
            "physical_cell_overlap": {"train_val": 0, "train_test": 0, "val_test": 0},
            "locator_overlap": {"train_val": 0, "train_test": 0, "val_test": 0},
            "proof": (
                "Every cache index appears exactly once; every plan row belongs to one owner; "
                "each shard occurs once and row_index_in_shard is strictly increasing; condition "
                "owners map to exactly one frozen split."
            ),
            "worker_shard_counts": [len(value) for value in worker_shards],
            "worker_shard_overlap": 0,
            "condition_owner_counts_match": True,
            "control_owner_counts_match": True,
            "sample_pooling_hidden_duplicate": False,
            "scan_elapsed_seconds": time.perf_counter() - started,
        },
        flat_samples,
    )


def raw_locator_spotcheck(
    locators: list[dict[str, Any]], genes: pd.DataFrame, token_lookup: np.ndarray
) -> dict[str, Any]:
    conditions = load_conditions(FORMAL_CONDITION_INDEX).to_dict("records")
    controls = pd.read_csv(
        CONTROL_INDEX, keep_default_na=False, encoding="utf-8-sig", low_memory=False
    ).sort_values("control_pool_index").to_dict("records")
    records: list[dict[str, Any]] = []
    totals = Counter()
    libraries: list[float] = []
    for locator in locators:
        source = pq.ParquetFile(PROJECT_ROOT / locator["shard_path"])
        table = source.read_row_group(locator["row_group_index"], columns=RAW_COLUMNS)
        row = locator["row_index_in_row_group"]
        raw = {column: table[column][row].as_py() for column in RAW_COLUMNS}
        group_start = sum(
            source.metadata.row_group(index).num_rows for index in range(locator["row_group_index"])
        )
        if group_start + row != locator["row_index_in_shard"]:
            raise AssertionError("row-group locator does not reproduce row_index_in_shard")
        if locator["category"] == "dmso":
            owner = controls[locator["owner_index"]]
            expected_drug = owner["control_drug"]
            expected_samples = str(owner["control_samples"]).split("|")
        else:
            owner = conditions[locator["owner_index"]]
            expected_drug = owner["drug"]
            expected_samples = str(owner["treated_samples"]).split("|")
        if (
            raw["plate"] != owner["plate"]
            or raw["cell_line_id"] != owner["cell_line_id"]
            or raw["drug"] != expected_drug
            or raw["sample"] not in expected_samples
        ):
            raise AssertionError("Locator raw metadata does not match its owner")
        _, _, audit = prepare_mapped_counts(raw["genes"], raw["expressions"], token_lookup)
        totals.update(
            {
                "sentinel_cells": int(audit["sentinel_removed"]),
                "unmapped_entries": audit["unmapped_entries"],
                "noninteger_count_entries": audit["noninteger_count_entries"],
                "duplicate_gene_entries": audit["duplicate_gene_entries_collapsed"],
            }
        )
        libraries.append(audit["library_size"])
        records.append(
            {
                "category": locator["category"],
                "embedding_index": locator["embedding_index"],
                "shard_path": locator["shard_path"],
                "row_group_index": locator["row_group_index"],
                "row_index_in_row_group": row,
                "plate": raw["plate"],
                "sample": raw["sample"],
                "drug": raw["drug"],
                "cell_line_id": raw["cell_line_id"],
                "sentinel_gene": raw["genes"][0],
                "sentinel_expression": raw["expressions"][0],
                "library_size": audit["library_size"],
            }
        )
    return {
        "status": "pass",
        "cells": len(records),
        "by_category": dict(Counter(record["category"] for record in records)),
        "unique_shards": len({record["shard_path"] for record in records}),
        "unique_source_row_groups": len(
            {(record["shard_path"], record["row_group_index"]) for record in records}
        ),
        "metadata_matches": True,
        "locator_arithmetic_matches": True,
        "sentinel_cells": totals["sentinel_cells"],
        "unmapped_entries": totals["unmapped_entries"],
        "noninteger_count_entries": totals["noninteger_count_entries"],
        "duplicate_gene_entries": totals["duplicate_gene_entries"],
        "zero_library_cells": int(sum(value <= 0 for value in libraries)),
        "library_size": {
            "min": min(libraries),
            "median": float(np.median(libraries)),
            "max": max(libraries),
        },
        "records": records,
    }


def read_raw_for_sample(sample: dict[str, Any]) -> tuple[list[int], list[float]]:
    source = pq.ParquetFile(PROJECT_ROOT / sample["shard_path"])
    table = source.read_row_group(
        int(sample["row_group_index"]), columns=["genes", "expressions"]
    )
    row = int(sample["row_index_in_row_group"])
    return table["genes"][row].as_py(), table["expressions"][row].as_py()


def manual_target(
    raw_genes: list[int],
    raw_expressions: list[float],
    token_to_index: dict[int, int],
    panel: np.ndarray,
) -> np.ndarray:
    start = 1 if raw_expressions[0] < 0 else 0
    count_by_index: dict[int, float] = {}
    library = 0.0
    for token_id, value in zip(raw_genes[start:], raw_expressions[start:], strict=True):
        if token_id not in token_to_index:
            continue
        index = token_to_index[token_id]
        numeric = float(value)
        library += numeric
        count_by_index[index] = count_by_index.get(index, 0.0) + numeric
    if library <= 0:
        raise AssertionError("Manual target encountered zero library")
    return np.asarray(
        [math.log1p(10_000.0 * count_by_index.get(int(index), 0.0) / library) for index in panel],
        dtype=np.float32,
    )


def dataset_smoke(genes: pd.DataFrame) -> dict[str, Any]:
    split_results: dict[str, Any] = {}
    token_to_index = dict(
        zip(genes["token_id"].astype(int), genes["genejepa_index"].astype(int), strict=True)
    )
    smoke_panel = pd.read_csv(SMOKE_PANEL, encoding="utf-8-sig", keep_default_na=False)
    panel = smoke_panel.sort_values("panel_rank")["genejepa_index"].to_numpy(np.int64)
    cache = np.load(FORMAL_EMBEDDINGS, mmap_mode="r")
    all_manual_errors: list[float] = []
    total_noninteger = 0
    total_zero_library = 0

    for split in ("train", "val"):
        kwargs = dict(
            split=split,
            panel_path=SMOKE_PANEL,
            shuffle_shards=False,
            max_cells=48,
            max_cells_per_shard=16,
            max_cells_per_source_row_group=2,
        )
        dataset = TahoeDecoderV1PairedDataset(**kwargs)
        samples = list(dataset)
        if len(samples) != 48:
            raise AssertionError(f"{split} Decoder dataset smoke did not yield 48 cells")
        batch = default_collate(samples[:8])
        if tuple(batch["latent"].shape) != (8, LATENT_DIM):
            raise AssertionError("Decoder latent batch shape mismatch")
        if tuple(batch["target_expression"].shape) != (8, len(panel)):
            raise AssertionError("Decoder target batch shape mismatch")
        if batch["latent"].dtype != torch.float32 or batch["target_expression"].dtype != torch.float32:
            raise AssertionError("Decoder batch dtype mismatch")
        if not torch.isfinite(batch["latent"]).all() or not torch.isfinite(
            batch["target_expression"]
        ).all():
            raise AssertionError("Decoder batch has non-finite values")
        if torch.any(batch["target_expression"] < 0):
            raise AssertionError("Decoder target has negative values")
        if not (torch.any(batch["latent"] < 0) and torch.any(batch["latent"] > 0)):
            raise AssertionError("Signed latent values were not preserved")

        conditions = dataset.conditions.set_index("pair_id")
        for sample in samples:
            index = int(sample["global_embedding_index"])
            if not np.array_equal(sample["latent"].numpy(), np.asarray(cache[index])):
                raise AssertionError("Dataset changed or misaligned a cached latent row")
            if conditions.loc[sample["pair_id"], "split"] != split:
                raise AssertionError("Dataset sample disagrees with frozen condition split")

        independently_constructed = TahoeDecoderV1PairedDataset(**{**kwargs, "max_cells": 8})
        second = list(independently_constructed)
        for left, right in zip(samples[:8], second, strict=True):
            for key in (
                "global_embedding_index",
                "pair_id",
                "shard_path",
                "row_group_index",
                "row_index_in_row_group",
            ):
                if left[key] != right[key]:
                    raise AssertionError("Independent Dataset construction is not deterministic")
            if not torch.equal(left["latent"], right["latent"]) or not torch.equal(
                left["target_expression"], right["target_expression"]
            ):
                raise AssertionError("Independent Dataset tensors are not deterministic")

        manual_errors: list[float] = []
        for sample in samples[:4]:
            raw_genes, raw_expressions = read_raw_for_sample(sample)
            reference = manual_target(raw_genes, raw_expressions, token_to_index, panel)
            error = float(np.max(np.abs(reference - sample["target_expression"].numpy())))
            manual_errors.append(error)
            _, _, audit = prepare_mapped_counts(raw_genes, raw_expressions, dataset.token_lookup)
            total_noninteger += int(audit["noninteger_count_entries"])
            total_zero_library += int(audit["library_size"] <= 0)
        all_manual_errors.extend(manual_errors)
        split_results[split] = {
            "cells": len(samples),
            "unique_embedding_indices": len({sample["global_embedding_index"] for sample in samples}),
            "unique_shards": len({sample["shard_path"] for sample in samples}),
            "unique_source_row_groups": len(
                {(sample["shard_path"], sample["row_group_index"]) for sample in samples}
            ),
            "latent_shape": [LATENT_DIM],
            "target_shape": [len(panel)],
            "batched_latent_shape": list(batch["latent"].shape),
            "batched_target_shape": list(batch["target_expression"].shape),
            "dtype": "float32",
            "finite": True,
            "signed_latent": True,
            "nonnegative_target": True,
            "latent_exact_cache_row": True,
            "metadata_exact": True,
            "split_exact": True,
            "independent_reconstruction_deterministic": True,
            "manual_transform_cells": len(manual_errors),
            "manual_transform_max_abs_error": max(manual_errors),
        }
    if max(all_manual_errors) > 1e-6:
        raise AssertionError("Independent target calculation exceeds tolerance")
    return {
        "status": "pass",
        "panel": source_record(SMOKE_PANEL),
        "panel_is_smoke_only": True,
        "splits": split_results,
        "manual_transform_max_abs_error": max(all_manual_errors),
        "manual_transform_tolerance": 1e-6,
        "manual_transform_noninteger_entries": total_noninteger,
        "manual_transform_zero_library_cells": total_zero_library,
        "test_expression_read": False,
        "model_training_run": False,
    }


def build_contract(audit: dict[str, Any]) -> dict[str, Any]:
    full_command = (
        "cd /mnt/c/SH/AIVC/GeneJEPA-main\n"
        ".venv/bin/python perturbation_scripts/tahoe_decoder_v1_data.py build-panel \\\n"
        "  --top-k 5000 \\\n"
        "  --output results/genejepa_decoder_v1_gene_panel_candidate.csv \\\n"
        "  --summary results/genejepa_decoder_v1_gene_panel_candidate.json \\\n"
        "  --state results/genejepa_decoder_v1_gene_panel_builder_state.npz \\\n"
        "  --progress results/genejepa_decoder_v1_gene_panel_builder_progress.json"
    )
    return {
        "schema": "genejepa_decoder_v1_contract_candidate_v1",
        "status": "candidate_not_frozen",
        "created_at": utc_now(),
        "source_provenance": audit["source_provenance"],
        "encoder": {
            "checkpoint": audit["latent_contract"]["checkpoint"],
            "encoder_branch": "Epoch25 frozen EMA teacher get_embedding(use_teacher=True)",
            "latent_dim": 768,
            "latent_dtype": "float32",
            "cache_path": display_path(FORMAL_EMBEDDINGS),
            "cache_shape": [30_839_089, 768],
            "row_semantics": "one physical Tahoe cell; row == global_embedding_index",
            "latent_transforms_after_encoder": [],
        },
        "gene_universe": {
            "vocabulary_size": 62_710,
            "genejepa_index_range": [0, 62_709],
            "ordering": "ascending token_id",
            "metadata_source": display_path(GENE_METADATA),
        },
        "target": {
            "status": "candidate_pending_human_freeze",
            "candidate_transform": "log1p(10000 * raw_count / mapped_gene_library_size)",
            "denominator_definition": "sum raw counts after sentinel removal over all mapped GeneJEPA genes",
            "order": "full mapped counts -> CP10000 -> log1p -> panel select",
            "output_dtype": "float32",
            "output_range": "finite and >= 0",
            "unresolved_questions": [
                "Human approval of CP10K/log1p as the formal Decoder target",
                "Full train scan must finish its exact zero-library/noninteger audit",
            ],
        },
        "gene_panel": {
            "status": "candidate_not_materialized",
            "candidate_top_k": 5000,
            "candidate_universe": "all 62,710 GeneJEPA genes",
            "fit_scope": "Experiment 1 train treated physical cells only",
            "expression_space": "candidate Decoder target space",
            "include_zeros": True,
            "ranking_rule": "population variance descending",
            "tie_rule": "genejepa_index ascending",
            "panel_file": None,
            "pending": True,
            "smoke_only_panel": display_path(SMOKE_PANEL),
            "exact_materialization_command": full_command,
        },
        "split": {
            "train_source": display_path(FROZEN_CONDITIONS),
            "val_source": display_path(FROZEN_CONDITIONS),
            "test_source": display_path(FROZEN_CONDITIONS),
            "ownership": "treated cell inherits its unique condition's frozen edge split",
            "DMSO_used_for_decoder_fitting": False,
            "unique_cell_counts": audit["split_leakage"]["treated_unique_cells_by_split"],
            "overlap_audit": audit["split_leakage"]["physical_cell_overlap"],
        },
        "dataset": {
            "class": "TahoeDecoderV1PairedDataset",
            "implementation": "perturbation_scripts/tahoe_decoder_v1_data.py",
            "cache_lookup_contract": "np.load(..., mmap_mode='r')[global_embedding_index]",
            "raw_expression_lookup_contract": "project-relative shard_path + row_group_index + row_index_in_row_group",
            "io_contract": "whole-shard plan order; each source row group read once; DDP/DataLoader workers partition shards",
            "rerun_genejepa": False,
        },
        "architecture_candidate": {
            "status": "candidate_pending_human_freeze",
            "input": 768,
            "hidden_dims": [1024, 1024, 512],
            "hidden_block": "Linear -> LayerNorm -> GELU -> Dropout(0.1)",
            "output": 5000,
            "final_activation": "Softplus candidate",
            "parameter_count_if_G_5000": 4_931_976,
            "state_reuse": "LatentToGeneDecoder body; current class needs optional final_activation because it is fixed to ReLU",
        },
        "loss_candidate": {
            "status": "candidate_pending_human_freeze",
            "name": "MSE",
            "prediction_and_target_dtype": "FP32 for loss/reduction",
            "space": "log1p(CP10000)",
        },
        "validation_candidate": "val cell-level macro MSE; exact checkpoint criterion still pending decision",
        "test_policy": "test ownership/schema may be audited; no test expression/statistic/performance informs panel, target, architecture, or hyperparameters",
        "cache_reuse": "reuse formal 30.84M mmap; no GeneJEPA rerun",
        "formal_training_requirements": audit["formal_training_requirements"],
        "pending_decisions": audit["pending_decisions"],
        "conflicts": audit["conflicts"],
        "ready_for_contract_freeze": False,
        "ready_for_formal_training": False,
    }


def write_markdown(audit: dict[str, Any], contract: dict[str, Any]) -> None:
    split = audit["split_leakage"]["treated_unique_cells_by_split"]
    target = audit["expression_target_audit"]
    d2 = audit["d2_dataset_smoke"]
    lines = [
        "# GeneJEPA Decoder v1 D1/D2 readiness audit",
        "",
        f"- Status: **{audit['status'].upper()}**",
        "- No GeneJEPA/ST/Decoder training was run; no embedding was re-extracted.",
        "- Candidate contract is not frozen and formal training is not ready yet.",
        "",
        "## Confirmed contracts",
        "",
        "- A cache row is one physical Tahoe cell's frozen Epoch25 EMA-teacher GeneJEPA embedding.",
        "- Internal encoder tokens are `[B,512,768]`; mean pooling over 512 latent tokens yields `[B,768]`.",
        "- ST uses 256 such cell embeddings as `[B,256,768]`; ST-A is absolute output, not a residual.",
        "- Merged cache is `[30,839,089,768]` float32 mmap, signed, and row equals global embedding index.",
        "- Gene vocabulary is exactly 62,710 entries, ordered by ascending token_id 3..62,712 into index 0..62,709.",
        "- Encoder preprocessing is sentinel removal, mapping, log1p(raw expression), then fixed global mean/std. It has no CP10K.",
        "",
        "## STATE decoder audit",
        "",
        "- `LatentToGeneDecoder` is an all-genes-at-once MLP with Linear/LayerNorm/GELU/Dropout hidden blocks and fixed final ReLU.",
        "- PyTorch last-dimension semantics were live-probed: `[2,3,768] -> [2,3,5000]`, 4,931,976 parameters for the candidate widths.",
        "- Its MLP body is reusable. Softplus needs one optional `final_activation` argument whose default remains ReLU; no change was made before contract freeze.",
        "- `FinetuneVCICountsDecoder` is gene-conditioned and combines a different SE/VCI cell latent with protein/gene embeddings, optional read depth and dataset embedding. No local pretrained VCI/SE checkpoint was found. Its weights are not compatible with GeneJEPA latent space.",
        "",
        "## Exact cache / split audit",
        "",
        f"- train treated unique cells: {split['train']:,}",
        f"- val treated unique cells: {split['val']:,}",
        f"- test treated unique cells: {split['test']:,}",
        f"- DMSO unique cells: {audit['split_leakage']['dmso_unique_cells']:,}",
        "- train/val/test embedding-index and physical-locator intersections: all exactly 0.",
        "- Both plan files were fully scanned; every global embedding index appeared exactly once and all per-condition/per-pool counts matched.",
        "",
        "## Expression target audit",
        "",
        f"- Candidate: `{target['candidate_transform']}`.",
        "- Denominator uses every post-sentinel mapped GeneJEPA gene before panel selection.",
        f"- Full-cache extraction proves unmapped genes per cell max = {target['full_cache_unmapped_gene_count_max']}; thus all usable post-sentinel Tahoe genes in this cache map.",
        f"- Locator sample: {target['locator_sample_cells']} cells, noninteger entries={target['locator_sample_noninteger_entries']}, zero-library={target['locator_sample_zero_library_cells']}.",
        f"- Independent Dataset/manual transform max absolute error: {d2['manual_transform_max_abs_error']:.3g}.",
        "- Exact all-train noninteger/zero-library counts remain part of the full panel scan; the builder rejects invalid cells rather than silently changing the denominator.",
        "",
        "## Gene panel",
        "",
        "- No pre-existing formal Tahoe train-only HVG/variance panel was found.",
        "- The exact builder is implemented and resumable. It accumulates float64 sum/sumsq over train treated cells, includes implicit zeros, ranks population variance descending, and breaks ties by GeneJEPA index ascending.",
        "- Only a 256-cell/64-gene `smoke_only` panel exists now. No formal top-5000 panel has been materialized.",
        "- Physical parquet reads will touch the genes/expressions columns of source row groups containing selected train cells (effectively most/all Tahoe shards), while only the 22,941,936 frozen train-treated locators enter statistics.",
        "",
        "## D2 Dataset smoke",
        "",
        f"- train: {d2['splits']['train']['cells']} cells, {d2['splits']['train']['unique_shards']} shards, {d2['splits']['train']['unique_source_row_groups']} source row groups.",
        f"- val: {d2['splits']['val']['cells']} cells, {d2['splits']['val']['unique_shards']} shards, {d2['splits']['val']['unique_source_row_groups']} source row groups.",
        "- latent `[768]`, smoke target `[64]`, batched `[8,768]` / `[8,64]`, float32, finite, signed latent, non-negative target, exact cache-row and metadata match.",
        "- Independent Dataset construction reproduced indices/tensors exactly. Test expression was not read.",
        "",
        "## Conflict and pending decisions",
        "",
        "- Declared source-of-truth `v2.1_20260911.md` is missing; only v1.0 exists. Current checkout code and frozen pass manifests were used as evidence, not v1.0 as a replacement authority.",
        "- Pending human freeze: target transform, top_k=5000 panel contract, Softplus, MLP widths, MSE and exact validation/checkpoint criterion.",
        "- Data blocker: exact train-only top-5000 candidate panel has not been materialized.",
        "",
        "## Readiness",
        "",
        "- `ready_for_contract_freeze: false`",
        "- `ready_for_formal_training: false`",
        "",
        "## Exact next step",
        "",
        "Run the exact panel builder command recorded in `genejepa_decoder_v1_contract_candidate.json`; after its candidate panel and full-scan integrity summary pass, review the small pending-decision list once and freeze Decoder v1. Do not start training before that freeze.",
        "",
    ]
    AUDIT_MD.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run() -> None:
    required = (
        FORMAL_EMBEDDINGS,
        FORMAL_EMBEDDING_MANIFEST,
        FORMAL_CACHE_SUMMARY,
        FORMAL_CONDITION_INDEX,
        *FORMAL_PLANS,
        GENE_METADATA,
        GLOBAL_STATS,
        SMOKE_PANEL,
        SMOKE_PANEL_SUMMARY,
        SMOKE_PANEL_STATE,
        *WORKER_MANIFESTS,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required audit inputs are missing: {missing}")

    code = static_code_audit()
    state_probe = live_state_decoder_probe()
    gene_audit = gene_and_preprocessing_audit()
    split_audit, locators = exact_plan_audit()
    genes, token_lookup = load_gene_universe(GENE_METADATA)
    raw_spot = raw_locator_spotcheck(locators, genes, token_lookup)
    d2 = dataset_smoke(genes)
    merged = read_json(FORMAL_EMBEDDING_MANIFEST)
    cache_summary = read_json(FORMAL_CACHE_SUMMARY)
    worker0 = read_json(WORKER_MANIFESTS[0])
    smoke_panel = read_json(SMOKE_PANEL_SUMMARY)
    with np.load(SMOKE_PANEL_STATE, allow_pickle=False) as state:
        smoke_resume_cells = int(state["cells"])
        smoke_resume_fingerprint = str(state["fingerprint"].item())
    if smoke_resume_cells != 256 or smoke_resume_fingerprint != smoke_panel["builder"]["fingerprint"]:
        raise AssertionError("Smoke panel resume state is not aligned with its summary")
    checkpoint = worker0["provenance"]["checkpoint"]
    full_cache = merged["embedding_statistics"]
    declared_source_exists = DECLARED_SOURCE.is_file()
    pretrained_files = [
        path
        for pattern in ("*.ckpt", "*vci*", "*se600*")
        for path in STATE_ROOT.rglob(pattern)
        if ".venv" not in path.parts and path.is_file()
    ]

    formal_training_requirements = [
        "2 x RTX A6000 native PyTorch DDP",
        "BF16 autocast with FP32 MSE/reduction",
        "DataLoader workers, pin_memory, non_blocking transfer",
        "gradient accumulation",
        "TensorBoard",
        "atomic best and last checkpoints",
        "resume including optimizer/epoch/RNG state",
        "validation every epoch and strict best-checkpoint selection",
        "max epochs, early stopping, fixed seed",
        "test excluded from model/panel/transform/hyperparameter selection",
    ]
    conflicts = [
        {
            "severity": "high_documentation",
            "issue": "Declared v2.1 source-of-truth document is absent from the workspace",
            "declared_path": str(DECLARED_SOURCE),
            "available_older_document": source_record(AVAILABLE_UNIFIED_SOURCE),
            "handling": "Used current code and frozen pass manifests; did not treat v1.0 as replacement authority",
        }
    ]
    pending_decisions = [
        "Freeze or reject log1p(CP10000) target with mapped-gene denominator",
        "Freeze top_k=5000 train-only population-variance panel after exact materialization",
        "Freeze MLP widths [1024,1024,512]",
        "Freeze Softplus final activation",
        "Freeze FP32 MSE and exact validation/checkpoint criterion",
        "Resolve or explicitly waive the missing v2.1 documentation provenance gap",
    ]
    audit: dict[str, Any] = {
        "schema": "genejepa_decoder_v1_readiness_audit_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "scope": "D1/D2 readiness/data-contract audit and paired Dataset implementation only",
        "prohibited_actions_confirmed_not_run": [
            "GeneJEPA training",
            "GeneJEPA embedding extraction",
            "ST-A/ST-R training",
            "Decoder training",
            "new perturbation benchmark",
            "test-driven gene/target/architecture selection",
        ],
        "source_provenance": {
            "task": source_record(TASK_PATH),
            "declared_source_of_truth": source_record(DECLARED_SOURCE),
            "declared_source_present": declared_source_exists,
            "available_older_unified_document": source_record(AVAILABLE_UNIFIED_SOURCE),
            "evidence_policy": "current checkout code + frozen pass manifests/results",
            "code": code,
            "cache_manifest": source_record(FORMAL_EMBEDDING_MANIFEST),
            "cache_plan_summary": source_record(FORMAL_CACHE_SUMMARY),
            "condition_split": source_record(FROZEN_CONDITIONS),
            "edge_split": source_record(FROZEN_EDGES),
        },
        "latent_contract": {
            "status": "confirmed",
            "internal_latent_tokens": ["B", 512, 768],
            "latents_L": 512,
            "one_cell_final_embedding": ["B", 768],
            "pooling": "final_norm(latent tokens).mean(dim=1)",
            "downstream_encoder": "EMA teacher",
            "checkpoint": checkpoint,
            "formal_cache": {
                "path": merged["output"]["path"],
                "sha256_from_completed_merge": merged["output"]["sha256"],
                "sha256_recomputed_this_audit": False,
                "size_bytes_verified": FORMAL_EMBEDDINGS.stat().st_size,
                "shape": merged["output"]["shape"],
                "dtype": merged["output"]["dtype"],
                "row_semantics": "one frozen physical Tahoe cell embedding",
                "row_equals_global_embedding_index": merged["output"]["row_equals_global_embedding_index"],
                "memmap_friendly": True,
                "finite": merged["integrity"]["finite"],
                "negative_ratio": full_cache["negative_ratio"],
                "min": full_cache["min"],
                "max": full_cache["max"],
            },
            "post_encoder_transforms": {
                "centering": False,
                "whitening": False,
                "l2_normalization": False,
                "clipping": False,
                "relu": False,
                "latent_standardization": False,
            },
            "st_contract": {
                "cell_set_dimension": 256,
                "input_output": ["B", 256, 768],
                "st_a": "Zpred = project_out(ST_hidden), absolute treated latent",
                "st_r": "Zpred = raw_Zctrl + project_out(ST_hidden), output-space residual",
                "best_predictor": "ST-A",
                "best_checkpoint": {
                    "path": "results/tahoe_experiment1_st_formal_checkpoints_v2/st-a/best.pt",
                    "exists": (RESULTS / "tahoe_experiment1_st_formal_checkpoints_v2/st-a/best.pt").is_file(),
                },
            },
        },
        "gene_and_encoder_preprocessing": gene_audit,
        "state_decoder_audit": {
            "LatentToGeneDecoder": {
                "input": "[..., latent_dim] (documented [B,D]; native Linear semantics also support [B,S,D])",
                "hidden": "per hidden dim: Linear -> LayerNorm -> GELU -> Dropout",
                "default_hidden_dims": [512, 1024],
                "output": "one whole gene vector [..., gene_dim]",
                "final_activation": "fixed ReLU in current checkout",
                "normalization": "LayerNorm after each hidden Linear; none at output",
                "residual_option": "optional hidden-block residual path",
                "loss_in_base_model": "same configured model loss; base path computes decoder loss against pert_cell_counts",
                "upstream_detach": "base PerturbationModel always detaches; StateTransition path detaches only when detach_decoder=True",
                "candidate_live_probe": state_probe,
                "reuse": "MLP body directly reusable; no pretrained latent-to-gene weights identified",
                "minimum_softplus_diff": "add final_activation={relu,softplus,identity}, default relu; activation modules have no state_dict parameters",
                "modified_this_round": False,
            },
            "SE_VCI_decoder": {
                "class": "FinetuneVCICountsDecoder",
                "gene_conditioned": True,
                "gene_embedding": "protein/ESM vectors projected by pretrained SE gene_embedding_layer; learned replacements for missing genes",
                "read_depth": "learnable scalar used when SE config enables RDA",
                "dataset_embedding": "optional ds_emb tail, default declared dimension 10",
                "output_grain": "binary decoder produces one scalar per cell-gene pair, reshaped to [B,S,G]",
                "additional_paths": "gene-vector post-projection + direct latent MLP residual, final ReLU",
                "external_checkpoint_and_config_required": True,
                "local_pretrained_decoder_found": bool(pretrained_files),
                "local_pretrained_matches": [str(path) for path in pretrained_files],
                "weights_reusable_for_genejepa": False,
                "reason": "SE/VCI cell latent basis/dimension and conditioning contract differ from GeneJEPA; no alignment or compatible checkpoint exists",
                "designs_borrowable": ["gene-conditioned decoding", "vectorized cell-gene pairing", "optional read-depth conditioning"],
                "code_directly_reusable_for_v1": False,
            },
        },
        "cache_locator_pairing": {
            "status": "pass",
            "cache": {
                "shape": merged["output"]["shape"],
                "dtype": merged["output"]["dtype"],
                "cells": merged["integrity"]["cells"],
                "mmap_opened": True,
                "file_size_bytes": FORMAL_EMBEDDINGS.stat().st_size,
                "row_equals_global_embedding_index": True,
            },
            "locator": {
                "version": cache_summary["locator"]["version"],
                "components": cache_summary["locator"]["canonical_components"],
                "deterministic_raw_lookup": True,
                "path": "worker0/worker1 plan Parquet",
            },
            "exact_plan_audit": split_audit,
            "raw_spotcheck": raw_spot,
        },
        "split_leakage": split_audit,
        "expression_target_audit": {
            "status": "candidate_supported_with_full-scan_checks_pending",
            "tahoe_expression_arrow_type": "list<float>",
            "semantic_evidence": "sparse non-negative count-like values after leading -2 sentinel; encoder applies log1p directly",
            "candidate_transform": "x_ig = log1p(10000 * count_ig / L_i)",
            "denominator": "sum raw counts over all post-sentinel mapped GeneJEPA genes",
            "all_tahoe_vs_mapped": "all usable post-sentinel genes map for every formal cache cell according to extraction manifests",
            "full_cache_unmapped_gene_count_max": gene_audit["sentinel_and_mapping_full_cache_evidence"]["unmapped_gene_count_max_per_cell"],
            "sentinel_removed_before_denominator": True,
            "panel_selected_after_full_library_normalization": True,
            "locator_sample_cells": raw_spot["cells"],
            "locator_sample_noninteger_entries": raw_spot["noninteger_count_entries"],
            "locator_sample_negative_after_sentinel": 0,
            "locator_sample_zero_library_cells": raw_spot["zero_library_cells"],
            "target_finite_and_nonnegative_in_dataset_smoke": True,
            "full_train_noninteger_and_zero_library_exact_counts": "pending exact panel scan",
            "noninteger_values_if_present": "accepted if finite/non-negative; CP10K is mathematically defined for count-like nonnegative floats",
        },
        "gene_panel_audit": {
            "preexisting_formal_panel_found": False,
            "preexisting_tahoe_train_only_variance_statistics_found": False,
            "search_scope": "current GeneJEPA checkout filenames/content excluding environments, raw cache, model binaries and smoke outputs",
            "candidate_contract": {
                "candidate_universe": 62_710,
                "statistics_scope": "Decoder train treated physical cells only",
                "expression_space": "candidate log1p(CP10000) target",
                "include_zeros": True,
                "ranking": "population variance descending",
                "top_k": 5000,
                "tie_breaker": "genejepa_index ascending",
            },
            "builder": {
                "implementation": "perturbation_scripts/tahoe_decoder_v1_data.py build-panel",
                "exact": True,
                "deterministic": True,
                "resumable": True,
                "accumulator_dtype": "float64 sum/sumsq",
                "accumulator_payload_bytes": 62_710 * 8 * 2,
                "accumulator_payload_MiB": 62_710 * 8 * 2 / 2**20,
                "selected_cells": 22_941_936,
                "physical_io": "genes/expressions columns from row groups in both frozen plans; selected train rows only enter accumulators",
                "expected_source_shards": 3388,
                "rough_wall_time": "approximately 7-15 hours on one CPU process; storage/parquet decoding dependent",
            },
            "smoke": smoke_panel,
            "smoke_resume_check": {
                "status": "pass",
                "rerun_processed_cells": smoke_resume_cells,
                "expected_cells": 256,
                "output_sha256_after_rerun": sha256_file(SMOKE_PANEL),
                "output_sha256_unchanged": sha256_file(SMOKE_PANEL)
                == smoke_panel["output"]["sha256"],
                "state_fingerprint_matches": True,
            },
            "formal_candidate_panel_path": display_path(DEFAULT_PANEL),
            "formal_candidate_panel_materialized": DEFAULT_PANEL.is_file(),
            "panel_pending": True,
        },
        "d2_dataset": {
            "implementation": source_record(PROJECT_ROOT / "perturbation_scripts/tahoe_decoder_v1_data.py"),
            "class": "TahoeDecoderV1PairedDataset",
            "unit": "one physical treated cell",
            "latent_lookup": "float32 mmap row at global_embedding_index",
            "target_lookup": "stable locator -> one Tahoe source row -> unified log1p_cp10k_target",
            "io_scalability": {
                "plan_order": "one plan row group per shard, strict physical row order",
                "parquet_open": "one source file per shard descriptor",
                "row_group_read": "once per selected source row group",
                "latent_read": "existing mmap; no embedding copy on disk",
                "ddp_and_loader_partition": "globally shuffled-or-canonical shard descriptors sliced disjointly across DDP rank x DataLoader worker",
                "per_cell_parquet_open": False,
            },
        },
        "d2_dataset_smoke": d2,
        "architecture_candidate_readiness": {
            "status": "reasonable first baseline; not frozen",
            "architecture": "768 -> 1024 -> 1024 -> 512 -> G, LayerNorm/GELU/Dropout(0.1), candidate Softplus",
            "G_candidate": 5000,
            "parameter_count": 4_931_976,
            "fp32_parameters_MiB": 4_931_976 * 4 / 2**20,
            "fp32_output_per_cell_KiB": 5000 * 4 / 2**10,
            "fp32_output_per_S256_set_MiB": 256 * 5000 * 4 / 2**20,
            "shape_issue": False,
            "dtype_issue": False,
            "memory_blocker": False,
            "state_minimal_reuse": "body yes; final activation option needed only after freeze",
            "state_modified": False,
        },
        "formal_training_requirements": formal_training_requirements,
        "conflicts": conflicts,
        "pending_decisions": pending_decisions,
        "blockers": [
            "Exact train-only top-5000 candidate panel has not been materialized/audited",
            "Candidate target/panel/activation/architecture/loss/validation criterion have not been human-frozen",
            "Declared v2.1 source-of-truth document is missing or must be explicitly waived",
        ],
        "ready_for_contract_freeze": False,
        "ready_for_formal_training": False,
    }
    contract = build_contract(audit)
    atomic_write_json(AUDIT_JSON, audit)
    atomic_write_json(CONTRACT_JSON, contract)
    write_markdown(audit, contract)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "audit_json": display_path(AUDIT_JSON),
                "audit_md": display_path(AUDIT_MD),
                "contract": display_path(CONTRACT_JSON),
                "split_cells": split_audit["treated_unique_cells_by_split"],
                "d2_smoke": d2["status"],
                "ready_for_contract_freeze": False,
                "ready_for_formal_training": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
