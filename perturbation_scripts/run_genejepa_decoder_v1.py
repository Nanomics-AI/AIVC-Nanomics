#!/usr/bin/env python3
"""Prepare, smoke, or formally train the frozen GeneJEPA Decoder v1."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn as nn
from state.tx.models.base import LatentToGeneDecoder
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, get_worker_info

from tahoe_decoder_v1_data import (
    FORMAL_EMBEDDING_MANIFEST,
    FORMAL_EMBEDDINGS,
    FORMAL_PLANS,
    GENE_METADATA,
    LATENT_DIM,
    OWNER_TREATED,
    PROJECT_ROOT,
    RESULTS,
    SPLIT_TO_CODE,
    TahoeDecoderV1PairedDataset,
    atomic_write_json,
    display_path,
    iter_descriptor_records,
    load_conditions,
    log1p_cp10k_target,
    plan_descriptors,
    sha256_file,
    utc_now,
)


SCRIPT_PATH = Path(__file__).resolve()
TASK_PATH = PROJECT_ROOT.parent / "当前任务.txt"
CONTRACT_PATH = RESULTS / "genejepa_decoder_v1_contract.json"
PANEL_PATH = RESULTS / "genejepa_decoder_v1_gene_panel.csv"
PANEL_AUDIT_PATH = RESULTS / "genejepa_decoder_v1_gene_panel_audit.json"
DATASET_PATH = PROJECT_ROOT / "perturbation_scripts/tahoe_decoder_v1_data.py"
STATE_DECODER_PATH = PROJECT_ROOT.parent / "state-main/src/state/tx/models/base.py"
CACHE_PLAN_SUMMARY = RESULTS / "tahoe_experiment1_cache_cap512_all_dmso_summary.json"
CONDITION_INDEX = RESULTS / "tahoe_experiment1_cache_cap512_all_dmso_condition_index.csv"
DESCRIPTOR_COUNTS_PATH = RESULTS / "genejepa_decoder_v1_descriptor_counts.npz"
PARTITION_AUDIT_PATH = RESULTS / "genejepa_decoder_v1_train_partition_audit.json"
TRAINING_CONFIG_PATH = RESULTS / "genejepa_decoder_v1_training_config.json"
SMOKE_RESULT_PATH = RESULTS / "genejepa_decoder_v1_exact_config_smoke.json"
SMOKE_CHECKPOINT_DIR = RESULTS / "genejepa_decoder_v1_exact_config_smoke_checkpoints"
FORMAL_CHECKPOINT_DIR = RESULTS / "genejepa_decoder_v1_checkpoints"
SMOKE_TENSORBOARD = RESULTS / "tensorboard/genejepa_decoder_v1/exact_config_smoke"
FORMAL_TENSORBOARD = RESULTS / "tensorboard/genejepa_decoder_v1/formal"
FORMAL_RESULT_PATH = RESULTS / "genejepa_decoder_v1_formal_training_result.json"

TASK_SHA256 = "a716f6692ba892b973a680b917c57b8f797a97e21cfd975135be978237d6099b"
PANEL_SHA256 = "c4f79cfb0a37ec278d04c3f0045fa568872961445a9f251fff37dfa3d38d790e"
EXPECTED_PARAMETER_COUNT = 4_931_976
GENE_DIM = 5_000
TRAIN_CELLS = 22_941_936
VAL_CELLS = 2_841_724
TEST_CELLS = 2_856_299
TREATED_CELLS = TRAIN_CELLS + VAL_CELLS + TEST_CELLS
DMSO_CELLS = 2_199_130
WORLD_SIZE = 2
PRIMARY_BATCH_SIZE = 1_024
PRIMARY_GRADIENT_ACCUMULATION = 1
FALLBACK_BATCH_SIZE = 512
FALLBACK_GRADIENT_ACCUMULATION = 2
EFFECTIVE_GLOBAL_BATCH = 2_048
FORMAL_STEPS_PER_EPOCH = 11_202
FORMAL_USED_CELLS = 22_941_696
FORMAL_TAIL_DROPPED = 240
FORMAL_CELLS_PER_RANK = 11_470_848
NUM_WORKERS_PER_RANK = 4
PREFETCH_FACTOR = 2
PIN_MEMORY = True
PERSISTENT_WORKERS = False
SEED = 42
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return display_path(path)


def artifact_record(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    record: dict[str, Any] = {"path": relative(path), "size_bytes": path.stat().st_size}
    if hash_file:
        record["sha256"] = sha256_file(path)
    return record


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_save_counts(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def frozen_inputs() -> dict[str, Any]:
    required = (
        TASK_PATH,
        CONTRACT_PATH,
        PANEL_PATH,
        PANEL_AUDIT_PATH,
        DATASET_PATH,
        STATE_DECODER_PATH,
        CACHE_PLAN_SUMMARY,
        CONDITION_INDEX,
        FORMAL_EMBEDDINGS,
        FORMAL_EMBEDDING_MANIFEST,
        GENE_METADATA,
        *FORMAL_PLANS,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen Decoder inputs: {missing}")
    contract = read_json(CONTRACT_PATH)
    panel_audit = read_json(PANEL_AUDIT_PATH)
    cache_summary = read_json(CACHE_PLAN_SUMMARY)
    cache_manifest = read_json(FORMAL_EMBEDDING_MANIFEST)
    if (
        contract.get("status") != "frozen"
        or contract.get("contract_frozen") is not True
        or contract.get("ready_for_exact_config_smoke") is not True
        or contract.get("ready_for_formal_training") is not False
    ):
        raise AssertionError("Decoder scientific contract is not at the frozen smoke-ready state")
    if sha256_file(PANEL_PATH) != PANEL_SHA256 or contract["panel"]["sha256"] != PANEL_SHA256:
        raise AssertionError("Frozen Decoder panel SHA changed")
    if panel_audit.get("status") != "pass" or panel_audit["panel"]["rows"] != GENE_DIM:
        raise AssertionError("Frozen Decoder panel audit is not PASS")
    if contract["dataset"]["implementation_sha256"] != sha256_file(DATASET_PATH):
        raise AssertionError("Frozen Decoder Dataset code changed")
    if contract["gene_universe"]["metadata_sha256"] != sha256_file(GENE_METADATA):
        raise AssertionError("Frozen GeneJEPA metadata changed")
    output = cache_manifest["output"]
    if (
        cache_manifest.get("status") != "pass"
        or output["shape"] != [30_839_089, LATENT_DIM]
        or output["dtype"] != "float32"
        or output.get("row_equals_global_embedding_index") is not True
    ):
        raise AssertionError("Formal latent cache manifest changed")
    workers = cache_summary["partition"]["workers"]
    for worker, plan in zip(workers, FORMAL_PLANS, strict=True):
        if plan.stat().st_size != worker["plan_size_bytes"]:
            raise AssertionError("Frozen locator plan size changed")
    return {
        "contract": contract,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "panel_audit": panel_audit,
        "cache_summary": cache_summary,
        "cache_manifest": cache_manifest,
    }


def build_decoder() -> LatentToGeneDecoder:
    seed_everything(SEED)
    model = LatentToGeneDecoder(
        latent_dim=LATENT_DIM,
        gene_dim=GENE_DIM,
        hidden_dims=[1024, 1024, 512],
        dropout=0.1,
        residual_decoder=False,
    )
    if not isinstance(model.decoder[-1], nn.ReLU):
        raise AssertionError("STATE LatentToGeneDecoder no longer has its expected default ReLU")
    model.decoder[-1] = nn.Softplus()
    assert_decoder(model)
    return model


def assert_decoder(model: LatentToGeneDecoder) -> None:
    if model.residual_decoder:
        raise AssertionError("Decoder v1 must not use residual blocks")
    expected_types = [
        nn.Linear,
        nn.LayerNorm,
        nn.GELU,
        nn.Dropout,
        nn.Linear,
        nn.LayerNorm,
        nn.GELU,
        nn.Dropout,
        nn.Linear,
        nn.LayerNorm,
        nn.GELU,
        nn.Dropout,
        nn.Linear,
        nn.Softplus,
    ]
    modules = list(model.decoder)
    if len(modules) != len(expected_types) or any(
        not isinstance(module, expected) for module, expected in zip(modules, expected_types)
    ):
        raise AssertionError("Decoder v1 module sequence changed")
    linears = [module for module in modules if isinstance(module, nn.Linear)]
    dimensions = [(module.in_features, module.out_features) for module in linears]
    if dimensions != [(768, 1024), (1024, 1024), (1024, 512), (512, 5000)]:
        raise AssertionError(f"Decoder v1 dimensions changed: {dimensions}")
    if any(module.p != 0.1 for module in modules if isinstance(module, nn.Dropout)):
        raise AssertionError("Decoder v1 dropout changed")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if parameters != EXPECTED_PARAMETER_COUNT or trainable != EXPECTED_PARAMETER_COUNT:
        raise AssertionError(f"Decoder v1 parameter count changed: {parameters}/{trainable}")


def descriptor_fingerprint(inputs: dict[str, Any]) -> dict[str, Any]:
    workers = inputs["cache_summary"]["partition"]["workers"]
    return {
        "schema": "genejepa_decoder_v1_descriptor_counts_v1",
        "plans": [
            {
                "path": relative(path),
                "sha256": worker["plan_sha256"],
                "size_bytes": worker["plan_size_bytes"],
            }
            for path, worker in zip(FORMAL_PLANS, workers, strict=True)
        ],
        "condition_index_sha256": sha256_file(CONDITION_INDEX),
        "dataset_code_sha256": sha256_file(DATASET_PATH),
    }


def scan_descriptor_counts(inputs: dict[str, Any]) -> tuple[dict[str, np.ndarray], float]:
    conditions = load_conditions(CONDITION_INDEX)
    split_codes = conditions["split"].map(SPLIT_TO_CODE).to_numpy(np.int8)
    descriptors = plan_descriptors(FORMAL_PLANS)
    arrays = {
        "plan_index": np.empty(len(descriptors), dtype=np.int8),
        "plan_row_group": np.empty(len(descriptors), dtype=np.int32),
        "train_cells": np.zeros(len(descriptors), dtype=np.int64),
        "val_cells": np.zeros(len(descriptors), dtype=np.int64),
        "test_cells": np.zeros(len(descriptors), dtype=np.int64),
        "dmso_cells": np.zeros(len(descriptors), dtype=np.int64),
        "total_cells": np.zeros(len(descriptors), dtype=np.int64),
    }
    plans = {index: pq.ParquetFile(path) for index, path in enumerate(FORMAL_PLANS)}
    started = time.perf_counter()
    for descriptor_index, (plan_index, row_group) in enumerate(descriptors):
        table = plans[plan_index].read_row_group(
            row_group, columns=["owner_type", "owner_index"]
        )
        owner_type = table["owner_type"].to_numpy(zero_copy_only=False)
        owner_index = table["owner_index"].to_numpy(zero_copy_only=False).astype(
            np.int64, copy=False
        )
        if np.any((owner_type != OWNER_TREATED) & (owner_type != 1)):
            raise AssertionError("Unknown cache owner_type")
        treated = owner_type == OWNER_TREATED
        treated_owner = owner_index[treated]
        if np.any((treated_owner < 0) | (treated_owner >= len(conditions))):
            raise AssertionError("Treated owner_index is out of range")
        counts = np.bincount(split_codes[treated_owner], minlength=3)
        arrays["plan_index"][descriptor_index] = plan_index
        arrays["plan_row_group"][descriptor_index] = row_group
        arrays["train_cells"][descriptor_index] = counts[SPLIT_TO_CODE["train"]]
        arrays["val_cells"][descriptor_index] = counts[SPLIT_TO_CODE["val"]]
        arrays["test_cells"][descriptor_index] = counts[SPLIT_TO_CODE["test"]]
        arrays["dmso_cells"][descriptor_index] = np.count_nonzero(~treated)
        arrays["total_cells"][descriptor_index] = len(owner_type)
        if (descriptor_index + 1) % 250 == 0:
            print(
                f"descriptor-count audit {descriptor_index + 1}/{len(descriptors)}",
                flush=True,
            )
    fingerprint = descriptor_fingerprint(inputs)
    arrays["fingerprint"] = np.asarray(
        hashlib.sha256(
            json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )
    return arrays, time.perf_counter() - started


def validate_descriptor_counts(
    arrays: dict[str, np.ndarray], inputs: dict[str, Any]
) -> None:
    descriptors = plan_descriptors(FORMAL_PLANS)
    if len(arrays["train_cells"]) != len(descriptors):
        raise AssertionError("Descriptor-count row count changed")
    observed_descriptors = list(
        zip(
            arrays["plan_index"].astype(int).tolist(),
            arrays["plan_row_group"].astype(int).tolist(),
            strict=True,
        )
    )
    if observed_descriptors != descriptors:
        raise AssertionError("Descriptor-count order changed")
    fingerprint = descriptor_fingerprint(inputs)
    expected_fingerprint = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if str(arrays["fingerprint"].item()) != expected_fingerprint:
        raise AssertionError("Descriptor-count input fingerprint changed")
    totals = {
        "train": int(arrays["train_cells"].sum()),
        "val": int(arrays["val_cells"].sum()),
        "test": int(arrays["test_cells"].sum()),
        "dmso": int(arrays["dmso_cells"].sum()),
        "all": int(arrays["total_cells"].sum()),
    }
    expected = {
        "train": TRAIN_CELLS,
        "val": VAL_CELLS,
        "test": TEST_CELLS,
        "dmso": DMSO_CELLS,
        "all": TREATED_CELLS + DMSO_CELLS,
    }
    if totals != expected:
        raise AssertionError(f"Descriptor-count totals changed: {totals} != {expected}")
    if np.any(
        arrays["train_cells"]
        + arrays["val_cells"]
        + arrays["test_cells"]
        + arrays["dmso_cells"]
        != arrays["total_cells"]
    ):
        raise AssertionError("Descriptor owner counts do not reconcile")


def ensure_descriptor_counts(inputs: dict[str, Any]) -> tuple[dict[str, np.ndarray], float]:
    if DESCRIPTOR_COUNTS_PATH.is_file():
        with np.load(DESCRIPTOR_COUNTS_PATH, allow_pickle=False) as source:
            arrays = {key: source[key].copy() for key in source.files}
        validate_descriptor_counts(arrays, inputs)
        return arrays, 0.0
    arrays, elapsed = scan_descriptor_counts(inputs)
    validate_descriptor_counts(arrays, inputs)
    atomic_save_counts(DESCRIPTOR_COUNTS_PATH, arrays)
    return arrays, elapsed


def epoch_descriptor_order(epoch: int, descriptor_count: int) -> np.ndarray:
    order = np.arange(descriptor_count, dtype=np.int32)
    np.random.Generator(np.random.PCG64(SEED + epoch)).shuffle(order)
    return order


def partition_layout(batch_size: int, gradient_accumulation: int) -> dict[str, Any]:
    if batch_size * WORLD_SIZE * gradient_accumulation != EFFECTIVE_GLOBAL_BATCH:
        raise AssertionError("Effective global batch changed")
    microbatches_per_rank = FORMAL_STEPS_PER_EPOCH * gradient_accumulation
    quotient, remainder = divmod(microbatches_per_rank, NUM_WORKERS_PER_RANK)
    worker_microbatches = [
        quotient + int(worker_id < remainder)
        for worker_id in range(NUM_WORKERS_PER_RANK)
    ]
    worker_cells = [value * batch_size for value in worker_microbatches]
    if sum(worker_cells) != FORMAL_CELLS_PER_RANK:
        raise AssertionError("Worker cell ownership does not sum to the frozen rank total")
    ranges: list[dict[str, int]] = []
    for rank in range(WORLD_SIZE):
        cursor = rank * FORMAL_CELLS_PER_RANK
        for worker_id, (microbatches, cells) in enumerate(
            zip(worker_microbatches, worker_cells, strict=True)
        ):
            ranges.append(
                {
                    "rank": rank,
                    "worker_id": worker_id,
                    "global_worker_id": rank * NUM_WORKERS_PER_RANK + worker_id,
                    "global_stream_start": cursor,
                    "global_stream_stop_exclusive": cursor + cells,
                    "cells": cells,
                    "microbatches": microbatches,
                }
            )
            cursor += cells
        if cursor != (rank + 1) * FORMAL_CELLS_PER_RANK:
            raise AssertionError("Rank ownership interval is not contiguous")
    return {
        "batch_size_per_gpu": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "effective_global_batch": EFFECTIVE_GLOBAL_BATCH,
        "optimizer_steps_per_epoch": FORMAL_STEPS_PER_EPOCH,
        "worker_microbatches_per_rank": worker_microbatches,
        "worker_cells_per_rank": worker_cells,
        "ranges": ranges,
    }


def rank_stride_counts(
    counts: np.ndarray, order: np.ndarray, *, batch_size: int | None
) -> dict[str, Any]:
    worker_cells = [int(counts[order[worker::8]].sum()) for worker in range(8)]
    if batch_size is None:
        worker_batches = None
        rank_batches = None
    else:
        worker_batches = [value // batch_size for value in worker_cells]
        rank_batches = [sum(worker_batches[:4]), sum(worker_batches[4:])]
    return {
        "worker_cells": worker_cells,
        "rank_cells": [sum(worker_cells[:4]), sum(worker_cells[4:])],
        "worker_full_batches": worker_batches,
        "rank_full_batches": rank_batches,
    }


def split_boundary_descriptors(counts: np.ndarray, order: np.ndarray) -> int:
    prefix = np.cumsum(counts[order], dtype=np.int64)
    descriptor_boundaries = set(prefix.tolist())
    boundaries = [
        item["global_stream_stop_exclusive"]
        for item in partition_layout(PRIMARY_BATCH_SIZE, 1)["ranges"][:-1]
    ]
    return sum(int(boundary not in descriptor_boundaries) for boundary in boundaries)


def partition_audit(arrays: dict[str, np.ndarray], scan_seconds: float) -> dict[str, Any]:
    train_counts = arrays["train_cells"]
    descriptor_count = len(train_counts)
    epoch0 = epoch_descriptor_order(0, descriptor_count)
    epoch0_repeat = epoch_descriptor_order(0, descriptor_count)
    epoch1 = epoch_descriptor_order(1, descriptor_count)
    if not np.array_equal(epoch0, epoch0_repeat) or np.array_equal(epoch0, epoch1):
        raise AssertionError("seed+epoch descriptor ordering is not deterministic and epoch-varying")
    old = rank_stride_counts(train_counts, epoch0, batch_size=PRIMARY_BATCH_SIZE)
    primary = partition_layout(PRIMARY_BATCH_SIZE, PRIMARY_GRADIENT_ACCUMULATION)
    fallback = partition_layout(FALLBACK_BATCH_SIZE, FALLBACK_GRADIENT_ACCUMULATION)
    validation = rank_stride_counts(
        arrays["val_cells"], np.arange(descriptor_count, dtype=np.int32), batch_size=None
    )
    if sum(validation["rank_cells"]) != VAL_CELLS:
        raise AssertionError("Validation descriptor union changed")
    ranges = primary["ranges"]
    if ranges[0]["global_stream_start"] != 0 or ranges[-1][
        "global_stream_stop_exclusive"
    ] != FORMAL_USED_CELLS:
        raise AssertionError("Formal train ownership does not cover the frozen global prefix")
    if any(
        left["global_stream_stop_exclusive"] != right["global_stream_start"]
        for left, right in zip(ranges, ranges[1:])
    ):
        raise AssertionError("Formal train ownership ranges overlap or have gaps")
    result = {
        "schema": "genejepa_decoder_v1_train_partition_audit_v1",
        "created_at_utc": utc_now(),
        "status": "pass",
        "descriptor_counts": {
            **artifact_record(DESCRIPTOR_COUNTS_PATH),
            "descriptors": descriptor_count,
            "scan_seconds_this_invocation": scan_seconds,
            "train_cells": int(arrays["train_cells"].sum()),
            "val_cells": int(arrays["val_cells"].sum()),
            "test_cells": int(arrays["test_cells"].sum()),
            "dmso_cells": int(arrays["dmso_cells"].sum()),
            "total_cells": int(arrays["total_cells"].sum()),
        },
        "current_descriptor_stride_epoch0": {
            **old,
            "safe_for_DDP_training": old["rank_full_batches"][0]
            == old["rank_full_batches"][1],
            "reason": "descriptor cell counts are unequal, so strided descriptor ownership does not guarantee rank step parity",
        },
        "formal_global_stream": {
            "definition": "epoch-specific deterministic descriptor order, physical row order within each descriptor",
            "shuffle": "PCG64(seed + epoch), seed=42",
            "epoch0_order_sha256": hashlib.sha256(epoch0.astype("<i4").tobytes()).hexdigest(),
            "epoch1_order_sha256": hashlib.sha256(epoch1.astype("<i4").tobytes()).hexdigest(),
            "same_seed_epoch_reproduces_exactly": True,
            "different_epoch_changes_order": True,
            "ownership": "rank-major contiguous global stream ranges; worker ranges are integral microbatch counts",
            "physical_locality": "whole descriptors except at at most seven rank/worker boundaries",
            "epoch0_boundary_descriptors_split_between_workers": split_boundary_descriptors(
                train_counts, epoch0
            ),
            "formal_train_total_cells": TRAIN_CELLS,
            "formal_global_batch": EFFECTIVE_GLOBAL_BATCH,
            "formal_steps_per_epoch": FORMAL_STEPS_PER_EPOCH,
            "formal_used_cells_per_epoch": FORMAL_USED_CELLS,
            "formal_global_tail_dropped": FORMAL_TAIL_DROPPED,
            "dropped_tail_policy": "only global stream positions [22,941,696, 22,941,936)",
            "duplicates_for_padding": 0,
            "rank_overlap": 0,
            "rank_union_cells": FORMAL_USED_CELLS,
        },
        "primary_1024": primary,
        "oom_fallback_512": fallback,
        "rank_parity": {
            "rank0_formal_cells_per_epoch": FORMAL_CELLS_PER_RANK,
            "rank1_formal_cells_per_epoch": FORMAL_CELLS_PER_RANK,
            "rank0_formal_optimizer_steps": FORMAL_STEPS_PER_EPOCH,
            "rank1_formal_optimizer_steps": FORMAL_STEPS_PER_EPOCH,
            "optimizer_step_parity": True,
        },
        "formal_validation_descriptor_ownership": {
            **validation,
            "descriptor_assignment": "unshuffled descriptor_index stride across rank x DataLoader worker",
            "rank_overlap": 0,
            "union_cells": VAL_CELLS,
            "all_validation_cells_exactly_once": True,
        },
        "input_provenance": {
            "task_sha256": sha256_file(TASK_PATH),
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "panel_sha256": sha256_file(PANEL_PATH),
            "dataset_code_sha256": sha256_file(DATASET_PATH),
            "runner_sha256": sha256_file(SCRIPT_PATH),
        },
        "test_dataset_constructed": False,
    }
    if PARTITION_AUDIT_PATH.is_file():
        existing = read_json(PARTITION_AUDIT_PATH)
        comparable_existing = json.loads(json.dumps(existing))
        comparable_result = json.loads(json.dumps(result))
        for value in (comparable_existing, comparable_result):
            value.pop("created_at_utc", None)
            value["descriptor_counts"]["scan_seconds_this_invocation"] = 0.0
        if comparable_existing != comparable_result:
            raise AssertionError("Existing Decoder partition audit differs from current inputs")
        return existing
    atomic_write_json(PARTITION_AUDIT_PATH, result)
    return result


class ExactTrainTahoeDecoderDataset(TahoeDecoderV1PairedDataset):
    """Use contiguous, exact-size slices of one global epoch descriptor stream."""

    def __init__(
        self,
        descriptor_train_counts: np.ndarray,
        *,
        batch_size: int,
        gradient_accumulation: int,
    ) -> None:
        super().__init__(split="train", panel_path=PANEL_PATH, shuffle_shards=False)
        self.descriptor_train_counts = np.asarray(descriptor_train_counts, dtype=np.int64)
        if self.descriptor_train_counts.shape != (len(self.descriptors),):
            raise AssertionError("Train descriptor counts do not match Dataset descriptors")
        self.batch_size = int(batch_size)
        self.gradient_accumulation = int(gradient_accumulation)
        self.layout = partition_layout(self.batch_size, self.gradient_accumulation)

    def __len__(self) -> int:
        return FORMAL_CELLS_PER_RANK

    def _sample(self, record: dict[str, Any], embeddings: np.memmap) -> dict[str, Any]:
        embedding_index = record["embedding_index"]
        if embedding_index < 0 or embedding_index >= len(embeddings):
            raise AssertionError("global embedding_index is out of range")
        latent = np.array(embeddings[embedding_index], dtype=np.float32, copy=True)
        target, target_audit = log1p_cp10k_target(
            record["genes"],
            record["expressions"],
            self.token_lookup,
            self.panel_indices,
            panel_lookup=self.panel_lookup,
        )
        condition = record["condition"]
        return {
            "latent": torch.from_numpy(latent),
            "target_expression": torch.from_numpy(target),
            "global_embedding_index": embedding_index,
            "split": "train",
            "condition_id": str(condition["pair_id"]),
            "pair_id": str(condition["pair_id"]),
            "edge_id": str(condition["edge_id"]),
            "cell_line_id": str(condition["cell_line_id"]),
            "drug": str(condition["drug"]),
            "dose_uM": float(condition["dose_uM"]),
            "plate": str(condition["plate"]),
            "sample": str(record["sample"]),
            "BARCODE_SUB_LIB_ID": str(record["BARCODE_SUB_LIB_ID"]),
            "shard_path": str(record["shard_path"]),
            "row_group_index": int(record["row_group_index"]),
            "row_index_in_row_group": int(record["row_index_in_row_group"]),
            "row_index_in_shard": int(record["row_index_in_shard"]),
            "target_library_size": float(target_audit["library_size"]),
        }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        if worker is None or worker.num_workers != NUM_WORKERS_PER_RANK:
            raise RuntimeError("Exact Decoder training requires four DataLoader workers per rank")
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size != WORLD_SIZE or rank not in (0, 1):
            raise RuntimeError("Exact Decoder training requires two torchrun ranks")
        global_worker_id = rank * NUM_WORKERS_PER_RANK + worker.id
        ownership = self.layout["ranges"][global_worker_id]
        start = ownership["global_stream_start"]
        stop = ownership["global_stream_stop_exclusive"]
        order = epoch_descriptor_order(self.epoch, len(self.descriptors))
        plans = {index: pq.ParquetFile(path) for index, path in enumerate(self.plan_paths)}
        embeddings = self._mmap()
        cursor = 0
        emitted = 0
        for descriptor_index in order:
            count = int(self.descriptor_train_counts[descriptor_index])
            descriptor_start, descriptor_stop = cursor, cursor + count
            cursor = descriptor_stop
            overlap_start = max(start, descriptor_start)
            overlap_stop = min(stop, descriptor_stop)
            if overlap_start >= overlap_stop:
                continue
            plan_index, plan_row_group = self.descriptors[int(descriptor_index)]
            records = iter_descriptor_records(
                plan=plans[plan_index],
                plan_row_group=plan_row_group,
                condition_records=self.condition_records,
                condition_split_codes=self.condition_split_codes,
                split="train",
            )
            local_start = overlap_start - descriptor_start
            local_stop = overlap_stop - descriptor_start
            for record in itertools.islice(records, local_start, local_stop):
                yield self._sample(record, embeddings)
                emitted += 1
        if cursor != TRAIN_CELLS or emitted != ownership["cells"]:
            raise AssertionError(
                f"Worker {global_worker_id} emitted {emitted}, expected {ownership['cells']}"
            )


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(
    dataset: TahoeDecoderV1PairedDataset,
    *,
    batch_size: int,
    train: bool,
    epoch: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(SEED + epoch)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=NUM_WORKERS_PER_RANK,
        pin_memory=PIN_MEMORY,
        drop_last=train,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTENT_WORKERS,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def model_contract() -> dict[str, Any]:
    model = build_decoder()
    result = {
        "source_class": "state.tx.models.base.LatentToGeneDecoder",
        "source_path": relative(STATE_DECODER_PATH),
        "source_sha256": sha256_file(STATE_DECODER_PATH),
        "reuse": "instantiate non-residual STATE decoder and replace only its terminal ReLU with Softplus",
        "dimensions": [768, 1024, 1024, 512, 5000],
        "hidden_block": "Linear -> LayerNorm -> GELU -> Dropout(0.1)",
        "residual_decoder": False,
        "final_activation": "Softplus(beta=1, threshold=20)",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    del model
    return result


def training_config_payload(
    inputs: dict[str, Any], partition: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema": "genejepa_decoder_v1_training_config_v1",
        "created_at_utc": utc_now(),
        "status": "frozen",
        "scientific_contract": {
            "path": relative(CONTRACT_PATH),
            "sha256": inputs["contract_sha256"],
            "status": "frozen",
        },
        "panel": {
            "path": relative(PANEL_PATH),
            "sha256": PANEL_SHA256,
            "genes": GENE_DIM,
        },
        "model": model_contract(),
        "data": {
            "dataset": "TahoeDecoderV1PairedDataset + exact train ownership adapter in this runner",
            "dataset_path": relative(DATASET_PATH),
            "dataset_sha256": sha256_file(DATASET_PATH),
            "latent": "raw signed frozen GeneJEPA Epoch25 EMA-teacher [768]",
            "target": "log1p(CP10000) [5000]",
            "train_treated_cells": TRAIN_CELLS,
            "val_treated_cells": VAL_CELLS,
            "test_treated_cells": TEST_CELLS,
            "DMSO_used": False,
            "test_dataset_constructed": False,
            "descriptor_counts": artifact_record(DESCRIPTOR_COUNTS_PATH),
            "partition_audit": artifact_record(PARTITION_AUDIT_PATH),
        },
        "training": {
            "world_size": WORLD_SIZE,
            "batch_size_per_gpu": PRIMARY_BATCH_SIZE,
            "global_batch": EFFECTIVE_GLOBAL_BATCH,
            "gradient_accumulation": PRIMARY_GRADIENT_ACCUMULATION,
            "precision": "BF16 autocast forward; FP32 prediction/target MSE",
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "scheduler": None,
            "seed": SEED,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience_full_val_epochs": EARLY_STOPPING_PATIENCE,
            "steps_per_epoch": FORMAL_STEPS_PER_EPOCH,
            "used_cells_per_epoch": FORMAL_USED_CELLS,
            "global_tail_dropped": FORMAL_TAIL_DROPPED,
            "rank_cells_per_epoch": [FORMAL_CELLS_PER_RANK, FORMAL_CELLS_PER_RANK],
            "rank_optimizer_steps": [FORMAL_STEPS_PER_EPOCH, FORMAL_STEPS_PER_EPOCH],
        },
        "oom_fallback_only": {
            "batch_size_per_gpu": FALLBACK_BATCH_SIZE,
            "gradient_accumulation": FALLBACK_GRADIENT_ACCUMULATION,
            "effective_global_batch": EFFECTIVE_GLOBAL_BATCH,
        },
        "dataloader": {
            "workers_per_rank": NUM_WORKERS_PER_RANK,
            "pin_memory": PIN_MEMORY,
            "non_blocking_transfer": True,
            "prefetch_factor": PREFETCH_FACTOR,
            "persistent_workers": PERSISTENT_WORKERS,
        },
        "validation": {
            "criterion": "global SSE / global cell-by-gene element count",
            "prediction_and_target": "FP32",
            "SSE_accumulation": "FP32 squared differences summed in float64",
            "DDP": "SUM local_sse and SUM local_element_count",
            "full_validation_cells": VAL_CELLS,
            "full_validation_elements": VAL_CELLS * GENE_DIM,
            "shuffle": False,
            "drop_last": False,
            "strict_minimum_checkpoint": True,
            "exact_tie_keeps_earlier_epoch": True,
        },
        "checkpoint": {
            "formal_best": relative(FORMAL_CHECKPOINT_DIR / "best.pt"),
            "formal_last": relative(FORMAL_CHECKPOINT_DIR / "last.pt"),
            "atomic": True,
            "resume_guarantee": "complete epoch boundaries in formal mode; optimizer-step boundary in smoke",
        },
        "tensorboard": {
            "rank0_only": True,
            "formal_path": relative(FORMAL_TENSORBOARD),
            "smoke_path": relative(SMOKE_TENSORBOARD),
        },
        "distributed": {
            "launcher": "torchrun --standalone --nproc_per_node=2",
            "backend": "nccl",
            "NCCL_CUMEM_ENABLE": "0",
            "NCCL_CUMEM_HOST_ENABLE": "0",
            "DDP_join": False,
        },
        "implementation": {
            "runner_path": relative(SCRIPT_PATH),
            "runner_sha256": sha256_file(SCRIPT_PATH),
            "task_sha256": sha256_file(TASK_PATH),
        },
        "formal_training_gate": {
            "artifact": relative(SMOKE_RESULT_PATH),
            "required": "status=pass and ready_for_formal_training=true",
        },
    }


def prepare() -> dict[str, Any]:
    if sha256_file(TASK_PATH) != TASK_SHA256:
        raise AssertionError("当前任务.txt changed before Decoder trainer preparation")
    inputs = frozen_inputs()
    arrays, scan_seconds = ensure_descriptor_counts(inputs)
    partition = partition_audit(arrays, scan_seconds)
    config = training_config_payload(inputs, partition)
    if TRAINING_CONFIG_PATH.exists():
        existing = read_json(TRAINING_CONFIG_PATH)
        existing.pop("created_at_utc", None)
        comparable = dict(config)
        comparable.pop("created_at_utc", None)
        if existing != comparable:
            raise AssertionError("Existing Decoder training config differs from current frozen inputs")
    else:
        atomic_write_json(TRAINING_CONFIG_PATH, config)
    result = {
        "status": "pass",
        "scope": "trainer preparation and exact ownership audit; no GPU/model training/test Dataset",
        "training_config": artifact_record(TRAINING_CONFIG_PATH),
        "partition_audit": artifact_record(PARTITION_AUDIT_PATH),
        "descriptor_counts": artifact_record(DESCRIPTOR_COUNTS_PATH),
        "model": config["model"],
        "current_stride_safe": partition["current_descriptor_stride_epoch0"][
            "safe_for_DDP_training"
        ],
        "formal_rank_parity": partition["rank_parity"],
        "ready_for_exact_config_smoke": True,
        "ready_for_formal_training": False,
        "formal_training_started": False,
        "test_dataset_constructed": False,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def load_runtime_contract() -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    inputs = frozen_inputs()
    if not TRAINING_CONFIG_PATH.is_file() or not PARTITION_AUDIT_PATH.is_file():
        raise FileNotFoundError("Run the single-process prepare command first")
    config = read_json(TRAINING_CONFIG_PATH)
    if config["scientific_contract"]["sha256"] != inputs["contract_sha256"]:
        raise AssertionError("Training config contract SHA changed")
    if config["panel"]["sha256"] != PANEL_SHA256:
        raise AssertionError("Training config panel SHA changed")
    if config["implementation"]["runner_sha256"] != sha256_file(SCRIPT_PATH):
        raise AssertionError("Production runner changed after training config freeze")
    if config["data"]["dataset_sha256"] != sha256_file(DATASET_PATH):
        raise AssertionError("Dataset code changed after training config freeze")
    if config["data"]["partition_audit"]["sha256"] != sha256_file(
        PARTITION_AUDIT_PATH
    ):
        raise AssertionError("Partition audit changed after training config freeze")
    with np.load(DESCRIPTOR_COUNTS_PATH, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    validate_descriptor_counts(arrays, inputs)
    if config["data"]["descriptor_counts"]["sha256"] != sha256_file(
        DESCRIPTOR_COUNTS_PATH
    ):
        raise AssertionError("Descriptor counts changed after training config freeze")
    return config, arrays, sha256_file(TRAINING_CONFIG_PATH)


def setup_distributed() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != WORLD_SIZE:
        raise RuntimeError("Decoder v1 requires exactly two torchrun ranks")
    if os.environ.get("NCCL_CUMEM_ENABLE") != "0" or os.environ.get(
        "NCCL_CUMEM_HOST_ENABLE"
    ) != "0":
        raise RuntimeError("Set both NCCL CUMEM workaround variables to 0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("Decoder v1 requires exactly two visible CUDA GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Visible GPU does not support BF16")
    return rank, local_rank, device


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": [value.cpu() for value in torch.cuda.get_rng_state_all()],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def gather_rng_states() -> list[dict[str, Any]]:
    gathered: list[dict[str, Any] | None] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, capture_rng_state())
    if any(item is None for item in gathered):
        raise AssertionError("Failed to gather both rank RNG states")
    return [item for item in gathered if item is not None]


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def model_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gradient_norm(model: nn.Module) -> float:
    squares = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is not None:
            squares += parameter.grad.detach().double().square().sum()
    return float(squares.sqrt())


def exact_parameter_sync_error(model: nn.Module, device: torch.device) -> float:
    maximum = torch.zeros((), device=device, dtype=torch.float32)
    for parameter in model.parameters():
        reference = parameter.detach().clone()
        dist.broadcast(reference, src=0)
        maximum = torch.maximum(maximum, (parameter.detach() - reference).abs().max())
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return float(maximum)


def loader_audit(loader: DataLoader, iterator: Any, batch: dict[str, Any]) -> dict[str, Any]:
    workers = list(getattr(iterator, "_workers", []))
    pids = [int(worker.pid) for worker in workers if worker.pid is not None]
    alive = [bool(worker.is_alive()) for worker in workers]
    result = {
        "configured_workers": loader.num_workers,
        "active_worker_processes": len(pids),
        "worker_pids": pids,
        "all_workers_alive": bool(alive) and all(alive),
        "pin_memory": loader.pin_memory,
        "latent_pinned": bool(batch["latent"].is_pinned()),
        "target_pinned": bool(batch["target_expression"].is_pinned()),
        "prefetch_factor": loader.prefetch_factor,
        "persistent_workers": loader.persistent_workers,
    }
    if (
        result["configured_workers"] != NUM_WORKERS_PER_RANK
        or result["active_worker_processes"] != NUM_WORKERS_PER_RANK
        or not result["all_workers_alive"]
        or result["pin_memory"] is not True
        or not result["latent_pinned"]
        or not result["target_pinned"]
        or result["prefetch_factor"] != PREFETCH_FACTOR
        or result["persistent_workers"] is not False
    ):
        raise AssertionError(f"DataLoader runtime contract failed: {result}")
    return result


def forward_mse(
    model: nn.Module,
    latent: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    linear_output_dtypes: list[str] = []

    def record_linear_dtype(
        _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        linear_output_dtypes.append(str(output.dtype).removeprefix("torch."))

    hooks = [
        module.register_forward_hook(record_linear_dtype)
        for module in model.modules()
        if isinstance(module, nn.Linear)
    ]
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            autocast_enabled = torch.is_autocast_enabled("cuda")
            prediction = model(latent)
    finally:
        for hook in hooks:
            hook.remove()
    prediction_fp32 = prediction.float()
    target_fp32 = target.float()
    if prediction.shape != target.shape or tuple(prediction.shape[1:]) != (GENE_DIM,):
        raise AssertionError("Decoder prediction/target shape is not [B,5000]")
    if len(linear_output_dtypes) != 4 or set(linear_output_dtypes) != {"bfloat16"}:
        raise AssertionError(
            f"Decoder Linear path did not execute in BF16: {linear_output_dtypes}"
        )
    if prediction.dtype not in (torch.bfloat16, torch.float32):
        raise AssertionError(f"Unexpected Decoder prediction dtype: {prediction.dtype}")
    if (
        not autocast_enabled
        or not torch.isfinite(prediction_fp32).all()
        or torch.any(prediction_fp32 < 0)
        or not torch.isfinite(target_fp32).all()
        or torch.any(target_fp32 < 0)
    ):
        raise AssertionError("Decoder forward/target contract failed")
    loss = torch.mean((prediction_fp32 - target_fp32).square())
    if not torch.isfinite(loss):
        raise AssertionError("Decoder FP32 MSE is non-finite")
    return prediction, loss, {
        "autocast_enabled": True,
        "autocast_dtype": "bfloat16",
        "linear_output_dtypes": linear_output_dtypes,
        "linear_layers_bfloat16": True,
        "prediction_dtype": str(prediction.dtype).removeprefix("torch."),
        "loss_prediction_dtype": "float32",
        "target_dtype": "float32",
        "shape": list(prediction.shape),
        "softplus_finite_nonnegative": True,
    }


def train_segment(
    model: DistributedDataParallel,
    raw_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset: ExactTrainTahoeDecoderDataset,
    device: torch.device,
    *,
    epoch: int,
    start_optimizer_step_in_epoch: int,
    optimizer_steps_to_run: int,
    batch_size: int,
    gradient_accumulation: int,
    rank: int,
    writer: Any | None,
    global_step: int,
    collect_indices: bool,
) -> dict[str, Any]:
    dataset.set_epoch(epoch)
    loader = make_loader(dataset, batch_size=batch_size, train=True, epoch=epoch)
    expected_microbatches = FORMAL_STEPS_PER_EPOCH * gradient_accumulation
    if len(loader) != expected_microbatches or not loader.drop_last:
        raise AssertionError(f"Exact train DataLoader length changed: {len(loader)}")
    iterator = iter(loader)
    skip_microbatches = start_optimizer_step_in_epoch * gradient_accumulation
    for _ in range(skip_microbatches):
        next(iterator)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    step_times: list[float] = []
    grad_norms: list[tuple[float, float]] = []
    consumed_indices: list[int] = []
    first_loader_audit: dict[str, Any] | None = None
    forward_audit: dict[str, Any] | None = None
    initial_fingerprint = model_fingerprint(raw_model)
    for optimizer_offset in range(optimizer_steps_to_run):
        window_started = time.perf_counter()
        window_losses: list[float] = []
        for microbatch_offset in range(gradient_accumulation):
            raw_batch = next(iterator)
            if first_loader_audit is None:
                first_loader_audit = loader_audit(loader, iterator, raw_batch)
            if set(raw_batch["split"]) != {"train"}:
                raise AssertionError("Non-train cell entered Decoder fitting")
            if raw_batch["latent"].shape != (batch_size, LATENT_DIM) or raw_batch[
                "target_expression"
            ].shape != (batch_size, GENE_DIM):
                raise AssertionError("Decoder train batch shape changed")
            if collect_indices:
                consumed_indices.extend(
                    raw_batch["global_embedding_index"].to(torch.int64).tolist()
                )
            latent = raw_batch["latent"].to(device, non_blocking=True)
            target = raw_batch["target_expression"].to(device, non_blocking=True)
            synchronization = (
                nullcontext()
                if microbatch_offset + 1 == gradient_accumulation
                else model.no_sync()
            )
            with synchronization:
                _prediction, loss, current_forward = forward_mse(model, latent, target)
                (loss / gradient_accumulation).backward()
            forward_audit = current_forward
            report_loss = loss.detach().clone()
            dist.all_reduce(report_loss, op=dist.ReduceOp.SUM)
            report_loss /= WORLD_SIZE
            window_losses.append(float(report_loss))
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise AssertionError("Decoder gradient became non-finite")
        before_clip = gradient_norm(model)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRADIENT_CLIP_NORM, error_if_nonfinite=True
        )
        after_clip = gradient_norm(model)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        mean_loss = float(np.mean(window_losses))
        losses.append(mean_loss)
        grad_norms.append((before_clip, after_clip))
        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - window_started)
        if rank == 0 and writer is not None:
            writer.add_scalar("train/loss", mean_loss, global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("train/grad_norm", after_clip, global_step)
            writer.add_scalar("train/global_step", global_step, global_step)
    final_fingerprint = model_fingerprint(raw_model)
    if final_fingerprint == initial_fingerprint:
        raise AssertionError("Decoder parameters did not update")
    if collect_indices and len(consumed_indices) != (
        optimizer_steps_to_run * batch_size * gradient_accumulation
    ):
        raise AssertionError("Smoke train index count changed")
    steady_times = step_times[1:] if len(step_times) > 1 else step_times
    mean_step_time = float(np.mean(steady_times))
    return {
        "losses": losses,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "all_losses_finite": bool(np.isfinite(losses).all()),
        "all_gradients_finite": True,
        "gradient_norm_before_clip_last": grad_norms[-1][0],
        "gradient_norm_after_clip_last": grad_norms[-1][1],
        "parameter_updated": True,
        "start_optimizer_step_in_epoch": start_optimizer_step_in_epoch,
        "end_optimizer_step_in_epoch": start_optimizer_step_in_epoch
        + optimizer_steps_to_run,
        "optimizer_steps": optimizer_steps_to_run,
        "global_step": global_step,
        "mean_steady_step_seconds": mean_step_time,
        "global_samples_per_second": EFFECTIVE_GLOBAL_BATCH / mean_step_time,
        "global_batches_per_second": 1.0 / mean_step_time,
        "consumed_indices": consumed_indices,
        "dataloader": first_loader_audit,
        "forward": forward_audit,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataset: TahoeDecoderV1PairedDataset,
    device: torch.device,
    *,
    batch_size: int,
    max_batches: int | None,
    expected_rank_cells: int,
    rank: int,
) -> dict[str, Any]:
    dataset.set_epoch(0)
    loader = make_loader(dataset, batch_size=batch_size, train=False, epoch=0)
    if loader.drop_last:
        raise AssertionError("Validation must use drop_last=False")
    iterator = iter(loader)
    was_training = model.training
    model.eval()
    local_sse = torch.zeros((), device=device, dtype=torch.float64)
    local_elements = torch.zeros((), device=device, dtype=torch.int64)
    local_cells = 0
    batches = 0
    local_indices: list[int] = []
    loader_runtime: dict[str, Any] | None = None
    forward_audit: dict[str, Any] | None = None
    while max_batches is None or batches < max_batches:
        try:
            raw_batch = next(iterator)
        except StopIteration:
            break
        if loader_runtime is None:
            loader_runtime = loader_audit(loader, iterator, raw_batch)
        if set(raw_batch["split"]) != {"val"}:
            raise AssertionError("Non-val cell entered Decoder validation")
        latent = raw_batch["latent"].to(device, non_blocking=True)
        target = raw_batch["target_expression"].to(device, non_blocking=True)
        prediction, _batch_mse, forward_audit = forward_mse(model, latent, target)
        difference = prediction.float() - target.float()
        local_sse += difference.square().sum(dtype=torch.float64)
        local_elements += difference.numel()
        local_cells += len(latent)
        batches += 1
        if max_batches is not None:
            local_indices.extend(raw_batch["global_embedding_index"].to(torch.int64).tolist())
    if was_training:
        model.train()
    if max_batches is None and local_cells != expected_rank_cells:
        raise AssertionError(
            f"Formal validation rank{rank} cells {local_cells} != {expected_rank_cells}"
        )
    gathered_local: list[dict[str, Any] | None] = [None] * WORLD_SIZE
    dist.all_gather_object(
        gathered_local,
        {
            "rank": rank,
            "cells": local_cells,
            "elements": int(local_elements),
            "sse": float(local_sse),
            "batches": batches,
            "indices": local_indices,
        },
    )
    complete = [item for item in gathered_local if item is not None]
    dist.all_reduce(local_sse, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_elements, op=dist.ReduceOp.SUM)
    global_sse = float(local_sse)
    global_elements = int(local_elements)
    global_mse = global_sse / global_elements
    reference_sse = sum(item["sse"] for item in complete)
    reference_elements = sum(item["elements"] for item in complete)
    reduction_correct = global_elements == reference_elements and math.isclose(
        global_sse, reference_sse, rel_tol=0.0, abs_tol=1e-8
    )
    if not reduction_correct or not math.isfinite(global_mse):
        raise AssertionError("Validation SSE/count all_reduce is incorrect or non-finite")
    if max_batches is None and global_elements != VAL_CELLS * GENE_DIM:
        raise AssertionError("Formal global validation element count changed")
    if max_batches is not None:
        rank_sets = [set(item["indices"]) for item in complete]
        overlap = len(rank_sets[0] & rank_sets[1])
        if overlap:
            raise AssertionError("Smoke validation ranks overlap")
    else:
        overlap = 0
    return {
        "mode": "formal_full" if max_batches is None else "smoke_capped",
        "max_batches_per_rank": max_batches,
        "per_rank": [
            {key: value for key, value in item.items() if key != "indices"}
            for item in complete
        ],
        "global_cells": sum(item["cells"] for item in complete),
        "global_elements": global_elements,
        "expected_full_global_elements": VAL_CELLS * GENE_DIM,
        "global_sse": global_sse,
        "global_val_mse": global_mse,
        "local_sse_count_all_reduce_correct": reduction_correct,
        "rank_index_overlap": overlap,
        "shuffle": False,
        "drop_last": False,
        "finite": True,
        "forward": forward_audit,
        "dataloader_per_rank_current": loader_runtime,
    }


def checkpoint_paths(mode: str) -> tuple[Path, Path]:
    root = SMOKE_CHECKPOINT_DIR if mode == "smoke" else FORMAL_CHECKPOINT_DIR
    return root / "best.pt", root / "last.pt"


def checkpoint_payload(
    *,
    mode: str,
    kind: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step_in_epoch: int,
    epoch_complete: bool,
    global_step: int,
    best_val_mse: float,
    best_epoch: int,
    early_stop_counter: int,
    training_config_sha256: str,
    batch_size: int,
    gradient_accumulation: int,
    rng_states: list[dict[str, Any]],
    history: list[dict[str, Any]],
    smoke_indices_by_rank: list[list[int]] | None,
) -> dict[str, Any]:
    return {
        "schema": "genejepa_decoder_v1_checkpoint_v1",
        "mode": mode,
        "checkpoint_kind": kind,
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "epoch_complete": epoch_complete,
        "global_optimizer_step": global_step,
        "model_state": cpu_tree(model.state_dict()),
        "optimizer_state": cpu_tree(optimizer.state_dict()),
        "model_fingerprint": model_fingerprint(model),
        "best_val_mse": best_val_mse,
        "best_epoch": best_epoch,
        "early_stop_counter": early_stop_counter,
        "training_config": {
            "path": relative(TRAINING_CONFIG_PATH),
            "sha256": training_config_sha256,
        },
        "frozen_contract_sha256": sha256_file(CONTRACT_PATH),
        "panel_sha256": PANEL_SHA256,
        "dataset_code_sha256": sha256_file(DATASET_PATH),
        "runner_sha256": sha256_file(SCRIPT_PATH),
        "descriptor_counts_sha256": sha256_file(DESCRIPTOR_COUNTS_PATH),
        "batch_size_per_gpu": batch_size,
        "world_size": WORLD_SIZE,
        "gradient_accumulation": gradient_accumulation,
        "effective_global_batch": EFFECTIVE_GLOBAL_BATCH,
        "precision": "bfloat16_autocast_forward_fp32_mse",
        "optimizer": {
            "class": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
        },
        "rng_states_by_rank": rng_states,
        "history": history,
        "smoke_consumed_embedding_indices_by_rank": smoke_indices_by_rank,
        "test_dataset_constructed": False,
    }


def validate_checkpoint(
    payload: dict[str, Any],
    *,
    mode: str,
    kind: str,
    config_sha: str,
    batch_size: int,
    gradient_accumulation: int,
) -> None:
    expected = {
        "schema": "genejepa_decoder_v1_checkpoint_v1",
        "mode": mode,
        "checkpoint_kind": kind,
        "frozen_contract_sha256": sha256_file(CONTRACT_PATH),
        "panel_sha256": PANEL_SHA256,
        "dataset_code_sha256": sha256_file(DATASET_PATH),
        "runner_sha256": sha256_file(SCRIPT_PATH),
        "descriptor_counts_sha256": sha256_file(DESCRIPTOR_COUNTS_PATH),
        "batch_size_per_gpu": batch_size,
        "world_size": WORLD_SIZE,
        "gradient_accumulation": gradient_accumulation,
        "effective_global_batch": EFFECTIVE_GLOBAL_BATCH,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise AssertionError(f"Checkpoint provenance mismatch for {key}")
    if payload.get("training_config", {}).get("sha256") != config_sha:
        raise AssertionError("Checkpoint training config SHA changed")
    required = {
        "epoch",
        "step_in_epoch",
        "epoch_complete",
        "global_optimizer_step",
        "model_state",
        "optimizer_state",
        "model_fingerprint",
        "best_val_mse",
        "best_epoch",
        "early_stop_counter",
        "rng_states_by_rank",
        "history",
    }
    if not required.issubset(payload) or len(payload["rng_states_by_rank"]) != WORLD_SIZE:
        raise AssertionError("Checkpoint lacks required training/resume state")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    mode: str,
    config_sha: str,
    batch_size: int,
    gradient_accumulation: int,
    rank: int,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    validate_checkpoint(
        payload,
        mode=mode,
        kind="last",
        config_sha=config_sha,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
    )
    model.load_state_dict(payload["model_state"], strict=True)
    if model_fingerprint(model) != payload["model_fingerprint"]:
        raise AssertionError("Checkpoint model did not restore exactly")
    optimizer.load_state_dict(payload["optimizer_state"])
    if not optimizer.state:
        raise AssertionError("Checkpoint optimizer state is empty")
    restore_rng_state(payload["rng_states_by_rank"][rank])
    return payload


def tensorboard_audit(log_dir: Path) -> dict[str, Any]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = sorted(log_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise AssertionError("TensorBoard event file was not created")
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = sorted(accumulator.Tags().get("scalars", []))
    required = {
        "train/loss",
        "train/lr",
        "train/grad_norm",
        "train/samples_per_second",
        "train/global_step",
        "val/mse",
        "epoch",
    }
    missing = sorted(required - set(tags))
    test_tags = [tag for tag in tags if tag.startswith("test/")]
    if missing or test_tags:
        raise AssertionError(f"TensorBoard audit failed: missing={missing}, test={test_tags}")
    return {
        "status": "pass",
        "log_dir": relative(log_dir),
        "event_files": [artifact_record(path) for path in event_files],
        "tags": tags,
        "required_tags_present": True,
        "test_tags": test_tags,
    }


def runtime_paths(mode: str) -> tuple[Path, Path, Path]:
    best, last = checkpoint_paths(mode)
    tensorboard = SMOKE_TENSORBOARD if mode == "smoke" else FORMAL_TENSORBOARD
    return best, last, tensorboard


def run_smoke(
    *,
    resume: bool,
    train_steps: int,
    val_batches: int,
    fallback_512: bool,
) -> dict[str, Any] | None:
    batch_size = FALLBACK_BATCH_SIZE if fallback_512 else PRIMARY_BATCH_SIZE
    accumulation = (
        FALLBACK_GRADIENT_ACCUMULATION
        if fallback_512
        else PRIMARY_GRADIENT_ACCUMULATION
    )
    rank, local_rank, device = setup_distributed()
    writer: Any | None = None
    try:
        config, arrays, config_sha = load_runtime_contract()
        best_path, last_path, log_dir = runtime_paths("smoke")
        if rank == 0:
            if resume and not last_path.is_file():
                raise FileNotFoundError("Smoke --resume requires the smoke last.pt")
            if not resume and (best_path.exists() or last_path.exists() or log_dir.exists()):
                raise FileExistsError("Fresh exact-config smoke artifacts already exist")
        dist.barrier()
        seed_everything(SEED)
        raw_model = build_decoder().to(device)
        optimizer = torch.optim.AdamW(
            raw_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        resume_audit = {
            "requested": resume,
            "model_restored": False,
            "optimizer_restored": False,
            "rng_restored": False,
            "global_step_continued": False,
            "best_metric_restored": False,
        }
        history: list[dict[str, Any]] = []
        prior_indices_by_rank: list[list[int]] = [[], []]
        if resume:
            checkpoint = load_checkpoint(
                last_path,
                raw_model,
                optimizer,
                mode="smoke",
                config_sha=config_sha,
                batch_size=batch_size,
                gradient_accumulation=accumulation,
                rank=rank,
                device=device,
            )
            if checkpoint["epoch"] != 0 or checkpoint["epoch_complete"]:
                raise AssertionError("Smoke resume checkpoint has invalid epoch semantics")
            start_step_in_epoch = int(checkpoint["step_in_epoch"])
            global_step = int(checkpoint["global_optimizer_step"])
            best_val_mse = float(checkpoint["best_val_mse"])
            best_epoch = int(checkpoint["best_epoch"])
            early_stop_counter = int(checkpoint["early_stop_counter"])
            history = list(checkpoint["history"])
            prior_indices_by_rank = checkpoint.get(
                "smoke_consumed_embedding_indices_by_rank", [[], []]
            )
            if global_step < 8 or train_steps < 2:
                raise AssertionError("Resume smoke requires >=8 prior and >=2 additional steps")
            resume_audit.update(
                {
                    "model_restored": True,
                    "optimizer_restored": True,
                    "rng_restored": True,
                    "prior_global_step": global_step,
                    "prior_step_in_epoch": start_step_in_epoch,
                    "restored_best_val_mse": best_val_mse,
                    "restored_best_epoch": best_epoch,
                }
            )
        else:
            if train_steps < 8 or train_steps > 10:
                raise ValueError("Initial exact-config smoke must run 8-10 optimizer steps")
            start_step_in_epoch = 0
            global_step = 0
            best_val_mse = float("inf")
            best_epoch = -1
            early_stop_counter = 0
        if val_batches < 4 or val_batches > 8:
            raise ValueError("Smoke validation must use 4-8 batches per rank")
        from torch.utils.tensorboard import SummaryWriter

        if rank == 0:
            log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(
                log_dir=str(log_dir), purge_step=global_step if resume else None
            )
        train_dataset = ExactTrainTahoeDecoderDataset(
            arrays["train_cells"],
            batch_size=batch_size,
            gradient_accumulation=accumulation,
        )
        val_dataset = TahoeDecoderV1PairedDataset(
            split="val", panel_path=PANEL_PATH, shuffle_shards=False
        )
        if val_dataset.max_cells is not None or val_dataset.shuffle_shards:
            raise AssertionError("Validation Dataset ownership was modified")
        model = DistributedDataParallel(raw_model, device_ids=[local_rank])
        torch.cuda.reset_peak_memory_stats(device)
        session_started = time.perf_counter()
        train = train_segment(
            model,
            raw_model,
            optimizer,
            train_dataset,
            device,
            epoch=0,
            start_optimizer_step_in_epoch=start_step_in_epoch,
            optimizer_steps_to_run=train_steps,
            batch_size=batch_size,
            gradient_accumulation=accumulation,
            rank=rank,
            writer=writer,
            global_step=global_step,
            collect_indices=True,
        )
        global_step = train["global_step"]
        if resume:
            resume_audit["global_step_continued"] = global_step > resume_audit[
                "prior_global_step"
            ]
            resume_audit["best_metric_restored"] = math.isfinite(
                resume_audit["restored_best_val_mse"]
            )
        local_train_indices = train.pop("consumed_indices")
        gathered_new_indices: list[list[int] | None] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_new_indices, local_train_indices)
        new_indices_by_rank = [item for item in gathered_new_indices if item is not None]
        if len(new_indices_by_rank) != WORLD_SIZE:
            raise AssertionError("Both ranks did not report smoke training ownership")
        new_sets = [set(item) for item in new_indices_by_rank]
        if (
            len(new_sets[0] & new_sets[1])
            or any(len(item) != len(new_indices_by_rank[index]) for index, item in enumerate(new_sets))
        ):
            raise AssertionError("Smoke train rank ownership overlaps or duplicates")
        prior_new_overlap = 0
        if resume:
            prior_set = set(prior_indices_by_rank[0]) | set(prior_indices_by_rank[1])
            new_set = new_sets[0] | new_sets[1]
            prior_new_overlap = len(prior_set & new_set)
            if prior_new_overlap:
                raise AssertionError("Resume smoke restarted previously consumed training cells")
        combined_indices = [
            prior_indices_by_rank[rank_id] + new_indices_by_rank[rank_id]
            for rank_id in range(WORLD_SIZE)
        ]
        partition = read_json(PARTITION_AUDIT_PATH)
        expected_val_rank_cells = partition["formal_validation_descriptor_ownership"][
            "rank_cells"
        ][rank]
        validation = validate(
            raw_model,
            val_dataset,
            device,
            batch_size=batch_size,
            max_batches=val_batches,
            expected_rank_cells=expected_val_rank_cells,
            rank=rank,
        )
        current_val_mse = validation["global_val_mse"]
        improved = current_val_mse < best_val_mse
        if improved:
            best_val_mse = current_val_mse
            best_epoch = 0
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        row = {
            "phase": "resume" if resume else "initial",
            "epoch": 0,
            "start_step_in_epoch": start_step_in_epoch,
            "end_step_in_epoch": train["end_optimizer_step_in_epoch"],
            "global_step": global_step,
            "train_initial_mse": train["initial_loss"],
            "train_final_mse": train["final_loss"],
            "val_mse": current_val_mse,
            "strict_improvement": improved,
            "best_val_mse": best_val_mse,
            "best_epoch": best_epoch,
        }
        history.append(row)
        sync_error = exact_parameter_sync_error(raw_model, device)
        if sync_error != 0.0:
            raise AssertionError(f"DDP parameters differ across ranks: {sync_error}")
        local_runtime = {
            "rank": rank,
            "local_rank": local_rank,
            "gpu": torch.cuda.get_device_name(device),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            "session_wall_seconds": time.perf_counter() - session_started,
            "dataloader": train["dataloader"],
        }
        gathered_runtime: list[dict[str, Any] | None] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_runtime, local_runtime)
        runtimes = [item for item in gathered_runtime if item is not None]
        if len(runtimes) != WORLD_SIZE:
            raise AssertionError("Both GPU ranks did not report runtime")
        if rank == 0 and writer is not None:
            writer.add_scalar("train/samples_per_second", train["global_samples_per_second"], global_step)
            writer.add_scalar("val/mse", current_val_mse, global_step)
            writer.add_scalar("epoch", 0, global_step)
            writer.flush()
        rng_states = gather_rng_states()
        if rank == 0:
            checkpoint_args = {
                "mode": "smoke",
                "model": raw_model,
                "optimizer": optimizer,
                "epoch": 0,
                "step_in_epoch": train["end_optimizer_step_in_epoch"],
                "epoch_complete": False,
                "global_step": global_step,
                "best_val_mse": best_val_mse,
                "best_epoch": best_epoch,
                "early_stop_counter": early_stop_counter,
                "training_config_sha256": config_sha,
                "batch_size": batch_size,
                "gradient_accumulation": accumulation,
                "rng_states": rng_states,
                "history": history,
                "smoke_indices_by_rank": combined_indices,
            }
            if improved:
                atomic_torch_save(
                    best_path, checkpoint_payload(kind="best", **checkpoint_args)
                )
            atomic_torch_save(last_path, checkpoint_payload(kind="last", **checkpoint_args))
        dist.barrier()
        if rank != 0:
            return None
        assert writer is not None
        writer.close()
        writer = None
        events = tensorboard_audit(log_dir)
        best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
        last_payload = torch.load(last_path, map_location="cpu", weights_only=False)
        validate_checkpoint(
            best_payload,
            mode="smoke",
            kind="best",
            config_sha=config_sha,
            batch_size=batch_size,
            gradient_accumulation=accumulation,
        )
        validate_checkpoint(
            last_payload,
            mode="smoke",
            kind="last",
            config_sha=config_sha,
            batch_size=batch_size,
            gradient_accumulation=accumulation,
        )
        resume_complete = resume and all(
            resume_audit[key]
            for key in (
                "model_restored",
                "optimizer_restored",
                "rng_restored",
                "global_step_continued",
                "best_metric_restored",
            )
        )
        status = "pass" if resume_complete else "resume_pending"
        result = {
            "schema": "genejepa_decoder_v1_exact_config_smoke_v1",
            "created_at_utc": utc_now(),
            "status": status,
            "scope": "production runner exact-config smoke; no formal training and no test Dataset",
            "configuration": {
                "batch_size_per_gpu": batch_size,
                "world_size": WORLD_SIZE,
                "global_microbatch": batch_size * WORLD_SIZE,
                "gradient_accumulation": accumulation,
                "effective_global_batch": EFFECTIVE_GLOBAL_BATCH,
                "workers_per_rank": NUM_WORKERS_PER_RANK,
                "precision": "BF16 autocast forward + FP32 MSE",
                "oom_fallback_used": fallback_512,
            },
            "provenance": {
                "contract": artifact_record(CONTRACT_PATH),
                "panel": artifact_record(PANEL_PATH),
                "dataset": artifact_record(DATASET_PATH),
                "runner": artifact_record(SCRIPT_PATH),
                "training_config": artifact_record(TRAINING_CONFIG_PATH),
                "partition_audit": artifact_record(PARTITION_AUDIT_PATH),
            },
            "model": config["model"],
            "formal_ownership": {
                **partition["formal_global_stream"],
                **partition["rank_parity"],
                "worker_ranges": partition[
                    "oom_fallback_512" if fallback_512 else "primary_1024"
                ]["ranges"],
            },
            "smoke_train": {
                **{key: value for key, value in train.items() if key != "losses"},
                "losses": train["losses"],
                "rank0_rank1_new_index_overlap": len(new_sets[0] & new_sets[1]),
                "prior_new_index_overlap_on_resume": prior_new_overlap,
                "new_global_cells": sum(len(item) for item in new_indices_by_rank),
                "DDP_parameter_max_abs_error": sync_error,
            },
            "validation": validation,
            "runtime": {
                "rank": runtimes,
                "global_samples_per_second": train["global_samples_per_second"],
                "global_batches_per_second": train["global_batches_per_second"],
                "peak_allocated_gib": max(item["peak_allocated_gib"] for item in runtimes),
                "peak_reserved_gib": max(item["peak_reserved_gib"] for item in runtimes),
            },
            "checkpoint": {
                "best": artifact_record(best_path),
                "last": artifact_record(last_path),
                "atomic_save": True,
                "schema_validated": True,
            },
            "resume": resume_audit,
            "history": history,
            "tensorboard": events,
            "acceptance": {
                "frozen_contract_sha_matches": True,
                "panel_sha_matches": True,
                "parameter_count_matches": True,
                "two_rank_DDP_initialized": True,
                "BF16_forward": True,
                "FP32_MSE_finite": True,
                "backward_and_gradients_finite": True,
                "optimizer_steps_executed": True,
                "softplus_prediction_finite_nonnegative": True,
                "rank_training_step_parity_proven": True,
                "validation_sse_count_reduction_correct": validation[
                    "local_sse_count_all_reduce_correct"
                ],
                "checkpoint_save": True,
                "checkpoint_resume": resume_complete,
                "tensorboard_event_written": True,
                "test_not_touched": True,
                "genejepa_not_rerun": True,
                "ST_A_not_used": True,
            },
            "ready_for_formal_training": resume_complete,
            "formal_training_started": False,
            "test_dataset_constructed": False,
            "blockers": [] if resume_complete else ["resume smoke still required"],
        }
        atomic_write_json(SMOKE_RESULT_PATH, result)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return result
    finally:
        if writer is not None:
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def run_formal(*, resume: bool, fallback_512: bool) -> dict[str, Any] | None:
    batch_size = FALLBACK_BATCH_SIZE if fallback_512 else PRIMARY_BATCH_SIZE
    accumulation = (
        FALLBACK_GRADIENT_ACCUMULATION
        if fallback_512
        else PRIMARY_GRADIENT_ACCUMULATION
    )
    rank, local_rank, device = setup_distributed()
    writer: Any | None = None
    try:
        smoke = read_json(SMOKE_RESULT_PATH)
        if smoke.get("status") != "pass" or smoke.get("ready_for_formal_training") is not True:
            raise RuntimeError("Exact-config smoke and resume gate have not passed")
        if smoke["configuration"]["oom_fallback_used"] != fallback_512:
            raise AssertionError("Formal batch choice differs from the accepted smoke")
        config, arrays, config_sha = load_runtime_contract()
        best_path, last_path, log_dir = runtime_paths("formal")
        if rank == 0:
            if resume and not last_path.is_file():
                raise FileNotFoundError("Formal --resume requires formal last.pt")
            if not resume and (
                best_path.exists()
                or last_path.exists()
                or FORMAL_RESULT_PATH.exists()
                or log_dir.exists()
            ):
                raise FileExistsError("Fresh formal Decoder artifacts already exist")
        dist.barrier()
        seed_everything(SEED)
        raw_model = build_decoder().to(device)
        optimizer = torch.optim.AdamW(
            raw_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        if resume:
            checkpoint = load_checkpoint(
                last_path,
                raw_model,
                optimizer,
                mode="formal",
                config_sha=config_sha,
                batch_size=batch_size,
                gradient_accumulation=accumulation,
                rank=rank,
                device=device,
            )
            if not checkpoint["epoch_complete"]:
                raise AssertionError("Formal v1 resume is only supported at epoch boundaries")
            start_epoch = int(checkpoint["epoch"]) + 1
            global_step = int(checkpoint["global_optimizer_step"])
            best_val_mse = float(checkpoint["best_val_mse"])
            best_epoch = int(checkpoint["best_epoch"])
            early_stop_counter = int(checkpoint["early_stop_counter"])
            history = list(checkpoint["history"])
        else:
            start_epoch = 0
            global_step = 0
            best_val_mse = float("inf")
            best_epoch = -1
            early_stop_counter = 0
            history: list[dict[str, Any]] = []
        from torch.utils.tensorboard import SummaryWriter

        if rank == 0:
            log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(
                log_dir=str(log_dir), purge_step=global_step if resume else None
            )
        train_dataset = ExactTrainTahoeDecoderDataset(
            arrays["train_cells"],
            batch_size=batch_size,
            gradient_accumulation=accumulation,
        )
        val_dataset = TahoeDecoderV1PairedDataset(
            split="val", panel_path=PANEL_PATH, shuffle_shards=False
        )
        model = DistributedDataParallel(raw_model, device_ids=[local_rank])
        partition = read_json(PARTITION_AUDIT_PATH)
        validation_rank_cells = partition["formal_validation_descriptor_ownership"][
            "rank_cells"
        ]
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        stop_reason = "max_epochs"
        for epoch in range(start_epoch, MAX_EPOCHS):
            train = train_segment(
                model,
                raw_model,
                optimizer,
                train_dataset,
                device,
                epoch=epoch,
                start_optimizer_step_in_epoch=0,
                optimizer_steps_to_run=FORMAL_STEPS_PER_EPOCH,
                batch_size=batch_size,
                gradient_accumulation=accumulation,
                rank=rank,
                writer=writer,
                global_step=global_step,
                collect_indices=False,
            )
            global_step = train["global_step"]
            validation = validate(
                raw_model,
                val_dataset,
                device,
                batch_size=batch_size,
                max_batches=None,
                expected_rank_cells=validation_rank_cells[rank],
                rank=rank,
            )
            val_mse = validation["global_val_mse"]
            improved = val_mse < best_val_mse
            if improved:
                best_val_mse = val_mse
                best_epoch = epoch
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "train_mse": float(np.mean(train["losses"])),
                "val_mse": val_mse,
                "strict_improvement": improved,
                "best_val_mse": best_val_mse,
                "best_epoch": best_epoch,
                "early_stop_counter": early_stop_counter,
            }
            history.append(row)
            if rank == 0 and writer is not None:
                writer.add_scalar("train/samples_per_second", train["global_samples_per_second"], global_step)
                writer.add_scalar("val/mse", val_mse, global_step)
                writer.add_scalar("epoch", epoch, global_step)
                writer.flush()
            rng_states = gather_rng_states()
            if rank == 0:
                args = {
                    "mode": "formal",
                    "model": raw_model,
                    "optimizer": optimizer,
                    "epoch": epoch,
                    "step_in_epoch": FORMAL_STEPS_PER_EPOCH,
                    "epoch_complete": True,
                    "global_step": global_step,
                    "best_val_mse": best_val_mse,
                    "best_epoch": best_epoch,
                    "early_stop_counter": early_stop_counter,
                    "training_config_sha256": config_sha,
                    "batch_size": batch_size,
                    "gradient_accumulation": accumulation,
                    "rng_states": rng_states,
                    "history": history,
                    "smoke_indices_by_rank": None,
                }
                if improved:
                    atomic_torch_save(best_path, checkpoint_payload(kind="best", **args))
                atomic_torch_save(last_path, checkpoint_payload(kind="last", **args))
                print(
                    f"decoder epoch={epoch} step={global_step} train_mse={row['train_mse']:.8f} "
                    f"val_mse={val_mse:.8f} best_epoch={best_epoch} "
                    f"early_stop_counter={early_stop_counter}",
                    flush=True,
                )
            dist.barrier()
            if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                stop_reason = "early_stopping"
                break
        sync_error = exact_parameter_sync_error(raw_model, device)
        if sync_error != 0.0:
            raise AssertionError("Formal Decoder parameters differ across ranks")
        if rank != 0:
            dist.barrier()
            return None
        result = {
            "schema": "genejepa_decoder_v1_formal_training_result_v1",
            "created_at_utc": utc_now(),
            "status": "pass",
            "stop_reason": stop_reason,
            "epochs_completed": len(history),
            "global_optimizer_step": global_step,
            "best_epoch": best_epoch,
            "best_val_mse": best_val_mse,
            "history": history,
            "configuration": artifact_record(TRAINING_CONFIG_PATH),
            "checkpoints": {
                "best": artifact_record(best_path),
                "last": artifact_record(last_path),
            },
            "elapsed_seconds": time.perf_counter() - started,
            "DDP_parameter_max_abs_error": sync_error,
            "test_dataset_constructed": False,
        }
        atomic_write_json(FORMAL_RESULT_PATH, result)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        dist.barrier()
        return result
    finally:
        if writer is not None:
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--resume", action="store_true")
    smoke.add_argument("--train-steps", type=int)
    smoke.add_argument("--val-batches", type=int, default=4)
    smoke.add_argument("--fallback-512", action="store_true")
    formal = subparsers.add_parser("run")
    formal.add_argument("--resume", action="store_true")
    formal.add_argument("--fallback-512", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError("Run prepare with one ordinary Python process")
        prepare()
    elif args.command == "smoke":
        steps = args.train_steps if args.train_steps is not None else (2 if args.resume else 8)
        run_smoke(
            resume=args.resume,
            train_steps=steps,
            val_batches=args.val_batches,
            fallback_512=args.fallback_512,
        )
    elif args.command == "run":
        run_formal(resume=args.resume, fallback_512=args.fallback_512)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
