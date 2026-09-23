#!/usr/bin/env python3
"""Read cached Tahoe GeneJEPA latent sets without expression transforms."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
SET_SIZE = 256
LATENT_DIM = 768
PERT_DIM = 380
FORMAL_CACHE_PREFIX = RESULTS / "tahoe_experiment1_cache_cap512_all_dmso"
FORMAL_EMBEDDINGS = Path(str(FORMAL_CACHE_PREFIX) + "_embeddings.npy")
FORMAL_EMBEDDING_MANIFEST = Path(str(FORMAL_CACHE_PREFIX) + "_embedding_manifest.json")
FORMAL_CACHE_SUMMARY = Path(str(FORMAL_CACHE_PREFIX) + "_summary.json")
FORMAL_CONDITION_INDEX = Path(str(FORMAL_CACHE_PREFIX) + "_condition_index.csv")
FORMAL_CONTROL_POOL_INDEX = Path(str(FORMAL_CACHE_PREFIX) + "_control_pool_index.csv")
CONDITION_SPLIT_MANIFEST = RESULTS / "tahoe_experiment1_condition_split_manifest.csv"
EDGE_SPLIT_MANIFEST = RESULTS / "tahoe_experiment1_edge_split_manifest.csv"
DRUG_VOCABULARY = RESULTS / "tahoe_experiment1_drug_vocabulary.csv"
PERTURBATION_SPECIFICATION = RESULTS / "tahoe_experiment1_perturbation_featurization.json"
PREPARATION_SUMMARY = RESULTS / "tahoe_experiment1_preparation_summary.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PerturbationFeaturizer:
    """Frozen 379-d drug one-hot plus one standardized log10-dose value."""

    def __init__(self, vocabulary_path: Path, specification_path: Path) -> None:
        self.specification = json.loads(specification_path.read_text(encoding="utf-8"))
        if self.specification["pert_dim"] != PERT_DIM:
            raise AssertionError("Perturbation specification is not 380-dimensional")
        if sha256_file(vocabulary_path) != self.specification["drug_vocabulary_sha256"]:
            raise AssertionError("Drug vocabulary SHA-256 does not match its specification")

        vocabulary = pd.read_csv(
            vocabulary_path, keep_default_na=False, encoding="utf-8-sig"
        )
        if len(vocabulary) != PERT_DIM - 1:
            raise AssertionError("Drug vocabulary must contain exactly 379 rows")
        ids = vocabulary["drug_id"].astype(np.int64).to_numpy()
        if not np.array_equal(ids, np.arange(PERT_DIM - 1)):
            raise AssertionError("drug_id must be contiguous from 0 to 378")
        if vocabulary["drug"].duplicated().any():
            raise AssertionError("Drug vocabulary contains duplicate exact strings")

        self.drug_to_id = dict(zip(vocabulary["drug"], ids, strict=True))
        dose = self.specification["dose_feature"]
        self.dose_index = int(dose["dimension_index"])
        self.log10_mean = float(dose["log10_mean"])
        self.log10_std = float(dose["log10_std"])
        self.allowed_doses = {float(value) for value in dose["allowed_dose_uM"]}
        if self.dose_index != PERT_DIM - 1 or self.log10_std <= 0:
            raise AssertionError("Invalid frozen dose transform")

    def validate_train_dose_statistics(self, condition_manifest_path: Path) -> dict[str, Any]:
        dose = self.specification["dose_feature"]
        if (
            dose.get("fit_scope") != "train eligible condition rows, one vote per condition"
            or dose.get("standard_deviation") != "population, ddof=0"
        ):
            raise AssertionError("Dose specification is not the frozen train-only transform")
        conditions = pd.read_csv(
            condition_manifest_path,
            usecols=["split", "drug", "dose_uM"],
            keep_default_na=False,
            encoding="utf-8-sig",
        )
        if set(conditions["split"]) != {"train", "val", "test"}:
            raise AssertionError("Condition manifest has unexpected split labels")
        train = conditions.loc[conditions["split"] == "train"]
        train_log_dose = np.log10(train["dose_uM"].astype(np.float64).to_numpy())
        observed_mean = float(train_log_dose.mean())
        observed_std = float(train_log_dose.std(ddof=0))
        if not math.isclose(observed_mean, self.log10_mean, rel_tol=0, abs_tol=1e-15):
            raise AssertionError("Frozen dose mean was not fitted from train conditions")
        if not math.isclose(observed_std, self.log10_std, rel_tol=0, abs_tol=1e-15):
            raise AssertionError("Frozen dose std was not fitted from train conditions")
        train_drugs = set(train["drug"])
        vocabulary_drugs = set(self.drug_to_id)
        if train_drugs != vocabulary_drugs:
            raise AssertionError("Train split does not cover the frozen drug vocabulary")
        for split in ("val", "test"):
            if not set(conditions.loc[conditions["split"] == split, "drug"]) <= train_drugs:
                raise AssertionError(f"{split} contains a drug absent from train")
        return {
            "fit_scope": dose["fit_scope"],
            "condition_votes": int(len(train)),
            "mean": observed_mean,
            "std": observed_std,
            "ddof": 0,
            "train_drugs": len(train_drugs),
            "val_unseen_drugs": 0,
            "test_unseen_drugs": 0,
        }

    def encode(self, drug: str, dose_uM: float) -> np.ndarray:
        if drug not in self.drug_to_id:
            raise KeyError(f"Drug is absent from the frozen vocabulary: {drug}")
        dose = float(dose_uM)
        if dose not in self.allowed_doses or not math.isfinite(dose) or dose <= 0:
            raise ValueError(f"Unsupported dose_uM: {dose_uM}")
        vector = np.zeros(PERT_DIM, dtype=np.float32)
        vector[self.drug_to_id[drug]] = 1.0
        vector[self.dose_index] = np.float32(
            (math.log10(dose) - self.log10_mean) / self.log10_std
        )
        return vector


class TahoeExperiment1LatentSetDataset(Dataset[dict[str, Any]]):
    """Sample formal S=256 sets directly from the merged Epoch25 latent cache."""

    def __init__(
        self,
        *,
        split: str | None,
        seed: int = 42,
        epoch: int = 0,
        embeddings_path: Path = FORMAL_EMBEDDINGS,
        embedding_manifest_path: Path = FORMAL_EMBEDDING_MANIFEST,
        cache_summary_path: Path = FORMAL_CACHE_SUMMARY,
        condition_index_path: Path = FORMAL_CONDITION_INDEX,
        control_pool_index_path: Path = FORMAL_CONTROL_POOL_INDEX,
        condition_manifest_path: Path = CONDITION_SPLIT_MANIFEST,
        edge_manifest_path: Path = EDGE_SPLIT_MANIFEST,
        vocabulary_path: Path = DRUG_VOCABULARY,
        featurization_path: Path = PERTURBATION_SPECIFICATION,
        preparation_summary_path: Path = PREPARATION_SUMMARY,
    ) -> None:
        if split not in {None, "train", "val", "test"}:
            raise ValueError("split must be train, val, test, or None")
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.split = split
        self.seed = int(seed)
        self.epoch = int(epoch)

        merged_manifest = json.loads(embedding_manifest_path.read_text(encoding="utf-8"))
        cache_summary = json.loads(cache_summary_path.read_text(encoding="utf-8"))
        preparation = json.loads(preparation_summary_path.read_text(encoding="utf-8"))
        if merged_manifest.get("status") != "pass" or cache_summary.get("status") != "pass":
            raise AssertionError("Formal cache manifests are not recorded as pass")
        integrity = merged_manifest.get("integrity", {})
        if any(
            (
                integrity.get("missing") != 0,
                integrity.get("duplicate") != 0,
                integrity.get("worker_overlap") != 0,
                integrity.get("finite") is not True,
            )
        ):
            raise AssertionError("Formal merged-cache integrity contract failed")

        plan_outputs = cache_summary["outputs"]
        prep_outputs = preparation["outputs"]
        small_inputs = (
            (
                condition_index_path,
                plan_outputs["condition_index"]["sha256"],
                "formal condition index",
            ),
            (
                control_pool_index_path,
                plan_outputs["control_pool_index"]["sha256"],
                "formal control-pool index",
            ),
            (
                condition_manifest_path,
                cache_summary["inputs"]["condition_manifest_sha256"],
                "frozen condition split manifest",
            ),
            (
                edge_manifest_path,
                prep_outputs[edge_manifest_path.relative_to(PROJECT_ROOT).as_posix()]["sha256"],
                "frozen edge split manifest",
            ),
            (
                vocabulary_path,
                prep_outputs[vocabulary_path.relative_to(PROJECT_ROOT).as_posix()]["sha256"],
                "frozen drug vocabulary",
            ),
            (
                featurization_path,
                prep_outputs[featurization_path.relative_to(PROJECT_ROOT).as_posix()]["sha256"],
                "frozen perturbation specification",
            ),
        )
        verified_hashes: dict[str, str] = {}
        for path, expected, label in small_inputs:
            observed = sha256_file(path)
            if observed != expected:
                raise AssertionError(f"SHA-256 changed for {label}: {path}")
            verified_hashes[label] = observed

        declared_output = merged_manifest["output"]
        if (PROJECT_ROOT / declared_output["path"]).resolve() != embeddings_path.resolve():
            raise AssertionError("Merged manifest points to a different embedding file")
        if embeddings_path.stat().st_size != int(declared_output["size_bytes"]):
            raise AssertionError("Merged embedding file size changed")
        self.embeddings = np.load(embeddings_path, mmap_mode="r")
        expected_shape = tuple(declared_output["shape"])
        if (
            not isinstance(self.embeddings, np.memmap)
            or self.embeddings.shape != expected_shape
            or expected_shape[1] != LATENT_DIM
            or self.embeddings.dtype != np.float32
        ):
            raise AssertionError("Formal cache must be a [N,768] float32 mmap")
        if not declared_output.get("row_equals_global_embedding_index"):
            raise AssertionError("Merged-cache row/global embedding_index contract is absent")
        total_cells = len(self.embeddings)
        if total_cells != int(cache_summary["embedding_index"]["total_cells"]):
            raise AssertionError("Merged cache and frozen index plan disagree on total cells")

        condition_columns = [
            "pair_id",
            "cache_condition_index",
            "edge_id",
            "split",
            "plate",
            "cell_line_id",
            "drug",
            "dose_uM",
            "control_drug",
            "control_samples",
            "control_pool_id",
            "control_pool_index",
            "eligible_S256",
            "treated_cached_cell_count",
            "treated_embedding_start",
            "treated_embedding_stop_exclusive",
            "control_cached_cell_count",
            "control_embedding_start",
            "control_embedding_stop_exclusive",
        ]
        conditions = pd.read_csv(
            condition_index_path,
            usecols=condition_columns,
            keep_default_na=False,
            encoding="utf-8-sig",
        ).sort_values("cache_condition_index").reset_index(drop=True)
        if conditions["pair_id"].duplicated().any():
            raise AssertionError("Formal condition index has duplicate pair_id")
        if not np.array_equal(
            conditions["cache_condition_index"].to_numpy(np.int64),
            np.arange(len(conditions), dtype=np.int64),
        ):
            raise AssertionError("cache_condition_index is not contiguous")
        if set(conditions["split"]) != {"train", "val", "test"}:
            raise AssertionError("Formal condition index has unexpected split labels")
        if not conditions["eligible_S256"].astype(bool).all():
            raise AssertionError("Formal condition index contains an ineligible condition")

        frozen_columns = [
            "pair_id",
            "edge_id",
            "split",
            "plate",
            "cell_line_id",
            "drug",
            "dose_uM",
            "control_drug",
            "control_samples",
            "control_pool_id",
            "eligible_S256",
        ]
        frozen = pd.read_csv(
            condition_manifest_path,
            usecols=frozen_columns,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
        if frozen["pair_id"].duplicated().any() or set(frozen["pair_id"]) != set(
            conditions["pair_id"]
        ):
            raise AssertionError("Formal and frozen condition manifests have different rows")
        frozen = frozen.set_index("pair_id").loc[conditions["pair_id"]].reset_index()
        for column in (
            "edge_id",
            "split",
            "plate",
            "cell_line_id",
            "drug",
            "control_drug",
            "control_samples",
            "control_pool_id",
        ):
            if not conditions[column].astype(str).eq(frozen[column].astype(str)).all():
                raise AssertionError(f"Condition index disagrees with frozen manifest: {column}")
        if not np.array_equal(
            conditions["dose_uM"].astype(np.float64).to_numpy(),
            frozen["dose_uM"].astype(np.float64).to_numpy(),
        ):
            raise AssertionError("Condition index disagrees with frozen dose_uM")
        if not np.array_equal(
            conditions["eligible_S256"].astype(np.int8).to_numpy(),
            frozen["eligible_S256"].astype(np.int8).to_numpy(),
        ):
            raise AssertionError("Condition index disagrees with frozen S=256 eligibility")

        edges = pd.read_csv(
            edge_manifest_path,
            usecols=["edge_id", "cell_line_id", "drug", "split"],
            keep_default_na=False,
            encoding="utf-8-sig",
        )
        if edges["edge_id"].duplicated().any():
            raise AssertionError("Frozen edge manifest has duplicate edge_id")
        edge_metadata = edges.set_index("edge_id").loc[conditions["edge_id"]].reset_index()
        for column in ("cell_line_id", "drug", "split"):
            if not conditions[column].eq(edge_metadata[column]).all():
                raise AssertionError(f"Condition-to-edge mapping changed: {column}")
        if conditions.groupby("edge_id")["split"].nunique().max() != 1:
            raise AssertionError("A treated edge crosses train/val/test")
        if conditions.groupby(["cell_line_id", "drug"])["split"].nunique().max() != 1:
            raise AssertionError("A frozen (cell_line_id, drug) key crosses splits")

        controls = pd.read_csv(
            control_pool_index_path,
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
        if controls["control_pool_id"].duplicated().any() or not np.array_equal(
            controls["control_pool_index"].to_numpy(np.int64),
            np.arange(len(controls), dtype=np.int64),
        ):
            raise AssertionError("Control-pool key/index is not unique and contiguous")

        treated_starts = conditions["treated_embedding_start"].to_numpy(np.int64)
        treated_stops = conditions["treated_embedding_stop_exclusive"].to_numpy(np.int64)
        treated_counts = conditions["treated_cached_cell_count"].to_numpy(np.int64)
        control_starts = controls["embedding_start"].to_numpy(np.int64)
        control_stops = controls["embedding_stop_exclusive"].to_numpy(np.int64)
        control_counts = controls["cached_cell_count"].to_numpy(np.int64)
        if (
            treated_starts[0] != 0
            or not np.array_equal(treated_starts[1:], treated_stops[:-1])
            or not np.array_equal(treated_stops - treated_starts, treated_counts)
            or (treated_counts < SET_SIZE).any()
        ):
            raise AssertionError("Treated embedding ranges are invalid")
        treated_stop = int(treated_stops[-1])
        if (
            control_starts[0] != treated_stop
            or not np.array_equal(control_starts[1:], control_stops[:-1])
            or not np.array_equal(control_stops - control_starts, control_counts)
            or (control_counts < SET_SIZE).any()
            or int(control_stops[-1]) != total_cells
        ):
            raise AssertionError("Control embedding ranges are invalid")

        control_lookup = controls.set_index("control_pool_id")
        matched_controls = control_lookup.loc[conditions["control_pool_id"]].reset_index()
        numeric_pairs = (
            ("control_pool_index", "control_pool_index"),
            ("control_cached_cell_count", "cached_cell_count"),
            ("control_embedding_start", "embedding_start"),
            ("control_embedding_stop_exclusive", "embedding_stop_exclusive"),
        )
        for condition_column, control_column in numeric_pairs:
            if not np.array_equal(
                conditions[condition_column].to_numpy(np.int64),
                matched_controls[control_column].to_numpy(np.int64),
            ):
                raise AssertionError(f"Condition/control-pool mismatch: {condition_column}")
        for column in ("plate", "cell_line_id", "control_drug", "control_samples"):
            if not conditions[column].eq(matched_controls[column]).all():
                raise AssertionError(f"Condition/control-pool metadata mismatch: {column}")

        featurizer = PerturbationFeaturizer(vocabulary_path, featurization_path)
        self.dose_statistics = featurizer.validate_train_dose_statistics(
            condition_manifest_path
        )
        self.featurizer = featurizer
        self.all_conditions = conditions
        self.controls = controls
        self.split_counts = {
            name: int((conditions["split"] == name).sum())
            for name in ("train", "val", "test")
        }
        edge_sets = {
            name: set(conditions.loc[conditions["split"] == name, "edge_id"])
            for name in ("train", "val", "test")
        }
        self.edge_overlap = {
            "train_val": len(edge_sets["train"] & edge_sets["val"]),
            "train_test": len(edge_sets["train"] & edge_sets["test"]),
            "val_test": len(edge_sets["val"] & edge_sets["test"]),
        }
        if any(self.edge_overlap.values()):
            raise AssertionError("Treated edges leak across train/val/test")
        self.provenance = {
            "verified_small_input_sha256": verified_hashes,
            "merged_embedding_sha256_from_pass_manifest": declared_output["sha256"],
            "merged_embedding_sha256_recomputed_on_dataset_init": False,
            "merged_embedding_size_bytes": int(declared_output["size_bytes"]),
            "row_equals_global_embedding_index": True,
        }
        self.embedding_transforms: tuple[str, ...] = ()
        selected = conditions if split is None else conditions.loc[conditions["split"] == split]
        self.conditions = selected.reset_index(drop=True)
        if self.conditions.empty:
            raise ValueError(f"No formal conditions are available for split={split}")

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def _sample_range(self, start: int, stop: int, pair_id: str, side: str) -> np.ndarray:
        count = int(stop) - int(start)
        if count < SET_SIZE:
            raise AssertionError(f"{pair_id}/{side} has fewer than {SET_SIZE} cached cells")
        payload = (
            f"tahoe_experiment1_set_v1|seed={self.seed}|epoch={self.epoch}|"
            f"pair_id={pair_id}|side={side}"
        )
        draw_seed = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
        offsets = np.random.Generator(np.random.PCG64(draw_seed)).choice(
            count, size=SET_SIZE, replace=False
        )
        return np.asarray(offsets, dtype=np.int64) + int(start)

    def __len__(self) -> int:
        return len(self.conditions)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.conditions.iloc[index]
        source_indices = self._sample_range(
            int(row["control_embedding_start"]),
            int(row["control_embedding_stop_exclusive"]),
            str(row["pair_id"]),
            "control",
        )
        target_indices = self._sample_range(
            int(row["treated_embedding_start"]),
            int(row["treated_embedding_stop_exclusive"]),
            str(row["pair_id"]),
            "treated",
        )
        ctrl = np.ascontiguousarray(self.embeddings[source_indices], dtype=np.float32)
        target = np.ascontiguousarray(self.embeddings[target_indices], dtype=np.float32)
        vector = self.featurizer.encode(str(row["drug"]), float(row["dose_uM"]))
        return {
            "ctrl_cell_emb": torch.from_numpy(ctrl),
            "pert_cell_emb": torch.from_numpy(target),
            "pert_emb": torch.from_numpy(vector).expand(SET_SIZE, -1),
            "source_embedding_index": torch.from_numpy(source_indices),
            "target_embedding_index": torch.from_numpy(target_indices),
            "condition_id": str(row["pair_id"]),
            "cache_condition_index": int(row["cache_condition_index"]),
            "edge_id": str(row["edge_id"]),
            "control_pool_id": str(row["control_pool_id"]),
            "split": str(row["split"]),
            "drug": str(row["drug"]),
            "drug_id": int(self.featurizer.drug_to_id[str(row["drug"])]),
            "dose_uM": float(row["dose_uM"]),
        }


def make_dataloader(
    dataset: Dataset[dict[str, Any]],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    sampler: Sampler[int] | None = None,
    drop_last: bool = True,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0")
    if prefetch_factor is not None and num_workers == 0:
        raise ValueError("prefetch_factor requires num_workers > 0")
    generator = torch.Generator().manual_seed(seed)
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available() if pin_memory is None else pin_memory,
        "persistent_workers": persistent_workers,
    }
    if prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=drop_last,
        generator=generator,
        **kwargs,
    )
