"""Cache-backed Phase-II ST-A v2 Dataset using the frozen Experiment-1 sampler."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from tahoe_experiment1_latent_data import (
    TahoeExperiment1LatentSetDataset,
    make_dataloader,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
CONDITIONS_PATH = RESULTS / "phase2_stav2_conditions.csv"
CONTROLS_PATH = RESULTS / "phase2_stav2_control_pools.csv"
SUBSET_MANIFEST = RESULTS / "phase2_stav2_subset_manifest.json"
CACHE_PLAN = RESULTS / "phase2_stav2_cache_plan.json"
DOSE_NORMALIZATION = RESULTS / "phase2_stav2_dose_normalization.json"
EMBEDDINGS_PATH = RESULTS / "phase2_author_genejepa_embeddings.npy"
EMBEDDING_MANIFEST = RESULTS / "phase2_author_genejepa_embedding_manifest.json"

SET_SIZE = 256
LATENT_DIM = 768
DRUG_COUNT = 379


class Phase2STAv2Dataset(TahoeExperiment1LatentSetDataset):
    """Reuse the exact seed/epoch/pair/side sampler on the Phase-II Author cache."""

    def __init__(
        self,
        *,
        split: str | None,
        seed: int = 42,
        epoch: int = 0,
        embeddings_path: Path = EMBEDDINGS_PATH,
    ) -> None:
        if split not in {None, "train", "val", "test"}:
            raise ValueError("split must be train, val, test, or None")
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.split = split
        self.seed = int(seed)
        self.epoch = int(epoch)

        subset = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))
        plan = json.loads(CACHE_PLAN.read_text(encoding="utf-8"))
        dose = json.loads(DOSE_NORMALIZATION.read_text(encoding="utf-8"))
        merged = json.loads(EMBEDDING_MANIFEST.read_text(encoding="utf-8"))
        if any(payload.get("status") != "pass" for payload in (subset, plan, merged)):
            raise AssertionError("Phase-II subset/cache/embedding manifest is not PASS")
        if dose.get("status") != "frozen" or dose.get("fit_scope") != "Phase-II selected train conditions only":
            raise AssertionError("Phase-II train-only dose normalization is not frozen")
        integrity = merged["integrity"]
        if any(
            (
                integrity.get("missing") != 0,
                integrity.get("duplicate") != 0,
                integrity.get("worker_overlap") != 0,
                integrity.get("finite") is not True,
                integrity.get("signed") is not True,
            )
        ):
            raise AssertionError("Phase-II merged Author cache integrity failed")
        output = merged["output"]
        if (PROJECT_ROOT / output["path"]).resolve() != embeddings_path.resolve():
            raise AssertionError("Embedding manifest points to a different file")
        if embeddings_path.stat().st_size != int(output["size_bytes"]):
            raise AssertionError("Phase-II embedding file size changed")
        if output.get("row_equals_phase2_cell_index") is not True:
            raise AssertionError("Phase-II row/index contract is absent")
        self.embeddings = np.load(embeddings_path, mmap_mode="r")
        if (
            not isinstance(self.embeddings, np.memmap)
            or self.embeddings.shape != tuple(output["shape"])
            or self.embeddings.shape != (int(plan["total_unique_cells"]), LATENT_DIM)
            or self.embeddings.dtype != np.float32
        ):
            raise AssertionError("Phase-II cache is not the declared [N,768] float32 mmap")

        conditions = pd.read_csv(CONDITIONS_PATH, encoding="utf-8-sig", keep_default_na=False)
        controls = pd.read_csv(CONTROLS_PATH, encoding="utf-8-sig", keep_default_na=False)
        if conditions["pair_id"].duplicated().any() or len(conditions) != 5000:
            raise AssertionError("Phase-II condition index changed")
        if controls["control_pool_id"].duplicated().any():
            raise AssertionError("Phase-II control pool index is not unique")
        if not np.array_equal(
            conditions["phase2_condition_index"].to_numpy(np.int64), np.arange(len(conditions))
        ):
            raise AssertionError("phase2_condition_index is not contiguous")
        if not np.array_equal(
            controls["control_pool_index"].to_numpy(np.int64), np.arange(len(controls))
        ):
            raise AssertionError("control_pool_index is not contiguous")
        treated_start = conditions["treated_embedding_start"].to_numpy(np.int64)
        treated_stop = conditions["treated_embedding_stop_exclusive"].to_numpy(np.int64)
        treated_count = conditions["treated_cached_cell_count"].to_numpy(np.int64)
        control_start = controls["embedding_start"].to_numpy(np.int64)
        control_stop = controls["embedding_stop_exclusive"].to_numpy(np.int64)
        control_count = controls["cached_cell_count"].to_numpy(np.int64)
        if (
            treated_start[0] != 0
            or not np.array_equal(treated_start[1:], treated_stop[:-1])
            or not np.array_equal(treated_stop - treated_start, treated_count)
            or np.any((treated_count < SET_SIZE) | (treated_count > 512))
        ):
            raise AssertionError("Phase-II treated pool ranges are invalid")
        if (
            control_start[0] != treated_stop[-1]
            or not np.array_equal(control_start[1:], control_stop[:-1])
            or not np.array_equal(control_stop - control_start, control_count)
            or np.any((control_count < SET_SIZE) | (control_count > 512))
            or control_stop[-1] != len(self.embeddings)
        ):
            raise AssertionError("Phase-II control pool ranges are invalid")
        control_lookup = controls.set_index("control_pool_id")
        matched = control_lookup.loc[conditions["control_pool_id"]].reset_index()
        for condition_column, control_column in (
            ("control_pool_index", "control_pool_index"),
            ("control_cached_cell_count", "cached_cell_count"),
            ("control_embedding_start", "embedding_start"),
            ("control_embedding_stop_exclusive", "embedding_stop_exclusive"),
        ):
            if not np.array_equal(
                conditions[condition_column].to_numpy(np.int64),
                matched[control_column].to_numpy(np.int64),
            ):
                raise AssertionError(f"Condition/control mismatch: {condition_column}")
        if conditions.groupby("edge_id")["split"].nunique().max() != 1:
            raise AssertionError("Phase-II edge split leakage")
        train_drugs = set(conditions.loc[conditions["split"].eq("train"), "drug_id"].astype(int))
        for held_out in ("val", "test"):
            if not set(conditions.loc[conditions["split"].eq(held_out), "drug_id"].astype(int)).issubset(train_drugs):
                raise AssertionError(f"Phase-II {held_out} includes an unseen-in-train drug ID")
        if conditions["drug_id"].astype(int).min() < 0 or conditions["drug_id"].astype(int).max() >= DRUG_COUNT:
            raise AssertionError("Phase-II drug_id is outside frozen 0..378")

        self.max_train_dose_uM = float(dose["max_train_dose_uM"])
        self.dose_denominator = math.log1p(self.max_train_dose_uM)
        if self.dose_denominator <= 0:
            raise AssertionError("Invalid Phase-II dose denominator")
        self.all_conditions = conditions
        self.controls = controls
        self.split_counts = {
            name: int(conditions["split"].eq(name).sum()) for name in ("train", "val", "test")
        }
        self.edge_overlap = {
            f"{left}_{right}": len(
                set(conditions.loc[conditions["split"].eq(left), "edge_id"])
                & set(conditions.loc[conditions["split"].eq(right), "edge_id"])
            )
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        }
        if any(self.edge_overlap.values()):
            raise AssertionError("Phase-II edge overlap is non-zero")
        self.embedding_transforms: tuple[str, ...] = ()
        self.provenance = {
            "row_equals_phase2_cell_index": True,
            "raw_signed_author_latent": True,
            "latent_transforms": [],
            "dose_fit_scope": dose["fit_scope"],
        }
        selected = conditions if split is None else conditions.loc[conditions["split"].eq(split)]
        self.conditions = selected.reset_index(drop=True)
        if self.conditions.empty:
            raise ValueError(f"No Phase-II conditions for split={split}")

    def __len__(self) -> int:
        return len(self.conditions)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.conditions.iloc[index]
        control_indices = self._sample_range(
            int(row["control_embedding_start"]),
            int(row["control_embedding_stop_exclusive"]),
            str(row["pair_id"]),
            "control",
        )
        treated_indices = self._sample_range(
            int(row["treated_embedding_start"]),
            int(row["treated_embedding_stop_exclusive"]),
            str(row["pair_id"]),
            "treated",
        )
        control = np.ascontiguousarray(self.embeddings[control_indices], dtype=np.float32)
        treated = np.ascontiguousarray(self.embeddings[treated_indices], dtype=np.float32)
        dose_uM = float(row["dose_uM"])
        dose_scaled = math.log1p(dose_uM) / self.dose_denominator
        if not 0 < dose_scaled <= 1:
            raise AssertionError("Positive Phase-II dose_scaled is outside (0,1]")
        return {
            "ctrl_cell_emb": torch.from_numpy(control),
            "pert_cell_emb": torch.from_numpy(treated),
            "drug_id": torch.tensor(int(row["drug_id"]), dtype=torch.long),
            "dose_scaled": torch.tensor([dose_scaled], dtype=torch.float32),
            "source_embedding_index": torch.from_numpy(control_indices.copy()),
            "target_embedding_index": torch.from_numpy(treated_indices.copy()),
            "condition_id": str(row["pair_id"]),
            "edge_id": str(row["edge_id"]),
            "control_pool_id": str(row["control_pool_id"]),
            "split": str(row["split"]),
            "plate": str(row["plate"]),
            "cell_line_id": str(row["cell_line_id"]),
            "drug": str(row["drug"]),
            "dose_uM": dose_uM,
        }


__all__ = ["Phase2STAv2Dataset", "make_dataloader"]
