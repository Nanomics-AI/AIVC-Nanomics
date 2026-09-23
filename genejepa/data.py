import os
import json
import math
import itertools
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
import lightning as L

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

from .configs import DataConfig, ExperimentConfig

log = logging.getLogger(__name__)


class Tahoe100MDataset(IterableDataset):
    def __init__(self, hf_dataset, gene_map: Dict[int, int]):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.gene_map = gene_map

    def __iter__(self):
        # Hugging Face datasets 2.19.2 already handles PyTorch DataLoader worker
        # sharding internally via get_worker_info(). Do NOT manually call
        # IterableDataset.shard(), because this datasets version has no public
        # .shard() method on IterableDataset.
        dataset_instance = self.hf_dataset

        for cell in dataset_instance:
            if "genes" not in cell or "expressions" not in cell:
                continue

            genes, expressions = cell["genes"], cell["expressions"]
            if not genes or not expressions:
                continue

            if (
                isinstance(expressions, list)
                and len(expressions) > 0
                and expressions[0] < 0
            ):
                genes, expressions = genes[1:], expressions[1:]

            if not genes:
                continue

            mapped_indices = [self.gene_map[g] for g in genes if g in self.gene_map]
            valid_expressions = [
                e for g, e in zip(genes, expressions) if g in self.gene_map
            ]

            if not mapped_indices:
                continue

            metadata = {
                "drug": cell.get("drug", "N/A"),
                "cell_line_name": cell.get("cell_line_name", "N/A"),
            }

            yield {
                "gene_indices": np.array(mapped_indices, dtype=np.int64),
                "counts": np.array(valid_expressions, dtype=np.float32),
                "metadata": metadata,
            }


class Tahoe100MDataModule(L.LightningDataModule):
    """
    Tahoe-100M DataModule.

    Key behavior:
    1. Downloads/cache files once and then uses the local manifest.
    2. Supports a frozen training-subset manifest for reproducible experiments.
    3. Uses fixed global normalization statistics when available.
    4. Splits parquet files across DDP ranks before creating HF streaming datasets.
    5. Lets Hugging Face IterableDataset handle PyTorch DataLoader worker sharding.
    6. Actual training DataLoader worker count can be controlled at runtime with:
           GENEJEPA_TRAIN_WORKERS=<N>
       Validation workers remain 0 for stability.
    """

    def __init__(self, data_config: DataConfig, exp_config: ExperimentConfig):
        super().__init__()
        self.save_hyperparameters()

        self.data_config = data_config
        self.exp_config = exp_config

        self.data_cache_dir = os.path.join(os.getcwd(), "hf_data_cache")
        self.manifest_path = os.path.join(
            self.data_cache_dir,
            "local_file_manifest.json",
        )
        self.stats_path = os.path.join(
            self.data_cache_dir,
            "global_stats.json",
        )
        self.repo_id = "vevotx/Tahoe-100M"

        self.gene_map: Optional[Dict[int, int]] = None
        self.train_files: Optional[List[str]] = None
        self.val_files: Optional[List[str]] = None
        self.metadata_file: Optional[str] = None

        self.global_mean: Optional[float] = None
        self.global_std: Optional[float] = None

        os.makedirs(self.data_cache_dir, exist_ok=True)

    @property
    def gene_vocab_size(self) -> int:
        if self.gene_map is None:
            raise RuntimeError(
                "gene_map is not initialized. "
                "Call setup() before accessing gene_vocab_size."
            )
        return len(self.gene_map)

    def prepare_data(self):
        """
        Runs once on rank 0. Downloads all necessary files from the Hub to the
        local cache and creates a manifest of local paths.
        """
        log.info("--- [prepare_data] Starting (runs on RANK 0 only) ---")

        if os.path.exists(self.manifest_path):
            log.info(
                f"Local file manifest found at {self.manifest_path}. "
                "Skipping downloads."
            )
            return

        log.info("Local manifest not found. Starting download process...")
        api = HfApi()

        try:
            repo_files = list(
                api.list_repo_tree(
                    self.repo_id,
                    repo_type="dataset",
                    recursive=True,
                )
            )
        except HfHubHTTPError as e:
            log.error(
                f"Failed to list files in repo {self.repo_id}. "
                f"Check connection and token. Error: {e}"
            )
            raise

        data_file_paths = sorted(
            [
                f.path
                for f in repo_files
                if f.path.startswith("data/") and f.path.endswith(".parquet")
            ]
        )
        metadata_file_path = next(
            (f.path for f in repo_files if f.path.endswith("gene_metadata.parquet")),
            None,
        )

        if not data_file_paths or not metadata_file_path:
            raise FileNotFoundError(
                f"Required .parquet files not found in {self.repo_id}."
            )

        files_to_download = data_file_paths + [metadata_file_path]
        local_file_manifest = {
            "data_files": [],
            "metadata_file": "",
        }

        log.info(
            f"Downloading {len(files_to_download)} files "
            f"to {self.data_cache_dir}..."
        )

        for i, filepath in enumerate(files_to_download):
            log.info(
                f"  ({i + 1}/{len(files_to_download)}) " f"Downloading {filepath}..."
            )

            try:
                local_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=filepath,
                    repo_type="dataset",
                    cache_dir=self.data_cache_dir,
                    local_dir=os.path.join(
                        self.data_cache_dir,
                        os.path.dirname(filepath),
                    ),
                    local_dir_use_symlinks=False,
                )

                if filepath == metadata_file_path:
                    local_file_manifest["metadata_file"] = local_path
                else:
                    local_file_manifest["data_files"].append(local_path)

            except HfHubHTTPError as e:
                log.error(f"Failed to download {filepath}. Error: {e}")
                raise

        local_file_manifest["data_files"].sort()

        with open(self.manifest_path, "w") as f:
            json.dump(local_file_manifest, f)

        log.info(
            "Successfully downloaded all files and saved local manifest "
            f"to {self.manifest_path}"
        )

    def setup(self, stage: Optional[str] = None):
        """
        Called on every DDP process. Loads the local manifest, freezes the
        train/validation split, optionally applies the fixed 25% training subset,
        loads normalization statistics, and builds the gene map.
        """
        log.info(
            f"--- [setup] Starting for stage '{stage}' " f"on PID: {os.getpid()} ---"
        )

        # STEP 1: Load manifest and determine train/val file splits.
        with open(self.manifest_path, "r") as f:
            manifest = json.load(f)

        all_local_data_files = manifest["data_files"]
        self.metadata_file = manifest["metadata_file"]

        if not all_local_data_files:
            raise ValueError(
                "The local file manifest is empty. " "prepare_data may have failed."
            )

        # Preserve the project's current validation split logic.
        total_files = len(all_local_data_files)
        samples_per_file = 100_000_000 / 3388

        num_files_for_samples = math.ceil(
            self.data_config.val_samples / samples_per_file
        )
        min_files_for_parallelism = self.data_config.num_workers * 2

        num_val_files = max(
            int(num_files_for_samples),
            int(min_files_for_parallelism),
        )
        num_val_files = min(
            num_val_files,
            total_files // 2,
        )

        if self.data_config.num_workers > 0 and num_val_files == 0 and total_files > 1:
            num_val_files = 1

        log.info(
            f"Data Splitting: {num_val_files} files for validation, "
            f"{total_files - num_val_files} for training."
        )

        self.val_files = all_local_data_files[:num_val_files]
        full_train_files = all_local_data_files[num_val_files:]
        self.train_files = full_train_files

        # Optional fixed training subset for reproducible experiments.
        subset_manifest = getattr(
            self.data_config,
            "train_subset_manifest",
            None,
        )

        if subset_manifest:
            if not os.path.isabs(subset_manifest):
                subset_manifest = os.path.join(
                    os.getcwd(),
                    subset_manifest,
                )

            log.info("Loading fixed training subset from: " f"{subset_manifest}")

            with open(subset_manifest, "r") as f:
                subset = json.load(f)

            selected_train_files = subset["selected_training_shards"]
            expected_val_files = subset.get("validation_shards")

            if expected_val_files is not None and expected_val_files != self.val_files:
                raise RuntimeError(
                    "Validation shard list does not match "
                    "the frozen subset manifest."
                )

            full_train_set = set(full_train_files)

            unknown_files = [p for p in selected_train_files if p not in full_train_set]

            if unknown_files:
                raise RuntimeError(
                    f"Subset manifest contains {len(unknown_files)} "
                    "files outside the training pool."
                )

            if len(selected_train_files) != len(set(selected_train_files)):
                raise RuntimeError(
                    "Duplicate training shards found " "in subset manifest."
                )

            self.train_files = selected_train_files

            log.info(
                f"Fixed training subset enabled: "
                f"{len(self.train_files)}/"
                f"{len(full_train_files)} shards "
                f"({len(self.train_files) / len(full_train_files):.2%})."
            )

        # STEP 2: Load or compute global statistics.
        self._setup_global_stats()

        # STEP 3: Load gene metadata.
        self._load_metadata(self.metadata_file)

        log.info(f"--- [setup] Finished on PID: {os.getpid()} ---")

    def _setup_global_stats(self):
        """
        Loads or computes global expression statistics, with DDP safety.
        """
        if self.global_mean is not None and self.global_std is not None:
            return

        is_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
        is_rank_zero = not is_ddp or torch.distributed.get_rank() == 0

        if os.path.exists(self.stats_path):
            if is_rank_zero:
                log.info("Loading pre-computed global stats from " f"{self.stats_path}")

            self._load_stats()
            return

        if not is_rank_zero:
            log.info(
                f"Rank {torch.distributed.get_rank()} waiting for "
                "rank 0 to compute stats..."
            )
            torch.distributed.barrier()
        else:
            log.info(
                "Global stats file not found. Computing on rank 0 "
                "from a subset of training data..."
            )

            self._compute_and_save_stats()

            if is_ddp:
                torch.distributed.barrier()

        if is_rank_zero:
            log.info("All ranks will now load the newly computed stats.")

        self._load_stats()

    def _load_stats(self):
        with open(self.stats_path, "r") as f:
            stats = json.load(f)

        self.global_mean = float(stats["mean"])
        self.global_std = float(stats["std"])

        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() == 0
        ):
            log.info(
                f"Successfully loaded stats: "
                f"mean={self.global_mean:.4f}, "
                f"std={self.global_std:.4f}"
            )

    def _compute_and_save_stats(self):
        num_samples_for_stats = 1_000_000
        samples_per_file = 100_000_000 / 3388

        num_files_to_load = math.ceil(num_samples_for_stats / samples_per_file)
        files_for_stats = self.train_files[:num_files_to_load]

        if not files_for_stats:
            raise RuntimeError("No training files available to compute statistics.")

        log.info(
            f"Using first {len(files_for_stats)} training files "
            "for streaming stats calculation..."
        )

        temp_ds = load_dataset(
            "parquet",
            data_files=files_for_stats,
            split="train",
        )

        n = 0
        mean = 0.0
        M2 = 0.0

        log.info("Starting streaming calculation of mean/std...")

        data_iterator = itertools.islice(
            temp_ds,
            num_samples_for_stats,
        )

        for i, cell in enumerate(data_iterator):
            if (i + 1) % 100_000 == 0:
                log.info(f"  ...processed {i + 1}/" f"{num_samples_for_stats} cells")

            if "expressions" not in cell or not cell["expressions"]:
                continue

            expressions = cell["expressions"]

            if expressions[0] < 0:
                expressions = expressions[1:]

            if not expressions:
                continue

            log1p_values = np.log1p(
                np.asarray(
                    expressions,
                    dtype=np.float64,
                )
            )

            new_count = log1p_values.size
            if new_count == 0:
                continue

            n_old = float(n)
            n_new = n_old + new_count

            current_mean = np.mean(log1p_values)
            delta = current_mean - mean

            mean += delta * (new_count / n_new)

            M2 += np.sum((log1p_values - current_mean) ** 2) + (delta**2) * (
                n_old * new_count / n_new
            )

            n += new_count

        if n < 2:
            raise RuntimeError(
                "Not enough data points (< 2) " "to compute standard deviation."
            )

        variance = M2 / (n - 1)
        std = np.sqrt(variance)

        log.info(
            f"Computed global stats from {n} total expression values: "
            f"mean={mean:.4f}, std={std:.4f}"
        )

        with open(self.stats_path, "w") as f:
            json.dump(
                {
                    "mean": float(mean),
                    "std": float(std),
                },
                f,
            )

        log.info(f"Saved global stats to {self.stats_path}")

    def _load_metadata(
        self,
        metadata_file_path: Optional[str],
    ):
        if self.gene_map:
            return

        if not metadata_file_path:
            raise ValueError("Metadata file path not set.")

        log.info("Loading gene metadata from LOCAL path: " f"{metadata_file_path}")

        gene_metadata_ds = load_dataset(
            "parquet",
            data_files=metadata_file_path,
            split="train",
        )

        sorted_genes = sorted(
            list(gene_metadata_ds),
            key=lambda x: x["token_id"],
        )

        self.gene_map = {entry["token_id"]: i for i, entry in enumerate(sorted_genes)}

    def _collate_fn(
        self,
        batch: List[Dict],
    ) -> Dict:
        if not batch:
            return {
                "indices": torch.empty(
                    0,
                    dtype=torch.long,
                ),
                "values": torch.empty(
                    0,
                    dtype=torch.float,
                ),
                "offsets": torch.tensor(
                    [0],
                    dtype=torch.long,
                ),
                "metadata": [],
            }

        indices_list = [torch.from_numpy(s["gene_indices"]) for s in batch]
        values_list = [torch.from_numpy(s["counts"]) for s in batch]

        indices = torch.cat(indices_list)
        values = torch.cat(values_list)

        values = torch.log1p(values.float())

        if self.global_mean is None or self.global_std is None:
            raise RuntimeError(
                "FATAL: Global normalization statistics are not "
                "available at collate time. This will lead to "
                "training instability. Check the DataModule's "
                "setup process."
            )

        values = (values - self.global_mean) / (self.global_std + 1e-6)

        if torch.rand(()) < 0.01:
            vstd = values.std().item()

            if not torch.isfinite(values).all() or vstd < 1e-6:
                print(
                    f"[NORM] WARNING: values std={vstd:.3e} "
                    f"finite={bool(torch.isfinite(values).all())}"
                )

        offsets = torch.tensor(
            [0] + [len(s["gene_indices"]) for s in batch],
            dtype=torch.long,
        ).cumsum(0)

        metadata = [s.get("metadata", {}) for s in batch]

        return {
            "indices": indices,
            "values": values,
            "offsets": offsets,
            "metadata": metadata,
        }

    def _create_dataloader(
        self,
        file_list: List[str],
        *,
        is_train: bool,
    ) -> DataLoader:
        if not file_list:
            log.warning(
                "Received an empty file list for dataloader creation. "
                "Returning an empty loader."
            )
            return DataLoader(
                [],
                batch_size=self.data_config.batch_size,
            )

        # DDP rank-level file sharding.
        # Each rank receives a disjoint subset of parquet files before
        # constructing the Hugging Face streaming dataset.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

            original_num_files = len(file_list)
            file_list = file_list[rank::world_size]

            log.info(
                f"DDP rank {rank}/{world_size}: "
                f"using {len(file_list)}/{original_num_files} "
                "parquet files."
            )

        hf_dataset = load_dataset(
            "parquet",
            data_files=file_list,
            streaming=True,
            split="train",
        )

        if is_train:
            rank = (
                torch.distributed.get_rank()
                if (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                )
                else 0
            )

            trainer = getattr(
                self,
                "trainer",
                None,
            )
            epoch = int(trainer.current_epoch) if trainer is not None else 0

            shuffle_seed = int(self.exp_config.random_seed) + epoch
            shuffle_buffer_size = 50_000

            log.info(
                f"Train shuffle: epoch={epoch}, " f"seed={shuffle_seed}, rank={rank}"
            )

            hf_dataset = hf_dataset.shuffle(
                seed=shuffle_seed,
                buffer_size=shuffle_buffer_size,
            )

            # Runtime-only performance knob.
            # Keep DataConfig.num_workers unchanged because that field is
            # also used by the project's validation-split logic.
            try:
                num_workers = int(
                    os.environ.get(
                        "GENEJEPA_TRAIN_WORKERS",
                        "0",
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    "GENEJEPA_TRAIN_WORKERS must be an integer >= 0."
                ) from exc

            if num_workers < 0:
                raise ValueError("GENEJEPA_TRAIN_WORKERS must be >= 0.")

            log.info("Training DataLoader using " f"num_workers={num_workers}")

        else:
            # Keep validation workers at 0 while benchmarking/training.
            # This avoids changing the validation pipeline while we tune
            # only training throughput.
            num_workers = 0

            log.info(
                "Setting validation dataloader workers to 0 "
                "for stability with streaming."
            )

        loader_kwargs = {
            "dataset": Tahoe100MDataset(
                hf_dataset,
                self.gene_map,
            ),
            "batch_size": self.data_config.batch_size,
            "collate_fn": self._collate_fn,
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
        }

        if num_workers > 0:
            # Each worker prefetches two batches.
            # Keep workers alive for the duration of the loader iterator.
            loader_kwargs.update(
                {
                    # "persistent_workers": True,
                    "persistent_workers": False,
                    "prefetch_factor": 2,
                }
            )
        else:
            loader_kwargs.update(
                {
                    "persistent_workers": False,
                }
            )

        return DataLoader(**loader_kwargs)

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(
            self.train_files,
            is_train=True,
        )

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(
            self.val_files,
            is_train=False,
        )
