import os
import json
import time
import argparse
import sys

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from genejepa.train import JepaLightningModule


# ============================================================
# Fixed project paths
# ============================================================

HLCA_PATH = (
    "benchmark_data/hlca/"
    "688185ad-11c2-4172-a53a-f4f1f4076860.h5ad"
)

BENCHMARK_CSV = (
    "benchmark_data/hlca/"
    "benchmark_ann_level3_all18_seed42.csv"
)

MANIFEST_PATH = "hf_data_cache/local_file_manifest.json"
STATS_PATH = "hf_data_cache/global_stats.json"

OUT_DIR = "benchmark_results/hlca18"


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--ckpt",
    required=True,
)

parser.add_argument(
    "--name",
    required=True,
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
)

parser.add_argument(
    "--max-cells",
    type=int,
    default=0,
    help="0 means use all benchmark cells",
)

parser.add_argument(
    "--num-shards",
    type=int,
    default=1,
)

parser.add_argument(
    "--shard-id",
    type=int,
    default=0,
)

args = parser.parse_args()


# ============================================================
# Setup
# ============================================================

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 72)
print("HLCA -> GeneJEPA BATCH EMBEDDING EXTRACTION")
print("=" * 72)

print("Checkpoint :", args.ckpt)
print("Name       :", args.name)
print("Batch size :", args.batch_size)
print("Max cells  :", args.max_cells if args.max_cells else "ALL")


if not os.path.isfile(args.ckpt):
    raise FileNotFoundError(args.ckpt)


# ============================================================
# 1. Load benchmark cohort
# ============================================================

benchmark = pd.read_csv(BENCHMARK_CSV)

required = {
    "cell_index",
    "label",
    "split",
}

if not required.issubset(benchmark.columns):
    raise RuntimeError(
        f"Benchmark CSV missing columns: "
        f"{required - set(benchmark.columns)}"
    )

if benchmark["cell_index"].duplicated().any():
    raise RuntimeError("Duplicate cell_index found.")


# Sort by original HLCA row number.
# This makes backed H5AD access much more sequential.

benchmark = (
    benchmark
    .sort_values("cell_index")
    .reset_index(drop=True)
)

# ------------------------------------------------------------
# Optional global smoke-test limit
# ------------------------------------------------------------

if args.max_cells > 0:
    benchmark = benchmark.iloc[:args.max_cells].copy()


# ------------------------------------------------------------
# Split the benchmark into independent contiguous shards.
#
# Example:
# num_shards=2
# shard_id=0 -> first half
# shard_id=1 -> second half
# ------------------------------------------------------------

if args.num_shards < 1:
    raise ValueError("--num-shards must be >= 1")

if not 0 <= args.shard_id < args.num_shards:
    raise ValueError(
        "--shard-id must satisfy "
        "0 <= shard_id < num_shards"
    )

total_before_shard = len(benchmark)

boundaries = np.linspace(
    0,
    total_before_shard,
    args.num_shards + 1,
    dtype=int,
)

shard_start = int(
    boundaries[args.shard_id]
)

shard_end = int(
    boundaries[args.shard_id + 1]
)

benchmark = benchmark.iloc[
    shard_start:shard_end
].copy()

N = len(benchmark)


print()
print(
    f"Shard          : "
    f"{args.shard_id}/{args.num_shards}"
)

print(
    f"Global rows    : "
    f"[{shard_start}, {shard_end})"
)

print()
print("Cells to process:", N)
print("Classes         :", benchmark["label"].nunique())

print(
    "Train/Test      :",
    benchmark["split"].value_counts().to_dict(),
)


# ============================================================
# 2. Save exact row ordering
# ============================================================

rows_path = os.path.join(
    OUT_DIR,
    f"{args.name}_rows.csv",
)

benchmark.to_csv(
    rows_path,
    index=False,
)

print()
print("Row metadata:")
print(rows_path)


# ============================================================
# 3. Tahoe vocabulary + normalization
# ============================================================

with open(MANIFEST_PATH, "r") as f:
    manifest = json.load(f)

meta = pd.read_parquet(
    manifest["metadata_file"]
)

meta = (
    meta
    .sort_values("token_id")
    .reset_index(drop=True)
)


with open(STATS_PATH, "r") as f:
    stats = json.load(f)

global_mean = float(stats["mean"])
global_std = float(stats["std"])

print()
print("Tahoe vocabulary:", len(meta))
print("Global mean     :", global_mean)
print("Global std      :", global_std)


# ============================================================
# 4. Build Tahoe lookup
#
# Formal rule:
#   1. Ensembl first
#   2. symbol fallback
# ============================================================

token_to_contiguous = {
    int(token_id): i
    for i, token_id in enumerate(
        meta["token_id"]
    )
}

ensembl_to_gj = {}
symbol_to_gj = {}

for symbol, ensembl, token_id in zip(
    meta["gene_symbol"],
    meta["ensembl_id"],
    meta["token_id"],
):

    gj_idx = token_to_contiguous[
        int(token_id)
    ]

    ens = (
        str(ensembl)
        .strip()
        .split(".")[0]
        .upper()
    )

    sym = (
        str(symbol)
        .strip()
        .upper()
    )

    if ens and ens != "NAN":
        ensembl_to_gj[ens] = gj_idx

    if sym and sym != "NAN":
        symbol_to_gj[sym] = gj_idx


# ============================================================
# 5. Open HLCA in backed mode
# ============================================================

adata = ad.read_h5ad(
    HLCA_PATH,
    backed="r",
)

hlca_ensembl = np.asarray(
    [
        str(x)
        .split(".")[0]
        .upper()
        for x in adata.var_names
    ]
)

hlca_symbols = (
    adata.var["feature_name"]
    .astype(str)
    .str.strip()
    .str.upper()
    .to_numpy()
)


# Direct array:
#
# HLCA column index
#       ↓
# GeneJEPA vocabulary index
#
hlca_to_gj = np.full(
    adata.n_vars,
    -1,
    dtype=np.int64,
)


for i, (ens, sym) in enumerate(
    zip(
        hlca_ensembl,
        hlca_symbols,
    )
):

    if ens in ensembl_to_gj:
        hlca_to_gj[i] = ensembl_to_gj[ens]

    elif sym in symbol_to_gj:
        hlca_to_gj[i] = symbol_to_gj[sym]


mapped = int(
    np.count_nonzero(
        hlca_to_gj >= 0
    )
)

print()
print("HLCA genes :", adata.n_vars)
print("Mapped     :", mapped)

if mapped != adata.n_vars:
    print(
        "WARNING:",
        adata.n_vars - mapped,
        "HLCA genes are unmapped.",
    )


# ============================================================
# 6. Load checkpoint
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print("Device:", device)

print()
print("Loading checkpoint...")

module = (
    JepaLightningModule
    .load_from_checkpoint(
        args.ckpt,
        map_location="cpu",
    )
)

module.eval()
module.to(device)

print("Checkpoint loaded.")


# ============================================================
# 7. Create disk-backed output
#
# We do NOT keep 578k × 768 in RAM.
# npy is written incrementally.
# ============================================================

embedding_path = os.path.join(
    OUT_DIR,
    f"{args.name}_embeddings.npy",
)

progress_path = os.path.join(
    OUT_DIR,
    f"{args.name}_progress.json",
)


embedding_dim = 768


# Fresh extraction for now.
# Refuse to silently overwrite an existing result.
if os.path.exists(embedding_path):
    raise RuntimeError(
        f"Output already exists:\n{embedding_path}\n"
        "Rename/remove it before starting a new extraction."
    )


embeddings_out = np.lib.format.open_memmap(
    embedding_path,
    mode="w+",
    dtype=np.float32,
    shape=(N, embedding_dim),
)


# ============================================================
# 8. Batch extraction
# ============================================================

cell_indices = (
    benchmark["cell_index"]
    .to_numpy(dtype=np.int64)
)

raw_X = adata.raw.X

batch_size = args.batch_size

total_batches = (
    N + batch_size - 1
) // batch_size


print()
print("=" * 72)
print("START EXTRACTION")
print("=" * 72)

print("Total cells  :", N)
print("Total batches:", total_batches)

start_time = time.time()


for batch_no, start in enumerate(
    range(0, N, batch_size),
    start=1,
):

    end = min(
        start + batch_size,
        N,
    )

    ids = cell_indices[start:end]


    # --------------------------------------------------------
    # Read this group of cells from the backed sparse matrix.
    # cell indices are sorted, which is important for efficient
    # HDF5 access.
    # --------------------------------------------------------

    sub = raw_X[ids, :]

    if hasattr(sub, "to_memory"):
        sub = sub.to_memory()

    if not sp.issparse(sub):
        sub = sp.csr_matrix(sub)

    sub = sub.tocsr()


    # --------------------------------------------------------
    # Convert HLCA columns -> GeneJEPA vocabulary indices.
    #
    # Since nearly all/all genes map, this can be done directly
    # on CSR nonzero entries without looping over genes.
    # --------------------------------------------------------

    mapped_indices_np = hlca_to_gj[
        sub.indices
    ]

    valid = mapped_indices_np >= 0

    # In our current mapping this should effectively be all
    # entries, but keep this check for safety.
    if not valid.all():

        new_rows = []

        for r in range(sub.shape[0]):

            s = sub.indptr[r]
            e = sub.indptr[r + 1]

            row_valid = valid[s:e]

            row_cols = mapped_indices_np[s:e][
                row_valid
            ]

            row_vals = sub.data[s:e][
                row_valid
            ]

            new_rows.append(
                sp.csr_matrix(
                    (
                        row_vals,
                        (
                            np.zeros(
                                len(row_cols),
                                dtype=np.int64,
                            ),
                            row_cols,
                        ),
                    ),
                    shape=(
                        1,
                        len(meta),
                    ),
                )
            )

        gj_sparse = sp.vstack(
            new_rows,
            format="csr",
        )

        gj_indices_np = (
            gj_sparse.indices
            .astype(
                np.int64,
                copy=False,
            )
        )

        counts_np = (
            gj_sparse.data
            .astype(
                np.float32,
                copy=False,
            )
        )

        offsets_np = (
            gj_sparse.indptr
            .astype(
                np.int64,
                copy=False,
            )
        )

    else:

        gj_indices_np = (
            mapped_indices_np
            .astype(
                np.int64,
                copy=False,
            )
        )

        counts_np = (
            sub.data
            .astype(
                np.float32,
                copy=False,
            )
        )

        offsets_np = (
            sub.indptr
            .astype(
                np.int64,
                copy=False,
            )
        )


    # --------------------------------------------------------
    # Same expression preprocessing used in training
    # --------------------------------------------------------

    indices = torch.from_numpy(
        gj_indices_np
    )

    values = torch.from_numpy(
        counts_np
    )

    offsets = torch.from_numpy(
        offsets_np
    )


    values = torch.log1p(
        values.float()
    )

    values = (
        values - global_mean
    ) / (
        global_std + 1e-6
    )


    if not torch.isfinite(values).all():
        raise RuntimeError(
            f"Non-finite values in batch {batch_no}"
        )


    indices = indices.to(
        device,
        non_blocking=True,
    )

    values = values.to(
        device,
        non_blocking=True,
    )

    offsets = offsets.to(
        device,
        non_blocking=True,
    )


    # --------------------------------------------------------
    # EMA Teacher inference
    # --------------------------------------------------------

    with torch.inference_mode():

        if device.type == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):

                emb = module.model.get_embedding(
                    indices=indices,
                    values=values,
                    offsets=offsets,
                    use_teacher=True,
                )

        else:

            emb = module.model.get_embedding(
                indices=indices,
                values=values,
                offsets=offsets,
                use_teacher=True,
            )


    emb = (
        emb
        .float()
        .cpu()
        .numpy()
    )


    expected_shape = (
        end - start,
        embedding_dim,
    )

    if emb.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected embedding shape "
            f"{emb.shape}, expected "
            f"{expected_shape}"
        )

    if not np.isfinite(emb).all():
        raise RuntimeError(
            f"Non-finite embedding "
            f"in batch {batch_no}"
        )


    embeddings_out[
        start:end
    ] = emb


    # --------------------------------------------------------
    # Flush periodically to disk
    # --------------------------------------------------------

    if (
        batch_no % 10 == 0
        or end == N
    ):

        embeddings_out.flush()

        elapsed = (
            time.time()
            - start_time
        )

        rate = (
            end / elapsed
            if elapsed > 0
            else 0.0
        )

        remaining = (
            (N - end) / rate
            if rate > 0
            else float("nan")
        )

        progress = {
            "processed_cells": int(end),
            "total_cells": int(N),
            "batch": int(batch_no),
            "total_batches": int(total_batches),
            "cells_per_second": float(rate),
            "elapsed_seconds": float(elapsed),
            "estimated_remaining_seconds": float(
                remaining
            ),
        }

        with open(
            progress_path,
            "w",
        ) as f:
            json.dump(
                progress,
                f,
                indent=2,
            )

        print(
            f"[{batch_no:5d}/{total_batches}] "
            f"cells={end:7d}/{N} "
            f"rate={rate:.2f} cells/s "
            f"ETA={remaining / 3600:.2f} h"
        )


# ============================================================
# 9. Final diagnostics
# ============================================================

embeddings_out.flush()

elapsed = time.time() - start_time


print()
print("=" * 72)
print("EXTRACTION COMPLETE")
print("=" * 72)

print("Embedding file:")
print(embedding_path)

print()
print("Rows file:")
print(rows_path)

print()
print(
    "Shape:",
    embeddings_out.shape,
)

print(
    "Size : %.2f GB"
    % (
        os.path.getsize(
            embedding_path
        )
        / 1024**3
    )
)

print(
    "Elapsed: %.2f minutes"
    % (
        elapsed / 60
    )
)

print(
    "Mean throughput: %.2f cells/s"
    % (
        N / elapsed
    )
)


adata.file.close()
