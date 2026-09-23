import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

try:
    import umap
except ImportError:
    raise RuntimeError(
        "umap-learn is not installed. "
        "Run: uv pip install umap-learn"
    )


# ============================================================
# Configuration
# ============================================================

SEED = 42
CELLS_PER_CLASS = 800

EMB_PATH = (
    "benchmark_results/hlca18/"
    "epoch25_all18_embeddings.npy"
)

ROWS_PATH = (
    "benchmark_results/hlca18/"
    "epoch25_all18_rows.csv"
)

OUT_DIR = "benchmark_results/hlca18"

os.makedirs(
    OUT_DIR,
    exist_ok=True,
)


# ============================================================
# 1. Load frozen Epoch25 embeddings + metadata
# ============================================================

print("=" * 72)
print("EPOCH25 UMAP / t-SNE VISUALIZATION")
print("=" * 72)

X = np.load(
    EMB_PATH,
    mmap_mode="r",
)

rows = pd.read_csv(
    ROWS_PATH,
)

print()
print("Embedding:", X.shape)
print("Rows     :", len(rows))

if len(rows) != X.shape[0]:
    raise RuntimeError(
        "Embedding / metadata length mismatch."
    )


# ============================================================
# 2. Use TEST cells only
# ============================================================

test_rows = rows[
    rows["split"] == "test"
].copy()

print()
print(
    "Test cells:",
    len(test_rows),
)

print(
    "Test classes:",
    test_rows["label"].nunique(),
)


# ============================================================
# 3. Balanced sampling:
#    exactly 800 test cells per class
# ============================================================

counts = (
    test_rows["label"]
    .value_counts()
)

print()
print("Minimum test class size:", counts.min())

if counts.min() < CELLS_PER_CLASS:
    raise RuntimeError(
        f"A class has fewer than "
        f"{CELLS_PER_CLASS} test cells."
    )


sampled = (
    test_rows
    .groupby(
        "label",
        group_keys=False,
    )
    .sample(
        n=CELLS_PER_CLASS,
        random_state=SEED,
    )
    .sort_values(
        "label"
    )
    .copy()
)

sample_idx = sampled.index.to_numpy()

labels = sampled[
    "label"
].to_numpy()


print()
print(
    "Visualization cells:",
    len(sampled),
)

print(
    "Cells per class:",
    sampled["label"]
    .value_counts()
    .unique(),
)


# ============================================================
# 4. Extract embedding rows
# ============================================================

print()
print("Loading sampled embeddings...")

X_sample = np.asarray(
    X[sample_idx],
    dtype=np.float32,
)

if not np.isfinite(
    X_sample
).all():
    raise RuntimeError(
        "Non-finite embedding detected."
    )

print(
    "Sample embedding:",
    X_sample.shape,
)


# ============================================================
# 5. L2 normalize
#
# GeneJEPA was trained with cosine-style objectives.
# This removes differences in vector magnitude and lets
# the visualization focus on embedding direction.
# ============================================================

print()
print("L2-normalizing embeddings...")

X_norm = normalize(
    X_sample,
    norm="l2",
).astype(
    np.float32,
    copy=False,
)


# ============================================================
# 6. PCA 768 -> 50
#
# Standard preprocessing before UMAP/t-SNE.
# ============================================================

print()
print("Running PCA: 768 -> 50...")

start = time.time()

pca = PCA(
    n_components=50,
    random_state=SEED,
)

X_pca = pca.fit_transform(
    X_norm
).astype(
    np.float32,
    copy=False,
)

print(
    "PCA explained variance:",
    f"{pca.explained_variance_ratio_.sum():.4f}",
)

print(
    "PCA time:",
    f"{time.time() - start:.2f} sec",
)


# ============================================================
# 7. UMAP
# ============================================================

print()
print("Running UMAP...")

start = time.time()

umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=30,
    min_dist=0.30,
    metric="cosine",
    random_state=SEED,
)

X_umap = umap_model.fit_transform(
    X_pca
)

print(
    "UMAP time:",
    f"{(time.time() - start) / 60:.2f} min",
)


# ============================================================
# 8. t-SNE
# ============================================================

print()
print("Running t-SNE...")

start = time.time()

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    max_iter=1000,
    random_state=SEED,
)

X_tsne = tsne.fit_transform(
    X_pca
)

print(
    "t-SNE time:",
    f"{(time.time() - start) / 60:.2f} min",
)


# ============================================================
# 9. Save coordinates
# ============================================================

result = sampled[
    [
        "cell_index",
        "label",
        "split",
    ]
].copy()

result[
    "UMAP1"
] = X_umap[:, 0]

result[
    "UMAP2"
] = X_umap[:, 1]

result[
    "TSNE1"
] = X_tsne[:, 0]

result[
    "TSNE2"
] = X_tsne[:, 1]


coord_path = os.path.join(
    OUT_DIR,
    "epoch25_test_balanced800_umap_tsne.csv",
)

result.to_csv(
    coord_path,
    index=False,
)


# ============================================================
# 10. Consistent cell-type colors
# ============================================================

classes = sorted(
    np.unique(labels)
)

cmap = plt.get_cmap(
    "tab20"
)

color_map = {
    label: cmap(i)
    for i, label in enumerate(classes)
}


# ============================================================
# 11. UMAP plot
# ============================================================

print()
print("Drawing UMAP...")

fig, ax = plt.subplots(
    figsize=(14, 11)
)

for label in classes:

    mask = labels == label

    ax.scatter(
        X_umap[mask, 0],
        X_umap[mask, 1],
        s=8,
        alpha=0.65,
        label=label,
        color=color_map[label],
        linewidths=0,
    )


ax.set_title(
    "GeneJEPA Epoch25 — HLCA Test Cells\n"
    "UMAP of Frozen Cell Embeddings "
    "(800 cells/class)"
)

ax.set_xlabel(
    "UMAP 1"
)

ax.set_ylabel(
    "UMAP 2"
)

ax.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    markerscale=2,
    frameon=False,
)

fig.tight_layout()


umap_path = os.path.join(
    OUT_DIR,
    "epoch25_test_balanced800_umap.png",
)

fig.savefig(
    umap_path,
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 12. t-SNE plot
# ============================================================

print("Drawing t-SNE...")

fig, ax = plt.subplots(
    figsize=(14, 11)
)

for label in classes:

    mask = labels == label

    ax.scatter(
        X_tsne[mask, 0],
        X_tsne[mask, 1],
        s=8,
        alpha=0.65,
        label=label,
        color=color_map[label],
        linewidths=0,
    )


ax.set_title(
    "GeneJEPA Epoch25 — HLCA Test Cells\n"
    "t-SNE of Frozen Cell Embeddings "
    "(800 cells/class)"
)

ax.set_xlabel(
    "t-SNE 1"
)

ax.set_ylabel(
    "t-SNE 2"
)

ax.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    markerscale=2,
    frameon=False,
)

fig.tight_layout()


tsne_path = os.path.join(
    OUT_DIR,
    "epoch25_test_balanced800_tsne.png",
)

fig.savefig(
    tsne_path,
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Final
# ============================================================

print()
print("=" * 72)
print("VISUALIZATION COMPLETE")
print("=" * 72)

print()
print("Coordinates:")
print(coord_path)

print()
print("UMAP:")
print(umap_path)

print()
print("t-SNE:")
print(tsne_path)

print()
print("Classes:", len(classes))
print("Cells  :", len(sampled))

print("=" * 72)
