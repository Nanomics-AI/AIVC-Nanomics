import os

import anndata as ad
import pandas as pd

from sklearn.model_selection import train_test_split


HLCA_PATH = (
    "benchmark_data/hlca/"
    "688185ad-11c2-4172-a53a-f4f1f4076860.h5ad"
)

OUT_PATH = (
    "benchmark_data/hlca/"
    "benchmark_ann_level3_all18_seed42.csv"
)

LABEL_COL = "ann_level_3"

MIN_CLASS_CELLS = 1000
TEST_SIZE = 0.20
SEED = 42


print("=" * 72)
print("BUILD FULL HLCA 18-CLASS BENCHMARK")
print("=" * 72)


# ============================================================
# 1. Open HLCA
# ============================================================

adata = ad.read_h5ad(
    HLCA_PATH,
    backed="r",
)

obs = adata.obs[[LABEL_COL]].copy()

obs["label"] = (
    obs[LABEL_COL]
    .astype(str)
    .str.strip()
)

obs["cell_index"] = range(adata.n_obs)


# ============================================================
# 2. Remove missing / invalid labels
# ============================================================

invalid_labels = {
    "None",
    "nan",
    "NaN",
    "",
}

obs = obs[
    ~obs["label"].isin(invalid_labels)
].copy()


print()
print("After removing invalid labels:")
print("Cells  :", len(obs))
print("Classes:", obs["label"].nunique())


# ============================================================
# 3. Remove rare classes
# ============================================================

class_counts = obs["label"].value_counts()

rare_classes = class_counts[
    class_counts < MIN_CLASS_CELLS
]

print()
print(
    f"Removing classes with < {MIN_CLASS_CELLS} cells:"
)

for label, count in rare_classes.sort_values().items():
    print(
        f"  {label:<35} {count:>6}"
    )


keep_classes = class_counts[
    class_counts >= MIN_CLASS_CELLS
].index

obs = obs[
    obs["label"].isin(keep_classes)
].copy()


print()
print("=" * 72)
print("FILTERED BENCHMARK POOL")
print("=" * 72)

print("Cells  :", len(obs))
print("Classes:", obs["label"].nunique())

print()
print("Class counts:")

print(
    obs["label"]
    .value_counts()
    .sort_values(ascending=False)
)


# ============================================================
# 4. Use ALL remaining cells
#    Only split into fixed train/test sets.
# ============================================================

train_df, test_df = train_test_split(
    obs,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=obs["label"],
)

train_df = train_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
test_df["split"] = "test"


benchmark = pd.concat(
    [
        train_df,
        test_df,
    ],
    axis=0,
)

benchmark = benchmark[
    [
        "cell_index",
        "label",
        "split",
    ]
].copy()

benchmark = benchmark.sort_values(
    [
        "split",
        "label",
        "cell_index",
    ]
).reset_index(drop=True)


# ============================================================
# 5. Diagnostics
# ============================================================

print()
print("=" * 72)
print("FINAL BENCHMARK")
print("=" * 72)

print("Total :", len(benchmark))

print(
    "Train :",
    int(
        (benchmark["split"] == "train").sum()
    ),
)

print(
    "Test  :",
    int(
        (benchmark["split"] == "test").sum()
    ),
)

print(
    "Classes:",
    benchmark["label"].nunique(),
)


print()
print("TRAIN class counts:")

print(
    benchmark.loc[
        benchmark["split"] == "train",
        "label",
    ]
    .value_counts()
    .sort_values()
)


print()
print("TEST class counts:")

print(
    benchmark.loc[
        benchmark["split"] == "test",
        "label",
    ]
    .value_counts()
    .sort_values()
)


# ============================================================
# 6. Save
# ============================================================

benchmark.to_csv(
    OUT_PATH,
    index=False,
)


print()
print("=" * 72)
print("SAVED")
print("=" * 72)

print(OUT_PATH)


adata.file.close()
