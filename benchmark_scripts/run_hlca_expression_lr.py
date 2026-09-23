import os
import time
import argparse
import warnings

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
)
from sklearn.exceptions import ConvergenceWarning


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--max-train",
    type=int,
    default=0,
    help="0 = use all training cells",
)

parser.add_argument(
    "--max-test",
    type=int,
    default=0,
    help="0 = use all test cells",
)

parser.add_argument(
    "--max-iter",
    type=int,
    default=300,
)

args = parser.parse_args()


# ============================================================
# Fixed benchmark configuration
# ============================================================

SEED = 42

HLCA_PATH = (
    "benchmark_data/hlca/"
    "688185ad-11c2-4172-a53a-f4f1f4076860.h5ad"
)

BENCHMARK_PATH = (
    "benchmark_data/hlca/"
    "benchmark_ann_level3_all18_seed42.csv"
)

OUT_DIR = "benchmark_results/hlca18"

os.makedirs(
    OUT_DIR,
    exist_ok=True,
)


print("=" * 72)
print("HLCA EXPRESSION-ONLY LOGISTIC REGRESSION BASELINE")
print("=" * 72)

print()
print("Input representation:")
print("  HLCA adata.X")
print("  normalized/log-like expression")
print("  NO GeneJEPA")
print("  NO PCA")
print()


# ============================================================
# 1. Load frozen benchmark metadata
# ============================================================

benchmark = pd.read_csv(
    BENCHMARK_PATH
)

required_cols = {
    "cell_index",
    "label",
    "split",
}

missing = required_cols - set(
    benchmark.columns
)

if missing:
    raise RuntimeError(
        f"Missing benchmark columns: {missing}"
    )


print("Benchmark cells :", len(benchmark))
print(
    "Classes         :",
    benchmark["label"].nunique(),
)

print(
    "Split counts    :",
    benchmark["split"]
    .value_counts()
    .to_dict(),
)


# ============================================================
# 2. Resolve train / test rows
# ============================================================

train_meta = benchmark[
    benchmark["split"] == "train"
].copy()

test_meta = benchmark[
    benchmark["split"] == "test"
].copy()


# ------------------------------------------------------------
# Deterministic stratified-like downsampling for smoke tests.
#
# We sample within every class so rare classes are preserved.
# Full run does NOT sample.
# ------------------------------------------------------------

def balanced_subsample(
    df,
    max_rows,
    seed,
):
    if (
        max_rows is None
        or max_rows <= 0
        or len(df) <= max_rows
    ):
        return df.copy()

    frac = max_rows / len(df)

    pieces = []

    for label, group in df.groupby(
        "label",
        sort=True,
    ):
        n = max(
            1,
            int(round(len(group) * frac)),
        )

        n = min(
            n,
            len(group),
        )

        pieces.append(
            group.sample(
                n=n,
                random_state=seed,
            )
        )

    out = pd.concat(
        pieces,
        axis=0,
    )

    # If rounding gave slightly too many rows,
    # trim deterministically.
    if len(out) > max_rows:
        out = out.sample(
            n=max_rows,
            random_state=seed,
        )

    return out.copy()


train_meta = balanced_subsample(
    train_meta,
    args.max_train,
    SEED,
)

test_meta = balanced_subsample(
    test_meta,
    args.max_test,
    SEED,
)


print()
print("Selected train :", len(train_meta))
print("Selected test  :", len(test_meta))

print(
    "Train classes  :",
    train_meta["label"].nunique(),
)

print(
    "Test classes   :",
    test_meta["label"].nunique(),
)


# ============================================================
# 3. Load adata.X directly as CSR
#
# Important:
# Do NOT densify.
#
# HLCA X is stored in the h5ad as CSR:
#
#   data
#   indices
#   indptr
#
# Loading the full CSR is ~12.7 GiB.
# ============================================================

print()
print("Loading HLCA adata.X sparse CSR...")
print(
    "This may take a little while."
)

load_start = time.time()


with h5py.File(
    HLCA_PATH,
    "r",
) as f:

    X_group = f["X"]

    shape = tuple(
        int(x)
        for x in X_group.attrs["shape"]
    )

    print()
    print("Full X shape:", shape)

    data = X_group[
        "data"
    ][:]

    indices = X_group[
        "indices"
    ][:]

    indptr = X_group[
        "indptr"
    ][:]


X_full = sp.csr_matrix(
    (
        data,
        indices,
        indptr,
    ),
    shape=shape,
)


# Release standalone references.
del data
del indices
del indptr


print(
    "Loaded CSR:"
)

print(
    "  shape :",
    X_full.shape,
)

print(
    "  nnz   :",
    f"{X_full.nnz:,}",
)

print(
    "  dtype :",
    X_full.dtype,
)

print(
    "  time  :",
    f"{time.time() - load_start:.2f} sec",
)


# ============================================================
# 4. Extract EXACT SAME benchmark cells
# ============================================================

train_cell_index = (
    train_meta["cell_index"]
    .to_numpy(
        dtype=np.int64
    )
)

test_cell_index = (
    test_meta["cell_index"]
    .to_numpy(
        dtype=np.int64
    )
)


print()
print("Extracting train expression...")

X_train = X_full[
    train_cell_index
].tocsr()


print("Extracting test expression...")

X_test = X_full[
    test_cell_index
].tocsr()


# Full matrix no longer needed.
del X_full


y_train = (
    train_meta["label"]
    .astype(str)
    .to_numpy()
)

y_test = (
    test_meta["label"]
    .astype(str)
    .to_numpy()
)


print()
print("Train X:", X_train.shape)
print(
    "Train nnz:",
    f"{X_train.nnz:,}",
)

print()
print("Test X :", X_test.shape)
print(
    "Test nnz:",
    f"{X_test.nnz:,}",
)


# ============================================================
# 5. Integrity checks
# ============================================================

if X_train.shape[0] != len(
    y_train
):
    raise RuntimeError(
        "Train X/y mismatch."
    )

if X_test.shape[0] != len(
    y_test
):
    raise RuntimeError(
        "Test X/y mismatch."
    )

if X_train.shape[1] != 27402:
    raise RuntimeError(
        "Unexpected gene dimension."
    )


if not np.isfinite(
    X_train.data
).all():
    raise RuntimeError(
        "Non-finite values in train X."
    )

if not np.isfinite(
    X_test.data
).all():
    raise RuntimeError(
        "Non-finite values in test X."
    )


# ============================================================
# 6. Logistic Regression
#
# Same conceptual classifier as GeneJEPA benchmark:
#
#   L2 Logistic Regression
#   C = 1.0
#   class_weight = None
#   seed = 42
#
# Difference:
#   solver = SAGA
#
# Reason:
#   27,402-dimensional sparse expression matrix is radically
#   larger than the dense 768-d GeneJEPA embedding.
#
# SAGA directly supports large sparse multinomial problems.
#
# We are changing the numerical optimizer, NOT the statistical
# model being fitted.
# ============================================================

print()
print("=" * 72)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 72)

print()
print("Penalty      : L2")
print("C            : 1.0")
print("class_weight : None")
print("solver       : saga")
print("max_iter     :", args.max_iter)
print("tol          : 1e-4")
print("random_state : 42")
print()
print("No GeneJEPA.")
print("No PCA.")
print("No feature scaling.")


clf = LogisticRegression(
    penalty="l2",
    C=1.0,
    class_weight=None,
    solver="saga",
    max_iter=args.max_iter,
    tol=1e-4,
    random_state=SEED,
    verbose=1,
)


print()
print("Fitting...")

start = time.time()


with warnings.catch_warnings(
    record=True
) as caught:

    warnings.simplefilter(
        "always",
        ConvergenceWarning,
    )

    clf.fit(
        X_train,
        y_train,
    )


elapsed = time.time() - start


convergence_warning = any(
    issubclass(
        w.category,
        ConvergenceWarning,
    )
    for w in caught
)


print()
print(
    "Training time:",
    f"{elapsed / 60:.2f} min",
)

print(
    "Iterations:",
    clf.n_iter_,
)

print(
    "ConvergenceWarning:",
    convergence_warning,
)


# ============================================================
# 7. Prediction
# ============================================================

print()
print("Predicting test set...")

pred = clf.predict(
    X_test
)


# ============================================================
# 8. Metrics
# ============================================================

accuracy = accuracy_score(
    y_test,
    pred,
)

macro_f1 = f1_score(
    y_test,
    pred,
    average="macro",
)

balanced_acc = balanced_accuracy_score(
    y_test,
    pred,
)


print()
print("=" * 72)
print("RESULT")
print("=" * 72)

print(
    f"Accuracy          : "
    f"{accuracy:.6f}"
)

print(
    f"Macro-F1          : "
    f"{macro_f1:.6f}"
)

print(
    f"Balanced Accuracy : "
    f"{balanced_acc:.6f}"
)


# ============================================================
# 9. Classification report
# ============================================================

print()
print("=" * 72)
print("CLASSIFICATION REPORT")
print("=" * 72)

print(
    classification_report(
        y_test,
        pred,
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# 10. Save outputs
# ============================================================

suffix = (
    "full"
    if (
        args.max_train == 0
        and args.max_test == 0
    )
    else (
        f"smoke_"
        f"{len(y_train)}train_"
        f"{len(y_test)}test"
    )
)


summary = pd.DataFrame(
    [
        {
            "representation":
                "HLCA_adata_X",

            "dimension":
                X_train.shape[1],

            "n_train":
                len(y_train),

            "n_test":
                len(y_test),

            "solver":
                "saga",

            "penalty":
                "l2",

            "C":
                1.0,

            "max_iter":
                args.max_iter,

            "iterations":
                int(
                    np.max(
                        clf.n_iter_
                    )
                ),

            "converged":
                not convergence_warning,

            "accuracy":
                accuracy,

            "macro_f1":
                macro_f1,

            "balanced_accuracy":
                balanced_acc,

            "training_seconds":
                elapsed,
        }
    ]
)


summary_path = os.path.join(
    OUT_DIR,
    (
        f"expression_lr_"
        f"{suffix}.csv"
    ),
)

summary.to_csv(
    summary_path,
    index=False,
)


prediction_df = pd.DataFrame(
    {
        "cell_index":
            test_meta[
                "cell_index"
            ].to_numpy(),

        "true_label":
            y_test,

        "predicted_label":
            pred,

        "correct":
            y_test == pred,
    }
)


prediction_path = os.path.join(
    OUT_DIR,
    (
        f"expression_lr_"
        f"predictions_"
        f"{suffix}.csv"
    ),
)

prediction_df.to_csv(
    prediction_path,
    index=False,
)


print()
print("Saved:")
print(summary_path)
print(prediction_path)

print()
print("=" * 72)
print("EXPRESSION LR COMPLETE")
print("=" * 72)
