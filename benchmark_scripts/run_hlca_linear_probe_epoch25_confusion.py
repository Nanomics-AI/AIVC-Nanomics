import os
import time
import argparse

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler


EMB_PATH = (
    "benchmark_results/hlca18/"
    "epoch25_all18_embeddings.npy"
)

ROWS_PATH = (
    "benchmark_results/hlca18/"
    "epoch25_all18_rows.csv"
)

OUT_DIR = "benchmark_results/hlca18"


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

args = parser.parse_args()


print("=" * 72)
print("GENEJEPA HLCA LINEAR PROBE")
print("=" * 72)


# ============================================================
# 1. Load embedding as memory map
# ============================================================

X = np.load(
    EMB_PATH,
    mmap_mode="r",
)

rows = pd.read_csv(
    ROWS_PATH,
)


print()
print("Embedding shape:", X.shape)
print("Metadata rows  :", len(rows))


if X.shape[0] != len(rows):
    raise RuntimeError(
        "Embedding / metadata row mismatch."
    )

if X.shape[1] != 768:
    raise RuntimeError(
        f"Unexpected embedding dimension: {X.shape[1]}"
    )


# ============================================================
# 2. Fixed train/test split
# ============================================================

train_idx = np.flatnonzero(
    rows["split"].to_numpy() == "train"
)

test_idx = np.flatnonzero(
    rows["split"].to_numpy() == "test"
)


# Optional deterministic smoke-test subset.
# Do NOT randomly resplit the benchmark.
if args.max_train > 0:
    train_idx = train_idx[:args.max_train]

if args.max_test > 0:
    test_idx = test_idx[:args.max_test]


X_train = np.asarray(
    X[train_idx],
    dtype=np.float32,
)

X_test = np.asarray(
    X[test_idx],
    dtype=np.float32,
)

y_train = rows.iloc[
    train_idx
]["label"].to_numpy()

y_test = rows.iloc[
    test_idx
]["label"].to_numpy()


print()
print("Train X:", X_train.shape)
print("Test X :", X_test.shape)

print(
    "Train classes:",
    len(np.unique(y_train)),
)

print(
    "Test classes :",
    len(np.unique(y_test)),
)


# ============================================================
# 3. Feature standardization
#
# Fit ONLY on training data.
# ============================================================

print()
print("Fitting StandardScaler...")

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
).astype(
    np.float32,
    copy=False,
)

X_test = scaler.transform(
    X_test
).astype(
    np.float32,
    copy=False,
)


# ============================================================
# 4. Strict linear probe
#
# No hidden layer.
# No MLP.
# L2-regularized multinomial Logistic Regression.
# ============================================================

print()
print("Training Logistic Regression...")

start = time.time()

clf = LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    max_iter=5000,
    tol=1e-4,
    random_state=42,
)

clf.fit(
    X_train,
    y_train,
)

elapsed = time.time() - start


# ============================================================
# 5. Prediction
# ============================================================

print()
print("Predicting test set...")

pred = clf.predict(
    X_test
)


# ============================================================
# 5.1 Save predictions + confusion matrix analysis
# ============================================================

print()
print("Building confusion matrix analysis...")

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

analysis_suffix = (
    "full"
    if args.max_train == 0
    else f"smoke_{len(y_train)}"
)

# Fixed alphabetical class order.
labels = np.array(
    sorted(np.unique(y_test))
)

# ------------------------------------------------------------
# Save per-cell predictions.
# test_position preserves the exact order of X_test / y_test.
# ------------------------------------------------------------

prediction_df = pd.DataFrame(
    {
        "test_position": np.arange(len(y_test)),
        "true_label": y_test,
        "predicted_label": pred,
        "correct": (y_test == pred),
    }
)

prediction_path = os.path.join(
    OUT_DIR,
    f"epoch25_predictions_{analysis_suffix}.csv",
)

prediction_df.to_csv(
    prediction_path,
    index=False,
)


# ------------------------------------------------------------
# Raw confusion matrix
# rows = true labels
# cols = predicted labels
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    pred,
    labels=labels,
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels,
)

cm_df.index.name = "true_label"
cm_df.columns.name = "predicted_label"

cm_path = os.path.join(
    OUT_DIR,
    f"epoch25_confusion_matrix_raw_{analysis_suffix}.csv",
)

cm_df.to_csv(cm_path)


# ------------------------------------------------------------
# Row-normalized confusion matrix
# Each row sums to 1.
# This answers:
# "For a true cell type, where did its cells get predicted?"
# ------------------------------------------------------------

row_sums = cm.sum(
    axis=1,
    keepdims=True,
)

cm_norm = np.divide(
    cm.astype(np.float64),
    row_sums,
    out=np.zeros_like(cm, dtype=np.float64),
    where=row_sums != 0,
)

cm_norm_df = pd.DataFrame(
    cm_norm,
    index=labels,
    columns=labels,
)

cm_norm_df.index.name = "true_label"
cm_norm_df.columns.name = "predicted_label"

cm_norm_path = os.path.join(
    OUT_DIR,
    f"epoch25_confusion_matrix_row_normalized_{analysis_suffix}.csv",
)

cm_norm_df.to_csv(cm_norm_path)


# ------------------------------------------------------------
# Top-3 wrong destinations for each true class
# ------------------------------------------------------------

top_errors = []

for i, true_label in enumerate(labels):

    candidates = []

    for j, predicted_label in enumerate(labels):

        if i == j:
            continue

        count = int(cm[i, j])

        if count == 0:
            continue

        candidates.append(
            (
                count,
                float(cm_norm[i, j]),
                predicted_label,
            )
        )

    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    for rank, (
        count,
        fraction,
        predicted_label,
    ) in enumerate(candidates[:3], start=1):

        top_errors.append(
            {
                "true_label": true_label,
                "rank": rank,
                "predicted_as": predicted_label,
                "count": count,
                "fraction_of_true_class": fraction,
            }
        )

top_errors_df = pd.DataFrame(
    top_errors
)

top_errors_path = os.path.join(
    OUT_DIR,
    f"epoch25_top3_misclassifications_{analysis_suffix}.csv",
)

top_errors_df.to_csv(
    top_errors_path,
    index=False,
)


# ------------------------------------------------------------
# Row-normalized heatmap
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(14, 12)
)

im = ax.imshow(
    cm_norm,
    aspect="auto",
)

fig.colorbar(
    im,
    ax=ax,
    label="Fraction of true class",
)

ax.set_xticks(
    np.arange(len(labels))
)

ax.set_yticks(
    np.arange(len(labels))
)

ax.set_xticklabels(
    labels,
    rotation=90,
)

ax.set_yticklabels(
    labels,
)

ax.set_xlabel(
    "Predicted cell type"
)

ax.set_ylabel(
    "True cell type"
)

ax.set_title(
    "GeneJEPA Epoch25 HLCA Linear Probe\\n"
    "Row-normalized confusion matrix"
)

fig.tight_layout()

heatmap_path = os.path.join(
    OUT_DIR,
    f"epoch25_confusion_matrix_row_normalized_{analysis_suffix}.png",
)

fig.savefig(
    heatmap_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)


print()
print("Confusion analysis saved:")
print("Predictions :", prediction_path)
print("Raw matrix  :", cm_path)
print("Normalized  :", cm_norm_path)
print("Top-3 errors:", top_errors_path)
print("Heatmap     :", heatmap_path)


# ============================================================
# 6. Metrics
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
    f"Accuracy          : {accuracy:.6f}"
)

print(
    f"Macro-F1          : {macro_f1:.6f}"
)

print(
    f"Balanced Accuracy : {balanced_acc:.6f}"
)

print(
    f"LR training time  : {elapsed / 60:.2f} min"
)

print(
    "Iterations        :",
    clf.n_iter_,
)


print()
print("=" * 72)
print("CLASSIFICATION REPORT")
print("=" * 72)

report = classification_report(
    y_test,
    pred,
    digits=4,
)

print(report)


# ============================================================
# 7. Save metrics
# ============================================================

os.makedirs(
    OUT_DIR,
    exist_ok=True,
)

result = pd.DataFrame(
    [
        {
            "checkpoint": "epoch25",
            "val_loss": 0.179450,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced_acc,
            "training_seconds": elapsed,
        }
    ]
)

suffix = (
    "full"
    if args.max_train == 0
    else f"smoke_{len(y_train)}"
)

result_path = os.path.join(
    OUT_DIR,
    f"epoch25_linear_probe_{suffix}.csv",
)

result.to_csv(
    result_path,
    index=False,
)

print()
print("Saved:")
print(result_path)

print()
print("=" * 72)
print("LINEAR PROBE COMPLETE")
print("=" * 72)
