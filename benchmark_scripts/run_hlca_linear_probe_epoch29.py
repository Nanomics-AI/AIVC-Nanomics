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
    "epoch29_all18_embeddings.npy"
)

ROWS_PATH = (
    "benchmark_results/hlca18/"
    "epoch29_all18_rows.csv"
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
            "checkpoint": "epoch29",
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
    f"epoch29_linear_probe_{suffix}.csv",
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
