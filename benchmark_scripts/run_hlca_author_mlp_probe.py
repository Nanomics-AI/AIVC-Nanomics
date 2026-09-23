import os
import time
import argparse
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
)


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--epoch",
    type=int,
    required=True,
    choices=[24, 25, 29],
)

args = parser.parse_args()


# ============================================================
# Fixed protocol
# ============================================================

SEED = 42

BATCH_SIZE = 32
TRAIN_EPOCHS = 5
LEARNING_RATE = 1e-3

OUT_DIR = "benchmark_results/hlca18"

VAL_LOSS = {
    24: 0.186290,
    25: 0.179450,
    29: 0.204405,
}


EMB_PATH = os.path.join(
    OUT_DIR,
    f"epoch{args.epoch}_all18_embeddings.npy",
)

ROWS_PATH = os.path.join(
    OUT_DIR,
    f"epoch{args.epoch}_all18_rows.csv",
)


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


print("=" * 72)
print(
    f"GENEJEPA HLCA AUTHOR-STYLE MLP PROBE — EPOCH {args.epoch}"
)
print("=" * 72)


# ============================================================
# 1. Load frozen embeddings
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


if X.shape != (578572, 768):
    raise RuntimeError(
        f"Unexpected embedding shape: {X.shape}"
    )

if len(rows) != X.shape[0]:
    raise RuntimeError(
        "Embedding / metadata mismatch."
    )


# ============================================================
# 2. Use EXACT SAME frozen benchmark split as LR
# ============================================================

split = rows["split"].to_numpy()

train_idx = np.flatnonzero(
    split == "train"
)

test_idx = np.flatnonzero(
    split == "test"
)


X_train = np.asarray(
    X[train_idx],
    dtype=np.float32,
)

X_test = np.asarray(
    X[test_idx],
    dtype=np.float32,
)


y_train_text = rows.iloc[
    train_idx
]["label"].to_numpy()

y_test_text = rows.iloc[
    test_idx
]["label"].to_numpy()


print()
print("Train X:", X_train.shape)
print("Test X :", X_test.shape)


# ============================================================
# 3. Encode 18 string labels -> integer class IDs
# ============================================================

encoder = LabelEncoder()

encoder.fit(
    rows["label"].to_numpy()
)

y_train = encoder.transform(
    y_train_text
).astype(
    np.int64
)

y_test = encoder.transform(
    y_test_text
).astype(
    np.int64
)


num_classes = len(
    encoder.classes_
)


print("Classes :", num_classes)

print()
print("Class mapping:")

for i, name in enumerate(
    encoder.classes_
):
    print(
        f"{i:2d} -> {name}"
    )


# ============================================================
# 4. Torch datasets
#
# IMPORTANT:
# No StandardScaler here.
# Author's probe begins with LayerNorm.
# ============================================================

train_tensor = torch.from_numpy(
    X_train
)

test_tensor = torch.from_numpy(
    X_test
)

y_train_tensor = torch.from_numpy(
    y_train
)

y_test_tensor = torch.from_numpy(
    y_test
)


train_dataset = TensorDataset(
    train_tensor,
    y_train_tensor,
)

test_dataset = TensorDataset(
    test_tensor,
    y_test_tensor,
)


generator = torch.Generator()
generator.manual_seed(SEED)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=512,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# 5. AUTHOR'S PROBE ARCHITECTURE
# ============================================================

class LinearProbeMLP(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_classes,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.LayerNorm(
                embedding_dim
            ),
            nn.Linear(
                embedding_dim,
                512,
            ),
            nn.GELU(),
            nn.Linear(
                512,
                num_classes,
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.model(x)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()
print("Device:", device)


model = LinearProbeMLP(
    embedding_dim=768,
    num_classes=num_classes,
).to(
    device
)


print()
print(model)


# ============================================================
# 6. AUTHOR'S OPTIMIZATION SETTINGS
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

loss_fn = nn.CrossEntropyLoss()


# ============================================================
# 7. Train for EXACTLY 5 epochs
# ============================================================

print()
print("=" * 72)
print("START MLP PROBE TRAINING")
print("=" * 72)


start_total = time.time()


for epoch in range(
    TRAIN_EPOCHS
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    epoch_start = time.time()


    for batch_idx, (
        xb,
        yb,
    ) in enumerate(
        train_loader
    ):

        xb = xb.to(
            device,
            non_blocking=True,
        )

        yb = yb.to(
            device,
            non_blocking=True,
        )


        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            xb
        )

        loss = loss_fn(
            logits,
            yb
        )

        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item()
            * len(yb)
        )

        pred = logits.argmax(
            dim=1
        )

        correct += (
            pred == yb
        ).sum().item()

        total += len(yb)


        if (
            batch_idx + 1
        ) % 2000 == 0:

            print(
                f"Epoch "
                f"{epoch + 1}/{TRAIN_EPOCHS} "
                f"| batch "
                f"{batch_idx + 1}/"
                f"{len(train_loader)} "
                f"| loss "
                f"{running_loss / total:.4f} "
                f"| acc "
                f"{correct / total:.4f}"
            )


    epoch_loss = (
        running_loss
        / total
    )

    epoch_acc = (
        correct
        / total
    )

    epoch_minutes = (
        time.time()
        - epoch_start
    ) / 60


    print()
    print(
        f"Epoch "
        f"{epoch + 1}/{TRAIN_EPOCHS} "
        f"complete"
    )

    print(
        f"  Train loss : "
        f"{epoch_loss:.6f}"
    )

    print(
        f"  Train acc  : "
        f"{epoch_acc:.6f}"
    )

    print(
        f"  Time       : "
        f"{epoch_minutes:.2f} min"
    )

    print()


training_seconds = (
    time.time()
    - start_total
)


# ============================================================
# 8. Test
# ============================================================

print()
print("=" * 72)
print("TESTING")
print("=" * 72)


model.eval()

all_pred = []
all_true = []


with torch.no_grad():

    for xb, yb in test_loader:

        xb = xb.to(
            device,
            non_blocking=True,
        )

        logits = model(
            xb
        )

        pred = logits.argmax(
            dim=1
        )

        all_pred.append(
            pred.cpu().numpy()
        )

        all_true.append(
            yb.numpy()
        )


pred = np.concatenate(
    all_pred
)

true = np.concatenate(
    all_true
)


# ============================================================
# 9. Same metrics as our LR benchmark
# ============================================================

accuracy = accuracy_score(
    true,
    pred,
)

macro_f1 = f1_score(
    true,
    pred,
    average="macro",
)

balanced_acc = balanced_accuracy_score(
    true,
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

print(
    f"Training time     : "
    f"{training_seconds / 60:.2f} min"
)


# ============================================================
# 10. Classification report
# ============================================================

print()
print("=" * 72)
print("CLASSIFICATION REPORT")
print("=" * 72)

print(
    classification_report(
        true,
        pred,
        labels=np.arange(
            num_classes
        ),
        target_names=encoder.classes_,
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# 11. Save summary
# ============================================================

result = pd.DataFrame(
    [
        {
            "checkpoint":
                f"epoch{args.epoch}",

            "val_loss":
                VAL_LOSS[
                    args.epoch
                ],

            "probe":
                "author_style_mlp",

            "n_train":
                len(train_idx),

            "n_test":
                len(test_idx),

            "train_epochs":
                TRAIN_EPOCHS,

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "accuracy":
                accuracy,

            "macro_f1":
                macro_f1,

            "balanced_accuracy":
                balanced_acc,

            "training_seconds":
                training_seconds,
        }
    ]
)


result_path = os.path.join(
    OUT_DIR,
    (
        f"epoch{args.epoch}_"
        f"author_mlp_probe_full.csv"
    ),
)

result.to_csv(
    result_path,
    index=False,
)


# ============================================================
# 12. Save predictions
# ============================================================

pred_text = encoder.inverse_transform(
    pred
)

true_text = encoder.inverse_transform(
    true
)


prediction_df = pd.DataFrame(
    {
        "true_label":
            true_text,

        "predicted_label":
            pred_text,

        "correct":
            true_text
            == pred_text,
    }
)


prediction_path = os.path.join(
    OUT_DIR,
    (
        f"epoch{args.epoch}_"
        f"author_mlp_predictions_full.csv"
    ),
)

prediction_df.to_csv(
    prediction_path,
    index=False,
)


print()
print("Saved:")
print(result_path)
print(prediction_path)

print()
print("=" * 72)
print("AUTHOR-STYLE MLP PROBE COMPLETE")
print("=" * 72)
