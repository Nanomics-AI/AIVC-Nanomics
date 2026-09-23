import os
import glob
import csv

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ============================================================
# Paths
# ============================================================

TB_ROOT = (
    "logs/tensorboard/"
    "GeneJEPA-quarter-d12-h6-700k-e30-seed42-run1"
)

OUT_DIR = "results"

CSV_PATH = os.path.join(
    OUT_DIR,
    "genejepa_val_loss_by_epoch.csv",
)

PNG_PATH = os.path.join(
    OUT_DIR,
    "genejepa_val_loss_curve.png",
)


# ============================================================
# Read val_loss from all TensorBoard versions
# ============================================================

os.makedirs(OUT_DIR, exist_ok=True)

records = []

version_dirs = sorted(
    glob.glob(os.path.join(TB_ROOT, "version_*")),
    key=lambda p: int(os.path.basename(p).split("_")[1]),
)

for version_dir in version_dirs:

    version = os.path.basename(version_dir)

    event_files = glob.glob(
        os.path.join(
            version_dir,
            "events.out.tfevents.*",
        )
    )

    for event_file in event_files:

        try:
            ea = EventAccumulator(
                event_file,
                size_guidance={"scalars": 0},
            )
            ea.Reload()

            scalar_tags = ea.Tags().get(
                "scalars",
                [],
            )

            if "val_loss" not in scalar_tags:
                continue

            for event in ea.Scalars("val_loss"):

                records.append(
                    {
                        "version": version,
                        "global_step": int(event.step),
                        "val_loss": float(event.value),
                        "wall_time": float(event.wall_time),
                    }
                )

        except Exception as exc:

            print(
                f"WARNING: failed to read {event_file}"
            )
            print(exc)


# ============================================================
# Remove duplicate global steps
#
# If one global_step appears in multiple TensorBoard versions,
# keep the newest record.
# ============================================================

by_step = {}

for record in records:

    step = record["global_step"]

    if step not in by_step:
        by_step[step] = record

    elif record["wall_time"] > by_step[step]["wall_time"]:
        by_step[step] = record


rows = sorted(
    by_step.values(),
    key=lambda r: r["global_step"],
)


# Assign Epoch 0...29
for epoch, row in enumerate(rows):
    row["epoch"] = epoch


print()
print("Number of validation records:", len(rows))

if len(rows) != 30:
    print(
        "WARNING: expected 30 epochs, "
        f"but found {len(rows)}."
    )


# ============================================================
# Print all values
# ============================================================

print()
print(
    f"{'Epoch':<8}"
    f"{'Global step':<15}"
    f"{'Val loss':<15}"
    f"{'Version'}"
)

print("-" * 55)

for row in rows:

    print(
        f"{row['epoch']:<8}"
        f"{row['global_step']:<15}"
        f"{row['val_loss']:<15.6f}"
        f"{row['version']}"
    )


# ============================================================
# Save CSV
# ============================================================

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "epoch",
            "global_step",
            "val_loss",
            "tensorboard_version",
        ]
    )

    for row in rows:

        writer.writerow(
            [
                row["epoch"],
                row["global_step"],
                row["val_loss"],
                row["version"],
            ]
        )


# ============================================================
# Find best epoch
# ============================================================

best = min(
    rows,
    key=lambda r: r["val_loss"],
)

epochs = [
    row["epoch"]
    for row in rows
]

losses = [
    row["val_loss"]
    for row in rows
]


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 6.5)
)


# Main validation curve
ax.plot(
    epochs,
    losses,
    marker="o",
    linewidth=2,
    markersize=5,
    label="Validation loss",
)


# ------------------------------------------------------------
# Mark single-GPU interval
#
# Epoch 19, 20, 21 were trained using one GPU.
# The shaded region starts halfway between 18 and 19
# and ends halfway between 21 and 22.
# ------------------------------------------------------------

ax.axvspan(
    18.5,
    21.5,
    alpha=0.12,
    label="Single GPU (Epoch 19-21)",
)


# ------------------------------------------------------------
# Mark best epoch
# ------------------------------------------------------------

ax.axvline(
    best["epoch"],
    linestyle="--",
    linewidth=2,
    alpha=0.7,
)


# Put the annotation to the RIGHT of the best point,
# so it does not overlap with the single-GPU region.
ax.annotate(
    (
        f'Best: Epoch {best["epoch"]}\n'
        f'val_loss = {best["val_loss"]:.6f}'
    ),
    xy=(
        best["epoch"],
        best["val_loss"],
    ),
    xytext=(
        best["epoch"] + 0.8,
        best["val_loss"] + 0.10,
    ),
    arrowprops={
        "arrowstyle": "->",
        "linewidth": 1.5,
    },
    fontsize=11,
    ha="left",
    va="center",
)


# ============================================================
# Styling
# ============================================================

ax.set_xlabel(
    "Epoch",
    fontsize=12,
)

ax.set_ylabel(
    "Validation Loss",
    fontsize=12,
)

ax.set_title(
    "GeneJEPA Validation Loss Across 30 Epochs",
    fontsize=15,
)

ax.set_xticks(
    range(0, 30, 2)
)

ax.grid(
    True,
    alpha=0.25,
)

ax.legend(
    loc="best",
)

fig.tight_layout()


# ============================================================
# Save figure
# ============================================================

fig.savefig(
    PNG_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Final summary
# ============================================================

print()
print("=" * 55)

print(
    f"Best epoch     : {best['epoch']}"
)

print(
    f"Best val_loss  : {best['val_loss']:.6f}"
)

print(
    f"Best step      : {best['global_step']}"
)

print()
print(
    f"CSV saved to   : {CSV_PATH}"
)

print(
    f"Figure saved to: {PNG_PATH}"
)

print("=" * 55)
