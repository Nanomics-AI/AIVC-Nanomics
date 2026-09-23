import os
import glob

import numpy as np
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TB_ROOT = (
    "logs/tensorboard/"
    "GeneJEPA-quarter-d12-h6-700k-e30-seed42-run1"
)

OUT_DIR = "results"
OUT_PATH = os.path.join(
    OUT_DIR,
    "genejepa_train_sim_loss_curve.png",
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 1. Read train_loss/sim from all TensorBoard versions
# ============================================================

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

            tags = ea.Tags().get("scalars", [])

            if "train_loss/sim" not in tags:
                continue

            for event in ea.Scalars("train_loss/sim"):

                records.append(
                    {
                        "version": version,
                        "global_step": int(event.step),
                        "value": float(event.value),
                        "wall_time": float(event.wall_time),
                    }
                )

        except Exception as exc:

            print(
                f"WARNING: failed to read {event_file}"
            )
            print(exc)


print("Raw train_loss/sim records:", len(records))


# ============================================================
# 2. Resolve duplicated global steps
#
# Because:
# - gradient accumulation can produce repeated logger steps
# - crash/resume can also generate overlapping TensorBoard logs
#
# For each global_step, keep the newest record.
# ============================================================

by_step = {}

for r in records:

    step = r["global_step"]

    if (
        step not in by_step
        or r["wall_time"] > by_step[step]["wall_time"]
    ):
        by_step[step] = r


rows = sorted(
    by_step.values(),
    key=lambda r: r["global_step"],
)

steps = np.array(
    [r["global_step"] for r in rows],
    dtype=np.int64,
)

losses = np.array(
    [r["value"] for r in rows],
    dtype=np.float64,
)


print(
    "Unique global steps:",
    len(rows),
)

print(
    "Step range:",
    int(steps.min()),
    "->",
    int(steps.max()),
)


# ============================================================
# 3. Smooth curve
#
# Rolling mean over 200 optimizer steps.
# This is visualization only.
# Raw data are not modified.
# ============================================================

window = 200

kernel = np.ones(window) / window

smooth_losses = np.convolve(
    losses,
    kernel,
    mode="valid",
)

smooth_steps = steps[
    window - 1:
]


# ============================================================
# 4. Epoch boundaries
#
# Training schedule:
# ~1903 optimizer updates per epoch.
# ============================================================

UPDATES_PER_EPOCH = 1903


# ============================================================
# 5. Plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 6.5)
)


# Raw loss: faint background
ax.plot(
    steps,
    losses,
    linewidth=0.5,
    alpha=0.18,
    label="Raw train_loss/sim",
)


# Smoothed curve
ax.plot(
    smooth_steps,
    smooth_losses,
    linewidth=2.2,
    label=f"Rolling mean ({window} steps)",
)


# ------------------------------------------------------------
# Single-GPU period: Epoch 19-21
#
# Approximate global-step boundaries:
#
# Epoch 19 starts around 19*1903
# Epoch 22 starts around 22*1903
# ------------------------------------------------------------

single_gpu_start = 19 * UPDATES_PER_EPOCH
single_gpu_end = 22 * UPDATES_PER_EPOCH

ax.axvspan(
    single_gpu_start,
    single_gpu_end,
    alpha=0.10,
    label="Single GPU (Epoch 19-21)",
)


# ------------------------------------------------------------
# Best validation checkpoint: Epoch 25
#
# Its actual validation global_step was 49477.
# ------------------------------------------------------------

best_step = 49477

ax.axvline(
    best_step,
    linestyle="--",
    linewidth=2,
    alpha=0.7,
    label="Best val checkpoint (Epoch 25)",
)


# ============================================================
# Add epoch labels on top axis
# ============================================================

ax2 = ax.twiny()

epoch_ticks = list(
    range(
        0,
        30,
        2,
    )
)

epoch_step_ticks = [
    epoch * UPDATES_PER_EPOCH
    for epoch in epoch_ticks
]

ax2.set_xlim(
    ax.get_xlim()
)

ax2.set_xticks(
    epoch_step_ticks
)

ax2.set_xticklabels(
    epoch_ticks
)

ax2.set_xlabel(
    "Epoch",
    fontsize=11,
)


# ============================================================
# Styling
# ============================================================

ax.set_xlabel(
    "Global optimizer step",
    fontsize=12,
)

ax.set_ylabel(
    "train_loss/sim",
    fontsize=12,
)

ax.set_title(
    "GeneJEPA Training Similarity Loss",
    fontsize=15,
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
# Save
# ============================================================

fig.savefig(
    OUT_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 60)

print(
    "First smoothed loss :",
    f"{smooth_losses[0]:.6f}",
)

print(
    "Final smoothed loss :",
    f"{smooth_losses[-1]:.6f}",
)

print(
    "Minimum smoothed loss:",
    f"{smooth_losses.min():.6f}",
)

print()
print(
    "Figure saved to:",
    OUT_PATH,
)

print("=" * 60)
