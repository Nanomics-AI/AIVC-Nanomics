import os
import numpy as np
import pandas as pd


DIR = "benchmark_results/hlca18"

EMB0 = os.path.join(DIR, "epoch29_all18_part0_embeddings.npy")
EMB1 = os.path.join(DIR, "epoch29_all18_part1_embeddings.npy")

ROWS0 = os.path.join(DIR, "epoch29_all18_part0_rows.csv")
ROWS1 = os.path.join(DIR, "epoch29_all18_part1_rows.csv")

BENCHMARK = (
    "benchmark_data/hlca/"
    "benchmark_ann_level3_all18_seed42.csv"
)

OUT_EMB = os.path.join(DIR, "epoch29_all18_embeddings.npy")
OUT_ROWS = os.path.join(DIR, "epoch29_all18_rows.csv")


print("=" * 72)
print("MERGE EPOCH29 HLCA EMBEDDINGS")
print("=" * 72)


# ============================================================
# 1. Load two embedding shards without loading into RAM
# ============================================================

x0 = np.load(EMB0, mmap_mode="r")
x1 = np.load(EMB1, mmap_mode="r")

r0 = pd.read_csv(ROWS0)
r1 = pd.read_csv(ROWS1)


print()
print("Part0 embedding:", x0.shape)
print("Part1 embedding:", x1.shape)

print("Part0 rows     :", len(r0))
print("Part1 rows     :", len(r1))


if x0.shape != (len(r0), 768):
    raise RuntimeError("Part0 shape / row mismatch.")

if x1.shape != (len(r1), 768):
    raise RuntimeError("Part1 shape / row mismatch.")


# ============================================================
# 2. Merge metadata
# ============================================================

rows = pd.concat(
    [r0, r1],
    ignore_index=True,
)

print()
print("Combined rows:", len(rows))

if len(rows) != 578572:
    raise RuntimeError(
        f"Expected 578572 rows, got {len(rows)}"
    )

if rows["cell_index"].duplicated().any():
    raise RuntimeError("Duplicate cell_index detected.")


# ============================================================
# 3. Compare with original frozen benchmark
# ============================================================

original = pd.read_csv(BENCHMARK)

original = (
    original
    .sort_values("cell_index")
    .reset_index(drop=True)
)

rows_check = (
    rows
    .sort_values("cell_index")
    .reset_index(drop=True)
)


if not np.array_equal(
    rows_check["cell_index"].to_numpy(),
    original["cell_index"].to_numpy(),
):
    raise RuntimeError(
        "Merged cell_index does not match frozen benchmark."
    )


if not np.array_equal(
    rows_check["label"].to_numpy(),
    original["label"].to_numpy(),
):
    raise RuntimeError(
        "Merged labels do not match frozen benchmark."
    )


if not np.array_equal(
    rows_check["split"].to_numpy(),
    original["split"].to_numpy(),
):
    raise RuntimeError(
        "Merged train/test split does not match frozen benchmark."
    )


print()
print("Frozen benchmark match: OK")


# ============================================================
# 4. Diagnostics
# ============================================================

print()
print("Classes:", rows["label"].nunique())

print(
    "Split counts:",
    rows["split"].value_counts().to_dict(),
)


# ============================================================
# 5. Check embeddings in chunks
# ============================================================

print()
print("Checking finite embeddings...")

for name, X in [
    ("part0", x0),
    ("part1", x1),
]:
    for start in range(0, len(X), 10000):

        end = min(start + 10000, len(X))

        block = np.asarray(
            X[start:end]
        )

        if not np.isfinite(block).all():
            raise RuntimeError(
                f"Non-finite values in {name} "
                f"rows {start}:{end}"
            )

print("Finite check: OK")


# ============================================================
# 6. Create final disk-backed embedding array
# ============================================================

if os.path.exists(OUT_EMB):
    raise RuntimeError(
        f"Output already exists: {OUT_EMB}"
    )


total = len(x0) + len(x1)

out = np.lib.format.open_memmap(
    OUT_EMB,
    mode="w+",
    dtype=np.float32,
    shape=(total, 768),
)


print()
print("Writing part0...")

out[:len(x0)] = x0
out.flush()


print("Writing part1...")

out[len(x0):] = x1
out.flush()


# ============================================================
# 7. Save exact corresponding metadata order
# ============================================================

rows.to_csv(
    OUT_ROWS,
    index=False,
)


# ============================================================
# 8. Final verification
# ============================================================

final = np.load(
    OUT_EMB,
    mmap_mode="r",
)

print()
print("=" * 72)
print("FINAL RESULT")
print("=" * 72)

print("Embedding shape :", final.shape)
print("Rows            :", len(rows))
print("Classes         :", rows["label"].nunique())

print(
    "Train           :",
    int((rows["split"] == "train").sum()),
)

print(
    "Test            :",
    int((rows["split"] == "test").sum()),
)

print(
    "Embedding size  : %.2f GB"
    % (
        os.path.getsize(OUT_EMB)
        / 1024**3
    )
)

print()
print("Embedding:")
print(OUT_EMB)

print()
print("Metadata:")
print(OUT_ROWS)

print()
print("=" * 72)
print("MERGE COMPLETE")
print("=" * 72)
