#!/usr/bin/env python3
"""Count each selected DMSO sample by cell line directly from Tahoe parquet."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import pyarrow.dataset as ds

from make_tahoe_latent_audit_cohort import (
    CELL_LINES,
    DEFAULT_INPUT,
    SET_SIZE,
    plate_number,
    read_selected_conditions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "hf_data_cache" / "data" / "data"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "tahoe_latent_audit_dmso_counts.csv"
OUTPUT_COLUMNS = (
    "plate",
    "cell_line_id",
    "control_sample",
    "control_sample_role",
    "cell_count",
    "eligible_S256",
)


def expected_controls() -> dict[tuple[str, str, str], str]:
    controls: dict[tuple[str, str, str], str] = {}
    for row in read_selected_conditions(DEFAULT_INPUT):
        samples = row["control_samples"].split("|")
        if len(samples) != 2 or len(set(samples)) != 2:
            raise ValueError(
                f"Expected two distinct DMSO samples in {row['pair_id']}: {samples}"
            )
        for role, sample in zip(("A", "B"), samples, strict=True):
            key = (row["plate"], row["cell_line_id"], sample)
            existing = controls.setdefault(key, role)
            if existing != role:
                raise ValueError(f"Inconsistent A/B role for {key}")
    return controls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    expected = expected_controls()
    sample_ids = sorted({key[2] for key in expected})
    dataset = ds.dataset(str(args.data_dir), format="parquet", exclude_invalid_files=True)
    table = dataset.to_table(
        columns=["plate", "sample", "drug", "cell_line_id"],
        filter=(
            (ds.field("drug") == "DMSO_TF")
            & ds.field("sample").isin(sample_ids)
            & ds.field("cell_line_id").isin(CELL_LINES)
        ),
    )
    counts = Counter(
        zip(
            table["plate"].to_pylist(),
            table["cell_line_id"].to_pylist(),
            table["sample"].to_pylist(),
            strict=True,
        )
    )

    missing = set(expected) - set(counts)
    unexpected = set(counts) - set(expected)
    if missing or unexpected:
        raise ValueError(
            f"DMSO count key mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    rows = []
    for key in sorted(expected, key=lambda value: (plate_number(value[0]), value[1], value[2])):
        count = counts[key]
        rows.append(
            {
                "plate": key[0],
                "cell_line_id": key[1],
                "control_sample": key[2],
                "control_sample_role": expected[key],
                "cell_count": count,
                "eligible_S256": int(count >= SET_SIZE),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    failing = [row for row in rows if not row["eligible_S256"]]
    print(f"Wrote: {args.output}")
    print(f"DMSO sample/cell-line groups: {len(rows)}")
    print(f"Minimum single-sample cell count: {min(counts.values())}")
    print(f"Below S={SET_SIZE}: {len(failing)}")
    if failing:
        raise RuntimeError("At least one selected DMSO sample is below S=256")


if __name__ == "__main__":
    main()
