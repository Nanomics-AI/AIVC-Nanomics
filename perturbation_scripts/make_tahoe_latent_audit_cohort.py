#!/usr/bin/env python3
"""Build the CPU-only Tahoe cohort plan for GeneJEPA latent Experiment 0.

Selection rule:
1. A cell-line/drug group must contain all three doses, a plate6/plate14
   repeated dose, and at least 256 cells on both sides of every condition.
2. Rank cell lines by eligible-drug count, then median worst-case capacity.
3. Across the top three cell lines, keep drugs with a non-unclear MOA, select
   the most robust drug per MOA, then take the five strongest MOA groups.

Dose and plate are confounded in this panel: 0.05 uM is on plate4, 0.5 uM on
plate5, and 5.0 uM on plate6 plus its plate14 repeat. Any dose-response result
must therefore use plate-matched DMSO and cannot identify a pure dose effect.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import statistics
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "tahoe_set_to_set_pairs.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "tahoe_latent_audit_cohort.csv"
DEFAULT_DMSO_COUNTS = PROJECT_ROOT / "results" / "tahoe_latent_audit_dmso_counts.csv"

SET_SIZE = 256
BASE_SEEDS = (42, 43, 44, 45, 46)
SAMPLING_METHOD = "repeated_subsampling_without_replacement"
SELECTION_RULE_ID = "eligible_3dose_rep_capacity_moa_v1"
DOSE_PLATE_DESIGN = "0.05=plate4|0.5=plate5|5.0=plate6,plate14"
EXPECTED_DOSE_PLATES = {
    0.05: {"plate4"},
    0.5: {"plate5"},
    5.0: {"plate6", "plate14"},
}
CELL_LINES = ("CVCL_0546", "CVCL_0459", "CVCL_0480")
DRUGS = (
    "Artesunate",
    "Trimetrexate",
    "Retinoic acid",
    "Clonidine (hydrochloride)",
    "Tucidinostat",
)
EXPECTED_MOA = {
    "Artesunate": "JAK/STAT inhibitor",
    "Trimetrexate": "DNA synthesis/repair inhibitor",
    "Retinoic acid": "Retinoic receptor agonist",
    "Clonidine (hydrochloride)": "Adrenoceptor agonist",
    "Tucidinostat": "HDAC inhibitor",
}
EXPECTED_DOSES = {0.05, 0.5, 5.0}
REQUIRED_COLUMNS = {
    "pair_id",
    "plate",
    "cell_line_id",
    "drug",
    "dose_uM",
    "treated_samples",
    "treated_cell_count",
    "control_drug",
    "control_samples",
    "control_cell_count",
    "matched_capacity",
    "moa_fine",
    "canonical_smiles",
    "pubchem_cid",
}
OUTPUT_COLUMNS = (
    "audit_pair_id",
    "comparison_type",
    "selection_rule_id",
    "condition_id",
    "parent_pair_id",
    "null_group_id",
    "repeat_index",
    "repeat_base_seed",
    "source_seed",
    "target_seed",
    "set_size",
    "sampling_method",
    "source_pool_kind",
    "target_pool_kind",
    "plate",
    "cell_line_id",
    "source_drug",
    "source_samples",
    "source_pool_cell_count",
    "target_drug",
    "target_dose_uM",
    "target_samples",
    "target_pool_cell_count",
    "matched_capacity",
    "moa_fine",
    "canonical_smiles",
    "pubchem_cid",
    "dose_series_group_id",
    "replicate_group_id",
    "is_cross_plate_replicate",
    "dose_plate_confound",
    "dose_plate_design",
)


def stable_seed(base_seed: int, role: str, key: str) -> int:
    """Derive a stable uint32 seed without Python's randomized hash()."""
    payload = f"{base_seed}|{role}|{key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def plate_number(plate: str) -> int:
    return int(plate.removeprefix("plate"))


def read_selected_conditions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Input is missing columns: {sorted(missing)}")
        all_rows = list(reader)

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        groups[(row["cell_line_id"], row["drug"])].append(row)

    eligible: dict[tuple[str, str], list[dict[str, str]]] = {}
    for key, items in groups.items():
        doses = {float(row["dose_uM"]) for row in items}
        plate6_doses = {
            float(row["dose_uM"]) for row in items if row["plate"] == "plate6"
        }
        plate14_doses = {
            float(row["dose_uM"]) for row in items if row["plate"] == "plate14"
        }
        if (
            doses == EXPECTED_DOSES
            and plate6_doses & plate14_doses
            and min(int(row["matched_capacity"]) for row in items) >= SET_SIZE
        ):
            eligible[key] = items

    cell_line_scores = []
    for cell_line in {key[0] for key in eligible}:
        capacities = [
            min(int(row["matched_capacity"]) for row in items)
            for (cell, _), items in eligible.items()
            if cell == cell_line
        ]
        cell_line_scores.append(
            (len(capacities), statistics.median(capacities), cell_line)
        )
    selected_cell_lines = tuple(
        score[2]
        for score in sorted(
            cell_line_scores,
            key=lambda score: (-score[0], -score[1], score[2]),
        )[:3]
    )

    drug_candidates = []
    for drug in {key[1] for key in eligible}:
        keys = [(cell_line, drug) for cell_line in selected_cell_lines]
        if not all(key in eligible for key in keys):
            continue
        moa_values = {
            row["moa_fine"] for key in keys for row in eligible[key]
        }
        if len(moa_values) != 1 or "unclear" in moa_values:
            continue
        robust_capacity = min(
            int(row["matched_capacity"])
            for key in keys
            for row in eligible[key]
        )
        drug_candidates.append((robust_capacity, drug, moa_values.pop()))

    best_by_moa: dict[str, tuple[int, str, str]] = {}
    for candidate in drug_candidates:
        existing = best_by_moa.get(candidate[2])
        if existing is None or (-candidate[0], candidate[1]) < (
            -existing[0],
            existing[1],
        ):
            best_by_moa[candidate[2]] = candidate
    selected_drugs = tuple(
        candidate[1]
        for candidate in sorted(
            best_by_moa.values(), key=lambda value: (-value[0], value[1])
        )[:5]
    )

    if selected_cell_lines != CELL_LINES:
        raise ValueError(
            f"Objective cell-line selection changed: {selected_cell_lines} != {CELL_LINES}"
        )
    if selected_drugs != DRUGS:
        raise ValueError(f"Objective drug selection changed: {selected_drugs} != {DRUGS}")

    rows = [
        row
        for row in all_rows
        if row["cell_line_id"] in selected_cell_lines and row["drug"] in selected_drugs
    ]

    rows.sort(
        key=lambda row: (
            plate_number(row["plate"]),
            CELL_LINES.index(row["cell_line_id"]),
            DRUGS.index(row["drug"]),
            float(row["dose_uM"]),
            row["pair_id"],
        )
    )
    return rows


def read_dmso_counts(path: Path) -> dict[tuple[str, str, str], int]:
    required = {
        "plate",
        "cell_line_id",
        "control_sample",
        "cell_count",
        "eligible_S256",
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"DMSO count input is missing columns: {sorted(missing)}")
        rows = list(reader)

    counts: dict[tuple[str, str, str], int] = {}
    for row in rows:
        key = (row["plate"], row["cell_line_id"], row["control_sample"])
        if key in counts:
            raise ValueError(f"Duplicate DMSO count key: {key}")
        count = int(row["cell_count"])
        if count < SET_SIZE or row["eligible_S256"] != "1":
            raise ValueError(f"DMSO sample below S={SET_SIZE}: {key} has {count}")
        counts[key] = count
    return counts


def validate_conditions(rows: list[dict[str, str]]) -> dict[str, bool]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    replicate_counts: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    for row in rows:
        if row["control_drug"] != "DMSO_TF":
            raise ValueError(f"Unexpected control in {row['pair_id']}")
        if int(row["matched_capacity"]) < SET_SIZE:
            raise ValueError(f"Condition below S={SET_SIZE}: {row['pair_id']}")
        if row["moa_fine"] != EXPECTED_MOA[row["drug"]]:
            raise ValueError(f"MOA changed for {row['drug']}: {row['moa_fine']}")
        dose = float(row["dose_uM"])
        if row["plate"] not in EXPECTED_DOSE_PLATES[dose]:
            raise ValueError(
                f"Unexpected dose/plate layout in {row['pair_id']}: "
                f"{dose} on {row['plate']}"
            )

        groups[(row["cell_line_id"], row["drug"])].append(row)
        replicate_counts[
            (row["cell_line_id"], row["drug"], row["dose_uM"])
        ].add(row["plate"])

    expected_groups = len(CELL_LINES) * len(DRUGS)
    if len(groups) != expected_groups:
        raise ValueError(f"Expected {expected_groups} cell-line/drug groups, got {len(groups)}")

    for key, items in groups.items():
        doses = {float(row["dose_uM"]) for row in items}
        if doses != EXPECTED_DOSES:
            raise ValueError(f"Incomplete dose series for {key}: {sorted(doses)}")
        repeated_doses = {
            dose
            for cell, drug, dose in replicate_counts
            if (cell, drug) == key and len(replicate_counts[(cell, drug, dose)]) > 1
        }
        if not repeated_doses:
            raise ValueError(f"No cross-plate replicate for {key}")

    if len(rows) != 60:
        raise ValueError(f"Expected 60 perturbation conditions, got {len(rows)}")

    return {
        "|".join(key): len(plates) > 1
        for key, plates in replicate_counts.items()
    }


def perturbation_record(
    row: dict[str, str],
    repeat_index: int,
    base_seed: int,
    is_replicate: bool,
) -> dict[str, object]:
    condition_id = row["pair_id"]
    null_group_id = f"null|{row['plate']}|{row['cell_line_id']}"
    key = f"{condition_id}|r{repeat_index:02d}"
    return {
        "audit_pair_id": f"pert|{key}",
        "comparison_type": "perturbation",
        "selection_rule_id": SELECTION_RULE_ID,
        "condition_id": condition_id,
        "parent_pair_id": row["pair_id"],
        "null_group_id": null_group_id,
        "repeat_index": repeat_index,
        "repeat_base_seed": base_seed,
        "source_seed": stable_seed(base_seed, "control_source", condition_id),
        "target_seed": stable_seed(base_seed, "perturbed_target", condition_id),
        "set_size": SET_SIZE,
        "sampling_method": SAMPLING_METHOD,
        "source_pool_kind": "pooled_dmso_samples",
        "target_pool_kind": "treated_sample_pool",
        "plate": row["plate"],
        "cell_line_id": row["cell_line_id"],
        "source_drug": row["control_drug"],
        "source_samples": row["control_samples"],
        "source_pool_cell_count": row["control_cell_count"],
        "target_drug": row["drug"],
        "target_dose_uM": row["dose_uM"],
        "target_samples": row["treated_samples"],
        "target_pool_cell_count": row["treated_cell_count"],
        "matched_capacity": row["matched_capacity"],
        "moa_fine": row["moa_fine"],
        "canonical_smiles": row["canonical_smiles"],
        "pubchem_cid": row["pubchem_cid"],
        "dose_series_group_id": f"dose|{row['cell_line_id']}|{row['drug']}",
        "replicate_group_id": (
            f"rep|{row['cell_line_id']}|{row['drug']}|{row['dose_uM']}"
        ),
        "is_cross_plate_replicate": int(is_replicate),
        "dose_plate_confound": 1,
        "dose_plate_design": DOSE_PLATE_DESIGN,
    }


def null_record(
    control: dict[str, str],
    dmso_counts: dict[tuple[str, str, str], int],
    repeat_index: int,
    base_seed: int,
) -> dict[str, object]:
    null_group_id = f"null|{control['plate']}|{control['cell_line_id']}"
    key = f"{null_group_id}|r{repeat_index:02d}"
    source_sample, target_sample = control["control_samples"].split("|")
    source_count = dmso_counts[
        (control["plate"], control["cell_line_id"], source_sample)
    ]
    target_count = dmso_counts[
        (control["plate"], control["cell_line_id"], target_sample)
    ]
    return {
        "audit_pair_id": key,
        "comparison_type": "dmso_null",
        "selection_rule_id": SELECTION_RULE_ID,
        "condition_id": null_group_id,
        "parent_pair_id": "",
        "null_group_id": null_group_id,
        "repeat_index": repeat_index,
        "repeat_base_seed": base_seed,
        "source_seed": stable_seed(base_seed, "null_source", null_group_id),
        "target_seed": stable_seed(base_seed, "null_target", null_group_id),
        "set_size": SET_SIZE,
        "sampling_method": SAMPLING_METHOD,
        "source_pool_kind": "single_dmso_sample_A",
        "target_pool_kind": "single_dmso_sample_B",
        "plate": control["plate"],
        "cell_line_id": control["cell_line_id"],
        "source_drug": control["control_drug"],
        "source_samples": source_sample,
        "source_pool_cell_count": source_count,
        "target_drug": control["control_drug"],
        "target_dose_uM": "",
        "target_samples": target_sample,
        "target_pool_cell_count": target_count,
        "matched_capacity": min(source_count, target_count),
        "moa_fine": "null",
        "canonical_smiles": "",
        "pubchem_cid": "",
        "dose_series_group_id": "",
        "replicate_group_id": "",
        "is_cross_plate_replicate": 0,
        "dose_plate_confound": 0,
        "dose_plate_design": "",
    }


def build_cohort(
    rows: list[dict[str, str]],
    dmso_counts: dict[tuple[str, str, str], int],
) -> list[dict[str, object]]:
    replicate_flags = validate_conditions(rows)
    records: list[dict[str, object]] = []

    for row in rows:
        replicate_key = "|".join(
            (row["cell_line_id"], row["drug"], row["dose_uM"])
        )
        for repeat_index, base_seed in enumerate(BASE_SEEDS, start=1):
            records.append(
                perturbation_record(
                    row,
                    repeat_index,
                    base_seed,
                    replicate_flags[replicate_key],
                )
            )

    controls: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["plate"], row["cell_line_id"])
        existing = controls.setdefault(key, row)
        if (
            existing["control_samples"] != row["control_samples"]
            or existing["control_cell_count"] != row["control_cell_count"]
        ):
            raise ValueError(f"Inconsistent DMSO pool for {key}")

        samples = row["control_samples"].split("|")
        if len(samples) != 2 or len(set(samples)) != 2:
            raise ValueError(f"Expected two distinct DMSO samples for {key}")
        individual_total = sum(
            dmso_counts[(row["plate"], row["cell_line_id"], sample)]
            for sample in samples
        )
        if individual_total != int(row["control_cell_count"]):
            raise ValueError(
                f"Individual DMSO counts do not sum to pooled count for {key}: "
                f"{individual_total} != {row['control_cell_count']}"
            )

    for key in sorted(controls, key=lambda value: (plate_number(value[0]), value[1])):
        control = controls[key]
        for repeat_index, base_seed in enumerate(BASE_SEEDS, start=1):
            records.append(
                null_record(control, dmso_counts, repeat_index, base_seed)
            )

    expected = 60 * len(BASE_SEEDS) + 12 * len(BASE_SEEDS)
    if len(records) != expected:
        raise AssertionError(f"Expected {expected} cohort rows, got {len(records)}")
    if len({record["audit_pair_id"] for record in records}) != len(records):
        raise AssertionError("audit_pair_id is not unique")
    if any(record["source_seed"] == record["target_seed"] for record in records):
        raise AssertionError("Source and target seeds must be independent")

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dmso-counts", type=Path, default=DEFAULT_DMSO_COUNTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = read_selected_conditions(args.input)
    dmso_counts = read_dmso_counts(args.dmso_counts)
    records = build_cohort(rows, dmso_counts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    perturbation_count = sum(
        record["comparison_type"] == "perturbation" for record in records
    )
    null_count = len(records) - perturbation_count
    print(f"Wrote: {args.output}")
    print(f"Underlying perturbation conditions: {len(rows)}")
    print(f"Repeated-set rows: {perturbation_count} perturbation + {null_count} null")
    print(f"Set size: {SET_SIZE}; base seeds: {BASE_SEEDS}")
    print(f"Sampling: {SAMPLING_METHOD}")


if __name__ == "__main__":
    main()
