#!/usr/bin/env python3
"""Read-only global metadata audit for the local Tahoe-100M parquet shards."""

import argparse
import ast
import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


READ_COLUMNS = [
    "drug",
    "sample",
    "cell_line_id",
    "moa-fine",
    "canonical_smiles",
    "pubchem_cid",
    "plate",
]
CONDITION_COLUMNS = ["plate", "sample", "drug", "cell_line_id"]
MAPPING_COLUMNS = [
    "drug",
    "sample",
    "canonical_smiles",
    "pubchem_cid",
    "moa-fine",
]

CONTROL_RE = re.compile(
    r"^(?:control|ctrl|vehicle|dmso(?:_tf)?|dimethyl\s*sulfoxide|untreated|mock|negative(?: control)?|positive(?: control)?)$",
    re.IGNORECASE,
)
DOSE_RE = re.compile(
    r"dose|dosage|concentration|\b\d+(?:\.\d+)?\s*(?:nm|um|µm|μm|mm|mg/ml|ug/ml|µg/ml)\b",
    re.IGNORECASE,
)
TIME_RE = re.compile(
    r"timepoint|treatment.?time|\b\d+(?:\.\d+)?\s*(?:h|hr|hrs|hour|hours|min|day|days)\b",
    re.IGNORECASE,
)


def summarize_counts(counter):
    values = np.fromiter(counter.values(), dtype=np.int64)
    if not len(values):
        return {}
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": int(values.min()),
        "percentiles": {
            str(p): float(np.percentile(values, p)) for p in percentiles
        },
        "max": int(values.max()),
        "conditions_at_least": {
            str(size): int((values >= size).sum())
            for size in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        },
    }


def clean_set(values):
    return {value for value in values if value is not None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--sample-metadata", type=Path)
    parser.add_argument("--condition-csv", type=Path)
    args = parser.parse_args()

    if args.condition_csv is not None and args.sample_metadata is None:
        parser.error("--condition-csv requires --sample-metadata")

    paths = sorted(args.data_dir.glob("train-*.parquet"))
    if args.max_files is not None:
        paths = paths[: args.max_files]
    if not paths:
        raise FileNotFoundError(f"No train parquet shards under {args.data_dir}")

    started = time.time()
    expected_schema = None
    schema_mismatches = []
    total_rows = 0
    file_rows = []
    null_counts = Counter()
    unique_values = {column: set() for column in READ_COLUMNS}
    condition_counts = Counter()
    drug_counts = Counter()
    drug_to_samples = defaultdict(set)
    sample_to_drugs = defaultdict(set)
    drug_to_smiles = defaultdict(set)
    drug_to_pubchem = defaultdict(set)
    drug_to_moa = defaultdict(set)

    for index, path in enumerate(paths, start=1):
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow.remove_metadata()
        if expected_schema is None:
            expected_schema = schema
        elif schema != expected_schema:
            schema_mismatches.append(path.name)

        missing_columns = sorted(set(READ_COLUMNS) - set(schema.names))
        if missing_columns:
            raise ValueError(f"{path.name} is missing columns: {missing_columns}")

        table = parquet.read(columns=READ_COLUMNS)
        if table.num_rows != parquet.metadata.num_rows:
            raise AssertionError(f"Projected row count mismatch in {path.name}")

        total_rows += table.num_rows
        file_rows.append(table.num_rows)

        for column in READ_COLUMNS:
            values = table[column]
            null_counts[column] += values.null_count
            unique_values[column].update(clean_set(values.unique().to_pylist()))

        mappings = table.select(MAPPING_COLUMNS).group_by(MAPPING_COLUMNS).aggregate([])
        for row in mappings.to_pylist():
            drug = row["drug"]
            sample = row["sample"]
            drug_to_samples[drug].add(sample)
            sample_to_drugs[sample].add(drug)
            drug_to_smiles[drug].add(row["canonical_smiles"])
            drug_to_pubchem[drug].add(row["pubchem_cid"])
            drug_to_moa[drug].add(row["moa-fine"])

        grouped = table.select(CONDITION_COLUMNS).group_by(CONDITION_COLUMNS).aggregate(
            [("sample", "count")]
        )
        grouped_total = 0
        for row in grouped.to_pylist():
            count = int(row["sample_count"])
            key = tuple(row[column] for column in CONDITION_COLUMNS)
            condition_counts[key] += count
            drug_counts[row["drug"]] += count
            grouped_total += count
        if grouped_total != table.num_rows:
            raise AssertionError(
                f"Condition grouping lost rows in {path.name}: "
                f"{grouped_total} != {table.num_rows}"
            )

        if index % 100 == 0 or index == len(paths):
            print(
                f"scanned {index}/{len(paths)} shards, "
                f"rows={total_rows:,}, elapsed={(time.time() - started) / 60:.1f} min",
                flush=True,
            )

    if sum(condition_counts.values()) != total_rows:
        raise AssertionError("Global condition counts do not sum to total rows")

    pooled_condition_counts = Counter()
    pooled_condition_plates = defaultdict(set)
    for (plate, sample, drug, cell_line), count in condition_counts.items():
        pooled_key = (sample, drug, cell_line)
        pooled_condition_counts[pooled_key] += count
        pooled_condition_plates[pooled_key].add(plate)

    control_drugs = sorted(
        drug
        for drug in unique_values["drug"]
        if isinstance(drug, str) and CONTROL_RE.search(drug)
    )
    control_details = []
    for drug in control_drugs:
        matching_conditions = [
            key for key in condition_counts if key[2] == drug
        ]
        control_details.append(
            {
                "drug": drug,
                "cells": int(drug_counts[drug]),
                "samples": sorted(clean_set(drug_to_samples[drug])),
                "plates": sorted({key[0] for key in matching_conditions}),
                "cell_lines": sorted({key[3] for key in matching_conditions}),
                "condition_count": len(matching_conditions),
            }
        )

    mapping_exceptions = {
        "drugs_with_multiple_samples": {
            str(key): sorted(clean_set(values))
            for key, values in drug_to_samples.items()
            if len(clean_set(values)) != 1
        },
        "samples_with_multiple_drugs": {
            str(key): sorted(clean_set(values))
            for key, values in sample_to_drugs.items()
            if len(clean_set(values)) != 1
        },
        "drugs_with_multiple_smiles": {
            str(key): sorted(clean_set(values))
            for key, values in drug_to_smiles.items()
            if len(clean_set(values)) != 1
        },
        "drugs_with_multiple_pubchem_ids": {
            str(key): sorted(clean_set(values))
            for key, values in drug_to_pubchem.items()
            if len(clean_set(values)) != 1
        },
        "drugs_with_multiple_moa_values": {
            str(key): sorted(clean_set(values))
            for key, values in drug_to_moa.items()
            if len(clean_set(values)) != 1
        },
    }

    sample_metadata_audit = None
    if args.sample_metadata is not None:
        sample_metadata = pq.read_table(args.sample_metadata).to_pandas()
        required = {"sample", "plate", "drug", "drugname_drugconc"}
        missing = sorted(required - set(sample_metadata.columns))
        if missing:
            raise ValueError(f"Sample metadata is missing columns: {missing}")

        parsed = sample_metadata["drugname_drugconc"].map(ast.literal_eval)
        invalid_entry_counts = int((parsed.map(len) != 1).sum())
        if invalid_entry_counts:
            raise ValueError(
                f"Expected one drug-dose tuple per sample; found "
                f"{invalid_entry_counts} invalid rows"
            )

        sample_metadata = sample_metadata.copy()
        sample_metadata["parsed_drug"] = parsed.map(lambda entries: entries[0][0].strip())
        sample_metadata["concentration"] = parsed.map(lambda entries: entries[0][1])
        sample_metadata["concentration_unit"] = parsed.map(lambda entries: entries[0][2])
        parsed_name_mismatches = int(
            (
                sample_metadata["parsed_drug"]
                != sample_metadata["drug"].str.strip()
            ).sum()
        )

        duplicated_sample_rows = int(sample_metadata["sample"].duplicated().sum())
        if duplicated_sample_rows:
            raise ValueError(
                f"Sample metadata has {duplicated_sample_rows} duplicate sample rows"
            )

        sample_lookup = {
            row.sample: (
                row.plate,
                row.drug.strip(),
                float(row.concentration),
                row.concentration_unit,
            )
            for row in sample_metadata.itertuples(index=False)
        }
        expression_samples = unique_values["sample"]
        missing_metadata_samples = sorted(expression_samples - set(sample_lookup))
        unused_metadata_samples = sorted(set(sample_lookup) - expression_samples)

        sample_plate_drug_mismatches = []
        plate_drug_dose_condition_counts = Counter()
        drug_dose_condition_counts = Counter()
        samples_by_plate_drug_dose_cell = defaultdict(set)
        for (plate, sample, drug, cell_line), count in condition_counts.items():
            metadata_value = sample_lookup.get(sample)
            if metadata_value is None:
                continue
            metadata_plate, metadata_drug, concentration, unit = metadata_value
            if plate != metadata_plate or drug != metadata_drug:
                sample_plate_drug_mismatches.append(
                    {
                        "sample": sample,
                        "expression_plate": plate,
                        "metadata_plate": metadata_plate,
                        "expression_drug": drug,
                        "metadata_drug": metadata_drug,
                    }
                )
                continue
            plate_drug_dose_condition_counts[
                (plate, drug, concentration, unit, cell_line)
            ] += count
            samples_by_plate_drug_dose_cell[
                (plate, drug, concentration, unit, cell_line)
            ].add(sample)
            drug_dose_condition_counts[
                (drug, concentration, unit, cell_line)
            ] += count

        control_name = "DMSO_TF"
        control_sample_conditions = Counter(
            {
                key: count
                for key, count in condition_counts.items()
                if key[2] == control_name
            }
        )
        control_plate_conditions = Counter(
            {
                key: count
                for key, count in plate_drug_dose_condition_counts.items()
                if key[1] == control_name
            }
        )
        active_plate_conditions = Counter(
            {
                key: count
                for key, count in plate_drug_dose_condition_counts.items()
                if key[1] != control_name
            }
        )
        active_drug_dose_conditions = Counter(
            {
                key: count
                for key, count in drug_dose_condition_counts.items()
                if key[0] != control_name
            }
        )
        dmso_by_plate_cell = {
            (plate, cell_line): count
            for (plate, _drug, _dose, _unit, cell_line), count
            in control_plate_conditions.items()
        }
        missing_matched_controls = []
        matched_pair_capacities = Counter()
        for key, treated_count in active_plate_conditions.items():
            plate, drug, concentration, unit, cell_line = key
            control_count = dmso_by_plate_cell.get((plate, cell_line))
            if control_count is None:
                missing_matched_controls.append(
                    {
                        "plate": plate,
                        "drug": drug,
                        "concentration": concentration,
                        "unit": unit,
                        "cell_line_id": cell_line,
                    }
                )
                continue
            matched_pair_capacities[key] = min(treated_count, control_count)

        concentration_counts = sample_metadata["concentration"].value_counts().sort_index()
        unit_counts = sample_metadata["concentration_unit"].value_counts()
        samples_per_plate = sample_metadata.groupby("plate")["sample"].nunique()
        drug_dose_combinations = sample_metadata[
            ["drug", "concentration", "concentration_unit"]
        ].drop_duplicates()
        active_drug_dose_combinations = drug_dose_combinations[
            drug_dose_combinations["drug"] != control_name
        ]

        sample_metadata_audit = {
            "path": str(args.sample_metadata.resolve()),
            "rows": int(len(sample_metadata)),
            "distinct_samples": int(sample_metadata["sample"].nunique()),
            "distinct_drugs": int(sample_metadata["drug"].nunique()),
            "distinct_plates": int(sample_metadata["plate"].nunique()),
            "samples_per_plate": {
                str(key): int(value) for key, value in samples_per_plate.items()
            },
            "concentration_sample_counts": {
                str(key): int(value) for key, value in concentration_counts.items()
            },
            "concentration_unit_counts": {
                str(key): int(value) for key, value in unit_counts.items()
            },
            "drug_dose_combination_count_including_control": int(
                len(drug_dose_combinations)
            ),
            "drug_dose_combination_count_excluding_control": int(
                len(active_drug_dose_combinations)
            ),
            "parsed_name_mismatch_count": parsed_name_mismatches,
            "missing_expression_samples_in_metadata": missing_metadata_samples,
            "unused_metadata_samples": unused_metadata_samples,
            "sample_plate_drug_mismatch_count": len(sample_plate_drug_mismatches),
            "sample_plate_drug_mismatch_examples": sample_plate_drug_mismatches[:20],
            "plate_drug_dose_condition_counts": summarize_counts(
                plate_drug_dose_condition_counts
            ),
            "active_plate_drug_dose_condition_counts": summarize_counts(
                active_plate_conditions
            ),
            "drug_dose_condition_counts_across_plates": summarize_counts(
                drug_dose_condition_counts
            ),
            "active_drug_dose_condition_counts_across_plates": summarize_counts(
                active_drug_dose_conditions
            ),
            "dmso_sample_condition_counts": summarize_counts(
                control_sample_conditions
            ),
            "dmso_plate_condition_counts": summarize_counts(
                control_plate_conditions
            ),
            "active_conditions_without_matched_plate_cell_dmso_count": len(
                missing_matched_controls
            ),
            "active_conditions_without_matched_plate_cell_dmso_examples": (
                missing_matched_controls[:20]
            ),
            "matched_treated_control_pair_capacity": summarize_counts(
                matched_pair_capacities
            ),
        }

        if args.condition_csv is not None:
            args.condition_csv.parent.mkdir(parents=True, exist_ok=True)
            temporary_csv = args.condition_csv.with_suffix(
                args.condition_csv.suffix + ".tmp"
            )
            columns = [
                "pair_id",
                "plate",
                "cell_line_id",
                "drug",
                "dose_uM",
                "treated_samples",
                "treated_sample_count",
                "treated_cell_count",
                "control_drug",
                "control_samples",
                "control_sample_count",
                "control_cell_count",
                "matched_capacity",
                "eligible_S64",
                "eligible_S128",
                "eligible_S256",
                "eligible_S512",
                "moa_fine",
                "canonical_smiles",
                "pubchem_cid",
            ]

            def plate_number(plate):
                return int(plate.removeprefix("plate"))

            ordered_conditions = sorted(
                active_plate_conditions,
                key=lambda key: (
                    plate_number(key[0]),
                    key[4],
                    key[1].casefold(),
                    key[2],
                ),
            )
            with temporary_csv.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                for pair_index, key in enumerate(ordered_conditions, start=1):
                    plate, drug, concentration, _unit, cell_line = key
                    control_key = (plate, control_name, 0.0, "uM", cell_line)
                    treated_count = active_plate_conditions[key]
                    control_count = control_plate_conditions[control_key]
                    capacity = min(treated_count, control_count)
                    treated_samples = sorted(samples_by_plate_drug_dose_cell[key])
                    control_samples = sorted(
                        samples_by_plate_drug_dose_cell[control_key]
                    )
                    writer.writerow(
                        {
                            "pair_id": f"pair_{pair_index:05d}",
                            "plate": plate,
                            "cell_line_id": cell_line,
                            "drug": drug,
                            "dose_uM": concentration,
                            "treated_samples": "|".join(treated_samples),
                            "treated_sample_count": len(treated_samples),
                            "treated_cell_count": treated_count,
                            "control_drug": control_name,
                            "control_samples": "|".join(control_samples),
                            "control_sample_count": len(control_samples),
                            "control_cell_count": control_count,
                            "matched_capacity": capacity,
                            "eligible_S64": int(capacity >= 64),
                            "eligible_S128": int(capacity >= 128),
                            "eligible_S256": int(capacity >= 256),
                            "eligible_S512": int(capacity >= 512),
                            "moa_fine": "|".join(
                                sorted(clean_set(drug_to_moa[drug]))
                            ),
                            "canonical_smiles": "|".join(
                                sorted(clean_set(drug_to_smiles[drug]))
                            ),
                            "pubchem_cid": "|".join(
                                sorted(clean_set(drug_to_pubchem[drug]))
                            ),
                        }
                    )
            temporary_csv.replace(args.condition_csv)
            sample_metadata_audit["condition_csv"] = str(
                args.condition_csv.resolve()
            )
            sample_metadata_audit["condition_csv_rows"] = len(ordered_conditions)

    text_columns = ["drug", "sample", "cell_line_id", "moa-fine", "plate"]
    result = {
        "source": str(args.data_dir.resolve()),
        "scanned_shards": len(paths),
        "total_rows": total_rows,
        "elapsed_seconds": time.time() - started,
        "schema_columns": expected_schema.names,
        "schema_mismatch_count": len(schema_mismatches),
        "schema_mismatch_examples": schema_mismatches[:20],
        "file_rows": {
            "min": min(file_rows),
            "mean": sum(file_rows) / len(file_rows),
            "max": max(file_rows),
        },
        "null_counts": {column: int(null_counts[column]) for column in READ_COLUMNS},
        "distinct_counts": {
            column: len(unique_values[column]) for column in READ_COLUMNS
        },
        "all_drugs": sorted(unique_values["drug"]),
        "all_plates": sorted(unique_values["plate"]),
        "all_cell_lines": sorted(unique_values["cell_line_id"]),
        "control_candidates": control_details,
        "dose_like_values": {
            column: sorted(
                value
                for value in unique_values[column]
                if isinstance(value, str) and DOSE_RE.search(value)
            )
            for column in text_columns
        },
        "time_like_values": {
            column: sorted(
                value
                for value in unique_values[column]
                if isinstance(value, str) and TIME_RE.search(value)
            )
            for column in text_columns
        },
        "mapping_exception_counts": {
            key: len(value) for key, value in mapping_exceptions.items()
        },
        "mapping_exceptions": mapping_exceptions,
        "plate_specific_condition_counts": summarize_counts(condition_counts),
        "pooled_across_plates_condition_counts": summarize_counts(
            pooled_condition_counts
        ),
        "pooled_conditions_on_multiple_plates": int(
            sum(len(plates) > 1 for plates in pooled_condition_plates.values())
        ),
        "sample_metadata_audit": sample_metadata_audit,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
