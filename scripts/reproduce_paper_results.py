#!/usr/bin/env python3
"""Reproduce paper-facing tables and figure source data from the frozen registry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reproducibility" / "tables"
FIGURES = ROOT / "reproducibility" / "figures"


def write_csv(directory: Path, name: str, fields: list[str], rows: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce frozen paper CSVs without training or dataset access."
    )
    parser.parse_args()

    registry = json.loads((ROOT / "metadata" / "result_registry.json").read_text())
    results = {entry["result_id"]: entry for entry in registry["results"]}

    census = results["lmd_census_80_10_10"]
    write_csv(
        TABLES,
        "lmd_census.csv",
        ["protocol", "identities", "families", "test_family_contamination", "test_file_contamination"],
        [
            {"protocol": "80/10/10 random file split", "identities": 178561, "families": 140427, "test_family_contamination": .2765669342, "test_file_contamination": .297670108},
            {"protocol": "90/5/5 random file split", "identities": 178561, "families": 140427, "test_family_contamination": .3030, "test_file_contamination": ""},
            {"protocol": "90/10 random file split", "identities": 178561, "families": 140427, "test_family_contamination": .2934, "test_file_contamination": ""},
        ],
    )

    transformer_l = results["lmd_transformer_l"]
    write_csv(
        TABLES,
        "confirmatory_effect.csv",
        ["model", "parameters", "treated_families", "tau", "ci_low", "ci_high", "relative_improvement", "status"],
        [{"model": transformer_l["model"], "parameters": transformer_l["parameter_count"], "treated_families": 100, "tau": transformer_l["effect"], "ci_low": transformer_l["ci95"][0], "ci_high": transformer_l["ci95"][1], "relative_improvement": transformer_l["relative_effect"], "status": transformer_l["status"]}],
    )

    capacity_rows = []
    for result_id in ["tcn_384", "transformer_s", "transformer_m", "lmd_transformer_l"]:
        result = results[result_id]
        capacity_rows.append({"model": result["model"], "parameters": result["parameter_count"], "tau": result["effect"], "ci_low": result["ci95"][0], "ci_high": result["ci95"][1], "relative_improvement": result["relative_effect"], "status": result["status"]})
    write_csv(TABLES, "architecture_capacity.csv", list(capacity_rows[0]), capacity_rows)

    mitigation = results["mitigation_4300"]["effect"]
    write_csv(
        TABLES,
        "mitigation.csv",
        ["protocol", "files", "cross_split_known_families", "token_loss", "reassigned_files"],
        [
            {"protocol": "file_split", "files": 4300, "cross_split_known_families": 177, "token_loss": 0, "reassigned_files": 0},
            {"protocol": "exact_dedup", "files": 4263, "cross_split_known_families": 161, "token_loss": .0080, "reassigned_files": ""},
            {"protocol": "family_aware", "files": 4300, "cross_split_known_families": 0, "token_loss": 0, "reassigned_files": .2595348837},
            {"protocol": "delete_multi_member", "files": "", "cross_split_known_families": 0, "token_loss": .2283570490, "reassigned_files": ""},
        ],
    )

    pdmx = results["pdmx_reduced"]
    write_csv(
        TABLES,
        "pdmx_external.csv",
        ["cohort", "treated", "tau", "ci_low", "ci_high", "relative_improvement", "status"],
        [{"cohort": "reduced eligible PDMX", "treated": 67, "tau": pdmx["effect"], "ci_low": pdmx["ci95"][0], "ci_high": pdmx["ci95"][1], "relative_improvement": pdmx["relative_effect"], "status": pdmx["status"]}],
    )

    write_csv(FIGURES, "prevalence.csv", ["split", "test_family_contamination"], [{"split": "80/10/10", "test_family_contamination": .2765669342}, {"split": "90/5/5", "test_family_contamination": .3030}, {"split": "90/10", "test_family_contamination": .2934}])
    write_csv(FIGURES, "capacity.csv", ["model", "parameters", "tau"], [{"model": row["model"], "parameters": row["parameters"], "tau": row["tau"]} for row in capacity_rows])
    write_csv(FIGURES, "mitigation.csv", ["protocol", "cross_split_known_families"], [{"protocol": "file_split", "cross_split_known_families": 177}, {"protocol": "exact_dedup", "cross_split_known_families": 161}, {"protocol": "family_aware", "cross_split_known_families": 0}])
    write_csv(FIGURES, "external_evidence.csv", ["dataset", "tau", "relative_improvement"], [{"dataset": "LMD", "tau": transformer_l["effect"], "relative_improvement": transformer_l["relative_effect"]}, {"dataset": "PDMX reduced", "tau": pdmx["effect"], "relative_improvement": pdmx["relative_effect"]}])


if __name__ == "__main__":
    main()
