#!/usr/bin/env python3
"""Combine six detector benchmark result files into paper-ready tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DATASETS = ("shs", "asap", "atepp", "lmd-clean", "vienna4x22", "pianovam")
EXPECTED_SCOPE = {
    "shs": (306, 80),
    "asap": (1067, 4988),
    "atepp": (1527, 3339),
    "lmd-clean": (2547, 2037),
    "vienna4x22": (88, 924),
    "pianovam": (94, 77),
}


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}"


def ci_text(value: list[float]) -> str:
    return f"[{100.0 * value[0]:.2f}, {100.0 * value[1]:.2f}]"


def load_results(input_dir: Path) -> list[dict]:
    output = []
    for dataset in DATASETS:
        path = input_dir / dataset / "results.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing benchmark result: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("dataset") != dataset:
            raise ValueError(f"dataset identity mismatch: {path}")
        if value.get("feature_failures") != 0:
            raise ValueError(f"feature extraction failures recorded in {path}")
        if value.get("backend") != "exact":
            raise ValueError(f"non-reference retrieval backend recorded in {path}")
        scope = (value.get("files"), value.get("reference_pairs"))
        if scope != EXPECTED_SCOPE[dataset]:
            raise ValueError(
                f"evaluation scope mismatch in {path}: {scope} != {EXPECTED_SCOPE[dataset]}"
            )
        output.append(value)
    return output


def flatten(results: list[dict]) -> list[dict[str, str | int]]:
    rows = []
    for item in results:
        row: dict[str, str | int] = {
            "Dataset": item["dataset"],
            "Files": item["files"],
            "Reference pairs": item["reference_pairs"],
            "Metric": item.get("primary_metric", item.get("metric", "pair_micro")),
        }
        for key, label in (("fixed", "Fixed"), ("optimal", "Optimal")):
            result = item[key]
            ci = result.get("pair_micro_ci95_work_bootstrap", {})
            row[f"{label} threshold"] = f"{result['threshold']:.6f}"
            for metric, title in (
                ("precision", "P"),
                ("recall", "R"),
                ("f1", "F1"),
            ):
                row[f"{label} {title} (%)"] = percent(result[metric])
                row[f"{label} {title} 95% CI"] = (
                    ci_text(ci[metric]) if metric in ci else ""
                )
        rows.append(row)
    return rows


def write_tables(results: list[dict], output_dir: Path) -> None:
    rows = flatten(results)
    fields = list(rows[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "six_dataset_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    markdown = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    markdown.extend("| " + " | ".join(str(row[field]) for field in fields) + " |" for row in rows)
    (output_dir / "six_dataset_results.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    (output_dir / "six_dataset_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the six-dataset result table.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    results = load_results(args.input_dir)
    write_tables(results, args.output_dir)
    print(args.output_dir / "six_dataset_results.csv")


if __name__ == "__main__":
    main()
