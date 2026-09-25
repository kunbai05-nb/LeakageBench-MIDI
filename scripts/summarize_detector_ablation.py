#!/usr/bin/env python3
"""Create and verify the four reported detector-ablation tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DATASETS = ("lmd-clean", "atepp", "shs", "asap")
PAIR_ORDER = (
    ("full", "Full v1.9"),
    ("without_structural", "w/o structural features"),
    ("without_retrieval", "w/o retrieval evidence"),
    ("without_local_alignment", "w/o local alignment"),
    ("rule_based_verifier", "w/o classifier (rule baseline)"),
)
DIMENSIONS = {
    "full": "57",
    "without_structural": "28",
    "without_retrieval": "49",
    "without_local_alignment": "37",
    "rule_based_verifier": "-",
}


def read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> float:
    return round(100.0 * float(value) + 1e-12, 2)


def one_decimal(value: float) -> float:
    return round(float(value) + 1e-12, 1)


def derive(raw: Path) -> dict:
    table1 = read(raw / "TABLE1_VIEW_COUNT.json")
    table2 = read(raw / "TABLE2_PAIR_ABLATION.json")
    table4 = read(raw / "TABLE4_TOPK_SENSITIVITY.json")
    table5 = read(raw / "TABLE5_CLASSIFIER_COUNT.json")

    views = {}
    for count, row in table1["rows"].items():
        views[count] = {
            **{name: pct(row["datasets"][name]["candidate_recall"]) for name in DATASETS},
            "macro_recall": pct(row["macro_candidate_recall"]),
            "candidates_per_file": one_decimal(row["macro_candidates_per_file"]),
        }

    pairs = {}
    for key, _ in PAIR_ORDER:
        value = table2["macro"][key]
        if "retrained_recalibrated" in value:
            value = value["retrained_recalibrated"]
        pairs[key] = {
            "dimensions": DIMENSIONS[key],
            "precision": pct(value["precision"]),
            "recall": pct(value["recall"]),
            "f1": pct(value["f1"]),
            "auprc": pct(value["auprc_candidate_pool"]),
        }

    baseline_cost = table4["rows"]["50"]["macro_candidates_per_file"]
    topk = {}
    for k, row in table4["rows"].items():
        topk[k] = {
            **{name: pct(row["datasets"][name]["candidate_recall"]) for name in DATASETS},
            "macro_recall": pct(row["macro_candidate_recall"]),
            "candidates_per_file": one_decimal(row["macro_candidates_per_file"]),
            "relative_cost": round(row["macro_candidates_per_file"] / baseline_cost + 1e-12, 2),
        }

    classifiers = {}
    for count, row in table5["rows"].items():
        summary = row["summary"]
        if "threshold" in summary:
            threshold = summary["threshold"]
        else:
            values = [entry["threshold"]["threshold"] for entry in row["combinations"]]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            threshold = {"mean": mean, "standard_deviation": variance**0.5}
        classifiers[count] = {
            "precision": [pct(summary["precision"]["mean"]), pct(summary["precision"]["standard_deviation"])],
            "recall": [pct(summary["recall"]["mean"]), pct(summary["recall"]["standard_deviation"])],
            "f1": [pct(summary["f1"]["mean"]), pct(summary["f1"]["standard_deviation"])],
            "auprc": [pct(summary["auprc_candidate_pool"]["mean"]), pct(summary["auprc_candidate_pool"]["standard_deviation"])],
            "threshold": [round(threshold["mean"], 4), round(threshold["standard_deviation"], 4)],
        }
    return {
        "table1_view_count": views,
        "table2_pairwise_verifier": pairs,
        "table4_top_k": topk,
        "table5_classifier_count": classifiers,
    }


def verify(actual: dict, expected: dict) -> None:
    failures = []
    for table, rows in actual.items():
        expected_rows = expected.get(table, {})
        for key, row in rows.items():
            if key not in expected_rows:
                failures.append(f"{table}/{key}: missing expected row")
            elif row != expected_rows[key]:
                failures.append(
                    f"{table}/{key}:\n  generated={row}\n  expected ={expected_rows[key]}"
                )
    if failures:
        raise RuntimeError("paper-value verification failed:\n" + "\n".join(failures))


def write_table(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    markdown = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    markdown.extend("| " + " | ".join(str(row[field]) for field in fields) + " |" for row in rows)
    path.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def write_outputs(actual: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    table1 = []
    for count in (5, 4, 3, 2, 1):
        row = actual["table1_view_count"][str(count)]
        table1.append({"Views": count, **row})
    write_table(output / "table1_view_count", table1)

    table2 = []
    for key, label in PAIR_ORDER:
        table2.append({"Setting": label, **actual["table2_pairwise_verifier"][key]})
    write_table(output / "table2_pairwise_verifier", table2)

    table4 = []
    for k in (10, 25, 50, 75, 100):
        table4.append({"Top-K": k, **actual["table4_top_k"][str(k)]})
    write_table(output / "table4_top_k", table4)

    table5 = []
    for count in (1, 2, 3, 4, 5):
        row = actual["table5_classifier_count"][str(count)]
        table5.append(
            {
                "Classifiers": count,
                **{key: f"{value[0]:.2f} ± {value[1]:.2f}" for key, value in row.items() if key != "threshold"},
                "threshold": f"{row['threshold'][0]:.4f} ± {row['threshold'][1]:.4f}",
            }
        )
    write_table(output / "table5_classifier_count", table5)
    (output / "verification.json").write_text(
        json.dumps({"status": "PASS", "tables": [1, 2, 4, 5], "generated": actual}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_results", type=Path)
    parser.add_argument("expected_results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    actual = derive(args.raw_results)
    expected = read(args.expected_results)
    verify(actual, expected)
    write_outputs(actual, args.output)
    print(f"PASS: generated values match the frozen paper values ({args.output})")


if __name__ == "__main__":
    main()
