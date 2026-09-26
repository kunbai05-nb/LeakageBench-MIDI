#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


DATASETS = ("atepp", "lmd-clean", "asap")
EXPECTED = {
    "atepp": (1527, 3339, "pair_micro"),
    "lmd-clean": (2547, 2037, "pair_micro"),
    "asap": (1067, 4988, "pair_micro"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def truth_pairs(rows: list[dict]) -> set[tuple[int, int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["reference_work_id"])].append(index)
    return {
        (left, right)
        for members in groups.values()
        for position, left in enumerate(members)
        for right in members[position + 1 :]
        if rows[left]["recording_group"] != rows[right]["recording_group"]
    }


def fast_pairs(path: Path, rows: list[dict]) -> set[tuple[int, int]]:
    identities = {str(row["identity"]): index for index, row in enumerate(rows)}
    predicted: set[tuple[int, int]] = set()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        next(handle)
        for line_number, line in enumerate(handle, 2):
            body, score = line.rstrip("\n").rsplit(",", 1)
            float(score)
            matches = []
            for position, character in enumerate(body):
                if character != ",":
                    continue
                left, right = body[:position], body[position + 1 :]
                if left in identities and right in identities:
                    matches.append((identities[left], identities[right]))
            if len(matches) != 1:
                raise ValueError(f"ambiguous edge at {path}:{line_number}")
            left, right = matches[0]
            if rows[left]["recording_group"] != rows[right]["recording_group"]:
                predicted.add((min(left, right), max(left, right)))
    return predicted


def pair_micro(truth: set[tuple[int, int]], predicted: set[tuple[int, int]]) -> dict:
    tp = len(truth & predicted)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def query_macro(truth: set[tuple[int, int]], predicted: set[tuple[int, int]]) -> dict:
    relevant: dict[int, set[int]] = defaultdict(set)
    guesses: dict[int, set[int]] = defaultdict(set)
    for source, target in ((truth, relevant), (predicted, guesses)):
        for left, right in source:
            target[left].add(right)
            target[right].add(left)
    precision, recall = [], []
    for query, expected in relevant.items():
        found = guesses[query]
        hit = len(expected & found)
        precision.append(hit / len(found) if found else 0.0)
        recall.append(hit / len(expected))
    p = float(np.mean(precision)) if precision else 0.0
    r = float(np.mean(recall)) if recall else 0.0
    return {
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "query_coverage": sum(bool(guesses[q]) for q in relevant) / max(1, len(relevant)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = {"status": "PASS", "datasets": {}}
    table = []
    for dataset in DATASETS:
        rows = load_manifest(args.evidence / "manifests" / f"{dataset}.jsonl")
        truth = truth_pairs(rows)
        fast_meta = json.loads((args.evidence / "fast" / dataset / "RESULTS.json").read_text())
        fast = fast_pairs(args.evidence / "fast" / dataset / "predicted_edges.csv.gz", rows)
        v19 = json.loads((args.evidence / "v1.9" / dataset / "results.json").read_text())
        candidate = json.loads((args.evidence / "v1.9" / dataset / "candidate_summary.json").read_text())
        expected_files, expected_pairs, metric = EXPECTED[dataset]
        if len(rows) != expected_files or len(truth) != expected_pairs:
            raise RuntimeError(f"frozen cohort mismatch for {dataset}")
        fast_metrics = pair_micro(truth, fast)
        v19_metrics = v19["fixed"]["pair_micro"]
        records["datasets"][dataset] = {
            "files": len(rows),
            "reference_pairs": len(truth),
            "manifest_sha256": sha256(args.evidence / "manifests" / f"{dataset}.jsonl"),
            "fast": fast_metrics,
            "v1.9": v19_metrics,
        }
        for method, metrics, recall, query in (
            ("Fast", fast_metrics, fast_meta["candidate_reference_metrics"]["reference_pair_recall"], query_macro(truth, fast)),
            ("v1.9", v19_metrics, candidate["candidate_recall"], v19["fixed"]["query_macro"]),
        ):
            table.append({
                "dataset": dataset,
                "method": method,
                "metric": metric,
                "candidate_recall": recall,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "query_coverage": query["query_coverage"],
            })
    (args.output / "fast_vs_v19_results.json").write_text(json.dumps(records, indent=2) + "\n")
    with (args.output / "fast_vs_v19.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)
    lines = [
        "| Dataset | Method | Candidate Recall (%) | Precision (%) | Recall (%) | F1 (%) | Query Coverage (%) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f'| {row["dataset"]} | {row["method"]} | {100*row["candidate_recall"]:.2f} | '
            f'{100*row["precision"]:.2f} | {100*row["recall"]:.2f} | '
            f'{100*row["f1"]:.2f} | {100*row["query_coverage"]:.2f} |'
        )
    (args.output / "fast_vs_v19.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "PASS", "rows": table}, indent=2))


if __name__ == "__main__":
    main()
