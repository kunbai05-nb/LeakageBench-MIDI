#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reproduction" / "data" / "detector"
EXPECTED = ROOT / "reproduction" / "frozen" / "detector_asap_summary.json"
SEED = 20260822 + 901
BOOTSTRAPS = 10_000


def load_rows():
    with (DATA / "asap_files.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    work = np.asarray([int(row["work_id"]) for row in rows], dtype=np.int32)
    component = np.asarray([int(row["component_id"]) for row in rows], dtype=np.int32)
    direct = set()
    with gzip.open(
        DATA / "asap_candidate_pairs.csv.gz", "rt", newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            if int(row["direct_edge"]):
                direct.add((int(row["left"]), int(row["right"])))
    groups = defaultdict(list)
    for file_id, label in enumerate(component):
        groups[int(label)].append(file_id)
    closed = {
        (left, right)
        for members in groups.values()
        for offset, left in enumerate(members)
        for right in members[offset + 1 :]
    }
    return work, direct, closed, groups


def positive_pairs(work: np.ndarray) -> set[tuple[int, int]]:
    groups = defaultdict(list)
    for file_id, work_id in enumerate(work):
        groups[int(work_id)].append(file_id)
    return {
        (left, right)
        for members in groups.values()
        for offset, left in enumerate(members)
        for right in members[offset + 1 :]
    }


def point_metrics(work: np.ndarray, predicted: set[tuple[int, int]]) -> dict:
    positives = positive_pairs(work)
    true_positive = len(predicted & positives)
    false_positive = len(predicted - positives)
    precision = true_positive / len(predicted)
    recall = true_positive / len(positives)
    totals = defaultdict(int)
    found = defaultdict(int)
    for pair in positives:
        work_id = int(work[pair[0]])
        totals[work_id] += 1
        found[work_id] += pair in predicted
    work_recall = np.asarray([found[key] / total for key, total in totals.items()])
    return {
        "predicted_pairs": len(predicted),
        "true_positive_pairs": true_positive,
        "false_positive_pairs": false_positive,
        "total_positive_pairs": len(positives),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
        "eligible_multi_recording_works": len(totals),
        "work_macro_recall": float(work_recall.mean()),
        "work_median_recall": float(np.median(work_recall)),
        "works_with_missed_positive": int(np.count_nonzero(work_recall < 1)),
    }


def bootstrap(work: np.ndarray, systems: dict[str, set[tuple[int, int]]]) -> dict:
    works = np.unique(work)
    position = {int(work_id): index for index, work_id in enumerate(works)}
    positives = positive_pairs(work)
    total = np.zeros(len(works))
    for left, _ in positives:
        total[position[int(work[left])]] += 1

    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(works), size=(BOOTSTRAPS, len(works)))
    total_draw = total[draws].sum(axis=1)
    summaries = {}
    for name, predicted in systems.items():
        true_positive = np.zeros(len(works))
        false_positive = np.zeros(len(works))
        for left, right in predicted:
            a, b = int(work[left]), int(work[right])
            if a == b:
                true_positive[position[a]] += 1
            else:
                false_positive[position[a]] += 0.5
                false_positive[position[b]] += 0.5
        tp = true_positive[draws].sum(axis=1)
        fp = false_positive[draws].sum(axis=1)
        precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0)
        recall = np.divide(tp, total_draw, out=np.zeros_like(tp), where=total_draw > 0)
        per_work = true_positive[total > 0] / total[total > 0]
        macro = rng.choice(
            per_work, size=(BOOTSTRAPS, len(per_work)), replace=True
        ).mean(axis=1)
        summaries[name] = {
            "precision_ci95": np.quantile(precision, (0.025, 0.975)).tolist(),
            "pair_recall_ci95": np.quantile(recall, (0.025, 0.975)).tolist(),
            "clique_macro_recall": float(per_work.mean()),
            "clique_macro_recall_ci95": np.quantile(macro, (0.025, 0.975)).tolist(),
            "clique_median_recall": float(np.median(per_work)),
        }
    return {
        "cluster_unit": "ASAP composition (composer, title)",
        "bootstrap_samples": BOOTSTRAPS,
        "cross_clique_false_positive_attribution": "half to each endpoint clique",
        "systems": summaries,
    }


def compare(actual, expected, path="result") -> list[str]:
    errors = []
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key not in actual:
                errors.append(f"{path}.{key}: missing")
            else:
                errors.extend(compare(actual[key], value, f"{path}.{key}"))
    elif isinstance(expected, list):
        for index, value in enumerate(expected):
            errors.extend(compare(actual[index], value, f"{path}[{index}]"))
    elif isinstance(expected, float):
        if not np.isclose(actual, expected, rtol=0, atol=1e-12):
            errors.append(f"{path}: {actual} != {expected}")
    elif actual != expected:
        errors.append(f"{path}: {actual} != {expected}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    work, direct, closed, groups = load_rows()
    result = {
        "analysis_unit": "ASAP composition (composer, title)",
        "bootstrap": bootstrap(
            work,
            {
                "improved_direct_edges": direct,
                "improved_guarded_components": closed,
            },
        ),
        "point_estimates": {
            "direct_edges": point_metrics(work, direct),
            "guarded_components": {
                **point_metrics(work, closed),
                "largest_component": max(map(len, groups.values())),
            },
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.verify:
        errors = compare(result, json.loads(EXPECTED.read_text()))
        if errors:
            raise SystemExit("\n".join(errors))
        print("Detector statistics: PASS")
    elif not args.output:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
