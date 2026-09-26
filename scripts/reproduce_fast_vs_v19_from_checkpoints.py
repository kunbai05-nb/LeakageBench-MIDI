#!/usr/bin/env python3
"""Reproduce the Fast-versus-v1.9 comparison from checkpoints and candidate features."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np


DATASETS = ("atepp", "lmd-clean", "asap")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def truth_pairs(records: list[dict]) -> set[tuple[int, int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[str(record["reference_work_id"])].append(index)
    return {
        (left, right)
        for members in groups.values()
        for offset, left in enumerate(members)
        for right in members[offset + 1 :]
        if records[left]["recording_group"] != records[right]["recording_group"]
    }


def eligible(records: list[dict], pairs: np.ndarray) -> np.ndarray:
    return np.asarray(
        [records[int(a)]["recording_group"] != records[int(b)]["recording_group"] for a, b in pairs],
        dtype=bool,
    )


def pair_set(pairs: np.ndarray, mask: np.ndarray) -> set[tuple[int, int]]:
    return {(min(int(a), int(b)), max(int(a), int(b))) for a, b in pairs[mask]}


def pair_micro(truth: set[tuple[int, int]], predicted: set[tuple[int, int]]) -> dict:
    tp = len(truth & predicted)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def query_coverage(truth: set[tuple[int, int]], predicted: set[tuple[int, int]]) -> float:
    relevant: dict[int, set[int]] = defaultdict(set)
    guesses: dict[int, set[int]] = defaultdict(set)
    for source, target in ((truth, relevant), (predicted, guesses)):
        for left, right in source:
            target[left].add(right)
            target[right].add(left)
    return sum(bool(guesses[query]) for query in relevant) / max(1, len(relevant))


def model_scores(features: np.ndarray, model_root: Path, config: dict) -> np.ndarray:
    scores = []
    for item in config["ensemble_models"]:
        path = model_root / item["artifact"]
        if item.get("sha256") and sha256(path) != item["sha256"]:
            raise RuntimeError(f"Fast checkpoint checksum mismatch: {path.name}")
        scores.append(joblib.load(path).predict_proba(features)[:, 1])
    return np.mean(np.vstack(scores), axis=0)


def v19_scores(features: np.ndarray, model_root: Path, config: dict) -> np.ndarray:
    raw = []
    for name, expected in zip(config["model_files"], config["model_sha256"]):
        path = model_root / name
        if sha256(path) != expected:
            raise RuntimeError(f"v1.9 checkpoint checksum mismatch: {name}")
        raw.append(joblib.load(path).predict_proba(features)[:, 1])
    mean = np.mean(np.vstack(raw), axis=0)
    calibration = config["score_calibration"]
    logits = calibration["coefficient"] * mean + calibration["intercept"]
    return 1.0 / (1.0 + np.exp(-logits))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fast_config = json.loads((args.evidence / "checkpoints" / "fast" / "FROZEN_CONFIG.json").read_text())
    v19_config = json.loads((args.evidence / "checkpoints" / "v1.9" / "MODEL_CONFIG.json").read_text())
    table = []
    details = {"status": "PASS", "source": "fresh checkpoint inference", "datasets": {}}

    for dataset in DATASETS:
        records = rows(args.evidence / "manifests" / f"{dataset}.jsonl")
        truth = truth_pairs(records)
        dataset_rows = {}
        for method in ("fast", "v1.9"):
            stored = np.load(args.evidence / "features" / method / f"{dataset}.npz", allow_pickle=False)
            pairs = stored["pairs"].astype(np.int32)
            features = stored["features"].astype(np.float32)
            names = stored["feature_names"].astype(str).tolist()
            if method == "fast":
                if names != fast_config["feature_names"]:
                    raise RuntimeError(f"Fast feature schema mismatch: {dataset}")
                scores = model_scores(features, args.evidence / "checkpoints" / "fast", fast_config)
                threshold = float(fast_config["threshold_selection"]["threshold"])
            else:
                if names != v19_config["feature_names"]:
                    raise RuntimeError(f"v1.9 feature schema mismatch: {dataset}")
                scores = v19_scores(features, args.evidence / "checkpoints" / "v1.9", v19_config)
                threshold = float(v19_config["decision_threshold"])
            valid = eligible(records, pairs)
            candidates = pair_set(pairs, valid)
            predicted = pair_set(pairs, valid & (scores >= threshold))
            metrics = pair_micro(truth, predicted)
            metrics.update(
                candidate_recall=len(truth & candidates) / len(truth),
                query_coverage=query_coverage(truth, predicted),
                threshold=threshold,
            )
            dataset_rows[method] = metrics
            table.append({"dataset": dataset, "method": method, **metrics})
        details["datasets"][dataset] = dataset_rows

    (args.output / "fast_vs_v19_results.json").write_text(json.dumps(details, indent=2) + "\n")
    fields = ("dataset", "method", "candidate_recall", "precision", "recall", "f1", "query_coverage", "threshold", "tp", "fp", "fn")
    with (args.output / "fast_vs_v19.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    lines = [
        "| Dataset | Method | Candidate recall | Precision | Recall | F1 | Query coverage |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            "| {dataset} | {method} | {candidate_recall:.2%} | {precision:.2%} | "
            "{recall:.2%} | {f1:.2%} | {query_coverage:.2%} |".format(**row)
        )
    (args.output / "fast_vs_v19.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output / "fast_vs_v19.csv")


if __name__ == "__main__":
    main()
