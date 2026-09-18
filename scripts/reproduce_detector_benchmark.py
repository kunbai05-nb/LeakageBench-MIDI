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

from leakagebench_midi.detector import detect, load_detector


ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "reproduction" / "detector_benchmark"
DATASETS = ("shs", "asap", "atepp", "lmd-clean", "vienna4x22", "pianovam")
GRID = np.round(np.arange(0.0, 1.001, 0.01), 2)


def read_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_identity(path: Path, dataset: str) -> str:
    data = path.read_bytes()
    if dataset == "asap":
        return hashlib.sha256(data).hexdigest()[:32]
    return hashlib.md5(data).hexdigest()


def verify_files(
    records: list[dict[str, str]], midi_root: Path, dataset: str
) -> list[Path]:
    paths = [midi_root / row["relative_path"] for row in records]
    for path, row in zip(paths, records):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = file_identity(path, dataset)
        if digest != row["file_md5"]:
            raise ValueError(f"MIDI checksum mismatch: {row['relative_path']}")
    return paths


def reference_pairs(records: list[dict[str, str]]) -> set[tuple[int, int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        groups[row["work_group"]].append(index)
    return {
        (left, right)
        for members in groups.values()
        for position, left in enumerate(members)
        for right in members[position + 1 :]
        if records[left]["recording_group"] != records[right]["recording_group"]
    }


def eligible_predictions(
    records: list[dict[str, str]], predicted: set[tuple[int, int]]
) -> set[tuple[int, int]]:
    return {
        (left, right)
        for left, right in predicted
        if records[left]["recording_group"] != records[right]["recording_group"]
    }


def predictions(
    pairs: np.ndarray, scores: np.ndarray, threshold: float
) -> set[tuple[int, int]]:
    return {
        tuple(map(int, pair))
        for pair, score in zip(pairs, scores)
        if float(score) >= threshold
    }


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


def query_macro(
    records: list[dict[str, str]],
    truth: set[tuple[int, int]],
    predicted: set[tuple[int, int]],
) -> dict:
    relevant: dict[int, set[int]] = defaultdict(set)
    guesses: dict[int, set[int]] = defaultdict(set)
    for source, target in ((truth, relevant), (predicted, guesses)):
        for left, right in source:
            target[left].add(right)
            target[right].add(left)
    precision, recall = [], []
    tp = fp = fn = 0
    for query, expected in relevant.items():
        found = guesses[query]
        hit = len(expected & found)
        tp += hit
        fp += len(found) - hit
        fn += len(expected) - hit
        recall.append(hit / len(expected))
        if found:
            precision.append(hit / len(found))
    p = float(np.mean(precision)) if precision else 0.0
    r = float(np.mean(recall)) if recall else 0.0
    return {
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce fixed and optimal threshold results."
    )
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("midi_root", type=Path)
    parser.add_argument("detector_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--backend", choices=("exact", "faiss"), default="exact")
    args = parser.parse_args()

    records = read_rows(SPECS / f"{args.dataset}.csv.gz")
    paths = verify_files(records, args.midi_root, args.dataset)
    result = detect(paths, args.detector_dir, args.workers, args.backend, threshold=0.0)
    if result["failures"]:
        raise RuntimeError(
            f"{len(result['failures'])} feature extractions failed; refusing partial evaluation"
        )
    pairs, scores = result["pairs"], result["scores"]
    truth = reference_pairs(records)
    metric = query_macro if args.dataset == "lmd-clean" else None

    def evaluate(threshold: float) -> dict:
        predicted = eligible_predictions(
            records, predictions(pairs, scores, threshold)
        )
        values = (
            metric(records, truth, predicted)
            if metric
            else pair_micro(truth, predicted)
        )
        return {"threshold": threshold, **values, "predicted_pairs": len(predicted)}

    metadata, _ = load_detector(args.detector_dir)
    fixed = evaluate(float(metadata["decision_threshold"]))
    optimal = max(
        (evaluate(float(threshold)) for threshold in GRID),
        key=lambda row: (row["f1"], row["recall"], row["precision"], -row["threshold"]),
    )
    summary = {
        "dataset": args.dataset,
        "files": len(records),
        "reference_pairs": len(truth),
        "metric": "query_macro" if metric else "pair_micro",
        "backend": args.backend,
        "detector": metadata["detector_id"],
        "feature_failures": 0,
        "unsupported_files": len(
            {item["index"] for item in result.get("unsupported", [])}
        ),
        "file_identity": "sha256-prefix-32" if args.dataset == "asap" else "md5",
        "candidate_diagnostics": result["candidate_diagnostics"],
        "fixed": fixed,
        "optimal": optimal,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "pair_scores.npz", pairs=pairs, scores=scores)
    (args.output_dir / "results.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
