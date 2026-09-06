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

from leakagebench_midi.detector import detect


ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "reproduction" / "detector_benchmark"


def rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_files(records: list[dict[str, str]], midi_root: Path) -> list[Path]:
    paths = [midi_root / row["relative_path"] for row in records]
    for path, row in zip(paths, records):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest != row["file_md5"]:
            raise ValueError(f"MIDI checksum mismatch: {row['relative_path']}")
    return paths


def pair_set(pairs: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, pair)) for pair in np.asarray(pairs).reshape(-1, 2)}


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
    for index in relevant:
        local_tp = len(relevant[index] & guesses[index])
        local_predictions = len(guesses[index])
        tp += local_tp
        fp += local_predictions - local_tp
        fn += len(relevant[index]) - local_tp
        recall.append(local_tp / len(relevant[index]))
        if local_predictions:
            precision.append(local_tp / local_predictions)
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


def pair_micro(
    truth: set[tuple[int, int]], predicted: set[tuple[int, int]]
) -> dict:
    tp = len(truth & predicted)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("shs", "asap", "lmd-clean"))
    parser.add_argument("midi_root", type=Path)
    parser.add_argument("detector_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--backend", choices=("exact", "faiss"), default="faiss")
    args = parser.parse_args()

    records = rows(SPECS / f"{args.dataset}.csv.gz")
    paths = verify_files(records, args.midi_root)
    result = detect(paths, args.detector_dir, args.workers, args.backend)
    truth = reference_pairs(records)
    predicted = eligible_predictions(
        records, pair_set(result["pairs"][result["selected"]])
    )
    convention = "query_macro" if args.dataset == "lmd-clean" else "pair_micro"
    metrics = (
        query_macro(records, truth, predicted)
        if convention == "query_macro"
        else pair_micro(truth, predicted)
    )
    summary = {
        "dataset": args.dataset,
        "files": len(records),
        "reference_pairs": len(truth),
        "metric_convention": convention,
        **metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(
        args.output_dir / "predicted_pairs.csv.gz",
        "wt",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("left", "right", "score"))
        for index in result["selected"]:
            left, right = result["pairs"][int(index)]
            writer.writerow((int(left), int(right), f"{result['scores'][int(index)]:.9f}"))
    (args.output_dir / "results.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
