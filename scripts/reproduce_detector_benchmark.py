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
BOOTSTRAP_SEED = 20260921
BOOTSTRAP_REPLICATES = 2000


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
        # A relevant query with no returned item has precision zero. Omitting
        # such queries inflates macro precision and makes coverage invisible.
        precision.append(hit / len(found) if found else 0.0)
    p = float(np.mean(precision)) if precision else 0.0
    r = float(np.mean(recall)) if recall else 0.0
    return {
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "query_count": len(relevant),
        "covered_queries": sum(bool(guesses[query]) for query in relevant),
        "query_coverage": (
            sum(bool(guesses[query]) for query in relevant) / len(relevant)
            if relevant
            else 0.0
        ),
    }


def cluster_bootstrap_ci(
    records: list[dict[str, str]],
    truth: set[tuple[int, int]],
    predicted: set[tuple[int, int]],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, list[float]]:
    """Work-family bootstrap CIs for pair-micro metrics.

    Counts are accumulated per query and therefore reproduce pair-micro
    precision/recall/F1 exactly (every unordered pair is counted twice), while
    resampling whole work families respects within-family dependence.
    """
    relevant: dict[int, set[int]] = defaultdict(set)
    guesses: dict[int, set[int]] = defaultdict(set)
    for source, target in ((truth, relevant), (predicted, guesses)):
        for left, right in source:
            target[left].add(right)
            target[right].add(left)
    groups: dict[str, list[int]] = defaultdict(list)
    for query, row in enumerate(records):
        groups[row["work_group"]].append(query)
    family_counts = []
    for queries in groups.values():
        tp = fp = fn = 0
        for query in queries:
            expected, found = relevant[query], guesses[query]
            hit = len(expected & found)
            tp += hit
            fp += len(found) - hit
            fn += len(expected) - hit
        family_counts.append((tp, fp, fn))
    counts = np.asarray(family_counts, dtype=np.int64)
    rng = np.random.default_rng(seed)
    values = np.empty((replicates, 3), dtype=np.float64)
    for index in range(replicates):
        sampled = counts[rng.integers(0, len(counts), size=len(counts))].sum(axis=0)
        tp, fp, fn = map(int, sampled)
        values[index, 0] = tp / (tp + fp) if tp + fp else 0.0
        values[index, 1] = tp / (tp + fn) if tp + fn else 0.0
        values[index, 2] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        name: [float(x) for x in np.quantile(values[:, column], [0.025, 0.975])]
        for column, name in enumerate(("precision", "recall", "f1"))
    }


def query_macro_cluster_bootstrap_ci(
    records: list[dict[str, str]],
    truth: set[tuple[int, int]],
    predicted: set[tuple[int, int]],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, list[float]]:
    relevant: dict[int, set[int]] = defaultdict(set)
    guesses: dict[int, set[int]] = defaultdict(set)
    for source, target in ((truth, relevant), (predicted, guesses)):
        for left, right in source:
            target[left].add(right)
            target[right].add(left)
    groups: dict[str, list[int]] = defaultdict(list)
    for query in relevant:
        groups[records[query]["work_group"]].append(query)
    family_values = []
    for queries in groups.values():
        p_sum = r_sum = 0.0
        for query in queries:
            expected, found = relevant[query], guesses[query]
            hit = len(expected & found)
            p_sum += hit / len(found) if found else 0.0
            r_sum += hit / len(expected)
        family_values.append((p_sum, r_sum, len(queries)))
    values = np.asarray(family_values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled_metrics = np.empty((replicates, 3), dtype=np.float64)
    for index in range(replicates):
        sampled = values[rng.integers(0, len(values), size=len(values))].sum(axis=0)
        precision = sampled[0] / sampled[2] if sampled[2] else 0.0
        recall = sampled[1] / sampled[2] if sampled[2] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        sampled_metrics[index] = (precision, recall, f1)
    return {
        name: [float(x) for x in np.quantile(sampled_metrics[:, column], [0.025, 0.975])]
        for column, name in enumerate(("precision", "recall", "f1"))
    }


def optimal_threshold(
    records: list[dict[str, str]],
    truth: set[tuple[int, int]],
    pairs: np.ndarray,
    scores: np.ndarray,
    use_query_macro: bool,
) -> float:
    """Find the exact attainable F1 optimum in one score-ordered pass."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    ordered_pairs = np.asarray(pairs, dtype=np.int64)[order]
    ordered_scores = np.asarray(scores, dtype=np.float64)[order]
    relevant: dict[int, set[int]] = defaultdict(set)
    for left, right in truth:
        relevant[left].add(right)
        relevant[right].add(left)
    found: dict[int, set[int]] = defaultdict(set)
    hit_by_query: dict[int, int] = defaultdict(int)
    precision_sum = recall_sum = 0.0
    query_count = len(relevant)
    pair_tp = pair_fp = 0
    best_key = (-1.0, -1.0, -1.0, 0.0)
    best_threshold = float("inf")
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and ordered_scores[end] == ordered_scores[start]:
            end += 1
        for left, right in ordered_pairs[start:end]:
            pair = (int(min(left, right)), int(max(left, right)))
            if pair in truth:
                pair_tp += 1
            else:
                pair_fp += 1
            for query, target in ((int(left), int(right)), (int(right), int(left))):
                expected = relevant[query]
                old_found = len(found[query])
                old_hit = hit_by_query[query]
                precision_sum -= old_hit / old_found if old_found else 0.0
                recall_sum -= old_hit / len(expected) if expected else 0.0
                found[query].add(target)
                if target in expected:
                    hit_by_query[query] += 1
                new_found = len(found[query])
                new_hit = hit_by_query[query]
                precision_sum += new_hit / new_found if new_found else 0.0
                recall_sum += new_hit / len(expected) if expected else 0.0
        if use_query_macro:
            precision = precision_sum / query_count if query_count else 0.0
            recall = recall_sum / query_count if query_count else 0.0
        else:
            fn = len(truth) - pair_tp
            precision = pair_tp / (pair_tp + pair_fp) if pair_tp + pair_fp else 0.0
            recall = pair_tp / (pair_tp + fn) if pair_tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        threshold = float(ordered_scores[start])
        key = (f1, recall, precision, -threshold)
        if key > best_key:
            best_key, best_threshold = key, threshold
        start = end
    return best_threshold


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
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Frozen evaluation manifest override (for example, family-disjoint SHS).",
    )
    args = parser.parse_args()

    manifest = args.manifest or (SPECS / f"{args.dataset}.csv.gz")
    records = read_rows(manifest)
    paths = verify_files(records, args.midi_root, args.dataset)
    result = detect(paths, args.detector_dir, args.workers, args.backend, threshold=0.0)
    if result["failures"]:
        raise RuntimeError(
            f"{len(result['failures'])} feature extractions failed; refusing partial evaluation"
        )
    pairs, scores = result["pairs"], result["scores"]
    eligible = np.asarray(
        [
            records[int(left)]["recording_group"]
            != records[int(right)]["recording_group"]
            for left, right in pairs
        ],
        dtype=bool,
    )
    pairs, scores = pairs[eligible], scores[eligible]
    truth = reference_pairs(records)
    def evaluate(threshold: float, include_ci: bool = True) -> dict:
        predicted = eligible_predictions(
            records, predictions(pairs, scores, threshold)
        )
        micro = pair_micro(truth, predicted)
        macro = query_macro(records, truth, predicted)
        primary = micro
        output = {
            "threshold": threshold,
            **primary,
            "predicted_pairs": len(predicted),
            "pair_micro": micro,
            "query_macro": macro,
        }
        if include_ci:
            output["pair_micro_ci95_work_bootstrap"] = cluster_bootstrap_ci(
                records, truth, predicted
            )
            output["query_macro_ci95_work_bootstrap"] = query_macro_cluster_bootstrap_ci(
                records, truth, predicted
            )
        return output

    metadata, _ = load_detector(args.detector_dir)
    fixed = evaluate(float(metadata["decision_threshold"]))
    oracle_threshold = optimal_threshold(
        records,
        truth,
        pairs,
        scores,
        use_query_macro=False,
    )
    optimal = evaluate(oracle_threshold)
    summary = {
        "dataset": args.dataset,
        "files": len(records),
        "evaluation_manifest": str(manifest),
        "evaluation_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "reference_pairs": len(truth),
        "primary_metric": "pair_micro",
        "secondary_metrics": ["pair_micro", "query_macro"],
        "backend": args.backend,
        "detector": metadata["detector_id"],
        "feature_failures": 0,
        "unsupported_files": len(
            {item["index"] for item in result.get("unsupported", [])}
        ),
        "file_identity": "sha256-prefix-32" if args.dataset == "asap" else "md5",
        "bootstrap": {
            "unit": "work_group",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "optimal_threshold_role": "target-label oracle upper bound; not used for deployment",
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
