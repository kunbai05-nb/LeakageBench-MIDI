#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summary(values: list[float]) -> dict:
    array = np.asarray(values)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "empirical_95_interval": np.quantile(array, [0.025, 0.975]).tolist(),
    }


def split_metrics(labels: np.ndarray, seeds: int, seed_base: int) -> dict:
    counts = np.bincount(labels)
    family_rates, file_rates = [], []
    train_size = int(len(labels) * 0.8)
    validation_size = int(len(labels) * 0.1)
    for offset in range(seeds):
        order = np.random.default_rng(seed_base + offset).permutation(len(labels))
        assignment = np.full(len(labels), 2, dtype=np.int8)
        assignment[order[:train_size]] = 0
        assignment[order[train_size:train_size + validation_size]] = 1
        train = np.bincount(labels[assignment == 0], minlength=len(counts)) > 0
        test_counts = np.bincount(labels[assignment == 2], minlength=len(counts))
        test = test_counts > 0
        contaminated = train & test
        family_rates.append(contaminated.sum() / max(1, test.sum()))
        file_rates.append(test_counts[contaminated].sum() / max(1, test_counts.sum()))
    return {
        "seeds": seeds,
        "seed_base": seed_base,
        "contaminated_test_family_rate": summary(family_rates),
        "contaminated_test_file_rate": summary(file_rates),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("feature_cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--seed-base", type=int, default=20260822000)
    args = parser.parse_args()

    valid = np.load(args.feature_cache / "dense_features.npz")["valid"].astype(bool)
    labels = np.load(args.result_dir / "component_labels.npz")["labels"][valid]
    _, labels = np.unique(labels, return_inverse=True)
    sizes = np.bincount(labels)
    multi = sizes > 1
    result = {
        "files": int(valid.size),
        "valid_files": int(valid.sum()),
        "valid_file_rate": float(valid.mean()),
        "components": int(len(sizes)),
        "multi_member_components": int(multi.sum()),
        "files_in_multi_member_components": int(sizes[multi].sum()),
        "largest_component": int(sizes.max(initial=0)),
        "random_file_split": split_metrics(labels, args.seeds, args.seed_base),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
