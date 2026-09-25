#!/usr/bin/env python3
"""Recompute detector ablations reported in Tables 1, 2, 4, and 5."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


SEEDS = (20260911, 20260912, 20260913, 20260914, 20260915)
DATASETS = ("lmd-clean", "atepp", "shs", "asap")
VIEWS = ("bass", "harmony", "motif", "interval_hist", "duration_hist")
VALIDATION_VIEW_INDICES = np.asarray([1, 3, 4, 5, 6], dtype=np.int64)
FEATURE_GROUPS = {
    "full": np.arange(57),
    "without_structural": np.r_[29:57],
    "without_retrieval": np.r_[0:29, 35:55],
    "without_local_alignment": np.r_[0:35, 55:57],
}
FIT_NAMES = ("fit_00_14", "fit_14_28", "fit_28_42", "fit_42_56", "fit_56_70")


def load(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=260,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=80,
        l2_regularization=8.0,
        class_weight="balanced",
        early_stopping=False,
        random_state=seed,
    )


def ensemble_scores(models: list, features: np.ndarray, columns: np.ndarray) -> np.ndarray:
    return np.mean(
        np.vstack([model.predict_proba(features[:, columns])[:, 1] for model in models]),
        axis=0,
    )


def calibration_masks(manifest: Path, pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with gzip.open(manifest, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    side = np.asarray(
        [int(hashlib.sha256(row["work_group"].encode()).hexdigest(), 16) & 1 for row in rows],
        dtype=np.int8,
    )
    pair_side = side[pairs]
    return (
        (pair_side[:, 0] == 0) & (pair_side[:, 1] == 0),
        (pair_side[:, 0] == 1) & (pair_side[:, 1] == 1),
    )


def fit_calibrator(scores: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(
        scores[:, None], labels
    )


def calibrated(model: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    return model.predict_proba(scores[:, None])[:, 1]


def choose_threshold(scores: np.ndarray, labels: np.ndarray, target: float = 0.97) -> dict:
    order = np.argsort(-scores, kind="stable")
    values = scores[order]
    truth = labels[order].astype(bool)
    tp = np.cumsum(truth, dtype=np.int64)
    boundaries = np.flatnonzero(np.r_[values[1:] != values[:-1], True])
    positives = int(labels.sum())
    feasible = [int(i) for i in boundaries if i + 1 >= 20 and tp[i] / (i + 1) >= target]
    if not feasible:
        raise RuntimeError("no calibration threshold reaches the precision target")

    def objective(index: int) -> tuple[float, float, float]:
        precision = float(tp[index] / (index + 1))
        recall = float(tp[index] / positives)
        f1 = 2 * precision * recall / (precision + recall)
        return f1, recall, precision

    best = max(feasible, key=objective)
    precision = float(tp[best] / (best + 1))
    recall = float(tp[best] / positives)
    return {
        "threshold": float(values[best]),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
    }


def pair_metric(scores: np.ndarray, labels: np.ndarray, threshold: float, reference: int) -> dict:
    accepted = scores >= threshold
    tp = int(labels[accepted].sum())
    fp = int(accepted.sum()) - tp
    fn = int(reference) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / reference if reference else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "auprc_candidate_pool": float(average_precision_score(labels, scores)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def eligible(data: dict[str, np.ndarray], key: str = "pairs") -> np.ndarray:
    pairs = data[key]
    recordings = data["recording_codes"]
    return recordings[pairs[:, 0]] != recordings[pairs[:, 1]]


def macro(rows: dict[str, dict], keys: tuple[str, ...]) -> dict:
    return {key: float(np.mean([rows[name][key] for name in DATASETS])) for key in keys}


def retrieval_tables(test: dict[str, dict], validation_path: Path) -> tuple[dict, dict]:
    validation = load(validation_path)
    ranks = validation["ranks"][:, VALIDATION_VIEW_INDICES, :]
    truth = validation["truth"].astype(bool)
    chosen: dict[str, dict] = {}
    for count in range(1, 6):
        options = []
        for subset in itertools.combinations(range(5), count):
            mask = np.min(ranks[:, subset, :], axis=(1, 2)) <= 50
            options.append(
                {
                    "indices": list(subset),
                    "views": [VIEWS[index] for index in subset],
                    "retrieved": int(truth[mask].sum()),
                    "candidate_pairs": int(mask.sum()),
                }
            )
        chosen[str(count)] = max(
            options,
            key=lambda row: (row["retrieved"], -row["candidate_pairs"], tuple(row["views"])),
        )

    view_rows = {}
    for count in range(1, 6):
        per_dataset = {}
        indices = chosen[str(count)]["indices"]
        for name, data in test.items():
            mask = (np.min(data["all_ranks"][:, indices, :], axis=(1, 2)) <= 50) & eligible(data, "all_pairs")
            per_dataset[name] = {
                "candidate_recall": float(data["all_labels"][mask].sum() / int(data["reference_pairs"])),
                "candidates_per_file": float(2 * mask.sum() / int(data["file_count"])),
            }
        view_rows[str(count)] = {
            "views": chosen[str(count)]["views"],
            "datasets": per_dataset,
            "macro_candidate_recall": float(np.mean([row["candidate_recall"] for row in per_dataset.values()])),
            "macro_candidates_per_file": float(np.mean([row["candidates_per_file"] for row in per_dataset.values()])),
        }

    topk_rows = {}
    for k in (10, 25, 50, 75, 100):
        per_dataset = {}
        for name, data in test.items():
            mask = (np.min(data["all_ranks"], axis=(1, 2)) <= k) & eligible(data, "all_pairs")
            per_dataset[name] = {
                "candidate_recall": float(data["all_labels"][mask].sum() / int(data["reference_pairs"])),
                "candidates_per_file": float(2 * mask.sum() / int(data["file_count"])),
            }
        topk_rows[str(k)] = {
            "datasets": per_dataset,
            "macro_candidate_recall": float(np.mean([row["candidate_recall"] for row in per_dataset.values()])),
            "macro_candidates_per_file": float(np.mean([row["candidates_per_file"] for row in per_dataset.values()])),
        }
    return {"selection_set": "MAESTRO-validation", "rows": view_rows}, {"rows": topk_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path, help="extracted detector-ablation-v1.9 feature bundle")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    train_dir = args.input_root / "train"
    test_dir = args.input_root / "test"
    fit_parts = [load(train_dir / f"{name}.npz") for name in FIT_NAMES]
    calibration = load(train_dir / "calibration_70_85.npz")
    test = {name: load(test_dir / f"{name}.npz") for name in DATASETS}
    names = fit_parts[0]["feature_names"].astype(str).tolist()
    if any(part["feature_names"].astype(str).tolist() != names for part in fit_parts[1:]):
        raise RuntimeError("fit feature schemas differ")
    if calibration["feature_names"].astype(str).tolist() != names:
        raise RuntimeError("calibration feature schema differs")
    if any(data["feature_names"].astype(str).tolist() != names for data in test.values()):
        raise RuntimeError("test feature schemas differ")

    x = np.concatenate([part["features"].astype(np.float32) for part in fit_parts])
    y = np.concatenate([part["labels"].astype(np.int8) for part in fit_parts])
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    selected_by_seed = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(negative, min(len(negative), 50 * len(positive)), replace=False)
        selected = np.r_[positive, chosen]
        rng.shuffle(selected)
        selected_by_seed[seed] = selected

    probability_mask, threshold_mask = calibration_masks(
        args.input_root / "calibration_70_85.csv.gz", calibration["pairs"]
    )
    if not probability_mask.any() or not threshold_mask.any():
        raise RuntimeError("invalid family-disjoint calibration partition")

    models_by_variant, calibrators, thresholds = {}, {}, {}
    pair_datasets = {name: {} for name in DATASETS}
    for variant, columns in FEATURE_GROUPS.items():
        models = [
            classifier(seed).fit(x[selected_by_seed[seed]][:, columns], y[selected_by_seed[seed]])
            for seed in SEEDS[:3]
        ]
        models_by_variant[variant] = models
        raw = ensemble_scores(models, calibration["features"], columns)
        calibrator = fit_calibrator(raw, calibration["labels"])
        calibrators[variant] = calibrator
        scores = calibrated(calibrator, raw)
        thresholds[variant] = choose_threshold(scores, calibration["labels"])
        for name, data in test.items():
            values = calibrated(calibrator, ensemble_scores(models, data["features"], columns))
            mask = eligible(data)
            pair_datasets[name][variant] = pair_metric(
                values[mask], data["labels"][mask], thresholds[variant]["threshold"], int(data["reference_pairs"])
            )

    for name, data in test.items():
        mask = eligible(data)
        reciprocal = data["reciprocal_support"].astype(float) / 3.0
        pair_datasets[name]["rule_based_verifier"] = pair_metric(
            reciprocal[mask], data["labels"][mask], 2 / 3, int(data["reference_pairs"])
        )
    pair_macro = {
        variant: macro(
            {name: pair_datasets[name][variant] for name in DATASETS},
            ("precision", "recall", "f1", "auprc_candidate_pool"),
        )
        for variant in (*FEATURE_GROUPS, "rule_based_verifier")
    }
    table2 = {"thresholds": thresholds, "datasets": pair_datasets, "macro": pair_macro}

    full_models = models_by_variant["full"]
    all_models = full_models + [
        classifier(seed).fit(x[selected_by_seed[seed]], y[selected_by_seed[seed]])
        for seed in SEEDS[3:]
    ]
    classifier_rows = {}
    for count in range(1, 6):
        combinations = []
        for indices in itertools.combinations(range(5), count):
            models = [all_models[index] for index in indices]
            raw = ensemble_scores(models, calibration["features"], FEATURE_GROUPS["full"])
            score_calibrator = fit_calibrator(raw, calibration["labels"])
            calibration_scores = calibrated(score_calibrator, raw)
            selected_threshold = choose_threshold(calibration_scores, calibration["labels"])
            per_dataset = {}
            for name, data in test.items():
                values = calibrated(
                    score_calibrator,
                    ensemble_scores(models, data["features"], FEATURE_GROUPS["full"]),
                )
                mask = eligible(data)
                per_dataset[name] = pair_metric(
                    values[mask], data["labels"][mask], selected_threshold["threshold"], int(data["reference_pairs"])
                )
            combinations.append(
                {
                    "model_indices": list(indices),
                    "threshold": selected_threshold,
                    "datasets": per_dataset,
                    "macro": macro(per_dataset, ("precision", "recall", "f1", "auprc_candidate_pool")),
                }
            )
        summary = {}
        for key in ("precision", "recall", "f1", "auprc_candidate_pool"):
            values = np.asarray([row["macro"][key] for row in combinations])
            summary[key] = {
                "mean": float(values.mean()),
                "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        threshold_values = np.asarray([row["threshold"]["threshold"] for row in combinations])
        summary["threshold"] = {
            "mean": float(threshold_values.mean()),
            "standard_deviation": float(threshold_values.std(ddof=0)),
        }
        classifier_rows[str(count)] = {
            "combination_count": len(combinations),
            "summary": summary,
            "combinations": combinations,
        }

    table1, table4 = retrieval_tables(test, args.input_root / "MAESTRO-validation.npz")
    outputs = {
        "TABLE1_VIEW_COUNT.json": table1,
        "TABLE2_PAIR_ABLATION.json": table2,
        "TABLE4_TOPK_SENSITIVITY.json": table4,
        "TABLE5_CLASSIFIER_COUNT.json": {"rows": classifier_rows},
        "INTEGRITY_AUDIT.json": {
            "status": "PASS",
            "detector": "Same-Work Detector v1.9",
            "tables": [1, 2, 4, 5],
            "test_tuning": False,
            "feature_variants_retrained": list(FEATURE_GROUPS),
            "classifier_count_uses_all_seed_combinations": True,
            "family_merging_table_included": False,
        },
    }
    for filename, value in outputs.items():
        (args.output / filename).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(outputs["INTEGRITY_AUDIT.json"], indent=2))


if __name__ == "__main__":
    main()
