#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from leakagebench_midi.detector import model_config, sha256


SEEDS = (20260911, 20260912, 20260913)


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as stored:
        output = {
            "features": stored["features"].astype(np.float32),
            "labels": stored["labels"].astype(np.int8),
            "feature_names": stored["feature_names"].astype(str).tolist(),
        }
        return output


def choose_threshold(scores: np.ndarray, labels: np.ndarray, target: float) -> dict:
    order = np.argsort(-scores, kind="stable")
    values = scores[order]
    truth = labels[order].astype(bool)
    tp = np.cumsum(truth, dtype=np.int64)
    boundaries = np.flatnonzero(np.r_[values[1:] != values[:-1], True])
    feasible = [
        int(i)
        for i in boundaries
        if i + 1 >= 20 and tp[i] / (i + 1) >= target
    ]
    if not feasible:
        raise RuntimeError("calibration cannot satisfy the precision target")
    positives = int(labels.sum())

    def objective(index: int) -> tuple[float, float, float]:
        accepted = index + 1
        precision = float(tp[index] / accepted)
        recall = float(tp[index] / positives) if positives else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return f1, recall, precision

    best = max(feasible, key=objective)
    accepted = best + 1
    precision = float(tp[best] / accepted)
    recall = float(tp[best] / positives) if positives else 0.0
    return {
        "threshold": float(values[best]),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "accepted": accepted,
        "true_positives": int(tp[best]),
        "candidate_positives": positives,
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Same-Work Detector v1.9.")
    parser.add_argument("--fit", type=Path, nargs="+", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision-target", type=float, default=0.97)
    parser.add_argument("--negative-ratio", type=int, default=50)
    args = parser.parse_args()

    parts = [load(path) for path in args.fit]
    calibration = load(args.calibration)
    names = parts[0]["feature_names"]
    if any(part["feature_names"] != names for part in parts[1:]) or calibration["feature_names"] != names:
        raise ValueError("feature schemas differ across candidate pools")
    x = np.concatenate([part["features"] for part in parts])
    y = np.concatenate([part["labels"] for part in parts])
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    models = []
    selected_counts = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        negative_count = min(len(negative), args.negative_ratio * len(positive))
        chosen_negative = rng.choice(negative, negative_count, replace=False)
        selected = np.r_[positive, chosen_negative]
        rng.shuffle(selected)
        models.append(classifier(seed).fit(x[selected], y[selected]))
        selected_counts.append(len(selected))

    raw_calibration_scores = np.mean(
        [model.predict_proba(calibration["features"])[:, 1] for model in models],
        axis=0,
    )
    raw_selection = choose_threshold(
        raw_calibration_scores, calibration["labels"], args.precision_target
    )
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(
        raw_calibration_scores[:, None], calibration["labels"]
    )
    calibrated_scores = calibrator.predict_proba(
        raw_calibration_scores[:, None]
    )[:, 1]
    calibrated_selection = choose_threshold(
        calibrated_scores, calibration["labels"], args.precision_target
    )
    args.output.mkdir(parents=True, exist_ok=True)
    model_paths = []
    for seed, fitted in zip(SEEDS, models):
        path = args.output / f"same-work-detector-v1.9-seed-{seed}.joblib"
        joblib.dump(fitted, path, compress=3)
        model_paths.append(path)
    metadata = model_config(
        [path.name for path in model_paths],
        [sha256(path) for path in model_paths],
        calibrated_selection["threshold"],
        names,
        version="1.9",
    )
    metadata["training_protocol"] = {
        "candidate_generation": "one-way Top-50 union across five views",
        "fit_pools": [str(path) for path in args.fit],
        "calibration_pool": str(args.calibration),
        "work_disjoint_splits": True,
        "precision_target": args.precision_target,
        "negative_to_positive_cap": args.negative_ratio,
        "class_weight": "balanced",
        "fit_candidate_rows": int(len(y)),
        "fit_positive_rows": int(len(positive)),
        "selected_rows_per_seed": selected_counts,
        "calibration": raw_selection,
        "score_semantics": (
            "Base HGB outputs are ranking scores under case-control negative sampling "
            "and class-balanced fitting; released outputs are Platt-calibrated on the "
            "natural-prevalence calibration pool."
        ),
        "calibrated_threshold": calibrated_selection,
    }
    metadata["score_calibration"] = {
        "method": "platt",
        "coefficient": float(calibrator.coef_[0, 0]),
        "intercept": float(calibrator.intercept_[0]),
        "fit_rows": int(len(calibration["labels"])),
        "fit_positives": int(calibration["labels"].sum()),
        "source_score": "mean ensemble ranking score",
    }
    (args.output / "MODEL_CONFIG.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata["training_protocol"], indent=2))


if __name__ == "__main__":
    main()
