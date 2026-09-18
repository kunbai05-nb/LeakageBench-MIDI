#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from leakagebench_midi.detector import model_config, sha256


SEEDS = (20260911, 20260912, 20260913)


def choose_threshold(
    scores: np.ndarray, labels: np.ndarray, target: float = 0.97
) -> float:
    order = np.argsort(scores, kind="mergesort")[::-1]
    scores, labels = scores[order], labels[order]
    boundaries = np.flatnonzero(np.r_[scores[1:] != scores[:-1], True])
    feasible = []
    for boundary in boundaries:
        count = int(boundary + 1)
        precision = float(labels[:count].mean())
        if count >= 20 and precision >= target:
            feasible.append(float(scores[boundary]))
    if not feasible:
        raise RuntimeError("calibration cannot satisfy the requested precision")
    return feasible[-1]


def frozen_features(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as stored:
        return {
            "fit_features": stored["fit_features"].astype(np.float32),
            "fit_labels": stored["fit_labels"].astype(np.int8),
            "calibration_features": stored["calibration_features"].astype(np.float32),
            "calibration_labels": stored["calibration_labels"].astype(np.int8),
            "feature_names": stored["feature_names"].astype(str).tolist(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Same-Work Detector v1.8.")
    parser.add_argument(
        "source", type=Path, help="Frozen v1.8 training features (.npz)"
    )
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.source.suffix != ".npz":
        parser.error("source must be the released v1.8 .npz feature file")
    data = frozen_features(args.source)

    x, y = data["fit_features"], data["fit_labels"]
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    models = []
    for seed in SEEDS:
        selected_negative = np.random.default_rng(seed).choice(
            negative, max(1, int(0.8 * len(negative))), replace=False
        )
        selected = np.r_[positive, selected_negative]
        models.append(
            HistGradientBoostingClassifier(
                max_iter=260,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=80,
                l2_regularization=8.0,
                class_weight="balanced",
                early_stopping=False,
                random_state=seed,
            ).fit(x[selected], y[selected])
        )
    calibration_scores = np.mean(
        [model.predict_proba(data["calibration_features"])[:, 1] for model in models],
        axis=0,
    )
    threshold = choose_threshold(calibration_scores, data["calibration_labels"])

    args.output.mkdir(parents=True, exist_ok=True)
    model_paths = []
    for seed, model in zip(SEEDS, models):
        model_path = args.output / f"same-work-detector-v1.8-seed-{seed}.joblib"
        joblib.dump(model, model_path, compress=3)
        model_paths.append(model_path)
    metadata = model_config(
        [path.name for path in model_paths],
        [sha256(path) for path in model_paths],
        threshold,
        data["feature_names"],
    )
    (args.output / "MODEL_CONFIG.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"models": [str(path) for path in model_paths], "threshold": threshold},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
