#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from leakagebench_midi.detector import extract_pair_features, model_config, sha256


SEED = 20260911


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


def labels(rows: list[dict], pairs: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            rows[int(left)]["work_group"] == rows[int(right)]["work_group"]
            and rows[int(left)]["recording_group"]
            != rows[int(right)]["recording_group"]
            for left, right in pairs
        ],
        dtype=np.int8,
    )


def raw_features(index: Path, midi_root: Path, workers: int, backend: str) -> dict:
    rows = list(csv.DictReader(index.open(encoding="utf-8", newline="")))
    required = {"file_md5", "work_group", "recording_group", "experiment_split"}
    if not rows or not required <= set(rows[0]):
        raise ValueError("unsupported detector index")
    output = {}
    names = None
    for split in ("fit", "calibration"):
        subset = [row for row in rows if row["experiment_split"] == split]
        paths = [
            midi_root / row["file_md5"][0] / f"{row['file_md5']}.mid" for row in subset
        ]
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(f"missing MIDI files in {split}")
        data = extract_pair_features(paths, workers, backend)
        if data["failures"]:
            raise RuntimeError(f"unparseable MIDI files in {split}")
        names = data["feature_names"] if names is None else names
        if names != data["feature_names"]:
            raise RuntimeError("feature schema changed between splits")
        output[f"{split}_features"] = data["features"]
        output[f"{split}_labels"] = labels(subset, data["pairs"])
    output["feature_names"] = names
    return output


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
    parser = argparse.ArgumentParser(description="Train Same-Work Detector v1.7.")
    parser.add_argument(
        "source", type=Path, help="Frozen .npz features or detector index CSV"
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--midi-root", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--backend", choices=("exact", "faiss"), default="exact")
    args = parser.parse_args()
    if args.source.suffix == ".npz":
        data = frozen_features(args.source)
    else:
        if args.midi_root is None:
            parser.error("--midi-root is required with a CSV index")
        data = raw_features(args.source, args.midi_root, args.workers, args.backend)

    x, y = data["fit_features"], data["fit_labels"]
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    selected_negative = np.random.default_rng(SEED).choice(
        negative, max(1, int(0.8 * len(negative))), replace=False
    )
    selected = np.r_[positive, selected_negative]
    model = HistGradientBoostingClassifier(
        max_iter=260,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=80,
        l2_regularization=8.0,
        class_weight="balanced",
        early_stopping=False,
        random_state=SEED,
    ).fit(x[selected], y[selected])
    calibration_scores = model.predict_proba(data["calibration_features"])[:, 1]
    threshold = choose_threshold(calibration_scores, data["calibration_labels"])

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "same-work-detector-v1.7.joblib"
    joblib.dump(model, model_path, compress=3)
    metadata = model_config(
        model_path.name, sha256(model_path), threshold, data["feature_names"]
    )
    (args.output / "MODEL_CONFIG.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"model": str(model_path), "threshold": threshold}, indent=2))


if __name__ == "__main__":
    main()
