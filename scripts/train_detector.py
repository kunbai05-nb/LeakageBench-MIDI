#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from leakagebench_midi.alignment import (
    AlignmentConfig,
    alignment_feature_matrix,
    extract_alignment_bundle,
)
from leakagebench_midi.content import (
    canonical_chroma,
    compact_candidate_ranks,
    compact_pair_feature_matrix,
    extract_feature_bundle,
)
from leakagebench_midi.detector import SIGNALS, _enrich, sha256


def fold_map(rows: list[dict], folds: int, seed: int) -> dict[int, int]:
    works = sorted({int(row["work_id"]) for row in rows})
    rng = np.random.default_rng(seed)
    tie = {work: float(rng.random()) for work in works}
    sizes = Counter(int(row["work_id"]) for row in rows)
    works.sort(key=lambda work: (-sizes[work], tie[work]))
    load = np.zeros(folds, dtype=int)
    output = {}
    for work in works:
        fold = int(np.argmin(load))
        output[work] = fold
        load[fold] += sizes[work]
    return output


def sample(
    features: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    names: list[str],
    seed: int,
) -> np.ndarray:
    positive = np.flatnonzero(mask & (labels == 1))
    negative = np.flatnonzero(mask & (labels == 0))
    limit = min(len(negative), max(25_000, 60 * max(1, len(positive))))
    if len(negative) <= limit:
        return np.r_[positive, negative]
    columns = [
        names.index(name)
        for name in (
            "motif_set_containment",
            "align_best_score_per_match",
            "align_coverage_hmean",
            "robust_alignment_agreement_min",
        )
    ]
    hardness = np.max(features[negative][:, columns], axis=1)
    hard_count = limit // 2
    hard = negative[np.argpartition(hardness, -hard_count)[-hard_count:]]
    remaining = np.setdiff1d(negative, hard, assume_unique=False)
    random = np.random.default_rng(seed).choice(
        remaining, limit - hard_count, replace=False
    )
    return np.r_[positive, hard, random]


def select_threshold(scores: np.ndarray, labels: np.ndarray, folds: np.ndarray) -> dict:
    order = np.argsort(scores, kind="mergesort")[::-1]
    scores, labels, folds = scores[order], labels[order], folds[order]
    true_positive = np.cumsum(labels)
    best = None
    boundaries = np.flatnonzero(np.r_[scores[1:] != scores[:-1], True])
    for boundary in boundaries:
        count = boundary + 1
        tp = int(true_positive[boundary])
        precision = tp / count
        fold_precision = []
        for fold in range(5):
            local = folds[:count] == fold
            if local.sum() >= 5:
                fold_precision.append(float(labels[:count][local].mean()))
        worst = min(fold_precision, default=0.0)
        if precision < 0.95 or worst < 0.80 or count < 50:
            continue
        key = (tp, worst, precision)
        if best is None or key > best[0]:
            best = (
                key,
                {
                    "threshold": float(scores[boundary]),
                    "precision": precision,
                    "true_positive_pairs": tp,
                    "false_positive_pairs": count - tp,
                    "predicted_pairs": count,
                    "worst_fold_precision": worst,
                },
            )
    if best is None:
        raise RuntimeError("no threshold met the precision constraints")
    return best[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--midi-root", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    template = json.loads(args.template.read_text())
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    rows = [row for row in rows if row["split"] != "test"]
    paths = [args.midi_root / row["relative_path"] for row in rows]
    bundle, failures = extract_feature_bundle(paths, workers=args.workers)
    if failures:
        raise RuntimeError(f"failed to parse {len(failures)} development MIDI files")
    valid = np.arange(len(rows), dtype=np.int64)
    matrices = {
        "chroma": canonical_chroma(bundle["chroma"]),
        **{name: bundle[name] for name in SIGNALS if name != "chroma"},
    }
    candidates = compact_candidate_ranks(matrices, valid, 240, workers=args.workers)
    pairs = candidates["pairs"]
    sequences, failures = extract_alignment_bundle(
        paths, AlignmentConfig(**template["alignment_config"]), args.workers
    )
    if failures:
        raise RuntimeError(f"failed to align {len(failures)} development MIDI files")
    zero = np.zeros((len(rows), 1), dtype=np.float32)
    global_values, global_names, pairs = compact_pair_feature_matrix(
        bundle, zero, zero, candidates, workers=args.workers
    )
    aligned, alignment_names = alignment_feature_matrix(
        sequences, pairs, AlignmentConfig(**template["alignment_config"]), args.workers
    )
    features, names = _enrich(
        np.column_stack((global_values, aligned)), global_names + alignment_names
    )
    columns = [names.index(name) for name in template["feature_names"]]
    features = features[:, columns]
    names = template["feature_names"]
    labels = np.asarray(
        [
            rows[int(left)]["work_id"] == rows[int(right)]["work_id"]
            and rows[int(left)]["track_id"] != rows[int(right)]["track_id"]
            for left, right in pairs
        ],
        dtype=np.int8,
    )
    mapping = fold_map(rows, 5, 20260827)
    left_fold = np.asarray([mapping[int(rows[int(left)]["work_id"])] for left in pairs])
    right_fold = np.asarray(
        [mapping[int(rows[int(right)]["work_id"])] for right in pairs]
    )

    args.output.mkdir(parents=True, exist_ok=True)
    scores, score_labels, score_folds, models = [], [], [], []
    for fold in range(5):
        fit = (left_fold != fold) & (right_fold != fold)
        validation = np.flatnonzero((left_fold == fold) & (right_fold == fold))
        selected = sample(features, labels, fit, names, 20261327 + fold)
        model = HistGradientBoostingClassifier(
            max_iter=260,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=80,
            l2_regularization=8.0,
            class_weight="balanced",
            random_state=20261327 + fold,
        )
        model.fit(features[selected], labels[selected])
        path = args.output / f"fold_{fold}.joblib"
        joblib.dump(model, path, compress=3)
        models.append({"artifact": path.name, "sha256": sha256(path)})
        scores.append(model.predict_proba(features[validation])[:, 1])
        score_labels.append(labels[validation])
        score_folds.append(np.full(len(validation), fold, dtype=np.int8))
    threshold = select_threshold(
        np.concatenate(scores),
        np.concatenate(score_labels),
        np.concatenate(score_folds),
    )
    config = {
        **template,
        "ensemble_models": models,
        "threshold": threshold["threshold"],
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "training_summary.json").write_text(
        json.dumps(
            {
                "files": len(rows),
                "pairs": len(pairs),
                "positive_pairs": int(labels.sum()),
                "features": len(names),
                "threshold_selection": threshold,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(threshold, indent=2))


if __name__ == "__main__":
    main()
