#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
    extract_feature_bundle,
    faiss_mutual_candidate_ranks,
    structural_pair_feature_matrix,
)
from leakagebench_midi.local_evidence import (
    EVIDENCE_NAMES,
    SIGNALS,
    LocalEvidenceConfig,
    assemble_base_features,
    evidence_feature_matrix,
    select_candidates,
)
from leakagebench_midi.local_evidence_batch import sparse_local_candidates
from leakagebench_midi.detector import sha256


def read_index(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "file_md5",
        "work_group",
        "recording_group",
        "experiment_split",
    }
    if not rows or not required <= set(rows[0]):
        raise ValueError("detector index has an unsupported schema")
    if len({row["file_md5"] for row in rows}) != len(rows):
        raise ValueError("detector index contains duplicate files")
    return rows


def midi_paths(rows: list[dict], root: Path) -> list[Path]:
    paths = [root / row["file_md5"][0] / f"{row['file_md5']}.mid" for row in rows]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} indexed MIDI files are missing")
    return paths


def candidates(
    bundle: dict,
    sequences: list[dict | None],
    valid: np.ndarray,
    config: LocalEvidenceConfig,
    workers: int,
    backend: str,
) -> tuple[np.ndarray, np.ndarray]:
    matrices = {
        name: canonical_chroma(bundle[name]) if name == "chroma" else bundle[name]
        for name in SIGNALS
    }
    if backend == "faiss":
        compact, _ = faiss_mutual_candidate_ranks(
            matrices,
            valid,
            config.candidate_k,
            minimum_mutual_signals=config.minimum_mutual_views,
            threads=max(1, workers // len(SIGNALS)),
            signal_workers=min(max(1, workers), len(SIGNALS)),
        )
    elif backend == "exact":
        if len(valid) > 5000:
            raise ValueError("exact backend is limited to 5,000 files")
        compact = compact_candidate_ranks(
            matrices, valid, config.candidate_k, workers=workers
        )
        support = (np.max(compact["ranks"], axis=2) <= config.candidate_k).sum(axis=1)
        keep = support >= config.minimum_mutual_views
        compact = {**compact, "pairs": compact["pairs"][keep], "ranks": compact["ranks"][keep]}
    else:
        raise ValueError(f"unknown backend: {backend}")

    ranks = compact["ranks"][:, [compact["signals"].index(name) for name in SIGNALS]]
    local_pairs, local_scores, _ = sparse_local_candidates(
        sequences, config, workers=workers
    )
    chosen = select_candidates(
        compact["pairs"], ranks, len(sequences), local_pairs, local_scores, config
    )["proposed"]
    lookup = {
        int(left) * len(sequences) + int(right): rank
        for (left, right), rank in zip(compact["pairs"], ranks)
    }
    fallback = np.full((len(SIGNALS), 2), config.candidate_k + 1, dtype=np.int16)
    pair_ranks = np.asarray(
        [lookup.get(int(left) * len(sequences) + int(right), fallback) for left, right in chosen],
        dtype=np.int16,
    ).reshape(-1, len(SIGNALS), 2)
    return chosen, pair_ranks


def pair_features(
    paths: list[Path],
    config: LocalEvidenceConfig,
    alignment: AlignmentConfig,
    workers: int,
    backend: str,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    bundle, failures = extract_feature_bundle(paths, workers=workers)
    sequences, alignment_failures = extract_alignment_bundle(paths, alignment, workers)
    if failures or alignment_failures:
        raise RuntimeError("all indexed files must be parseable for detector training")
    valid = np.flatnonzero(bundle["valid"])
    pairs, ranks = candidates(bundle, sequences, valid, config, workers, backend)
    structural, structural_names = structural_pair_feature_matrix(bundle, pairs)
    aligned, alignment_names = alignment_feature_matrix(sequences, pairs, alignment, workers)
    base, base_names = assemble_base_features(
        structural, structural_names, ranks, aligned, config.candidate_k
    )
    extra = evidence_feature_matrix(sequences, pairs, config, workers)
    return np.column_stack((base, extra)), base_names + list(EVIDENCE_NAMES), pairs


def labels(rows: list[dict], pairs: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            rows[int(left)]["work_group"] == rows[int(right)]["work_group"]
            and rows[int(left)]["recording_group"] != rows[int(right)]["recording_group"]
            for left, right in pairs
        ],
        dtype=np.int8,
    )


def select_threshold(scores: np.ndarray, labels: np.ndarray, target: float) -> dict:
    order = np.argsort(scores, kind="mergesort")[::-1]
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    boundaries = np.flatnonzero(np.r_[ordered_scores[1:] != ordered_scores[:-1], True])
    valid = []
    for boundary in boundaries:
        count = int(boundary + 1)
        precision = float(ordered_labels[:count].mean())
        if count < 20 or precision < target:
            continue
        valid.append((count, float(ordered_scores[boundary]), precision))
    if not valid:
        raise RuntimeError("calibration set has no threshold at the requested precision")
    count, threshold, precision = valid[-1]
    return {
        "threshold": threshold,
        "target_precision": target,
        "predicted_pairs": count,
        "precision": precision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the released same-work detector.")
    parser.add_argument("--midi-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--backend", choices=("exact", "faiss"), default="exact")
    parser.add_argument("--precision-target", type=float, default=0.95)
    args = parser.parse_args()
    if args.workers < 1 or not 0 < args.precision_target <= 1:
        raise ValueError("invalid worker count or precision target")

    rows = read_index(args.index)
    config = LocalEvidenceConfig()
    alignment = AlignmentConfig()
    fit_rows = [row for row in rows if row["experiment_split"] == "fit"]
    calibration_rows = [row for row in rows if row["experiment_split"] == "calibration"]
    if len(fit_rows) < 4 or len(calibration_rows) < 4:
        raise ValueError("fit and calibration splits are required")
    fit_features, names, fit_pairs = pair_features(
        midi_paths(fit_rows, args.midi_root), config, alignment, args.workers, args.backend
    )
    calibration_features, calibration_names, calibration_pairs = pair_features(
        midi_paths(calibration_rows, args.midi_root), config, alignment, args.workers, args.backend
    )
    if names != calibration_names:
        raise RuntimeError("feature schemas differ between fit and calibration")
    fit_labels = labels(fit_rows, fit_pairs)
    calibration_labels = labels(calibration_rows, calibration_pairs)
    if not fit_labels.any() or not calibration_labels.any():
        raise RuntimeError("indexed candidates contain no positive family pairs")

    args.output.mkdir(parents=True, exist_ok=True)
    model_dir = args.output / "MODEL"
    model_dir.mkdir(exist_ok=True)
    models = []
    model_objects = []
    for member in range(5):
        positive = np.flatnonzero(fit_labels == 1)
        negative = np.flatnonzero(fit_labels == 0)
        limit = min(len(negative), max(1, int(0.8 * len(negative))))
        selected_negative = np.random.default_rng(20260905 + member).choice(
            negative, limit, replace=False
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
            random_state=20260905 + member,
        )
        model.fit(fit_features[selected], fit_labels[selected])
        path = model_dir / f"local87_{member}.joblib"
        joblib.dump(model, path, compress=3)
        models.append({"file": f"MODEL/{path.name}", "sha256": sha256(path), "seed": 20260905 + member})
        model_objects.append(model)

    calibration_scores = np.mean(
        [model.predict_proba(calibration_features)[:, 1] for model in model_objects],
        axis=0,
    )
    threshold = select_threshold(
        calibration_scores, calibration_labels, args.precision_target
    )
    metadata = {
        "format_version": "1.3",
        "detector_id": "same-work-detector-v1.3.0",
        "feature_names": names,
        "base_feature_names": names[:-len(EVIDENCE_NAMES)],
        "method": {
            "window_segments": list(config.window_segments),
            "windows_per_file": config.windows_per_file,
            "tokens_per_window": config.tokens_per_window,
            "max_token_postings": config.max_token_postings,
            "max_document_frequency": config.max_document_frequency,
            "neighbours_per_file": config.neighbours_per_file,
            "min_window_similarity": config.min_window_similarity,
            "pairs_per_file": config.pairs_per_file,
            "rescue_fraction": config.rescue_fraction,
            "candidate_k": config.candidate_k,
            "minimum_mutual_views": config.minimum_mutual_views,
            "evidence_max_segments": config.evidence_max_segments,
            "background_quantile": config.background_quantile,
            "context_weight": config.context_weight,
            "initial_shifts": config.initial_shifts,
            "maximum_shifts": config.maximum_shifts,
            "shift_boundary_margin": config.shift_boundary_margin,
        },
        "alignment_config": {
            "segment_beats": alignment.segment_beats,
            "onset_bins": alignment.onset_bins,
            "max_segments": alignment.max_segments,
            "max_paths": alignment.max_paths,
            "min_path_matches": alignment.min_path_matches,
            "transposition_candidates": alignment.transposition_candidates,
            "match_baseline": alignment.match_baseline,
            "gap_open": alignment.gap_open,
            "gap_extend": alignment.gap_extend,
        },
        "ensemble_models": models,
        "threshold": threshold["threshold"],
        "calibration_status": "CALIBRATED",
        "component_max_size": 50,
        "candidate_seed": 20260824,
        "training": {
            "fit_files": len(fit_rows),
            "fit_positive_pairs": int(fit_labels.sum()),
            "fit_negative_pairs": int((fit_labels == 0).sum()),
            "calibration_files": len(calibration_rows),
            "calibration_positive_pairs": int(calibration_labels.sum()),
            "precision_target": args.precision_target,
        },
    }
    (args.output / "MODEL_CONFIG.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"status": "COMPLETE", "models": 5, "features": len(names), **threshold}, indent=2))


if __name__ == "__main__":
    main()
