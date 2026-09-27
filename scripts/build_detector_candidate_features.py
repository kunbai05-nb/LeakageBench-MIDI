#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from leakagebench_midi.alignment import alignment_feature_matrix, extract_alignment_bundle
from leakagebench_midi.content import (
    compact_candidate_ranks,
    extract_feature_bundle,
    structural_pair_feature_matrix,
)
from leakagebench_midi.detector import (
    DetectorConfig,
    RETRIEVAL_VIEWS,
    assemble_features,
    candidate_union_mask,
    extract_pair_features,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"relative_path", "file_md5", "work_group", "recording_group"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"manifest must contain {sorted(required)}")
    return rows


def verify(rows: list[dict[str, str]], root: Path) -> list[Path]:
    paths = []
    for row in rows:
        path = root / row["relative_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        data = path.read_bytes()
        expected = row["file_md5"]
        if expected not in {
            hashlib.md5(data).hexdigest(),
            hashlib.sha256(data).hexdigest()[:32],
        }:
            raise ValueError(f"MIDI checksum mismatch: {row['relative_path']}")
        paths.append(path)
    return paths


def reference_pair_count(rows: list[dict[str, str]], recordings: np.ndarray) -> int:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["work_group"], []).append(index)
    return sum(
        1
        for members in groups.values()
        for position, left in enumerate(members)
        for right in members[position + 1 :]
        if recordings[left] != recordings[right]
    )


def build_ablation(
    rows: list[dict[str, str]], paths: list[Path], workers: int
) -> tuple[dict[str, np.ndarray], dict]:
    """Keep ranked Top-100 evidence required by view and Top-K ablations."""
    config = DetectorConfig()
    bundle, content_failures = extract_feature_bundle(paths, workers=workers)
    sequences, alignment_failures = extract_alignment_bundle(
        paths, config.alignment, workers
    )
    valid = np.flatnonzero(bundle["valid"])
    matrices = {name: bundle[name] for name in RETRIEVAL_VIEWS}
    compact = compact_candidate_ranks(
        matrices, valid, config.rank_k, workers=workers
    )
    all_pairs = compact["pairs"].astype(np.int32)
    all_ranks = compact["ranks"].astype(np.int16)
    candidate_mask = candidate_union_mask(all_ranks, config.top_k)
    pairs = all_pairs[candidate_mask].astype(np.int64)
    ranks = all_ranks[candidate_mask]
    structural, structural_names = structural_pair_feature_matrix(bundle, pairs)
    aligned, alignment_names = alignment_feature_matrix(
        sequences, pairs, config.alignment, workers
    )
    features, feature_names = assemble_features(
        structural,
        structural_names,
        ranks,
        tuple(compact["signals"]),
        aligned,
        alignment_names,
        config.rank_k,
    )
    works = np.asarray([row["work_group"] for row in rows], dtype=str)
    recordings = np.asarray([row["recording_group"] for row in rows], dtype=str)
    _, work_codes = np.unique(works, return_inverse=True)
    _, recording_codes = np.unique(recordings, return_inverse=True)
    labels = (works[pairs[:, 0]] == works[pairs[:, 1]]) & (
        recordings[pairs[:, 0]] != recordings[pairs[:, 1]]
    )
    all_labels = (works[all_pairs[:, 0]] == works[all_pairs[:, 1]]) & (
        recordings[all_pairs[:, 0]] != recordings[all_pairs[:, 1]]
    )
    failures = content_failures + alignment_failures
    unsupported = [
        item for item in failures if item["error"].startswith("TruncatedMidiError:")
    ]
    hard_failures = [item for item in failures if item not in unsupported]
    if hard_failures:
        raise RuntimeError(f"{len(hard_failures)} feature extractions failed")
    reference = reference_pair_count(rows, recordings)
    arrays = {
        "features": features.astype(np.float32),
        "labels": labels.astype(np.int8),
        "pairs": pairs.astype(np.int32),
        "feature_names": np.asarray(feature_names, dtype="U80"),
        "work_codes": work_codes.astype(np.int32),
        "recording_codes": recording_codes.astype(np.int32),
        "reference_pairs": np.asarray(reference, dtype=np.int64),
        "file_count": np.asarray(len(rows), dtype=np.int64),
        "reciprocal_support": features[:, [30, 32, 34]].sum(axis=1).astype(np.int8),
        "all_pairs": all_pairs,
        "all_ranks": all_ranks,
        "all_labels": all_labels.astype(np.int8),
        "rank_signals": np.asarray(compact["signals"], dtype="U40"),
    }
    summary = {
        "files": len(rows),
        "works": len(set(works)),
        "reference_pairs": reference,
        "ranked_pairs_top100": len(all_pairs),
        "candidate_pairs_top50": len(pairs),
        "candidate_positive_pairs": int(labels.sum()),
        "candidate_recall": float(labels.sum() / reference) if reference else None,
        "feature_failures": 0,
        "unsupported_files": len({item["index"] for item in unsupported}),
        "backend": "exact",
        "output_mode": "ablation",
    }
    return arrays, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deployment-aligned candidate features for detector training."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("midi_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--backend", choices=("exact", "faiss"), default="exact")
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="retain exhaustive ranked-pair evidence for sensitivity experiments",
    )
    args = parser.parse_args()

    rows = read_rows(args.manifest)
    paths = verify(rows, args.midi_root)
    if args.ablation:
        if args.backend != "exact":
            parser.error("--ablation requires --backend exact")
        arrays, summary = build_ablation(rows, paths, args.workers)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **arrays)
        args.output.with_suffix(".json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        return

    extracted = extract_pair_features(
        paths,
        workers=args.workers,
        backend=args.backend,
        config=DetectorConfig(),
    )
    pairs = extracted["pairs"]
    works = np.asarray([row["work_group"] for row in rows], dtype=str)
    recordings = np.asarray([row["recording_group"] for row in rows], dtype=str)
    labels = (
        (works[pairs[:, 0]] == works[pairs[:, 1]])
        & (recordings[pairs[:, 0]] != recordings[pairs[:, 1]])
    )
    reference = reference_pair_count(rows, recordings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=extracted["features"].astype(np.float32),
        labels=labels.astype(np.int8),
        pairs=pairs.astype(np.int32),
        feature_names=np.asarray(extracted["feature_names"], dtype="U80"),
    )
    summary = {
        "manifest": str(args.manifest),
        "files": len(rows),
        "works": len(set(works)),
        "reference_pairs": int(reference),
        "candidate_pairs": len(pairs),
        "candidate_positive_pairs": int(labels.sum()),
        "candidate_recall": float(labels.sum() / reference) if reference else None,
        "feature_failures": len(extracted["failures"]),
        "backend": args.backend,
        "candidate_rule": "one-way Top-50 union across five views",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
