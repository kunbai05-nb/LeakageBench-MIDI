#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from leakagebench_midi.detector import DetectorConfig, extract_pair_features


def read_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else path.open
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
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest != row["file_md5"]:
            raise ValueError(f"MIDI checksum mismatch: {row['relative_path']}")
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deployment-aligned candidate features for detector training."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("midi_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--backend", choices=("exact", "faiss"), default="exact")
    args = parser.parse_args()

    rows = read_rows(args.manifest)
    paths = verify(rows, args.midi_root)
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
    reference = sum(count * (count - 1) // 2 for count in Counter(works).values())
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
