#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from leakagebench_midi.detector import detect


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--midi-root", type=Path, required=True)
    parser.add_argument("--detector-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--backend", choices=("exact", "faiss"), default="faiss")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    paths = [args.midi_root / row["relative_path"] for row in rows]
    missing = next((path for path in paths if not path.is_file()), None)
    if missing:
        raise FileNotFoundError(missing)
    result = detect(paths, args.detector_dir, args.workers, args.backend)
    args.output.mkdir(parents=True, exist_ok=True)

    with gzip.open(args.output / "components.csv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("identity", "component"))
        for row, label in zip(rows, result["component_labels"]):
            writer.writerow((row["identity"], int(label)))

    accepted = set(map(int, result["accepted"]))
    with gzip.open(args.output / "relation_edges.csv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("left_identity", "right_identity", "score", "accepted"))
        for index in result["selected"]:
            left, right = result["pairs"][int(index)]
            writer.writerow(
                (
                    rows[int(left)]["identity"],
                    rows[int(right)]["identity"],
                    f"{result['scores'][int(index)]:.9f}",
                    int(index) in accepted,
                )
            )

    summary = {
        "status": "PASS",
        "files": len(paths),
        "valid_files": len(paths) - len(result["failures"]),
        "candidate_pairs": len(result["pairs"]),
        "direct_edges": len(result["selected"]),
        "components": len(set(map(int, result["component_labels"]))),
        "largest_component": int(np.bincount(result["component_labels"]).max()),
        "rejected_by_size": len(result["rejected_by_size"]),
        "candidate_backend": result["candidate_diagnostics"]["backend"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
