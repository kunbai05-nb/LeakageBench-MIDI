#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

from leakagebench_midi.detector import detect


def midi_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.suffix.lower() in {".mid", ".midi"}
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a reference relation graph with the released detector."
    )
    parser.add_argument("midi_dir", type=Path)
    parser.add_argument("detector_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--backend", choices=("exact", "faiss"), default="faiss")
    args = parser.parse_args()

    paths = midi_files(args.midi_dir)
    if len(paths) < 2:
        raise SystemExit("fewer than two MIDI files found")
    result = detect(paths, args.detector_dir, args.workers, args.backend)
    identities = [path.relative_to(args.midi_dir).as_posix() for path in paths]
    labels = [int(value) for value in result["component_labels"]]
    family_map = {
        identity: f"component_{label:08d}"
        for identity, label in zip(identities, labels)
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output_dir / "family_edges.csv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("left", "right"))
        for index in result["accepted"]:
            left, right = result["pairs"][int(index)]
            writer.writerow((identities[int(left)], identities[int(right)]))
    with gzip.open(args.output_dir / "family_map.json.gz", "wt") as handle:
        json.dump(family_map, handle, sort_keys=True, separators=(",", ":"))

    sizes = Counter(labels)
    summary = {
        "files": len(paths),
        "parsed_files": len(paths) - len(result["failures"]),
        "components": len(sizes),
        "multi_member_components": sum(size > 1 for size in sizes.values()),
        "files_in_multi_member_components": sum(
            size for size in sizes.values() if size > 1
        ),
        "largest_component": max(sizes.values()),
        "accepted_edges": len(result["accepted"]),
        "rejected_by_size": len(result["rejected_by_size"]),
        "candidate_backend": result["candidate_diagnostics"]["backend"],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
