#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from leakagebench_midi.detector import detect


def midi_files(root: Path) -> list[Path]:
    suffixes = {".mid", ".midi"}
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in suffixes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("midi_dir", type=Path)
    parser.add_argument("detector_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--backend", choices=("exact", "faiss"), default="exact")
    args = parser.parse_args()

    paths = midi_files(args.midi_dir)
    if len(paths) < 2:
        raise SystemExit("fewer than two MIDI files found")
    result = detect(paths, args.detector_dir, args.workers, args.backend)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.output_dir / "components.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("file", "component"))
        for path, label in zip(paths, result["component_labels"]):
            writer.writerow((path.relative_to(args.midi_dir), int(label)))

    accepted = set(map(int, result["accepted"]))
    rejected = set(map(int, result["rejected_by_size"]))
    with (args.output_dir / "relation_edges.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("left", "right", "score", "decision"))
        for index in result["selected"]:
            left, right = result["pairs"][int(index)]
            decision = (
                "accepted"
                if int(index) in accepted
                else "size_guard" if int(index) in rejected else "transitive"
            )
            writer.writerow(
                (
                    paths[int(left)].relative_to(args.midi_dir),
                    paths[int(right)].relative_to(args.midi_dir),
                    f"{result['scores'][int(index)]:.9f}",
                    decision,
                )
            )

    summary = {
        "files": len(paths),
        "parsed_files": len(paths) - len(result["failures"]),
        "direct_edges": len(result["selected"]),
        "components": len(set(map(int, result["component_labels"]))),
        "rejected_by_size": len(result["rejected_by_size"]),
        "candidate_backend": result["candidate_diagnostics"]["backend"],
        "parse_failures": result["failures"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
