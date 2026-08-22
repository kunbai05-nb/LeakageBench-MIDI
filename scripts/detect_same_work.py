#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from leakagebench_midi.detector import detect


ROOT = Path(__file__).resolve().parents[1]


def midi_files(root: Path) -> list[Path]:
    return sorted(
        {
            *root.rglob("*.mid"),
            *root.rglob("*.midi"),
            *root.rglob("*.MID"),
            *root.rglob("*.MIDI"),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group structurally related MIDI files."
    )
    parser.add_argument("midi_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    paths = midi_files(args.midi_dir)
    if len(paths) < 2:
        raise SystemExit("fewer than two MIDI files found")
    result = detect(paths, ROOT / "artifacts" / "detector", workers=args.workers)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.output_dir / "components.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("file", "component"))
        writer.writeheader()
        for path, label in zip(paths, result["component_labels"]):
            writer.writerow(
                {"file": str(path.relative_to(args.midi_dir)), "component": int(label)}
            )

    accepted = set(map(int, result["accepted"]))
    rejected = set(map(int, result["rejected_by_size"]))
    with (args.output_dir / "candidate_pairs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = ("left", "right", "score", "mutual_signals", "decision")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, ((left, right), score, mutual) in enumerate(
            zip(result["pairs"], result["scores"], result["mutual_support"])
        ):
            decision = (
                "accepted"
                if index in accepted
                else "size_guard"
                if index in rejected
                else "none"
            )
            writer.writerow(
                {
                    "left": str(paths[int(left)].relative_to(args.midi_dir)),
                    "right": str(paths[int(right)].relative_to(args.midi_dir)),
                    "score": f"{float(score):.9f}",
                    "mutual_signals": int(mutual),
                    "decision": decision,
                }
            )

    summary = {
        "midi_files": len(paths),
        "parsed_files": len(paths) - len(result["failures"]),
        "components": len(set(map(int, result["component_labels"]))),
        "direct_edges": len(result["selected"]),
        "component_merges": len(result["merged"]),
        "rejected_by_size": len(result["rejected_by_size"]),
        "parse_failures": result["failures"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
