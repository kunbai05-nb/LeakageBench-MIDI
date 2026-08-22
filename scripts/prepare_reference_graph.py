#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

from leakagebench_midi import build_family_map


COMMIT = "c42fa1c3f881261b92c0cf0d58dba5b0e5955d26"
RELATION_URL = (
    "https://raw.githubusercontent.com/jech2/LMD_Deduplication/"
    f"{COMMIT}/lmd_filtering_list/CAugBERT_0.99_with_CLaMP_0.99.json"
)


def identities(source: Path) -> list[str]:
    if source.is_file():
        with tarfile.open(source, "r:gz") as archive:
            values = {
                Path(member.name).stem.lower()
                for member in archive
                if member.isfile() and member.name.lower().endswith(".mid")
            }
    else:
        values = {path.stem.lower() for path in source.rglob("*.mid")}
    return sorted(values)


def load_relations(path: Path | None) -> dict:
    if path is not None:
        return json.loads(path.read_text())
    with urllib.request.urlopen(RELATION_URL) as response:
        return json.load(response)


def content_id(value: str) -> str:
    return value.split("__", 1)[-1].lower()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the frozen LMD reference graph."
    )
    parser.add_argument("lmd_source", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--relations", type=Path)
    args = parser.parse_args()

    files = identities(args.lmd_source)
    universe = set(files)
    relations = load_relations(args.relations)
    edges = []
    seen = set()
    for anchor, members in sorted(relations.items()):
        for member in members:
            edge = tuple(sorted((content_id(anchor), content_id(member))))
            if edge[0] in universe and edge[1] in universe and edge not in seen:
                seen.add(edge)
                edges.append(edge)
    family_map = build_family_map(files, edges)
    sizes = Counter(family_map.values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output_dir / "family_edges.csv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("left", "right"))
        writer.writerows(edges)
    with gzip.open(args.output_dir / "family_map.json.gz", "wt") as handle:
        json.dump(family_map, handle, sort_keys=True, separators=(",", ":"))
    summary = {
        "files": len(files),
        "families": len(sizes),
        "multi_member_families": sum(size > 1 for size in sizes.values()),
        "files_in_multi_member_families": sum(
            size for size in sizes.values() if size > 1
        ),
        "largest_family": max(sizes.values()),
        "relation_edges": len(edges),
        "relation_commit": COMMIT,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
