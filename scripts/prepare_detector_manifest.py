#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np


def assignments(path: Path) -> dict[str, tuple[float, str]]:
    candidates = defaultdict(list)
    for track_id, matches in json.loads(path.read_text()).items():
        for identity, score in matches.items():
            candidates[identity].append((float(score), track_id))
    return {
        identity: max(values, key=lambda value: (value[0], value[1]))
        for identity, values in candidates.items()
    }


def works(path: Path) -> dict[str, tuple[int, int]]:
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "select track_id, shs_work, shs_perf from songs where shs_work != 0"
        ).fetchall()
    finally:
        connection.close()
    return {
        track_id: (int(work_id), int(performance_id))
        for track_id, work_id, performance_id in rows
    }


def work_splits(rows: list[dict], seed: int) -> dict[int, str]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["work_id"]].append(row)
    strata = {False: [], True: []}
    for work_id, members in grouped.items():
        strata[len({row["track_id"] for row in members}) >= 2].append(work_id)
    rng = np.random.default_rng(seed)
    output = {}
    for key in (False, True):
        ids = np.asarray(sorted(strata[key]), dtype=np.int64)
        rng.shuffle(ids)
        train = int(np.floor(0.6 * len(ids)))
        calibration = int(np.floor(0.2 * len(ids)))
        for index, work_id in enumerate(ids):
            output[int(work_id)] = (
                "train"
                if index < train
                else "calibration" if index < train + calibration else "test"
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-scores", type=Path, required=True)
    parser.add_argument("--track-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    best = assignments(args.match_scores)
    metadata = works(args.track_metadata)
    rows = []
    for identity, (score, track_id) in sorted(best.items()):
        if track_id not in metadata:
            continue
        work_id, performance_id = metadata[track_id]
        rows.append(
            {
                "identity": identity,
                "relative_path": f"{identity[0]}/{identity}.mid",
                "track_id": track_id,
                "work_id": work_id,
                "performance_id": performance_id,
                "match_score": score,
            }
        )
    split = work_splits(rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps({**row, "split": split[row["work_id"]]}, sort_keys=True)
                + "\n"
            )
    print(
        json.dumps(
            {"files": len(rows), "works": len(split), "output": str(args.output)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
