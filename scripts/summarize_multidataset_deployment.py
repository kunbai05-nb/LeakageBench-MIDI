#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ORDER = ("pop909", "asap", "pdmx", "gigamidi", "maestro", "lmd", "aria")
DISPLAY = {"pop909": "POP909", "asap": "ASAP", "pdmx": "PDMX", "gigamidi": "GigaMIDI", "maestro": "MAESTRO", "lmd": "LMD", "aria": "Aria-MIDI"}
PAPER_TABLE_SHA256 = "fc1d151653d44ad8969ce06b3513676d782ec84ce03a0f9a03f1b8d1d6622c1b"


def split_simulation(labels: np.ndarray, seeds: int, seed_base: int) -> dict:
    size = len(labels)
    counts = np.bincount(labels)
    family_rows = []
    train_size, validation_size = int(size * 0.8), int(size * 0.1)
    for offset in range(seeds):
        order = np.random.default_rng(seed_base + offset).permutation(size)
        assignment = np.full(size, 2, dtype=np.int8)
        assignment[order[:train_size]] = 0
        assignment[order[train_size : train_size + validation_size]] = 1
        train = np.bincount(labels[assignment == 0], minlength=len(counts)) > 0
        test = np.bincount(labels[assignment == 2], minlength=len(counts)) > 0
        family_rows.append(float(np.count_nonzero(train & test) / max(1, np.count_nonzero(test))))
    values = np.asarray(family_rows)
    return {"mean": float(values.mean()), "interval": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((args.evidence / "baseline_summary.json").read_text())
    baseline_by_name = {item["dataset"].lower().replace("-midi full", "").replace("-midi", ""): item for item in baseline["datasets"]}
    rows = []
    audit = {"status": "PASS", "datasets": {}}
    for name in ORDER:
        result = json.loads((args.evidence / name / "RESULTS.json").read_text())
        labels = np.load(args.evidence / name / "component_labels.npz")["labels"]
        if labels.ndim != 1 or len(labels) != int(result["files"]):
            raise RuntimeError(f"component-label length mismatch for {name}")
        if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
            raise RuntimeError(f"invalid component labels for {name}")
        frozen = result["random_file_split"]["contaminated_test_family_rate"]
        replay = split_simulation(labels, result["random_file_split"]["seeds"], result["random_file_split"]["seed_base"])
        expected_interval = frozen["empirical_95_interval"]
        if not np.allclose([replay["mean"], *replay["interval"]], [frozen["mean"], *expected_interval], atol=1e-12):
            raise RuntimeError(f"split replay mismatch for {name}")
        base_key = "aria" if name == "aria" else name
        old = baseline_by_name[base_key]
        official = result["official_file_split"].get("contaminated_test_family_rate")
        row = {
            "dataset": DISPLAY[name],
            "midi_files": result["files"],
            "valid_midi_percent": 100 * old["valid_file_rate"],
            "multi_version_files_percent": 100 * old["inferred_multi_member_file_rate"],
            "random_test_exposure_percent": 100 * replay["mean"],
            "random_test_exposure_low": 100 * replay["interval"][0],
            "random_test_exposure_high": 100 * replay["interval"][1],
            "official_exposure_percent": None if official is None else 100 * official,
            "wall_time_minutes": result["runtime_seconds"] / 60,
        }
        rows.append(row)
        audit["datasets"][name] = {"manifest_sha256": result["manifest_sha256"], "split_replay": "PASS"}
    with (args.output / "multidataset_deployment.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Dataset | MIDI files | Valid MIDI (%) | Multi-version files (%) | Random test exposure, % [95% EI] | Official exposure (%) | Wall time (min) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        official = "—" if row["official_exposure_percent"] is None else f'{row["official_exposure_percent"]:.1f}'
        lines.append(
            f'| {row["dataset"]} | {row["midi_files"]:,} | {row["valid_midi_percent"]:.1f} | '
            f'{row["multi_version_files_percent"]:.1f} | {row["random_test_exposure_percent"]:.1f} '
            f'[{row["random_test_exposure_low"]:.1f}, {row["random_test_exposure_high"]:.1f}] | '
            f'{official} | {row["wall_time_minutes"]:.1f} |'
        )
    markdown = "\n".join(lines) + "\n"
    (args.output / "multidataset_deployment.md").write_text(markdown)
    generated_digest = hashlib.sha256(markdown.encode()).hexdigest()
    audit["paper_table"] = {
        "status": "PASS" if generated_digest == PAPER_TABLE_SHA256 else "FAIL",
        "expected_sha256": PAPER_TABLE_SHA256,
        "generated_sha256": generated_digest,
    }
    audit["status"] = audit["paper_table"]["status"]
    (args.output / "multidataset_verification.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    if audit["status"] != "PASS":
        raise RuntimeError("generated deployment table does not match the frozen paper table")
    print(json.dumps({"status": "PASS", "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
