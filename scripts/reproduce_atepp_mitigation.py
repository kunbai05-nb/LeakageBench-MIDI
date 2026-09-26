#!/usr/bin/env python3
"""Regenerate the ATEPP v1.9 mitigation table from frozen checkpoint evaluations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CONDITIONS = (
    "uncorrected",
    "simple_dedup",
    "random_removal",
    "detector",
    "label_reference_matched",
)
REFERENCE = "label_reference_matched"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(root: Path) -> None:
    manifest = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        manifest[relative] = expected
    mismatches = []
    for relative, expected in manifest.items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError("ATEPP reproduction files failed verification: " + ", ".join(mismatches))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.input_root.resolve()
    output = args.output.resolve()
    verify_files(root)

    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    labels = json.loads((root / "evaluation_labels.json").read_text(encoding="utf-8"))
    preparation = json.loads((root / "PREPARATION_AUDIT.json").read_text(encoding="utf-8"))
    split = json.loads((root / "fixed_split.json").read_text(encoding="utf-8"))
    works = sorted({labels[name] for name in split["test"]})
    test_files = set(split["test"])
    scores: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    run_protocols: dict[str, set[str]] = defaultdict(set)
    run_identities: dict[int, list[tuple[str, int]]] = defaultdict(list)
    evaluation_counts = None

    for condition in CONDITIONS:
        values = []
        source_condition = "uncorrected" if condition == "simple_dedup" else condition
        for seed in protocol["seeds"]:
            run_root = root / "runs" / source_condition / f"seed_{seed}"
            report = json.loads((run_root / "run_report.json").read_text(encoding="utf-8"))
            if report["status"] != "COMPLETE" or report["steps"] != protocol["training"]["steps"]:
                raise RuntimeError(f"incomplete run: {source_condition}/{seed}")
            expected_stream = preparation["conditions"][condition]["stream_sha256"]
            if report["identity"]["stream_sha256"] != expected_stream:
                raise RuntimeError(f"stream mismatch: {condition}/{seed}")
            run_protocols[condition].add(report["identity"]["protocol_sha256"])
            run_identities[seed].append((report["initial_state_sha256"], report["trained_tokens"]))
            hashes[f"{condition}/{seed}"] = report["final_model_sha256"]

            evaluation = json.loads((run_root / "test_evaluation.json").read_text(encoding="utf-8"))
            if set(evaluation) != test_files:
                raise RuntimeError(f"test cohort mismatch: {condition}/{seed}")
            counts = {name: (row["tokens"], row["windows"]) for name, row in evaluation.items()}
            if evaluation_counts is None:
                evaluation_counts = counts
            elif counts != evaluation_counts:
                raise RuntimeError(f"scored-token mismatch: {condition}/{seed}")
            by_work: dict[str, list[float]] = defaultdict(list)
            for name, result in evaluation.items():
                by_work[labels[name]].append(result["nll"])
            if set(by_work) != set(works):
                raise RuntimeError(f"work cohort mismatch: {condition}/{seed}")
            values.append([float(np.mean(by_work[work])) for work in works])
        scores[condition] = np.asarray(values)

    if not all(len(value) == 1 for value in run_protocols.values()):
        raise RuntimeError("a condition used more than one training protocol")
    if not all(len(set(value)) == 1 for value in run_identities.values()):
        raise RuntimeError("paired conditions did not share initialization and token budget")

    config = protocol["analysis"]
    rng = np.random.default_rng(config["bootstrap_seed"])
    reference = scores[REFERENCE].mean(0)
    rows = []
    for condition in CONDITIONS:
        values = scores[condition]
        mean = values.mean(0)
        delta = mean - reference
        relative = 100 * delta / reference
        draws = np.empty(config["bootstrap_replicates"])
        for index in range(len(draws)):
            sample = rng.integers(0, len(works), len(works))
            draws[index] = relative[sample].mean()
        low, high = np.quantile(draws, [0.025, 0.975])
        info = preparation["conditions"][condition]
        rows.append(
            {
                "condition": condition,
                "work_macro_nll": float(mean.mean()),
                "seed_mean_sd": float(values.mean(1).std(ddof=1)),
                "nll_minus_reference": float(delta.mean()),
                "relative_difference_percent": float(relative.mean()),
                "relative_ci_low": float(low),
                "relative_ci_high": float(high),
                "test_work_exposure_percent": info["exposure_percent"],
                "actually_used_training_files": len(info["selected_files"]),
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "PASS",
        "reference": REFERENCE,
        "rows": rows,
        "test_works": len(works),
        "seeds": protocol["seeds"],
        "checkpoint_hashes": hashes,
        "uncertainty": "work-cluster bootstrap conditional on three trained seeds",
    }
    (output / "atepp_mitigation_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "atepp_mitigation_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with (output / "atepp_mitigation_results.md").open("w", encoding="utf-8") as handle:
        handle.write("| Condition | Work-macro NLL | Delta vs. label reference | Relative delta (%) | 95% CI (%) | Exposure (%) | Training files |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['condition']} | {row['work_macro_nll']:.3f} | "
                f"{row['nll_minus_reference']:+.3f} | {row['relative_difference_percent']:+.2f} | "
                f"[{row['relative_ci_low']:+.2f}, {row['relative_ci_high']:+.2f}] | "
                f"{row['test_work_exposure_percent']:.2f} | {row['actually_used_training_files']} |\n"
            )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
