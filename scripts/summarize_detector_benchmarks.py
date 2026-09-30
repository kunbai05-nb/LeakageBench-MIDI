#!/usr/bin/env python3
"""Combine six detector benchmark result files into paper-ready tables."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path


DATASETS = ("shs", "asap", "atepp", "lmd-clean", "vienna4x22", "pianovam")
EXPECTED_SCOPE = {
    "shs": (306, 80),
    "asap": (1067, 4988),
    "atepp": (1527, 3339),
    "lmd-clean": (2547, 2037),
    "vienna4x22": (88, 924),
    "pianovam": (94, 77),
}
MANIFEST_DIR = Path(__file__).resolve().parents[1] / "reproduction" / "detector_benchmark"
MANIFEST_SHA256 = {
    "shs": "a1390f92470e402cdfd9eb42d1204359bc9656f89333cdbff77bfbe951f086d4",
    "asap": "46d9eb87d7a17d62426c0980883cd9f223e08cb47244e3fc86ea4411692bee1e",
    "atepp": "8b701bab16bcb9c85cce015aa6cc51c2cadb725ef812e56803408224d0fb1f9c",
    "lmd-clean": "2ae9eec55a7b326e9c7279c62fa0088157216012a5243289ae16718161866920",
    "vienna4x22": "09f48bab4b2d8bc88c0e440e0aae770487bf098505dbc79aff3632dc9069da57",
    "pianovam": "3cf8a58daa44f45574e5716abaa62c213dbf4007fa1435a285da557893f03026",
}


def check_manifests() -> None:
    """Fail before detector inference if a registry differs from the paper cohort."""
    for dataset in DATASETS:
        path = MANIFEST_DIR / f"{dataset}.csv.gz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != MANIFEST_SHA256[dataset]:
            raise ValueError(f"frozen registry checksum mismatch: {path}")
        with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len({row["relative_path"] for row in rows}) != len(rows):
            raise ValueError(f"duplicate MIDI paths in {path}")
        groups: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            if not row["file_md5"] or not row["work_group"] or not row["recording_group"]:
                raise ValueError(f"incomplete registry row in {path}")
            groups[row["work_group"]].append(row["recording_group"])
        pairs = sum(
            left != right
            for recordings in groups.values()
            for position, left in enumerate(recordings)
            for right in recordings[position + 1 :]
        )
        scope = (len(rows), pairs)
        if scope != EXPECTED_SCOPE[dataset]:
            raise ValueError(f"registry scope mismatch in {path}: {scope} != {EXPECTED_SCOPE[dataset]}")
        print(f"{dataset}: {scope[0]} MIDI files, {scope[1]} reference pairs")


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}"


def ci_text(value: list[float]) -> str:
    return f"[{100.0 * value[0]:.2f}, {100.0 * value[1]:.2f}]"


def load_results(input_dir: Path) -> list[dict]:
    output = []
    for dataset in DATASETS:
        path = input_dir / dataset / "results.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing benchmark result: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("dataset") != dataset:
            raise ValueError(f"dataset identity mismatch: {path}")
        if value.get("feature_failures") != 0:
            raise ValueError(f"feature extraction failures recorded in {path}")
        if value.get("backend") != "exact":
            raise ValueError(f"non-reference retrieval backend recorded in {path}")
        scope = (value.get("files"), value.get("reference_pairs"))
        if scope != EXPECTED_SCOPE[dataset]:
            raise ValueError(
                f"evaluation scope mismatch in {path}: {scope} != {EXPECTED_SCOPE[dataset]}"
            )
        if value.get("evaluation_manifest_sha256") != MANIFEST_SHA256[dataset]:
            raise ValueError(f"evaluation registry checksum mismatch in {path}")
        output.append(value)
    return output


def flatten(results: list[dict]) -> list[dict[str, str | int]]:
    rows = []
    for item in results:
        row: dict[str, str | int] = {
            "Dataset": item["dataset"],
            "Files": item["files"],
            "Reference pairs": item["reference_pairs"],
            "Metric": item.get("primary_metric", item.get("metric", "pair_micro")),
        }
        for key, label in (("fixed", "Fixed"), ("optimal", "Optimal")):
            result = item[key]
            ci = result.get("pair_micro_ci95_work_bootstrap", {})
            row[f"{label} threshold"] = f"{result['threshold']:.6f}"
            for metric, title in (
                ("precision", "P"),
                ("recall", "R"),
                ("f1", "F1"),
            ):
                row[f"{label} {title} (%)"] = percent(result[metric])
                row[f"{label} {title} 95% CI"] = (
                    ci_text(ci[metric]) if metric in ci else ""
                )
        rows.append(row)
    return rows


def write_tables(results: list[dict], output_dir: Path) -> None:
    rows = flatten(results)
    fields = list(rows[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "six_dataset_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    markdown = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    markdown.extend("| " + " | ".join(str(row[field]) for field in fields) + " |" for row in rows)
    (output_dir / "six_dataset_results.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    (output_dir / "six_dataset_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the six-dataset result table.")
    parser.add_argument("input_dir", type=Path, nargs="?")
    parser.add_argument("output_dir", type=Path, nargs="?")
    parser.add_argument("--check-manifests", action="store_true", help="Verify frozen cohort sizes before inference.")
    args = parser.parse_args()
    if args.check_manifests:
        check_manifests()
        return
    if args.input_dir is None or args.output_dir is None:
        parser.error("input_dir and output_dir are required unless --check-manifests is used")
    results = load_results(args.input_dir)
    write_tables(results, args.output_dir)
    print(args.output_dir / "six_dataset_results.csv")


if __name__ == "__main__":
    main()
