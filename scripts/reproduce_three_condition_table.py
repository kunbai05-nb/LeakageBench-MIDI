#!/usr/bin/env python3
"""Evaluate released checkpoints and rebuild the multi-model three-condition table."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("clean", "unrelated_donor", "same_family_donor")
SEEDS = (202608040, 202608041, 202608042)
MODEL_ORDER = (
    "transformer_s",
    "transformer_m",
    "transformer_l",
    "midigpt",
    "lstm",
    "tcn",
    "conditional_vae",
    "latent_diffusion",
)
MODEL_LABELS = {
    "transformer_s": "Transformer-S",
    "transformer_m": "Transformer-M",
    "transformer_l": "Transformer-L",
    "midigpt": "MIDI-GPT",
    "lstm": "LSTM",
    "tcn": "TCN-384",
    "conditional_vae": "Conditional VAE",
    "latent_diffusion": "Latent Diffusion",
}
AUTOREGRESSIVE = {
    "transformer_s",
    "transformer_m",
    "transformer_l",
    "midigpt",
    "lstm",
    "tcn",
}


def classify_checkpoint(path: Path) -> str | None:
    lowered = "/".join(path.parts).lower().replace("-", "_")
    if "neutral_encoder" in lowered:
        return None
    for key in MODEL_ORDER:
        if key in lowered:
            return key
    if "tcn_384" in lowered:
        return "tcn"
    return None


def checkpoint_identity(path: Path) -> tuple[str, str, int] | None:
    model = classify_checkpoint(path)
    if model is None:
        return None
    normalized_parts = tuple(part.lower().replace("-", "_") for part in path.parts)
    condition = next(
        (
            value
            for value in CONDITIONS
            if any(part == value or part.endswith("_" + value) for part in normalized_parts)
        ),
        None,
    )
    match = re.search(r"seed[_-](\d+)", path.name)
    if condition is None or match is None:
        return None
    return model, condition, int(match.group(1))


def discover_checkpoints(root: Path) -> dict[tuple[str, str, int], Path]:
    found: dict[tuple[str, str, int], Path] = {}
    for path in sorted(root.rglob("*.pt")):
        identity = checkpoint_identity(path)
        if identity is None:
            continue
        if identity in found:
            raise ValueError(f"duplicate checkpoint identity: {identity}")
        found[identity] = path
    expected = {
        (model, condition, seed)
        for model in MODEL_ORDER
        for condition in CONDITIONS
        for seed in SEEDS
    }
    missing = sorted(expected - set(found))
    extra = sorted(set(found) - expected)
    if missing or extra:
        raise ValueError(
            f"checkpoint matrix is incomplete: missing={missing}, extra={extra}"
        )
    return found


def find_encoder(root: Path, seed: int) -> Path:
    candidates = [
        path
        for path in root.rglob("*.pt")
        if "neutral_encoder" in "/".join(path.parts).lower()
        and re.search(rf"seed[_-]{seed}(?:\D|$)", path.name)
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected one neutral encoder for seed {seed}, found {candidates}")
    return candidates[0]


def evaluate(
    checkpoints: dict[tuple[str, str, int], Path],
    checkpoint_root: Path,
    prepared_dir: Path,
    evaluation_dir: Path,
    device: str,
) -> None:
    def validate_summary(path: Path, condition: str, seed: int) -> None:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("condition") != condition or int(summary.get("seed", -1)) != seed:
            raise ValueError(f"checkpoint metadata mismatch: {path}")

    for model in MODEL_ORDER:
        for condition in CONDITIONS:
            for seed in SEEDS:
                output = evaluation_dir / model / condition / f"seed_{seed}"
                per_family = output / "per_family.csv"
                summary_path = output / "summary.json"
                if per_family.is_file() and summary_path.is_file():
                    validate_summary(summary_path, condition, seed)
                    print(f"[resume] {model}/{condition}/{seed}")
                    continue
                command = [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_checkpoint.py"),
                    str(checkpoints[(model, condition, seed)]),
                    str(prepared_dir),
                    str(output),
                    "--split",
                    "treated",
                    "--device",
                    device,
                ]
                if model == "latent_diffusion":
                    command.extend(["--encoder", str(find_encoder(checkpoint_root, seed))])
                print(f"[evaluate] {model}/{condition}/{seed}", flush=True)
                subprocess.run(command, check=True)
                validate_summary(summary_path, condition, seed)


def read_family_metrics(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty evaluation file: {path}")
    output: dict[str, float] = {}
    for row in rows:
        if row.get("split") not in (None, "", "treated"):
            continue
        family_id = row["family_id"]
        value = float(row.get("metric", row.get("nll", "nan")))
        if not math.isfinite(value):
            raise ValueError(f"non-finite metric in {path}: {family_id}")
        if family_id in output:
            raise ValueError(f"duplicate family in {path}: {family_id}")
        output[family_id] = value
    if not output:
        raise ValueError(f"no treated families in {path}")
    return output


def aggregate(evaluation_dir: Path) -> list[dict]:
    results = []
    for model in MODEL_ORDER:
        family_values: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
        seed_means: dict[str, list[float]] = defaultdict(list)
        reference_families: set[str] | None = None
        for condition in CONDITIONS:
            for seed in SEEDS:
                path = (
                    evaluation_dir
                    / model
                    / condition
                    / f"seed_{seed}"
                    / "per_family.csv"
                )
                values = read_family_metrics(path)
                families = set(values)
                if reference_families is None:
                    reference_families = families
                elif families != reference_families:
                    raise ValueError(f"family scope mismatch: {path}")
                family_values[condition][seed] = values
                seed_means[condition].append(statistics.fmean(values.values()))
        assert reference_families is not None
        condition_summary = {
            condition: {
                "mean": statistics.fmean(seed_means[condition]),
                "sd": statistics.stdev(seed_means[condition]),
            }
            for condition in CONDITIONS
        }
        relative_by_family = []
        for family_id in sorted(reference_families):
            unrelated = statistics.fmean(
                family_values["unrelated_donor"][seed][family_id] for seed in SEEDS
            )
            same = statistics.fmean(
                family_values["same_family_donor"][seed][family_id] for seed in SEEDS
            )
            if unrelated == 0:
                raise ValueError(f"zero unrelated-donor metric: {model}/{family_id}")
            relative_by_family.append(100.0 * (unrelated - same) / unrelated)
        same_mean = condition_summary["same_family_donor"]["mean"]
        results.append(
            {
                "model": model,
                "label": MODEL_LABELS[model],
                "families": len(reference_families),
                "conditions": condition_summary,
                "delta_nll": same_mean
                - condition_summary["unrelated_donor"]["mean"],
                "relative_reduction_percent": statistics.fmean(relative_by_family),
                "same_family_ppl": math.exp(same_mean)
                if model in AUTOREGRESSIVE
                else None,
            }
        )
    return results


def mean_sd(value: dict) -> str:
    return f"{value['mean']:.3f} ± {value['sd']:.3f}"


def write_tables(results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "Model",
        "Clean NLL",
        "Unrelated donor NLL",
        "Same-family donor NLL",
        "Delta NLL",
        "Relative reduction (%)",
        "Same-family PPL",
    ]
    rows = []
    for item in results:
        rows.append(
            {
                "Model": item["label"],
                "Clean NLL": mean_sd(item["conditions"]["clean"]),
                "Unrelated donor NLL": mean_sd(
                    item["conditions"]["unrelated_donor"]
                ),
                "Same-family donor NLL": mean_sd(
                    item["conditions"]["same_family_donor"]
                ),
                "Delta NLL": f"{item['delta_nll']:.3f}",
                "Relative reduction (%)": f"{item['relative_reduction_percent']:.2f}",
                "Same-family PPL": "—"
                if item["same_family_ppl"] is None
                else f"{item['same_family_ppl']:.2f}",
            }
        )
    with (output_dir / "three_condition_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    markdown = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    markdown.extend("| " + " | ".join(row[field] for field in fields) + " |" for row in rows)
    (output_dir / "three_condition_results.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    (output_dir / "three_condition_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the multi-model three-condition paper table."
    )
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("prepared_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="skip checkpoint evaluation and summarize existing per-family CSV files",
    )
    args = parser.parse_args()
    evaluation_dir = args.output_dir / "evaluation"
    if not args.summarize_only:
        checkpoints = discover_checkpoints(args.checkpoint_root)
        evaluate(
            checkpoints,
            args.checkpoint_root,
            args.prepared_dir,
            evaluation_dir,
            args.device,
        )
    results = aggregate(evaluation_dir)
    write_tables(results, args.output_dir / "tables")
    print(args.output_dir / "tables" / "three_condition_results.csv")


if __name__ == "__main__":
    main()
