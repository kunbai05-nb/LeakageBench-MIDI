from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "reproduce_three_condition_table.py"
    spec = importlib.util.spec_from_file_location("reproduce_three_condition_table", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_summary_module():
    path = ROOT / "scripts" / "summarize_detector_benchmarks.py"
    spec = importlib.util.spec_from_file_location("summarize_detector_benchmarks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_evaluation(root, model, condition, seed, values):
    path = root / model / condition / f"seed_{seed}" / "per_family.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("family_id", "metric", "split"))
        writer.writeheader()
        for family_id, value in values.items():
            writer.writerow(
                {"family_id": family_id, "metric": value, "split": "treated"}
            )


def test_three_condition_aggregation_and_output(tmp_path):
    module = load_module()
    for model in module.MODEL_ORDER:
        for seed_index, seed in enumerate(module.SEEDS):
            write_evaluation(
                tmp_path,
                model,
                "clean",
                seed,
                {"a": 1.0 + 0.1 * seed_index, "b": 2.0 + 0.1 * seed_index},
            )
            write_evaluation(
                tmp_path,
                model,
                "unrelated_donor",
                seed,
                {"a": 2.0, "b": 4.0},
            )
            write_evaluation(
                tmp_path,
                model,
                "same_family_donor",
                seed,
                {"a": 1.0, "b": 2.0},
            )
    results = module.aggregate(tmp_path)
    assert len(results) == 8
    assert results[0]["delta_nll"] == -1.5
    assert results[0]["relative_reduction_percent"] == 50.0
    assert math.isclose(results[0]["same_family_ppl"], math.exp(1.5))
    assert results[-1]["same_family_ppl"] is None
    module.write_tables(results, tmp_path / "tables")
    assert (tmp_path / "tables" / "three_condition_results.csv").is_file()
    assert (tmp_path / "tables" / "three_condition_results.md").is_file()
    assert (tmp_path / "tables" / "three_condition_results.json").is_file()


def test_six_dataset_table_output(tmp_path):
    module = load_summary_module()
    result = {
        "primary_metric": "pair_micro",
        "backend": "exact",
        "feature_failures": 0,
        "fixed": {
            "threshold": 0.9,
            "precision": 0.8,
            "recall": 0.6,
            "f1": 0.685714,
            "pair_micro_ci95_work_bootstrap": {
                "precision": [0.7, 0.9],
                "recall": [0.5, 0.7],
                "f1": [0.6, 0.75],
            },
        },
        "optimal": {
            "threshold": 0.8,
            "precision": 0.75,
            "recall": 0.75,
            "f1": 0.75,
            "pair_micro_ci95_work_bootstrap": {
                "precision": [0.65, 0.85],
                "recall": [0.65, 0.85],
                "f1": [0.65, 0.85],
            },
        },
    }
    for dataset in module.DATASETS:
        folder = tmp_path / "raw" / dataset
        folder.mkdir(parents=True)
        files, reference_pairs = module.EXPECTED_SCOPE[dataset]
        (folder / "results.json").write_text(
            json.dumps(
                {
                    **result,
                    "dataset": dataset,
                    "files": files,
                    "reference_pairs": reference_pairs,
                }
            ),
            encoding="utf-8",
        )
    results = module.load_results(tmp_path / "raw")
    module.write_tables(results, tmp_path / "tables")
    rows = module.flatten(results)
    assert len(rows) == 6
    assert rows[0]["Fixed P (%)"] == "80.00"
    assert rows[0]["Optimal F1 95% CI"] == "[65.00, 85.00]"
    assert (tmp_path / "tables" / "six_dataset_results.csv").is_file()
