from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "reproduce_three_condition_table.py"
    spec = importlib.util.spec_from_file_location("reproduce_three_condition_table", path)
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
