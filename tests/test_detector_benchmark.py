from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / "scripts" / "reproduce_detector_benchmark.py"
    spec = importlib.util.spec_from_file_location("detector_benchmark", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_pair_metrics():
    benchmark = module()
    truth = {(0, 1), (0, 2), (3, 4)}
    predicted = {(0, 1), (0, 3), (3, 4)}
    result = benchmark.pair_micro(truth, predicted)
    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["precision"] == result["recall"] == result["f1"] == 2 / 3


def test_query_macro_metrics():
    benchmark = module()
    records = [{"work_group": "a"}] * 3
    truth = {(0, 1), (0, 2)}
    predicted = {(0, 1)}
    result = benchmark.query_macro(records, truth, predicted)
    assert result["tp"] == 2
    assert result["fn"] == 2
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5


def test_public_benchmark_registries():
    benchmark = module()
    expected = {"shs": (733, 515), "asap": (1067, 4988), "lmd-clean": (16788, 14083)}
    for dataset, counts in expected.items():
        records = benchmark.rows(
            ROOT / "reproduction" / "detector_benchmark" / f"{dataset}.csv.gz"
        )
        assert (len(records), len(benchmark.reference_pairs(records))) == counts


def test_same_recording_predictions_are_excluded():
    benchmark = module()
    records = [
        {"recording_group": "x"},
        {"recording_group": "x"},
        {"recording_group": "y"},
    ]
    assert benchmark.eligible_predictions(records, {(0, 1), (0, 2)}) == {(0, 2)}
