from __future__ import annotations

import importlib.util
import hashlib
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
    expected = {
        "shs": (733, 515),
        "asap": (1067, 4988),
        "atepp": (1527, 3339),
        "lmd-clean": (2547, 2037),
        "vienna4x22": (88, 924),
        "pianovam": (94, 77),
    }
    for dataset, counts in expected.items():
        records = benchmark.read_rows(
            ROOT / "reproduction" / "detector_benchmark" / f"{dataset}.csv.gz"
        )
        assert (len(records), len(benchmark.reference_pairs(records))) == counts


def test_dataset_specific_file_identity(tmp_path):
    benchmark = module()
    path = tmp_path / "example.mid"
    path.write_bytes(b"MThd")
    assert benchmark.file_identity(path, "asap") == hashlib.sha256(b"MThd").hexdigest()[:32]
    assert benchmark.file_identity(path, "shs") == hashlib.md5(b"MThd").hexdigest()
