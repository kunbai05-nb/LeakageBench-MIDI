from __future__ import annotations

import csv
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np

from leakagebench_midi.data import PackedWindows
from leakagebench_midi.detector import _predict_probability, load_detector


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_portable_detector_is_stable():
    config, model = load_detector(ROOT / "artifacts" / "detector")
    features = np.asarray([[0.0] * 47, [0.5] * 47, [1.0] * 47], dtype=np.float32)
    expected = np.asarray(
        [
            0.014407912323173764,
            0.9480029526668894,
            0.9996333421184397,
        ]
    )
    assert len(config["selected_feature_names"]) == 47
    np.testing.assert_array_equal(_predict_probability(model, features), expected)


def test_formal_source_specs():
    source = ROOT / "reproduction" / "source_specs"
    summary = json.loads((source / "formal_data.json").read_text())
    assert summary["rows"] == {
        "clean": 38374,
        "same_family_replacements": 1264,
        "unrelated_replacements": 1264,
        "probe_windows": 9316,
        "probe_pieces": 700,
    }
    with gzip.open(source / "formal_windows.csv.gz", "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        assert "token_ids" not in reader.fieldnames
        assert sum(1 for _ in reader) == 50218


def test_packed_windows(tmp_path):
    prefix = tmp_path / "clean"
    np.asarray([1, 2, 3, 4, 5], dtype=np.uint16).tofile(prefix.with_suffix(".tokens"))
    np.savez_compressed(
        prefix.with_name("clean_index.npz"),
        offsets=np.asarray([0, 3, 5]),
        prompt=np.asarray([1, 2]),
    )
    rows = PackedWindows(prefix)
    assert len(rows) == 2
    assert rows[0]["token_ids"].tolist() == [1, 2, 3]
    assert rows[1]["prompt_token_count"] == 2


def test_public_schedules_are_deterministic():
    train = load_script("train_model")
    first = next(train.formal_batches(38374, 202608040))
    assert first[:5] == [4458, 2215, 9471, 741, 5656]
    slots = train.intervention_slots()
    assert len(slots) == 1264
    invariant = [index for index in range(38374) if index not in slots]
    neutral = next(train.neutral_batches(invariant, 202608040))
    assert neutral[:5] == [7790, 28673, 1615, 14817, 6789]


def test_imperfect_inference_metrics():
    simulation = load_script("simulate_imperfect_inference")
    families = [np.asarray([0, 1]), np.asarray([2, 3])]
    reference = {
        "files": 4,
        "family": np.asarray([0, 0, 1, 1], dtype=np.int32),
        "families": families,
        "sizes": np.asarray([2, 2], dtype=np.int32),
        "multi": np.asarray([0, 1], dtype=np.int32),
        "edges": [(0, 1), (2, 3)],
        "pairs": 2,
    }
    perfect = simulation.relation_metrics(reference, families)
    fragmented = simulation.relation_metrics(reference, [families[0]])
    assert perfect["pairwise_relation_f1"] == 1.0
    assert fragmented["pairwise_same_family_recall"] == 0.5
    assert fragmented["under_split_reference_family_count"] == 1
