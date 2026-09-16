from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from leakagebench_midi.data import PackedWindows
from leakagebench_midi.detector import (
    Components,
    build_components,
    candidate_gate_mask,
    load_detector,
    model_config,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]


def test_detector_checkpoint_loading(tmp_path):
    x = np.vstack((np.zeros((8, 57)), np.ones((8, 57))))
    y = np.asarray([0] * 8 + [1] * 8)
    model = HistGradientBoostingClassifier(min_samples_leaf=2, random_state=1).fit(x, y)
    model_path = tmp_path / "same-work-detector-v1.7.1.joblib"
    joblib.dump(model, model_path)
    metadata = model_config(
        model_path.name, sha256(model_path), 0.5, [f"f{i}" for i in range(57)]
    )
    (tmp_path / "MODEL_CONFIG.json").write_text(json.dumps(metadata), encoding="utf-8")
    loaded_metadata, loaded_model = load_detector(tmp_path)
    assert loaded_metadata["detector_id"] == "same-work-detector-v1.7.1"
    np.testing.assert_allclose(loaded_model.predict_proba(x), model.predict_proba(x))


def test_detector_rejects_modified_checkpoint(tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"not a model")
    metadata = model_config(
        model_path.name,
        hashlib.sha256(b"different").hexdigest(),
        0.5,
        [f"f{i}" for i in range(57)],
    )
    (tmp_path / "MODEL_CONFIG.json").write_text(json.dumps(metadata), encoding="utf-8")
    try:
        load_detector(tmp_path)
    except ValueError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("modified checkpoint was accepted")


def test_components():
    components = Components(4)
    assert components.union(0, 1, 3)
    assert components.union(1, 2, 3)
    assert not components.union(2, 3, 3)
    labels = components.labels()
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]


def test_component_guard_prioritizes_high_scores():
    pairs = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    scores = np.asarray([0.8, 0.9, 0.7])
    selected = np.arange(3, dtype=np.int64)
    accepted, rejected_by_size, rejected_by_bridge, labels = build_components(
        pairs,
        scores,
        selected,
        file_count=4,
        maximum_size=3,
        soft_size=3,
    )
    assert accepted.tolist() == [1, 0]
    assert rejected_by_size.tolist() == [2]
    assert rejected_by_bridge.tolist() == []
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]


def test_component_bridge_guard_requires_independent_edges():
    pairs = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    scores = np.asarray([0.9, 0.8, 0.7])
    selected = np.arange(3, dtype=np.int64)
    accepted, rejected_by_size, rejected_by_bridge, labels = build_components(
        pairs,
        scores,
        selected,
        file_count=4,
        maximum_size=10,
        soft_size=3,
        minimum_bridge_edges=2,
    )
    assert accepted.tolist() == [0, 1]
    assert rejected_by_size.tolist() == []
    assert rejected_by_bridge.tolist() == [2]
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]


def test_candidate_gate_requires_two_reciprocal_views():
    names = ["retrieval_mutual_support"]
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    assert candidate_gate_mask(features, names, 2).tolist() == [False, False, True, True]


def test_formal_source_specs():
    source = ROOT / "reproduction" / "source_specs"
    summary = json.loads((source / "formal_data.json").read_text())
    assert summary["rows"]["clean"] == 38374
    assert summary["rows"]["probe_windows"] == 9316
    with gzip.open(source / "formal_windows.csv.gz", "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        assert "token_ids" not in reader.fieldnames
        assert sum(1 for _ in reader) == 50218


def test_frozen_detector_training_features():
    path = ROOT / "reproduction" / "detector" / "training_features_v1_7.npz"
    with np.load(path, allow_pickle=False) as stored:
        assert stored["fit_features"].shape[1] == 57
        assert stored["calibration_features"].shape[1] == 57
        assert len(stored["feature_names"]) == 57


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
