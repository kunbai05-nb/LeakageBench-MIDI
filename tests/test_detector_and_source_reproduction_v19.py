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
    candidate_union_mask,
    load_detector,
    model_config,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]


def test_detector_checkpoint_loading(tmp_path):
    x = np.vstack((np.zeros((8, 57)), np.ones((8, 57))))
    y = np.asarray([0] * 8 + [1] * 8)
    model = HistGradientBoostingClassifier(min_samples_leaf=2, random_state=1).fit(x, y)
    model_paths = [
        tmp_path / f"same-work-detector-v1.9-seed-{seed}.joblib"
        for seed in range(3)
    ]
    for model_path in model_paths:
        joblib.dump(model, model_path)
    metadata = model_config(
        [path.name for path in model_paths],
        [sha256(path) for path in model_paths],
        0.5,
        [f"f{i}" for i in range(57)],
        version="1.9",
    )
    (tmp_path / "MODEL_CONFIG.json").write_text(json.dumps(metadata), encoding="utf-8")
    loaded_metadata, loaded_models = load_detector(tmp_path)
    assert loaded_metadata["detector_id"] == "same-work-detector-v1.9"
    assert len(loaded_models) == 3
    for loaded_model in loaded_models:
        np.testing.assert_allclose(loaded_model.predict_proba(x), model.predict_proba(x))


def test_detector_rejects_modified_checkpoint(tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"not a model")
    metadata = model_config(
        [model_path.name] * 3,
        [hashlib.sha256(b"different").hexdigest()] * 3,
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


def test_candidate_pool_is_one_way_top_50_union():
    ranks = np.asarray(
        [
            [[51, 51], [50, 51]],
            [[51, 51], [51, 51]],
            [[12, 51], [51, 51]],
        ],
        dtype=np.int16,
    )
    assert candidate_union_mask(ranks, 50).tolist() == [True, False, True]


def test_formal_source_specs():
    source = ROOT / "reproduction" / "source_specs"
    if not source.is_dir():
        manifest = json.loads((ROOT / "reproduction" / "ASSETS.json").read_text())
        asset = manifest["assets"]["three-condition-source-specs"]
        assert asset["sha256"] == "3069ac8e5ad078b9848e527f3cb41837214ffa7032578eb09ee43dfebc4c4b8b"
        assert "reproduction/source_specs/formal_windows.csv.gz" in asset["required_paths"]
        return
    summary = json.loads((source / "formal_data.json").read_text())
    assert summary["rows"]["clean"] == 38374
    assert summary["rows"]["probe_windows"] == 9316
    with gzip.open(source / "formal_windows.csv.gz", "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        assert "token_ids" not in reader.fieldnames
        assert sum(1 for _ in reader) == 50218


def test_frozen_detector_training_manifests():
    root = ROOT / "reproduction" / "detector_training"
    summary = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    expected = {
        "fit_00_14",
        "fit_14_28",
        "fit_28_42",
        "fit_42_56",
        "fit_56_70",
        "calibration_70_85",
        "test_85_100",
    }
    assert expected.issubset({path.stem.replace(".csv", "") for path in root.glob("*.csv.gz")})
    assert summary["work_disjoint"] is True
    for path in root.glob("*.csv.gz"):
        with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
            fields = set(csv.DictReader(handle).fieldnames or [])
        assert {"relative_path", "file_md5", "work_group", "recording_group"}.issubset(fields)


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
