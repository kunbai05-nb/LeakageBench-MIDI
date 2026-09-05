from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from leakagebench_midi.data import PackedWindows
from leakagebench_midi.detector import Components, _predict_probability, load_detector


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_detector_checkpoint_loading(tmp_path):
    x = np.vstack((np.zeros((4, 3)), np.ones((4, 3))))
    y = np.asarray([0] * 4 + [1] * 4)
    artifacts = []
    models = []
    for fold in range(5):
        model = LogisticRegression(random_state=fold).fit(x, y)
        path = tmp_path / f"fold_{fold}.joblib"
        joblib.dump(model, path)
        artifacts.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        models.append(model)
    (tmp_path / "MODEL_CONFIG.json").write_text(json.dumps({"ensemble_models": artifacts}))
    config, loaded = load_detector(tmp_path)
    observed = _predict_probability(
        loaded, np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    )
    expected = _predict_probability(
        models, np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    )
    assert config["ensemble_models"] == artifacts
    np.testing.assert_allclose(observed, expected)


def test_component_size_guard():
    components = Components(5)
    assert components.union(0, 1, 2)
    assert components.union(2, 3, 2)
    assert not components.union(0, 2, 2)
    labels = components.labels()
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_formal_source_specs():
    source = ROOT / "reproduction" / "source_specs"
    summary = json.loads((source / "formal_data.json").read_text())
    assert summary["rows"]["clean"] == 38374
    assert summary["rows"]["probe_windows"] == 9316
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
    assert next(train.formal_batches(38374, 202608040))[:5] == [
        4458,
        2215,
        9471,
        741,
        5656,
    ]
    slots = train.intervention_slots()
    invariant = [index for index in range(38374) if index not in slots]
    assert len(slots) == 1264
    assert next(train.neutral_batches(invariant, 202608040))[:5] == [
        7790,
        28673,
        1615,
        14817,
        6789,
    ]


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
