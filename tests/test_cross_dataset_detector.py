from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import mido
import numpy as np

from leakagebench_midi.detector import apply_tfidf, extract_count_features, extract_features


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def midi(path: Path, pitches: list[int]) -> None:
    value = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    value.tracks.append(track)
    for pitch in pitches:
        track.append(mido.Message("note_on", note=pitch, velocity=64, time=0))
        track.append(mido.Message("note_off", note=pitch, velocity=0, time=480))
    value.save(path)


def test_count_shards_match_direct_tfidf(tmp_path):
    paths = [tmp_path / "a.mid", tmp_path / "b.mid"]
    midi(paths[0], [60, 62, 64, 65])
    midi(paths[1], [67, 69, 71, 72])
    direct, direct_failures = extract_features(paths)
    counts, count_failures = extract_count_features(paths)
    sharded = apply_tfidf(counts)
    assert direct_failures == count_failures == []
    np.testing.assert_array_equal(direct["valid"], sharded["valid"])
    for name in ("chroma", "interval_hist", "duration_hist", "ioi_hist", "scalars"):
        np.testing.assert_allclose(direct[name], sharded[name])
    for name in ("melody", "bass", "rhythm", "harmony", "motif"):
        np.testing.assert_allclose(direct[name].toarray(), sharded[name].toarray())


def test_reference_metrics_and_component_guard():
    runner = load_script("run_cross_dataset_detector")
    reference = np.asarray([0, 0, 1, 2, 2], dtype=np.int32)
    left = np.asarray([0, 3], dtype=np.int32)
    right = np.asarray([1, 4], dtype=np.int32)
    direct = runner.pair_metrics(left, right, reference, True)
    assert direct["precision"] == 1.0
    assert direct["recall"] == 1.0

    components = runner.Components(5)
    assert components.union(0, 1, 2)
    assert components.union(3, 4, 2)
    assert not components.union(0, 3, 2)
    grouped = runner.component_metrics(components.labels(), reference, True)
    assert grouped["precision"] == 1.0
    assert grouped["recall"] == 1.0


def test_official_split_metrics():
    runner = load_script("run_cross_dataset_detector")
    rows = [{"split": "train"}, {"split": "test"}, {"split": "test"}]
    labels = np.asarray([0, 0, 1], dtype=np.int32)
    result = runner.official_split_metrics(rows, labels)
    assert result["known_train_test_family_count"] == 1
    assert result["contaminated_test_family_rate"] == 0.5
    assert result["contaminated_test_file_rate"] == 0.5


def test_gigamidi_manifest_rows(tmp_path):
    builder = load_script("build_cross_dataset_manifest")
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "file_path,md5,audio_text_matches_score,audio_text_matches_sid\n"
        "./Final/train/a.mid,aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,0.9,"
        "0123456789ABCDEFGHIJKL\n"
        "./Final/test/b.mid,bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,,\n"
    )
    rows = builder.gigamidi(metadata)
    assert rows[0]["reference_work_id"] == "spotify:0123456789ABCDEFGHIJKL"
    assert rows[0]["split"] == "train"
    assert rows[1]["reference_work_id"].startswith("file:")
    assert rows[1]["split"] == "test"


def test_asap_maestro_and_aria_manifest_rows(tmp_path):
    builder = load_script("build_cross_dataset_manifest")

    asap_root = tmp_path / "asap"
    asap_root.mkdir()
    midi(asap_root / "performance.mid", [60, 64])
    asap_metadata = tmp_path / "asap.csv"
    asap_metadata.write_text(
        "composer,title,midi_performance,maestro_midi_performance\n"
        "Bach,Prelude,performance.mid,{maestro}/2018/example.midi\n"
    )
    asap_rows = builder.asap(asap_root, asap_metadata)
    assert len(asap_rows) == 1

    maestro_metadata = tmp_path / "maestro.csv"
    maestro_metadata.write_text(
        "midi_filename,split\n2018/example.midi,test\n2018/unlinked.midi,train\n"
    )
    maestro_rows = builder.maestro(maestro_metadata, asap_metadata)
    assert maestro_rows[0]["reference_work_id"] == asap_rows[0]["reference_work_id"]
    assert maestro_rows[0]["split"] == "test"
    assert maestro_rows[1]["reference_work_id"].startswith("file:")

    aria_root = tmp_path / "aria"
    aria_root.mkdir()
    midi(aria_root / "000001_0.mid", [60, 64])
    midi(aria_root / "000002_0.mid", [62, 65])
    aria_metadata = tmp_path / "aria.json"
    aria_metadata.write_text(
        json.dumps(
            {
                "1": {"metadata": {"composer": "Bach", "opus": "1", "piece_number": 1}},
                "2": {"metadata": {"composer": "Bach"}},
            }
        )
    )
    aria_rows = builder.aria(aria_root, aria_metadata)
    assert len(aria_rows) == 2
    assert aria_rows[0]["reference_work_id"].startswith("aria-work:")
    assert aria_rows[1]["reference_work_id"].startswith("file:")


def test_valid_subset_split_summary_is_deterministic():
    summarizer = load_script("summarize_valid_detector_subset")
    labels = np.asarray([0, 0, 1, 2, 2, 2], dtype=np.int32)
    first = summarizer.split_metrics(labels, 20, 20260822000)
    second = summarizer.split_metrics(labels, 20, 20260822000)
    assert first == second
    assert 0 <= first["contaminated_test_file_rate"]["mean"] <= 1


def test_compact_detector_result(tmp_path, monkeypatch):
    compactor = load_script("compact_detector_result")
    path = tmp_path / "RESULTS.json"
    path.write_text(
        json.dumps(
            {
                "files": 3,
                "valid_files": 2,
                "parse_failures": [{"error": "ValueError: empty"}],
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["compact_detector_result", str(path)])
    compactor.main()
    compact = json.loads((tmp_path / "RESULTS_SUMMARY.json").read_text())
    assert compact["parse_failure_count"] == 1
    assert "parse_failures" not in compact
