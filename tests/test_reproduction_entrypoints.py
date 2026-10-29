from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import json
import hashlib
import shutil
import tarfile
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def shell(script: str, *args: str, data_root: Path | None = None):
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    if data_root is not None:
        env["LEAKAGEBENCH_DATA_DIR"] = str(data_root)
    return subprocess.run(
        ["bash", str(ROOT / "scripts" / script), *args],
        cwd=ROOT, env=env, text=True, capture_output=True,
    )


def test_all_shell_entries_parse():
    for path in (ROOT / "scripts").glob("*.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_three_condition_environment_check():
    result = shell("reproduce_three_condition.sh", "--check-only")
    assert result.returncode == 0, result.stderr
    assert "entry points are ready" in result.stdout


def test_full_midi_three_condition_reports_missing_source(tmp_path):
    result = shell("reproduce_three_condition_from_midi.sh", data_root=tmp_path)
    assert result.returncode == 1
    assert str(tmp_path / "lmd_full.tar.gz") in result.stderr
    assert "Download lmd_full.tar.gz" in result.stderr


def test_six_dataset_defaults_report_missing_source(tmp_path):
    result = shell("reproduce_six_dataset_table.sh", "--check-only", data_root=tmp_path)
    assert result.returncode == 1
    assert str(tmp_path / "shs") in result.stderr


def test_published_gpt_checkpoint_folder_is_recognized():
    spec = importlib.util.spec_from_file_location(
        "three_condition_entry", ROOT / "scripts" / "reproduce_three_condition_table.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.checkpoint_identity(Path("midigpt-clean/seed_202608040.pt")) == (
        "midigpt", "clean", 202608040
    )


def test_prepare_help_does_not_require_downloaded_specs(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("prepare_entry", ROOT / "scripts" / "prepare_lmd.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SPECS = tmp_path
    monkeypatch.setattr(sys, "argv", ["prepare_lmd.py", "--help"])
    with pytest.raises(SystemExit) as result:
        module.main()
    assert result.value.code == 0


def ablation_summary_module():
    spec = importlib.util.spec_from_file_location("ablation_summary", ROOT / "scripts/summarize_detector_ablation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ablation_difference_is_not_execution_failure(tmp_path, monkeypatch):
    module = ablation_summary_module()
    monkeypatch.setattr(module, "derive", lambda _: {"different": True})
    monkeypatch.setattr(module, "write_outputs", lambda _, output: output.mkdir(parents=True))
    monkeypatch.setattr(sys, "argv", ["summary", "raw", str(tmp_path / "output")])
    module.main()
    report = json.loads((tmp_path / "output/verification.json").read_text())
    assert report["generation_status"] == "SUCCESS"
    assert report["status"] == "DIFFERENT"
    assert report["matches_frozen_paper"] is False


def test_ablation_strict_difference_returns_error(tmp_path, monkeypatch):
    module = ablation_summary_module()
    monkeypatch.setattr(module, "derive", lambda _: {"different": True})
    monkeypatch.setattr(module, "write_outputs", lambda _, output: output.mkdir(parents=True))
    monkeypatch.setattr(sys, "argv", ["summary", "raw", str(tmp_path / "output"), "--strict-match"])
    with pytest.raises(RuntimeError, match="strict-match"):
        module.main()


def test_ablation_invalid_input_still_fails(tmp_path, monkeypatch):
    module = ablation_summary_module()
    monkeypatch.setattr(sys, "argv", ["summary", str(tmp_path / "missing"), str(tmp_path / "output")])
    with pytest.raises(FileNotFoundError):
        module.main()
    assert not (tmp_path / "output/verification.json").exists()


def test_metadata_patch_rechecks_cached_hashes(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("asset_fetch", ROOT / "scripts/fetch_reproduction_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    source = tmp_path / "protocol.json"
    source.write_bytes(b'{"conditions":5}')
    archive = tmp_path / "patch.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source, arcname="protocol.json")
    manifest = tmp_path / "ASSETS.json"
    manifest.write_text(json.dumps({"assets": {"patch": {
        "url": "test-patch", "sha256": module.sha256(archive), "archive_prefix": "",
        "required_paths": ["protocol.json"],
        "file_sha256": {"protocol.json": hashlib.sha256(source.read_bytes()).hexdigest()},
    }}}))
    monkeypatch.setattr(module, "ROOT", checkout)
    monkeypatch.setattr(module, "MANIFEST", manifest)
    calls = []
    def download(url, destination):
        calls.append(url)
        shutil.copyfile(archive, destination)
    monkeypatch.setattr(module, "download", download)
    (checkout / "protocol.json").write_bytes(b"outdated")
    module.fetch("patch")
    module.fetch("patch")
    assert calls == ["test-patch"]
    assert (checkout / "protocol.json").read_bytes() == source.read_bytes()
    (checkout / "protocol.json").write_bytes(b"outdated-again")
    module.fetch("patch")
    assert len(calls) == 2
