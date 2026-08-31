from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def public_files():
    ignored = {".git", "__pycache__", ".pytest_cache", ".venv"}
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not (ignored & set(path.parts))
    ]


def test_no_packaged_data_weights_or_logs():
    forbidden = {
        ".mid",
        ".midi",
        ".wav",
        ".flac",
        ".mp3",
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".log",
    }
    assert not [path for path in public_files() if path.suffix.lower() in forbidden]


def test_no_absolute_paths_or_obvious_secrets():
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in public_files()
        if path.stat().st_size < 5_000_000
    )
    assert "/home/" + "bk" not in text
    assert "/" + "root/" not in text
    patterns = (
        r"AKIA[0-9A-Z]{16}",
        r"ghp_[A-Za-z0-9]{30,}",
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    )
    assert not [pattern for pattern in patterns if re.search(pattern, text)]


def test_readme_has_primary_workflows():
    text = (ROOT / "README.md").read_text()
    required = (
        "scripts/detect_same_work.py",
        "scripts/train_detector.py",
        "scripts/prepare_lmd.py",
        "scripts/train_model.py",
        "scripts/evaluate_checkpoint.py",
        "scripts/verify_model_checkpoints.py",
        "/releases/tag/v1.2.1",
    )
    assert all(value in text for value in required)


def test_no_obsolete_detector_implementation():
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore").lower()
        for path in public_files()
        if path.stat().st_size < 5_000_000
    )
    forbidden = (
        "caug" + "bert",
        "clamp" + "_cosine",
        "shs" + "_structural_detector",
        "historical " + "rc2",
    )
    assert not [value for value in forbidden if value in text]
