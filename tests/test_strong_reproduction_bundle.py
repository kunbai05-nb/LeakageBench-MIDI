import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_reproduction_manifest_integrity():
    manifest = json.loads((ROOT / "reproduction/PUBLIC_REPRODUCTION_MANIFEST.json").read_text())
    assert manifest["contains_raw_midi"] is False
    assert manifest["contains_tokens"] is False
    assert manifest["contains_checkpoints"] is False
    for item in manifest["files"]:
        path = ROOT / item["path"]
        assert path.is_file()
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]


def test_public_family_ids_are_pseudonyms_not_original_hashes():
    for name in ("phase2_nll_rows.csv", "musical_family_metrics.csv", "pdmx_nll_rows.csv"):
        with (ROOT / "reproduction/data" / name).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        assert all(re.fullmatch(r"family_[0-9a-f]{16}", row["family_id"]) for row in rows)


def test_no_token_sequence_columns():
    for path in (ROOT / "reproduction/data").glob("*.csv"):
        with path.open(newline="") as handle:
            fields = next(csv.reader(handle), [])
        assert "token_ids" not in fields
        assert "tokens" not in fields


def test_strong_reproduction_cli_is_present():
    text = (ROOT / "scripts/reproduce_paper_statistics.py").read_text()
    assert "import torch" not in text
    assert ".cuda(" not in text
    assert "CUDA_VISIBLE_DEVICES" not in text
    assert "PUBLIC_REPRODUCTION_MANIFEST.json" in text
