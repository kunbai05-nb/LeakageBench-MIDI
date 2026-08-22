#!/usr/bin/env python3
"""Verify and materialize frozen public-safe result artifacts; no training."""

from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def verify():
    lock = RESULTS / "manuscript_results_v2_public.json"
    expected = (RESULTS / "manuscript_results_v2_public.sha256").read_text().split()[0]
    if sha(lock) != expected:
        raise ValueError("public v2 result checksum mismatch")
    payload = json.loads(lock.read_text())
    if (
        payload.get("formal_results_changed") is not False
        or payload.get("formal_protocol_changed") is not False
    ):
        raise ValueError("formal boundary metadata invalid")
    if len(payload.get("results", [])) != payload.get("result_count"):
        raise ValueError("public v2 result count mismatch")
    forbidden = {"source_artifact", "source_field", "source_sha256"}
    if any(forbidden & set(row) for row in payload["results"]):
        raise ValueError("internal provenance field present")
    manifest = json.loads((RESULTS / "RESULTS_MANIFEST.json").read_text())
    for item in manifest["files"]:
        p = RESULTS / item["path"]
        if (
            not p.is_file()
            or p.stat().st_size != item["bytes"]
            or sha(p) != item["sha256"]
        ):
            raise ValueError("result artifact integrity failure: " + item["path"])
    new_root = RESULTS / "new_experiments"
    new_manifest = json.loads((new_root / "MANIFEST.json").read_text())
    for item in new_manifest["files"]:
        path = new_root / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or sha(path) != item["sha256"]
        ):
            raise ValueError(
                "new experiment artifact integrity failure: " + item["path"]
            )
    manifest["new_experiments"] = new_manifest["files"]
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--output")
    a = ap.parse_args()
    manifest = verify()
    if a.output:
        target = Path(a.output)
        target.mkdir(parents=True, exist_ok=True)
        for item in manifest["files"]:
            src = RESULTS / item["path"]
            dst = target / item["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        for item in manifest["new_experiments"]:
            src = RESULTS / "new_experiments" / item["path"]
            dst = target / "new_experiments" / item["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    total = len(manifest["files"]) + len(manifest["new_experiments"])
    print(f"verified {total} public result artifacts")


if __name__ == "__main__":
    main()
