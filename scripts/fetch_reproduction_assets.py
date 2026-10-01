#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reproduction" / "ASSETS.json"
GITHUB_RELEASE = re.compile(
    r"https://github\.com/([^/]+)/([^/]+)/releases/download/([^/]+)/([^/]+)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(name: str, prefix: str) -> Path | None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member: {name}")
    if not name.startswith(prefix):
        return None
    relative = PurePosixPath(name[len(prefix) :])
    if not relative.parts:
        return None
    return Path(*relative.parts)


def download(url: str, destination: Path) -> None:
    try:
        with urllib.request.urlopen(url) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
        return
    except urllib.error.HTTPError as error:
        match = GITHUB_RELEASE.fullmatch(url)
        gh = shutil.which("gh")
        if error.code not in (403, 404) or match is None or gh is None:
            raise

    owner, repository, tag, asset = match.groups()
    subprocess.run(
        [
            gh,
            "release",
            "download",
            tag,
            "--repo",
            f"{owner}/{repository}",
            "--pattern",
            asset,
            "--dir",
            str(destination.parent),
        ],
        check=True,
    )
    downloaded = destination.parent / asset
    if not downloaded.is_file():
        raise FileNotFoundError(downloaded)
    downloaded.replace(destination)


def fetch(name: str, force: bool = False) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    try:
        spec = manifest["assets"][name]
    except KeyError as error:
        choices = ", ".join(sorted(manifest.get("assets", {})))
        raise ValueError(f"unknown asset {name!r}; choose from: {choices}") from error

    required = [ROOT / path for path in spec["required_paths"]]
    file_hashes = spec.get("file_sha256", {})
    hashes_match = all((ROOT / path).is_file() and sha256(ROOT / path) == expected for path, expected in file_hashes.items())
    if not force and all(path.is_file() for path in required) and hashes_match:
        print(f"{name}: already available")
        return

    with tempfile.TemporaryDirectory(prefix="leakagebench-assets-") as directory:
        archive_path = Path(directory) / "asset.tar.gz"
        print(f"Downloading {name} ...")
        download(spec["url"], archive_path)
        actual = sha256(archive_path)
        if actual != spec["sha256"]:
            raise ValueError(
                f"asset checksum mismatch: expected {spec['sha256']}, received {actual}"
            )

        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                relative = safe_relative_path(member.name, spec["archive_prefix"])
                if relative is None or member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f"unsupported archive member: {member.name}")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot read archive member: {member.name}")
                destination = ROOT / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("wb") as output:
                    shutil.copyfileobj(source, output)

    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"asset is incomplete; missing: {', '.join(missing)}")
    for path, expected in file_hashes.items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"installed asset file checksum mismatch: {path}")
    print(f"{name}: downloaded and verified")


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(
        description="Download and verify optional LeakageBench-MIDI reproduction assets."
    )
    parser.add_argument("asset", choices=sorted(manifest["assets"]))
    parser.add_argument("--force", action="store_true", help="replace an existing asset")
    args = parser.parse_args()
    fetch(args.asset, args.force)


if __name__ == "__main__":
    main()
