#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path


SPOTIFY_TRACK_ID = re.compile(r"[A-Za-z0-9]{22}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_family_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return {str(key): str(value) for key, value in json.load(handle).items()}


def reference_id(
    family_map: dict[str, str], identity: str, relative_path: str
) -> str | None:
    return family_map.get(identity, family_map.get(relative_path))


def normalized(value: object) -> str:
    return " ".join(str(value).casefold().split())


def stable_id(prefix: str, *parts: object) -> str:
    value = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return f"{prefix}:{hashlib.sha256(value.encode()).hexdigest()}"


def pop909(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("[0-9][0-9][0-9]/**/*.mid")):
        relative = path.relative_to(root).as_posix()
        path_digest = hashlib.sha256(relative.encode()).hexdigest()[:16]
        rows.append(
            {
                "identity": f"{sha256(path)}:{path_digest}",
                "relative_path": relative,
                "reference_work_id": relative.split("/", 1)[0],
            }
        )
    return rows


def pdmx(root: Path, metadata_csv: Path, family_map: dict[str, str]) -> list[dict]:
    rows = []
    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            if item["subset:no_license_conflict"].casefold() != "true":
                continue
            if item["subset:all_valid"].casefold() != "true":
                continue
            relative = item["mid"].removeprefix("./")
            if not (root / relative).is_file():
                raise FileNotFoundError(root / relative)
            identity = Path(relative).stem
            row = {"identity": identity, "relative_path": relative}
            work = reference_id(family_map, identity, relative)
            if work is not None:
                row["reference_work_id"] = work
            rows.append(row)
    return rows


def lmd(root: Path, family_map: dict[str, str]) -> list[dict]:
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.casefold() == ".mid"
    )
    rows = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        identity = path.stem.casefold()
        row = {"identity": identity, "relative_path": relative}
        work = reference_id(family_map, identity, relative)
        if work is not None:
            row["reference_work_id"] = work
        rows.append(row)
    return rows


def gigamidi(metadata_csv: Path) -> list[dict]:
    csv.field_size_limit(max(csv.field_size_limit(), 64 * 1024**2))
    rows = []
    with metadata_csv.open(newline="", encoding="utf-8-sig") as handle:
        for item in csv.DictReader(handle):
            relative = item["file_path"].removeprefix("./")
            identity = item["md5"].strip().casefold()
            sid = item["audio_text_matches_sid"].strip()
            try:
                score = float(item["audio_text_matches_score"])
            except ValueError:
                score = float("nan")
            reference = (
                f"spotify:{sid}"
                if SPOTIFY_TRACK_ID.fullmatch(sid) and score > 0.5
                else f"file:{identity}"
            )
            split = next(
                name
                for name in ("train", "validation", "test")
                if any(part.casefold().startswith(name) for part in Path(relative).parts)
            )
            rows.append(
                {
                    "identity": identity,
                    "relative_path": relative,
                    "reference_work_id": reference,
                    "split": split,
                }
            )
    return rows


def asap(root: Path, metadata_csv: Path) -> list[dict]:
    rows = []
    with metadata_csv.open(newline="", encoding="utf-8-sig") as handle:
        for item in csv.DictReader(handle):
            relative = item["midi_performance"].strip()
            if not relative:
                continue
            if not (root / relative).is_file():
                raise FileNotFoundError(root / relative)
            rows.append(
                {
                    "identity": stable_id("asap-file", relative),
                    "relative_path": relative,
                    "reference_work_id": stable_id(
                        "asap-work", normalized(item["composer"]), normalized(item["title"])
                    ),
                }
            )
    return rows


def maestro(metadata_csv: Path, asap_metadata_csv: Path) -> list[dict]:
    linked = {}
    with asap_metadata_csv.open(newline="", encoding="utf-8-sig") as handle:
        for item in csv.DictReader(handle):
            relative = item["maestro_midi_performance"].removeprefix("{maestro}/").strip()
            if relative:
                linked[relative] = stable_id(
                    "asap-work", normalized(item["composer"]), normalized(item["title"])
                )

    rows = []
    with metadata_csv.open(newline="", encoding="utf-8-sig") as handle:
        for item in csv.DictReader(handle):
            relative = item["midi_filename"].strip()
            identity = stable_id("maestro-file", relative)
            rows.append(
                {
                    "identity": identity,
                    "relative_path": relative,
                    "reference_work_id": linked.get(relative, f"file:{identity}"),
                    "split": item["split"].strip(),
                }
            )
    return rows


def aria(root: Path, metadata_json: Path) -> list[dict]:
    metadata = json.loads(metadata_json.read_text())
    rows = []
    for path in sorted(root.rglob("*.mid")):
        relative = path.relative_to(root).as_posix()
        file_id, separator, _ = path.stem.rpartition("_")
        if not separator:
            raise ValueError(f"unexpected Aria-MIDI filename: {relative}")
        item = metadata.get(str(int(file_id)))
        if item is None:
            raise KeyError(f"missing Aria-MIDI metadata for {relative}")
        fields = tuple(item["metadata"].get(name) for name in ("composer", "opus", "piece_number"))
        identity = stable_id("aria-file", relative)
        reference = (
            stable_id("aria-work", *(normalized(value) for value in fields))
            if all(value is not None for value in fields)
            else f"file:{identity}"
        )
        rows.append(
            {
                "identity": identity,
                "relative_path": relative,
                "reference_work_id": reference,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a detector input manifest.")
    parser.add_argument(
        "dataset", choices=("pop909", "pdmx", "lmd", "gigamidi", "asap", "maestro", "aria")
    )
    parser.add_argument("midi_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--asap-metadata-csv", type=Path)
    parser.add_argument("--family-map", type=Path)
    args = parser.parse_args()

    family_map = load_family_map(args.family_map)
    if args.dataset == "pop909":
        rows = pop909(args.midi_root)
    elif args.dataset == "pdmx":
        if args.metadata_csv is None:
            parser.error("PDMX requires --metadata-csv")
        rows = pdmx(args.midi_root, args.metadata_csv, family_map)
    elif args.dataset == "lmd":
        rows = lmd(args.midi_root, family_map)
    elif args.dataset == "gigamidi":
        if args.metadata_csv is None:
            parser.error("GigaMIDI requires --metadata-csv")
        rows = gigamidi(args.metadata_csv)
    elif args.dataset == "asap":
        if args.metadata_csv is None:
            parser.error("ASAP requires --metadata-csv")
        rows = asap(args.midi_root, args.metadata_csv)
    elif args.dataset == "maestro":
        if args.metadata_csv is None or args.asap_metadata_csv is None:
            parser.error("MAESTRO requires --metadata-csv and --asap-metadata-csv")
        rows = maestro(args.metadata_csv, args.asap_metadata_csv)
    else:
        if args.metadata_json is None:
            parser.error("Aria-MIDI requires --metadata-json")
        rows = aria(args.midi_root, args.metadata_json)
    if not rows:
        raise RuntimeError("no MIDI files found")
    if len({row["identity"] for row in rows}) != len(rows):
        raise RuntimeError("MIDI identities are not unique")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"DATASET={args.dataset}")
    print(f"FILES={len(rows)}")
    print(f"MANIFEST_SHA256={sha256(args.output)}")


if __name__ == "__main__":
    main()
