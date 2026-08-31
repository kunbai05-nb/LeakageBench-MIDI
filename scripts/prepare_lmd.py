#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from leakagebench_midi.models.tokenizer import MidiTokenizer


ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "reproduction" / "source_specs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def read_specs() -> list[dict]:
    with gzip.open(
        SPECS / "formal_windows.csv.gz", "rt", newline="", encoding="utf-8"
    ) as handle:
        return list(csv.DictReader(handle))


def read_slots() -> dict[str, dict[int, dict]]:
    output = defaultdict(dict)
    with gzip.open(
        SPECS / "condition_slots.csv.gz", "rt", newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            output[row["condition"]][int(row["slot_index"])] = row
    return output


def copy_required_midi(
    source: Path, target: Path, required: dict[str, str], archive_hash: str
) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        if sha256(source) != archive_hash:
            raise ValueError("LMD archive hash mismatch")
        found = set()
        with tarfile.open(source, "r:gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if not name.lower().endswith(".mid"):
                    continue
                identity = Path(name).stem.lower()
                if identity not in required:
                    continue
                destination = target / identity[0] / f"{identity}.mid"
                destination.parent.mkdir(parents=True, exist_ok=True)
                stream = archive.extractfile(member)
                if stream is None:
                    continue
                with destination.open("wb") as handle:
                    shutil.copyfileobj(stream, handle)
                found.add(identity)
    else:
        roots = (source, source / "lmd_full")
        sample = next(iter(required))
        midi_root = next(
            (root for root in roots if (root / sample[0] / f"{sample}.mid").is_file()),
            None,
        )
        if midi_root is None:
            raise FileNotFoundError("cannot find the extracted lmd_full directory")
        found = set()
        for identity in required:
            origin = midi_root / identity[0] / f"{identity}.mid"
            if not origin.is_file():
                continue
            destination = target / identity[0] / f"{identity}.mid"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, destination)
            found.add(identity)

    missing = sorted(set(required) - found)
    if missing:
        raise FileNotFoundError(f"{len(missing)} required LMD files are missing")
    for identity, expected in required.items():
        path = target / identity[0] / f"{identity}.mid"
        if sha256(path) != expected:
            raise ValueError(f"MIDI hash mismatch: {identity}")


def materialize_rows(
    specs: list[dict], midi_root: Path, source_role: str
) -> dict[int, dict]:
    tokenizer = MidiTokenizer()
    by_piece = defaultdict(list)
    for spec in specs:
        by_piece[spec["md5"]].append(spec)
    output = {}
    for piece_specs in by_piece.values():
        first = piece_specs[0]
        path = midi_root / first["md5"][0] / f"{first['md5']}.mid"
        tokens, metadata = tokenizer.encode_file(path)
        windows = {
            (row["start_bar"], row["end_bar"]): row
            for row in tokenizer.windows(tokens, metadata)
        }
        for spec in piece_specs:
            window = windows.get((int(spec["start_bar"]), int(spec["end_bar"])))
            if (
                window is None
                or window["token_count"] != int(spec["token_count"])
                or window["prompt_token_count"] != int(spec["prompt_token_count"])
                or canonical_hash(window["token_ids"]) != spec["token_ids_sha256"]
            ):
                raise ValueError(f"token mismatch: {spec['md5']}:{spec['start_bar']}")
            output[int(spec["slot_index"])] = {
                "schema_version": 1,
                "source_role": source_role,
                "window_id": spec["window_id"],
                "family_id": spec["family_id"],
                "piece_sha256": spec["piece_sha256"],
                "start_bar": int(spec["start_bar"]),
                "end_bar": int(spec["end_bar"]),
                "token_ids": window["token_ids"],
                "token_count": int(spec["token_count"]),
                "prompt_token_count": int(spec["prompt_token_count"]),
            }
    return output


def assignment_fields(row: dict, assignment: dict) -> dict:
    row = dict(row)
    row["replacement_slot_id"] = assignment["slot_id"]
    row["receiver_family_id"] = assignment["receiver_family_id"]
    row["donor_id"] = assignment["donor_id"] or None
    return row


def stream_identity(index: int, row: dict) -> dict:
    return {
        "index": index,
        "window_id": row["window_id"],
        "family_id": row["family_id"],
        "token_count": row["token_count"],
        "prompt_token_count": row["prompt_token_count"],
        "token_ids_sha256": canonical_hash(row["token_ids"]),
        "replacement_slot_id": row.get("replacement_slot_id"),
        "donor_id": row.get("donor_id"),
    }


def write_streams(
    specs: list[dict], slots: dict, midi_root: Path, output: Path, expected: dict
) -> dict:
    same = materialize_rows(
        [row for row in specs if row["role"] == "same_family_replacement"],
        midi_root,
        "confirmatory_treated_donor",
    )
    unrelated = materialize_rows(
        [row for row in specs if row["role"] == "unrelated_replacement"],
        midi_root,
        "unrelated_donor",
    )
    clean_specs = [row for row in specs if row["role"] == "clean"]
    paths = {condition: output / "streams" / condition for condition in expected}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    handles = {
        condition: path.with_suffix(".tokens").open("wb")
        for condition, path in paths.items()
    }
    packed = {condition: {"offsets": [0], "prompt": []} for condition in expected}
    identities = {condition: [] for condition in expected}
    totals = defaultdict(int)
    tokenizer = MidiTokenizer()
    current_md5 = None
    windows = {}
    try:
        for index, spec in enumerate(clean_specs):
            if int(spec["slot_index"]) != index:
                raise ValueError("clean stream slot order mismatch")
            if spec["md5"] != current_md5:
                current_md5 = spec["md5"]
                path = midi_root / current_md5[0] / f"{current_md5}.mid"
                tokens, metadata = tokenizer.encode_file(path)
                windows = {
                    (row["start_bar"], row["end_bar"]): row
                    for row in tokenizer.windows(tokens, metadata)
                }
            window = windows[(int(spec["start_bar"]), int(spec["end_bar"]))]
            if (
                window["token_count"] != int(spec["token_count"])
                or window["prompt_token_count"] != int(spec["prompt_token_count"])
                or canonical_hash(window["token_ids"]) != spec["token_ids_sha256"]
            ):
                raise ValueError(f"token mismatch: {spec['md5']}:{spec['start_bar']}")
            clean = {
                "schema_version": 1,
                "source_role": "base",
                "window_id": spec["window_id"],
                "family_id": spec["family_id"],
                "piece_sha256": spec["piece_sha256"],
                "start_bar": int(spec["start_bar"]),
                "end_bar": int(spec["end_bar"]),
                "token_ids": window["token_ids"],
                "token_count": int(spec["token_count"]),
                "prompt_token_count": int(spec["prompt_token_count"]),
            }
            rows = {
                "clean": clean,
                "same_family_donor": same.get(index, clean),
                "unrelated_donor": unrelated.get(index, clean),
            }
            for condition, row in rows.items():
                if index in slots[condition]:
                    row = assignment_fields(row, slots[condition][index])
                values = np.asarray(row["token_ids"], dtype=np.uint16)
                handles[condition].write(values.tobytes())
                packed[condition]["offsets"].append(
                    packed[condition]["offsets"][-1] + len(values)
                )
                packed[condition]["prompt"].append(row["prompt_token_count"])
                identities[condition].append(stream_identity(index, row))
                totals[condition] += row["token_count"]
    finally:
        for handle in handles.values():
            handle.close()

    result = {}
    for condition, rows in identities.items():
        np.savez_compressed(
            paths[condition].with_name(paths[condition].name + "_index.npz"),
            offsets=np.asarray(packed[condition]["offsets"], dtype=np.int64),
            prompt=np.asarray(packed[condition]["prompt"], dtype=np.int16),
        )
        result[condition] = {
            "rows": len(rows),
            "token_count": totals[condition],
            "sha256": canonical_hash(rows),
        }
        if result[condition] != expected[condition]:
            raise ValueError(f"stream identity mismatch: {condition}")
    return result


def write_probes(specs: list[dict], midi_root: Path, output: Path) -> int:
    path = output / "evaluation_windows.jsonl"
    tokenizer = MidiTokenizer()
    probe_specs = [row for row in specs if row["role"] == "probe"]
    path.parent.mkdir(parents=True, exist_ok=True)
    current_md5 = None
    windows = {}
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for spec in probe_specs:
            if spec["md5"] != current_md5:
                current_md5 = spec["md5"]
                midi = midi_root / current_md5[0] / f"{current_md5}.mid"
                tokens, metadata = tokenizer.encode_file(midi)
                windows = {
                    (row["start_bar"], row["end_bar"]): row
                    for row in tokenizer.windows(tokens, metadata)
                }
            window = windows[(int(spec["start_bar"]), int(spec["end_bar"]))]
            if (
                window["token_count"] != int(spec["token_count"])
                or window["prompt_token_count"] != int(spec["prompt_token_count"])
                or canonical_hash(window["token_ids"]) != spec["token_ids_sha256"]
            ):
                raise ValueError(
                    f"probe token mismatch: {spec['md5']}:{spec['start_bar']}"
                )
            row = {
                "split": spec["split"],
                "piece_id": spec["piece_id"],
                "family_id": spec["family_id"],
                **window,
            }
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the formal token streams from LMD-full."
    )
    parser.add_argument("lmd_source", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    config = json.loads((SPECS / "formal_data.json").read_text())
    specs = read_specs()
    required = {}
    for row in specs:
        previous = required.setdefault(row["md5"], row["piece_sha256"])
        if previous != row["piece_sha256"]:
            raise ValueError(f"conflicting source hash: {row['md5']}")
    midi_root = args.output_dir / "midi"
    copy_required_midi(
        args.lmd_source,
        midi_root,
        required,
        config["lmd_full_archive_sha256"],
    )
    streams = write_streams(
        specs, read_slots(), midi_root, args.output_dir, config["expected_streams"]
    )
    probe_windows = write_probes(specs, midi_root, args.output_dir)
    result = {
        "midi_files": len(required),
        "probe_windows": probe_windows,
        "streams": streams,
    }
    (args.output_dir / "PREPARATION_SUMMARY.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
