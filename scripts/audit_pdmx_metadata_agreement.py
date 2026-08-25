#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import unicodedata
from pathlib import Path

import numpy as np


CATALOG_PATTERNS = (
    ("bwv", re.compile(r"\bbwv\s*([0-9]+[a-z]?)\b", re.I)),
    ("kv", re.compile(r"\b(?:kv|k)\.?\s*([0-9]+[a-z]?)\b", re.I)),
    ("hob", re.compile(r"\bhob\.?\s*([ivxlcdm]+)\s*[:./-]\s*([0-9]+[a-z]?)\b", re.I)),
    ("hwv", re.compile(r"\bhwv\s*([0-9]+[a-z]?)\b", re.I)),
    ("rv", re.compile(r"\brv\s*([0-9]+[a-z]?)\b", re.I)),
    ("woo", re.compile(r"\bwoo\s*([0-9]+[a-z]?)\b", re.I)),
    ("sz", re.compile(r"\bsz\.?\s*([0-9]+[a-z]?)\b", re.I)),
    ("d", re.compile(r"\bd\.?\s*([0-9]+[a-z]?)\b", re.I)),
    ("s", re.compile(r"\bs\.?\s*([0-9]+[a-z]?)\b", re.I)),
)
OPUS_PATTERN = re.compile(
    r"\bop(?:us)?\.?\s*([0-9]+[a-z]?)\s*(?:[,./-]?\s*(?:no|nr)\.?\s*([0-9]+[a-z]?))?\b",
    re.I,
)
MOVEMENT_PATTERNS = (
    re.compile(r"\b(?:movement|mov|mvt)\.?\s*([ivxlcdm]+|[0-9]+)\b", re.I),
    re.compile(r"\b(?:no|nr)\.?\s*([0-9]+)\b", re.I),
)
PRESENTATION_PATTERNS = (
    re.compile(r"\b(?:sheet\s+music|musescore|full\s+score|score)\b", re.I),
    re.compile(r"\s*[\[(]\s*(?:arr(?:anged)?\.?|transcri(?:bed|ption)|version)\s+(?:by|for)?[^\])]*[\])]\s*$", re.I),
    re.compile(r"\s*[-–—:]\s*(?:arr(?:anged)?\.?|transcri(?:bed|ption)|version)\s+(?:by|for)?\s+.+$", re.I),
    re.compile(r"\s*[-–—:]\s*(?:piano|guitar|orchestra|orchestral|choir|choral|violin|cello|flute|solo|duet|quartet)\s+(?:solo|duet|trio|quartet|quintet|arrangement|version|score)\s*$", re.I),
)


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(text.split())


def normalize_composer(value: object) -> str:
    text = normalize_text(value)
    text = re.sub(r"\([^)]*(?:[12][0-9]{3}|b\.|d\.)[^)]*\)", " ", text)
    text = re.sub(r"\b(?:born|died|composer|music by|composed by)\b", " ", text)
    text = re.sub(r"[^\w\s'-]", " ", text)
    return " ".join(text.split()).strip(" '-")


def normalize_title(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    for pattern in PRESENTATION_PATTERNS:
        while match := pattern.search(text):
            text = text[: match.start()] + " " + text[match.end() :]
    text = re.sub(r"\.(?:mid|midi|mxl|musicxml|mscz)$", "", text.casefold())
    return " ".join(re.sub(r"[^\w\s]+", " ", text).split())


def extract_catalog(value: object) -> list[str]:
    text = normalize_text(value)
    found = []
    for label, pattern in CATALOG_PATTERNS:
        for match in pattern.finditer(text):
            found.append(label + ":" + ":".join(part.casefold() for part in match.groups()))
    for match in OPUS_PATTERN.finditer(text):
        opus, number = match.groups()
        found.append(f"op:{opus.casefold()}" + (f":no:{number.casefold()}" if number else ""))
    return sorted(set(found))


def extract_movement(value: object) -> str:
    text = normalize_text(value)
    hits = []
    for index, pattern in enumerate(MOVEMENT_PATTERNS):
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start() - 12) : match.start()]
            if index == 1 and re.search(r"op(?:us)?\.?\s*\d+\s*$", prefix):
                continue
            hits.append(match.group(1).casefold())
    unique = sorted(set(hits))
    return unique[0] if len(unique) == 1 else ("CONFLICT:" + ",".join(unique) if unique else "")


def metadata(csv_path: Path) -> dict[str, dict]:
    output = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            identity = Path(row["mid"]).stem
            title = (row["song_name"] or "").strip() or (row["title"] or "").strip()
            combined = " | ".join((row[name] or "").strip() for name in ("song_name", "title", "subtitle"))
            output[identity] = {
                "title": normalize_title(title),
                "composer": normalize_composer(row["composer_name"]),
                "catalogs": extract_catalog(combined),
                "movement": extract_movement(combined),
                "raw_title": title,
                "raw_composer": (row["composer_name"] or "").strip(),
            }
    return output


def agreement(left: dict, right: dict) -> dict[str, bool]:
    title = bool(left["title"] and left["title"] == right["title"])
    composer = bool(left["composer"] and left["composer"] == right["composer"])
    catalog = bool(set(left["catalogs"]) & set(right["catalogs"]))
    movement = bool(left["movement"] and left["movement"] == right["movement"])
    return {
        "title": title,
        "composer": composer,
        "title_and_composer": title and composer,
        "catalog_and_composer": catalog and composer,
        "catalog_movement_and_composer": catalog and movement and composer,
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"pairs": 0}
    keys = tuple(rows[0]["agreement"])
    return {
        "pairs": len(rows),
        **{
            f"{key}_rate": sum(row["agreement"][key] for row in rows) / len(rows)
            for key in keys
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PDMX detector edges with metadata.")
    parser.add_argument("metadata_csv", type=Path)
    parser.add_argument("edges", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--control-pairs", type=int)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    index = metadata(args.metadata_csv)
    predicted = []
    with gzip.open(args.edges, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            left_id, right_id = row["left_identity"], row["right_identity"]
            if left_id in index and right_id in index:
                predicted.append(
                    {
                        "left": left_id,
                        "right": right_id,
                        "score": float(row["score"]),
                        "agreement": agreement(index[left_id], index[right_id]),
                    }
                )
    if not predicted:
        raise RuntimeError("no detector edges matched PDMX metadata")

    control_pairs = args.control_pairs or len(predicted)
    identities = np.asarray(sorted(index))
    rng = np.random.default_rng(args.seed)
    controls = []
    while len(controls) < control_pairs:
        selected = rng.integers(0, len(identities), size=(control_pairs, 2))
        for left_pos, right_pos in selected:
            if left_pos == right_pos:
                continue
            controls.append(
                {
                    "agreement": agreement(
                        index[identities[left_pos]], index[identities[right_pos]]
                    )
                }
            )
            if len(controls) == control_pairs:
                break

    scores = np.asarray([row["score"] for row in predicted])
    cuts = np.quantile(scores, np.linspace(0, 1, 11))
    deciles = []
    for number, (low, high) in enumerate(zip(cuts, cuts[1:]), start=1):
        selected = [
            row
            for row in predicted
            if row["score"] >= low
            and (row["score"] <= high if number == 10 else row["score"] < high)
        ]
        deciles.append(
            {
                "decile": number,
                "score_low": float(low),
                "score_high": float(high),
                **summarize(selected),
            }
        )

    unmatched = [row for row in predicted if not row["agreement"]["title_and_composer"]]
    rng.shuffle(unmatched)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "predicted": summarize(predicted),
        "random_control": summarize(controls),
        "score_deciles": deciles,
        "metadata_records": len(index),
        "predicted_edges_with_metadata": len(predicted),
    }
    (args.output_dir / "metadata_agreement.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    with (args.output_dir / "unmatched_review_sample.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = (
            "left",
            "right",
            "score",
            "left_title",
            "right_title",
            "left_composer",
            "right_composer",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in unmatched[:200]:
            left, right = index[row["left"]], index[row["right"]]
            writer.writerow(
                {
                    "left": row["left"],
                    "right": row["right"],
                    "score": row["score"],
                    "left_title": left["raw_title"],
                    "right_title": right["raw_title"],
                    "left_composer": left["raw_composer"],
                    "right_composer": right["raw_composer"],
                }
            )
    print(json.dumps(result["predicted"], sort_keys=True))
    print(json.dumps(result["random_control"], sort_keys=True))


if __name__ == "__main__":
    main()
