#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import mido
from leakagebench_midi import stable, write_jsonl


def midi(path, notes, tracks=1, meta_text=None, reverse_tracks=False):
    m = mido.MidiFile(ticks_per_beat=480)
    ts = []
    for ti in range(tracks):
        t = mido.MidiTrack()
        t.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
        if meta_text:
            t.append(mido.MetaMessage("track_name", name=meta_text, time=0))
        for i, n in enumerate(notes[ti::tracks]):
            t.append(mido.Message("note_on", note=n, velocity=64, time=120 if i else 0))
            t.append(mido.Message("note_off", note=n, velocity=0, time=120))
        ts.append(t)
    if reverse_tracks:
        ts.reverse()
    for t in ts:
        m.tracks.append(t)
    path.parent.mkdir(parents=True, exist_ok=True)
    m.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    a = p.parse_args()
    o = Path(a.output)
    o.mkdir(parents=True, exist_ok=True)
    md = o / "midi"
    rows = []
    family_map = {}
    for fi in range(12):
        n = 2 + fi % 2
        for mi in range(n):
            i = f"syn_{fi:02d}_{mi}"
            r = {
                "id": i,
                "family_id": f"family_{fi:02d}",
                "tokens": 1024 + ((fi * 37 + mi * 13) % 97),
                "relative_source": f"midi/{i}.mid",
            }
            rows.append(r)
            family_map[i] = r["family_id"]
            midi(
                md / f"{i}.mid",
                [48 + ((fi * 5 + mi + j * 2) % 24) for j in range(16)],
                1,
            )
    write_jsonl(o / "input_manifest.jsonl", rows)
    (o / "family_map.json").write_text(
        json.dumps(family_map, indent=2, sort_keys=True) + "\n"
    )
    edges = []
    for fi in range(12):
        members = [x["id"] for x in rows if x["family_id"] == f"family_{fi:02d}"]
        edges += [{"a": members[0], "b": x} for x in members[1:]]
    write_jsonl(o / "family_edges.jsonl", edges)
    random = [
        dict(
            r,
            split=(
                "train"
                if int(stable("split", r["id"])[:8], 16) % 10 < 6
                else "validation"
                if int(stable("split", r["id"])[:8], 16) % 10 < 8
                else "test"
            ),
        )
        for r in rows
    ]
    write_jsonl(o / "random_split.jsonl", random)
    treated = []
    assign = []
    for fi in range(3):
        members = [dict(x) for x in rows if x["family_id"] == f"family_{fi:02d}"]
        treated.append(
            {
                "family_id": f"family_{fi:02d}",
                "members": members,
                "receiver_id": members[0]["id"],
                "donor_id": members[1]["id"],
            }
        )
        assign.append(
            {
                "family_id": f"family_{fi:02d}",
                "receiver_id": members[0]["id"],
                "donor_id": members[1]["id"],
            }
        )
        control = [
            dict(next(x for x in rows if x["family_id"] == f"family_{fi:02d}"))
            for fi in range(3, 6)
        ]
        validation = [
            dict(next(x for x in rows if x["family_id"] == f"family_{fi:02d}"))
            for fi in range(6, 9)
        ]
        base = [dict(x) for x in rows if int(x["family_id"].split("_")[1]) >= 9]
        for b, t in zip(base, [x["members"][1] for x in treated]):
            b["tokens"] = t["tokens"]
    write_jsonl(o / "treated.jsonl", treated)
    write_jsonl(o / "control.jsonl", control)
    write_jsonl(o / "clean_validation.jsonl", validation)
    write_jsonl(o / "base.jsonl", base)
    (o / "donor_receiver.json").write_text(
        json.dumps(assign, indent=2, sort_keys=True) + "\n"
    )
    clean = []
    leak = []
    for s in range(3):
        for split, group in [
            ("treated", treated),
            ("control", control),
            ("clean_validation", validation),
        ]:
            for j, x in enumerate(group):
                f = x["family_id"]
                base_nll = 2 + 0.01 * j + 0.02 * s
                effect = (
                    -0.12 - 0.005 * j
                    if split == "treated"
                    else -0.01
                    if split == "control"
                    else -0.005
                )
                common = {
                    "dataset": "synthetic",
                    "architecture": "toy",
                    "model_size": "tiny",
                    "seed": s,
                    "split": split,
                    "family_id": f,
                }
                clean.append({**common, "nll": base_nll})
                leak.append({**common, "nll": base_nll + effect})
    write_jsonl(o / "clean_results.jsonl", clean)
    write_jsonl(o / "leak_results.jsonl", leak)


if __name__ == "__main__":
    main()
