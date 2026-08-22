#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from leakagebench_midi import read_jsonl, build_family_map

p = argparse.ArgumentParser()
p.add_argument("--manifest", required=True)
p.add_argument("--edges", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
rows = read_jsonl(a.manifest)
edges = read_jsonl(a.edges)
m = build_family_map([x["id"] for x in rows], [(x["a"], x["b"]) for x in edges])
Path(a.output).write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
