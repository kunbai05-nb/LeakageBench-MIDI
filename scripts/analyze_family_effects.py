#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

from leakagebench_midi import analyze_effect, read_jsonl


parser = argparse.ArgumentParser(description="Analyze fully paired family effects with frozen-manifest validation.")
parser.add_argument("--clean_results", required=True)
parser.add_argument("--leak_results", required=True)
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--family_manifest", help="Required for formal analysis; family IDs and optional hash are validated.")
mode.add_argument("--exploratory_without_manifest", action="store_true", help="Explicit non-formal exploratory mode only.")
parser.add_argument("--bootstrap_samples", type=int, default=10000)
parser.add_argument("--bootstrap_seed", type=int, default=0)
parser.add_argument("--output", required=True)
args = parser.parse_args()

clean = read_jsonl(args.clean_results)
leak = read_jsonl(args.leak_results)
rows = [dict(row, condition="clean") for row in clean] + [dict(row, condition="family_leak") for row in leak]
manifest = json.loads(Path(args.family_manifest).read_text()) if args.family_manifest else None
result = analyze_effect(rows, args.bootstrap_samples, args.bootstrap_seed, family_manifest=manifest)
result["input_sha256"] = {
    "clean_results": hashlib.sha256(Path(args.clean_results).read_bytes()).hexdigest(),
    "family_leak_results": hashlib.sha256(Path(args.leak_results).read_bytes()).hexdigest(),
}
if args.family_manifest:
    result["family_manifest_sha256"] = hashlib.sha256(Path(args.family_manifest).read_bytes()).hexdigest()
Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
