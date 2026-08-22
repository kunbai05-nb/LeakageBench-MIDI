#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from leakagebench_midi import build_contamination, read_jsonl, write_jsonl


parser = argparse.ArgumentParser(
    description="Construct validated, approximately token-budget-matched contamination manifests."
)
for name in [
    "base_manifest",
    "treated_families",
    "control_families",
    "clean_validation",
    "donor_receiver",
    "output",
]:
    parser.add_argument("--" + name, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_total_token_rel_diff", type=float)
parser.add_argument("--max_pair_token_rel_diff", type=float)
args = parser.parse_args()

assignments_value = json.loads(Path(args.donor_receiver).read_text())
assignments = (
    assignments_value.get("assignments", assignments_value)
    if isinstance(assignments_value, dict)
    else assignments_value
)
result = build_contamination(
    read_jsonl(args.base_manifest),
    read_jsonl(args.treated_families),
    read_jsonl(args.control_families),
    read_jsonl(args.clean_validation),
    assignments,
    args.seed,
    max_total_token_rel_diff=args.max_total_token_rel_diff,
    max_pair_token_rel_diff=args.max_pair_token_rel_diff,
)
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
for name in [
    "train_clean",
    "train_family_leak",
    "treated_probe",
    "control_probe",
    "clean_validation",
]:
    write_jsonl(output / (name + ".jsonl"), result[name])
(output / "replacement_audit.json").write_text(
    json.dumps(result["replacements"], indent=2, sort_keys=True) + "\n"
)
(output / "token_budget_reconciliation.json").write_text(
    json.dumps(result["token_budget"], indent=2, sort_keys=True) + "\n"
)
(output / "integrity.json").write_text(
    json.dumps(result["integrity"], indent=2, sort_keys=True) + "\n"
)
