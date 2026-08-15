#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."; export PYTHONPATH=.
out="${1:-examples/synthetic_mini/generated}"; mkdir -p "$out"
if [[ -n "${PYTHON:-}" ]]; then python_bin="$PYTHON"; elif [[ -x .venv/bin/python ]]; then python_bin=.venv/bin/python; else python_bin=python3; fi
"$python_bin" scripts/generate_synthetic_mini.py --output "$out"
"$python_bin" scripts/build_family_map.py --manifest "$out/input_manifest.jsonl" --edges "$out/family_edges.jsonl" --output "$out/rebuilt_family_map.json"
"$python_bin" scripts/audit_split_leakage.py --manifest "$out/random_split.jsonl" --output "$out/random_split_audit.json"
"$python_bin" scripts/build_family_aware_split.py --input_manifest "$out/input_manifest.jsonl" --ratios 0.8 0.1 0.1 --seed 17 --output "$out/family_aware"
"$python_bin" scripts/audit_split_leakage.py --manifest "$out/family_aware/split_manifest.jsonl" --output "$out/family_aware_leakage_audit.json"
"$python_bin" scripts/build_family_contamination_experiment.py --base_manifest "$out/base.jsonl" --treated_families "$out/treated.jsonl" --control_families "$out/control.jsonl" --clean_validation "$out/clean_validation.jsonl" --donor_receiver "$out/donor_receiver.json" --output "$out/contamination"
"$python_bin" scripts/analyze_family_leakage_effect.py --clean_results "$out/clean_results.jsonl" --leak_results "$out/leak_results.jsonl" --bootstrap_seed 23 --output "$out/effect.json"
"$python_bin" scripts/run_leakage_census.py --family_map "$out/rebuilt_family_map.json" --ratios 0.8 0.1 0.1 --num_seeds 100 --seed 29 --output "$out/census.json"
