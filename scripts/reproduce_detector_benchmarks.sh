#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "Usage: $0 SHS_ROOT ASAP_ROOT LMD_CLEAN_ROOT DETECTOR_DIR OUTPUT_ROOT [WORKERS]" >&2
  exit 2
fi

shs_root=$1
asap_root=$2
lmd_clean_root=$3
detector_dir=$4
output_root=$5
workers=${6:-8}

python scripts/reproduce_detector_benchmark.py shs "$shs_root" "$detector_dir" "$output_root/shs" --workers "$workers"
python scripts/reproduce_detector_benchmark.py asap "$asap_root" "$detector_dir" "$output_root/asap" --workers "$workers"
python scripts/reproduce_detector_benchmark.py lmd-clean "$lmd_clean_root" "$detector_dir" "$output_root/lmd-clean" --workers "$workers"
