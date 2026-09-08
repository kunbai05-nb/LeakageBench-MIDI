#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 || $# -gt 8 ]]; then
  echo "Usage: $0 SHS_LMD_ROOT ASAP_ROOT ATEPP_ROOT MAESTRO_ROOT LMD_CLEAN_ROOT DETECTOR_DIR OUTPUT_ROOT [WORKERS]" >&2
  exit 2
fi

shs_root=$1
asap_root=$2
atepp_root=$3
maestro_root=$4
lmd_clean_root=$5
detector_dir=$6
output_root=$7
workers=${8:-8}

python scripts/reproduce_detector_benchmark.py shs "$shs_root" "$detector_dir" "$output_root/shs" --workers "$workers"
python scripts/reproduce_detector_benchmark.py asap "$asap_root" "$detector_dir" "$output_root/asap" --workers "$workers"
python scripts/reproduce_detector_benchmark.py atepp "$atepp_root" "$detector_dir" "$output_root/atepp" --workers "$workers"
python scripts/reproduce_detector_benchmark.py maestro "$maestro_root" "$detector_dir" "$output_root/maestro" --workers "$workers"
python scripts/reproduce_detector_benchmark.py lmd-clean "$lmd_clean_root" "$detector_dir" "$output_root/lmd-clean" --workers "$workers"
