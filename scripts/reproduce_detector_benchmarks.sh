#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 8 || $# -gt 9 ]]; then
  echo "Usage: $0 SHS_ROOT ASAP_ROOT ATEPP_ROOT LMD_CLEAN_ROOT VIENNA_ROOT PIANOVAM_ROOT DETECTOR_DIR OUTPUT_ROOT [WORKERS]" >&2
  exit 2
fi

shs_root=$1
asap_root=$2
atepp_root=$3
lmd_clean_root=$4
vienna_root=$5
pianovam_root=$6
detector_dir=$7
output_root=$8
workers=${9:-8}

python scripts/reproduce_detector_benchmark.py shs "$shs_root" "$detector_dir" "$output_root/shs" --workers "$workers"
python scripts/reproduce_detector_benchmark.py asap "$asap_root" "$detector_dir" "$output_root/asap" --workers "$workers"
python scripts/reproduce_detector_benchmark.py atepp "$atepp_root" "$detector_dir" "$output_root/atepp" --workers "$workers"
python scripts/reproduce_detector_benchmark.py lmd-clean "$lmd_clean_root" "$detector_dir" "$output_root/lmd-clean" --workers "$workers"
python scripts/reproduce_detector_benchmark.py vienna4x22 "$vienna_root" "$detector_dir" "$output_root/vienna4x22" --workers "$workers"
python scripts/reproduce_detector_benchmark.py pianovam "$pianovam_root" "$detector_dir" "$output_root/pianovam" --workers "$workers"
