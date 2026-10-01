#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
DATA_ROOT=${LEAKAGEBENCH_DATA_DIR:-$ROOT/data}
CHECK_ONLY=0
if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  echo "Usage: bash scripts/reproduce_six_dataset_table.sh [--check-only] [SHS_ROOT ASAP_ROOT ATEPP_ROOT LMD_CLEAN_ROOT VIENNA4X22_ROOT PIANOVAM_ROOT [OUTPUT_DIR] [WORKERS]]"
  echo "Default dataset directories: $DATA_ROOT/{shs,asap,atepp,lmd-clean,vienna4x22,pianovam}"
  exit 0
fi
if [[ ${1:-} == "--check-only" ]]; then
  CHECK_ONLY=1
  shift
fi
if [[ $# -eq 0 ]]; then
  set -- "$DATA_ROOT/shs" "$DATA_ROOT/asap" "$DATA_ROOT/atepp" \
    "$DATA_ROOT/lmd-clean" "$DATA_ROOT/vienna4x22" "$DATA_ROOT/pianovam"
fi

if [[ $# -lt 6 || $# -gt 8 ]]; then
  echo "Usage: bash scripts/reproduce_six_dataset_table.sh SHS_ROOT ASAP_ROOT ATEPP_ROOT LMD_CLEAN_ROOT VIENNA4X22_ROOT PIANOVAM_ROOT [OUTPUT_DIR] [WORKERS]" >&2
  exit 2
fi

for command in python realpath; do
  command -v "$command" >/dev/null || { echo "ERROR: $command is not installed." >&2; exit 1; }
done
python -c "import numpy, torch; import leakagebench_midi" >/dev/null || {
  echo "ERROR: dependencies are missing. Run: pip install -r requirements.txt && pip install -e '.[scale]'" >&2
  exit 1
}

DATASET_ROOTS=()
for source in "${@:1:6}"; do
  resolved=$(realpath "$source")
  [[ -d "$resolved" ]] || { echo "ERROR: dataset directory not found: $source" >&2; exit 1; }
  DATASET_ROOTS+=("$resolved")
done

OUTPUT=$(realpath -m "${7:-$ROOT/reproduced/six_dataset}")
WORKERS=${8:-8}
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: WORKERS must be a positive integer." >&2; exit 1; }
DETECTOR="$ROOT/checkpoints/same-work-detector-v1.9"

python "$ROOT/scripts/summarize_detector_benchmarks.py" --check-manifests
python "$ROOT/scripts/verify_detector_checkpoint.py" "$DETECTOR"
DATASETS=(shs asap atepp lmd-clean vienna4x22 pianovam)
# Validate all inputs before spending time on the first detector run.
for index in "${!DATASETS[@]}"; do
  python "$ROOT/scripts/reproduce_detector_benchmark.py" \
    "${DATASETS[$index]}" "${DATASET_ROOTS[$index]}" "$DETECTOR" "$OUTPUT/raw/${DATASETS[$index]}" \
    --check-only
done
if [[ "$CHECK_ONLY" -eq 1 ]]; then
  echo "All six dataset cohorts and detector checkpoint are ready. No inference was run."
  exit 0
fi
for index in "${!DATASETS[@]}"; do
  dataset=${DATASETS[$index]}
  python "$ROOT/scripts/reproduce_detector_benchmark.py" \
    "$dataset" "${DATASET_ROOTS[$index]}" "$DETECTOR" "$OUTPUT/raw/$dataset" \
    --workers "$WORKERS"
done
python "$ROOT/scripts/summarize_detector_benchmarks.py" "$OUTPUT/raw" "$OUTPUT/tables"

echo
echo "Done. Six-dataset tables are in: $OUTPUT/tables"
