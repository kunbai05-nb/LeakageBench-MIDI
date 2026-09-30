#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

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
for index in "${!DATASETS[@]}"; do
  dataset=${DATASETS[$index]}
  python "$ROOT/scripts/reproduce_detector_benchmark.py" \
    "$dataset" "${DATASET_ROOTS[$index]}" "$DETECTOR" "$OUTPUT/raw/$dataset" \
    --workers "$WORKERS"
done
python "$ROOT/scripts/summarize_detector_benchmarks.py" "$OUTPUT/raw" "$OUTPUT/tables"

echo
echo "Done. Six-dataset tables are in: $OUTPUT/tables"
