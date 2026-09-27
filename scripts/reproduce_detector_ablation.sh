#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
THREADS=${LEAKAGEBENCH_THREADS:-4}
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
  MKL_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"

PYTHON_BIN=$(command -v python || command -v python3 || true)
[[ -n "$PYTHON_BIN" ]] || { echo "ERROR: Python is not installed." >&2; exit 1; }
"$PYTHON_BIN" -c "import numpy, sklearn; import leakagebench_midi" >/dev/null || {
  echo "ERROR: dependencies are missing. Run: pip install -r requirements.txt && pip install -e '.[scale]'" >&2
  exit 1
}

if [[ "${1:-}" != "--from-midi" ]]; then
  if [[ $# -gt 1 ]]; then
    echo "Usage: bash scripts/reproduce_detector_ablation.sh [OUTPUT_DIR]" >&2
    echo "   or: bash scripts/reproduce_detector_ablation.sh --from-midi LMD_ROOT ATEPP_ROOT ASAP_ROOT [OUTPUT_DIR] [WORKERS]" >&2
    exit 2
  fi
  OUTPUT=${1:-$ROOT/reproduced/detector_ablation}
  INPUT="$ROOT/reproduction/detector_ablation_features/input"
  "$PYTHON_BIN" "$ROOT/scripts/fetch_reproduction_assets.py" detector-ablation-features-v1.9
else
  shift
  if [[ $# -lt 3 || $# -gt 5 ]]; then
    echo "Usage: bash scripts/reproduce_detector_ablation.sh --from-midi LMD_ROOT ATEPP_ROOT ASAP_ROOT [OUTPUT_DIR] [WORKERS]" >&2
    exit 2
  fi
  for command in realpath curl sha256sum; do
    command -v "$command" >/dev/null || { echo "ERROR: $command is not installed." >&2; exit 1; }
  done
  LMD=$(realpath "$1")
  ATEPP=$(realpath "$2")
  ASAP=$(realpath "$3")
  OUTPUT=$(realpath -m "${4:-$ROOT/reproduced/detector_ablation_from_midi}")
  WORKERS=${5:-8}
  [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: WORKERS must be a positive integer." >&2; exit 1; }
  for path in "$LMD" "$ATEPP" "$ASAP"; do
    [[ -d "$path" ]] || { echo "ERROR: dataset directory not found: $path" >&2; exit 1; }
  done

  INPUT="$OUTPUT/input"
  mkdir -p "$INPUT/train" "$INPUT/test"
  TRAIN_SPEC="$ROOT/reproduction/detector_training"
  for name in fit_00_14 fit_14_28 fit_28_42 fit_42_56 fit_56_70 calibration_70_85; do
    target="$INPUT/train/$name.npz"
    if [[ ! -f "$target" ]]; then
      "$PYTHON_BIN" "$ROOT/scripts/build_detector_candidate_features.py" \
        "$TRAIN_SPEC/$name.csv.gz" "$LMD" "$target" --workers "$WORKERS" --backend exact
    fi
  done

  declare -A DATA_ROOTS=(
    [lmd-clean]="$LMD"
    [atepp]="$ATEPP"
    [asap]="$ASAP"
  )
  for name in lmd-clean atepp asap; do
    target="$INPUT/test/$name.npz"
    if [[ ! -f "$target" ]]; then
      "$PYTHON_BIN" "$ROOT/scripts/build_detector_candidate_features.py" \
        "$ROOT/reproduction/detector_benchmark/$name.csv.gz" \
        "${DATA_ROOTS[$name]}" "$target" --workers "$WORKERS" --backend exact --ablation
    fi
  done

  cp "$TRAIN_SPEC/calibration_70_85.csv.gz" "$INPUT/calibration_70_85.csv.gz"
  if [[ ! -f "$INPUT/MAESTRO-validation.npz" ]]; then
    echo "Downloading the frozen MAESTRO view-selection input..."
    curl -L --fail --retry 3 \
      -o "$INPUT/MAESTRO-validation.npz" \
      "https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.9.0/MAESTRO-validation.npz"
  fi
  MAESTRO_SHA256="3ad63246fd8614826e5ac13ffc14df9fe2e5d1ee8f17d8e61670e9efd3b4316a"
  ACTUAL_MAESTRO_SHA256=$(sha256sum "$INPUT/MAESTRO-validation.npz" | awk '{print $1}')
  [[ "$ACTUAL_MAESTRO_SHA256" == "$MAESTRO_SHA256" ]] || {
    echo "ERROR: MAESTRO validation input checksum does not match the frozen release." >&2
    exit 1
  }
fi

mkdir -p "$OUTPUT/raw" "$OUTPUT/tables"
"$PYTHON_BIN" "$ROOT/scripts/run_detector_ablation.py" "$INPUT" "$OUTPUT/raw"
"$PYTHON_BIN" "$ROOT/scripts/summarize_detector_ablation.py" "$OUTPUT/raw" "$OUTPUT/tables"

echo
echo "Done. Tables 1, 2, 4, and 5 are in: $OUTPUT/tables"
