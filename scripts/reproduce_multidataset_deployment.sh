#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  echo "Usage: bash scripts/reproduce_multidataset_deployment.sh [OUTPUT_DIR]"
  exit 0
fi
[[ $# -le 1 ]] || { echo "ERROR: at most one output directory is accepted." >&2; exit 2; }
PYTHON_BIN=$(command -v python || command -v python3)
OUTPUT=${1:-$ROOT/reproduced/multidataset_deployment}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

cd "$ROOT"
"$PYTHON_BIN" scripts/fetch_reproduction_assets.py multidataset-deployment
(cd reproduction/multidataset_deployment && sha256sum -c SHA256SUMS)
"$PYTHON_BIN" scripts/summarize_multidataset_deployment.py \
  reproduction/multidataset_deployment "$OUTPUT"
echo "Done. The deployment table is in: $OUTPUT"
