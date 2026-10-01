#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  echo "Usage: bash scripts/reproduce_fast_vs_v19.sh [OUTPUT_DIR]"
  exit 0
fi
[[ $# -le 1 ]] || { echo "ERROR: at most one output directory is accepted." >&2; exit 2; }
PYTHON_BIN=$(command -v python || command -v python3)
OUTPUT=${1:-$ROOT/reproduced/fast_vs_v19}

cd "$ROOT"
"$PYTHON_BIN" scripts/fetch_reproduction_assets.py fast-vs-v19-fixed
(cd reproduction/fast_vs_v19_fixed && sha256sum -c SHA256SUMS)
"$PYTHON_BIN" scripts/reproduce_fast_vs_v19_from_checkpoints.py \
  reproduction/fast_vs_v19_fixed "$OUTPUT"
echo "Done. The comparison table is in: $OUTPUT"
