#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=$(command -v python || command -v python3)
OUTPUT=${1:-$ROOT/reproduced/fast_vs_v19}

cd "$ROOT"
"$PYTHON_BIN" scripts/fetch_reproduction_assets.py fast-vs-v19-fixed
"$PYTHON_BIN" scripts/reproduce_fast_vs_v19_from_checkpoints.py \
  reproduction/fast_vs_v19_fixed "$OUTPUT"
echo "Done. The comparison table is in: $OUTPUT"
