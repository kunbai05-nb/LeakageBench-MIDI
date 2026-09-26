#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=$(command -v python || command -v python3)
OUTPUT=${1:-$ROOT/reproduced/multidataset_deployment}

cd "$ROOT"
"$PYTHON_BIN" scripts/fetch_reproduction_assets.py multidataset-deployment
"$PYTHON_BIN" scripts/summarize_multidataset_deployment.py \
  reproduction/multidataset_deployment "$OUTPUT"
echo "Done. The deployment table is in: $OUTPUT"
