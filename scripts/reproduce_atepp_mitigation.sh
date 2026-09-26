#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}

cd "$ROOT"
"$PYTHON_BIN" scripts/fetch_reproduction_assets.py atepp-mitigation-v1.9
"$PYTHON_BIN" scripts/reproduce_atepp_mitigation.py \
  reproduction/atepp_mitigation_v1.9 \
  reproduced/atepp_mitigation

echo "ATEPP v1.9 mitigation table written to reproduced/atepp_mitigation/."
