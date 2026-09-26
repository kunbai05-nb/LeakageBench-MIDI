#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}

cd "$ROOT"
"$PYTHON_BIN" scripts/fetch_reproduction_assets.py atepp-mitigation-v1.9
PYTHON="$PYTHON_BIN" bash reproduction/atepp_mitigation_v1.9/reproduce.sh

echo "ATEPP v1.9 mitigation table written to reproduction/atepp_mitigation_v1.9/reproduced_results/."
