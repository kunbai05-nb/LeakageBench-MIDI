#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  echo "Usage: bash scripts/reproduce_atepp_mitigation.sh"
  echo "Downloads the checkpoint bundle and evaluates all five conditions and three seeds."
  exit 0
fi
[[ $# -eq 0 ]] || { echo "ERROR: this command takes no arguments." >&2; exit 2; }
PYTHON_BIN=${PYTHON_BIN:-python3}

cd "$ROOT"
"$PYTHON_BIN" scripts/fetch_reproduction_assets.py atepp-mitigation-v1.9
"$PYTHON_BIN" scripts/fetch_reproduction_assets.py atepp-mitigation-v1.9-protocol-fix
(cd reproduction/atepp_mitigation_v1.9 && sha256sum -c SHA256SUMS)
PYTHON="$PYTHON_BIN" bash reproduction/atepp_mitigation_v1.9/reproduce.sh

echo "ATEPP v1.9 mitigation table written to reproduction/atepp_mitigation_v1.9/reproduced_results/."
