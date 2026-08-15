#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."; export PYTHONPATH=.
if [[ -n "${PYTHON:-}" ]]; then python_bin="$PYTHON"; elif [[ -x .venv/bin/python ]]; then python_bin=.venv/bin/python; else python_bin=python3; fi
"$python_bin" scripts/reproduce_release_tables.py
