#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -n "${PYTHON:-}" ]]; then
  python_bin="$PYTHON"
elif [[ -x .venv/bin/python ]]; then
  python_bin=.venv/bin/python
else
  python_bin=python3
fi

PYTHONDONTWRITEBYTECODE=1 "$python_bin" -m pytest -q -p no:cacheprovider
