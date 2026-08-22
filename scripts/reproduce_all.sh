#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1

output_dir="${1:-./_reproduced_release}"
if [[ -n "${PYTHON:-}" ]]; then
  python_bin="$PYTHON"
elif [[ -x .venv/bin/python ]]; then
  python_bin=.venv/bin/python
else
  python_bin=python3
fi

"$python_bin" scripts/reproduce_paper_statistics.py \
  --output "${output_dir}/paper_statistics"
"$python_bin" scripts/reproduce_detector_statistics.py \
  --output "${output_dir}/detector_statistics.json" --verify
"$python_bin" scripts/reproduce_public_results.py --verify
"$python_bin" scripts/reproduce_public_results.py \
  --output "${output_dir}/typeset_artifacts"
"$python_bin" -m pytest -q -s -p no:cacheprovider

printf 'Reproduction bundle written to %s\n' "${output_dir}"
