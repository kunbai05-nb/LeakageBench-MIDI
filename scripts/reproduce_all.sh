#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

output_dir="${1:-./_reproduced_release}"

python scripts/reproduce_paper_statistics.py \
  --output "${output_dir}/paper_statistics"
python scripts/reproduce_public_results.py --verify
python scripts/reproduce_public_results.py \
  --output "${output_dir}/typeset_artifacts"
python -m pytest -q -s -p no:cacheprovider

printf 'Reproduction bundle written to %s\n' "${output_dir}"
