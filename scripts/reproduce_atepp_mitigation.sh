#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# Compatibility entry point: the current ATEPP study includes both models.
exec bash "$ROOT/scripts/reproduce_cleaning_mitigation.sh" --dataset atepp "$@"
