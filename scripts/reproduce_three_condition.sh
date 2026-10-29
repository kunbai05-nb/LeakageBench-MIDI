#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# Compatibility name; the current controlled-exposure study has four conditions.
exec bash "$ROOT/scripts/reproduce_four_condition.sh" "$@"
