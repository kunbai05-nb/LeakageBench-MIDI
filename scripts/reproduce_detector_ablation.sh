#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT=${1:-$ROOT/reproduced/detector_ablation}
ARCHIVE="$OUTPUT/cache/detector-ablation-v1.9-raw.tar.gz"
URL="https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.9.0/detector-ablation-v1.9-raw.tar.gz"
EXPECTED_SHA256="80d1e138f2357f1b65607d2ebb26ece543e990eae3bca25a2ee9ef01a12c5c5b"

PYTHON_BIN=$(command -v python || command -v python3 || true)
[[ -n "$PYTHON_BIN" ]] || { echo "ERROR: Python is not installed." >&2; exit 1; }
for command in curl tar sha256sum; do
  command -v "$command" >/dev/null || { echo "ERROR: $command is not installed." >&2; exit 1; }
done

mkdir -p "$OUTPUT/cache" "$OUTPUT/tables"
if [[ ! -f "$ARCHIVE" ]]; then
  echo "Downloading the frozen detector-ablation results..."
  curl -L --fail --retry 3 -o "$ARCHIVE" "$URL"
fi

ACTUAL_SHA256=$(sha256sum "$ARCHIVE" | awk '{print $1}')
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "ERROR: downloaded bundle checksum does not match the frozen release." >&2
  rm -f "$ARCHIVE"
  exit 1
fi

mkdir -p "$OUTPUT/raw"
tar -xzf "$ARCHIVE" --strip-components=1 -C "$OUTPUT/raw"

"$PYTHON_BIN" "$ROOT/scripts/summarize_detector_ablation.py" \
  "$OUTPUT/raw" \
  "$ROOT/reproduction/detector_ablation_v1.9/EXPECTED_RESULTS.json" \
  "$OUTPUT/tables"

echo
echo "PASS. Tables 1, 2, 4, and 5 are in: $OUTPUT/tables"
