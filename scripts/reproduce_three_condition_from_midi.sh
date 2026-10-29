#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=${LEAKAGEBENCH_DATA_DIR:-$ROOT/data}
RELEASE_BASE="https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  echo "Usage: bash scripts/reproduce_three_condition.sh [LMD_SOURCE] [OUTPUT_DIR]"
  echo "Default LMD source: $DATA_ROOT/lmd_full.tar.gz"
  echo "       bash scripts/reproduce_three_condition.sh --check-only"
}

check_environment() {
  command -v python >/dev/null || { echo "ERROR: Python is not installed." >&2; exit 1; }
  command -v curl >/dev/null || { echo "ERROR: curl is not installed." >&2; exit 1; }
  command -v tar >/dev/null || { echo "ERROR: tar is not installed." >&2; exit 1; }
  command -v sha256sum >/dev/null || { echo "ERROR: sha256sum is not installed." >&2; exit 1; }
  python -c "import numpy, torch; import leakagebench_midi" >/dev/null || {
    echo "ERROR: dependencies are missing. Run: pip install -r requirements.txt && pip install -e '.[scale]'" >&2
    exit 1
  }
}

if [[ ${1:-} == "--check-only" ]]; then
  check_environment
  python -m py_compile "$ROOT/scripts/reproduce_three_condition_table.py"
  python -m py_compile "$ROOT/scripts/summarize_detector_benchmarks.py"
  python "$ROOT/scripts/prepare_lmd.py" --help >/dev/null
  python "$ROOT/scripts/evaluate_checkpoint.py" --help >/dev/null
  echo "Environment and reproduction entry points are ready."
  exit 0
fi

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

check_environment
LMD_SOURCE=$(realpath -m "${1:-$DATA_ROOT/lmd_full.tar.gz}")
OUTPUT=$(realpath -m "${2:-$ROOT/reproduced/three_condition}")
[[ -e "$LMD_SOURCE" ]] || {
  echo "ERROR: LMD source not found: $LMD_SOURCE" >&2
  echo "Download lmd_full.tar.gz from https://colinraffel.com/projects/lmd/ into $DATA_ROOT, or supply an existing archive/directory as the first argument." >&2
  exit 1
}
python -c "import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'A BF16-capable CUDA GPU is required'"
mkdir -p "$OUTPUT/downloads" "$OUTPUT/checkpoints" "$OUTPUT/tables"

python "$ROOT/scripts/fetch_reproduction_assets.py" three-condition-source-specs

download() {
  local tag=$1 name=$2 expected=$3 target="$OUTPUT/downloads/$2"
  local url="$RELEASE_BASE/$tag/$name" staging
  if [[ -f "$target" ]] && echo "$expected  $target" | sha256sum --check --status; then
    return
  fi
  if [[ -f "$target" ]]; then
    echo "Resuming incomplete download: $name"
    if curl -fL --retry 5 --retry-delay 3 -C - "$url" -o "$target" &&
       echo "$expected  $target" | sha256sum --check --status; then
      return
    fi
  fi

  # The shared downloader falls back to authenticated `gh release download`
  # when a private GitHub release returns 404 to an anonymous request.
  staging=$(mktemp -d "$OUTPUT/downloads/.${name}.XXXXXX")
  echo "Downloading $name"
  if ! python -c 'from pathlib import Path; from scripts.fetch_reproduction_assets import download; import sys; download(sys.argv[1], Path(sys.argv[2]))' \
      "$url" "$staging/archive"; then
    rmdir "$staging" 2>/dev/null || true
    echo "ERROR: download failed. For a private repository, install GitHub CLI and run: gh auth login" >&2
    exit 1
  fi
  if ! echo "$expected  $staging/archive" | sha256sum --check --status; then
    echo "ERROR: checksum mismatch after download: $name" >&2
    exit 1
  fi
  mv -f "$staging/archive" "$target"
  rmdir "$staging"
}

MODEL_ASSETS=(
  "lmd-transformer-s.tar.gz|4b62a1a573490a9b69f6a888eb68de2c016b5cc6179d6b1b49ca87e822192af8"
  "lmd-transformer-m.tar.gz|3bd39c2feeaa9dd78d4f6df8a14388ca2340750a211a339c63298dd6864db710"
  "lmd-transformer-l.tar.gz|6a315de785004a50ff8022e80fbe71c28d38c2a5ad532d3a1aae6e8d04027042"
  "lmd-tcn.tar.gz|a1a846b38471c4fa3805c75f51e3931d0700e6ace3ea3d6ccbe00118d5b1474a"
  "conditional-vae.tar.gz|69aa6e7ab3841b224d2ee12cad9ff6d828b52f0b32ae16ce738309bfd8983ddd"
  "latent-diffusion-and-neutral-encoders.tar.gz|8a94e6cd8e45cb4f3d86f99e959fe60eca0ab006e560c4ec710250e4c77d5eb1"
  "midigpt-clean.tar.gz|a25aa9aaa39c1b0de519c37d9b0c1b9a90ba1fc12fcc8ae285180adf9cefbf5e"
  "midigpt-unrelated_donor.tar.gz|8fee29b5f3e107ae83f187c85e0a616a610207a0869b7f864f0b7cd6b42dbd92"
  "midigpt-same_family_donor.tar.gz|6db187f5ffdfdcc6bd58b9ef6830e9c0f3fe99cc336d544872703d5622f1575c"
  "lstm-clean.tar.gz|e4ea21ff7aab33ffc098453d17bbac9d482277d99590811999ac319c71cef50c"
  "lstm-unrelated_donor.tar.gz|f12d972ea15cdcac4fb66c76dadbdee7f24990a735318e9b3f85445b23cac3cb"
  "lstm-same_family_donor.tar.gz|9c7b13dc27339c687241d4ae1dd6b9fe6438981b70e48582545155cc804e274d"
)

for item in "${MODEL_ASSETS[@]}"; do
  IFS='|' read -r name digest <<< "$item"
  download v1.3.0 "$name" "$digest"
  marker="$OUTPUT/checkpoints/.${name}.extracted"
  if [[ ! -f "$marker" ]]; then
    tar -xzf "$OUTPUT/downloads/$name" -C "$OUTPUT/checkpoints"
    touch "$marker"
  fi
done

if [[ ! -f "$OUTPUT/prepared_lmd/PREPARATION_SUMMARY.json" ]]; then
  echo "Preparing frozen LMD streams"
  python "$ROOT/scripts/prepare_lmd.py" "$LMD_SOURCE" "$OUTPUT/prepared_lmd" --evaluation-only
fi

echo "Reproducing the multi-model three-condition table"
python "$ROOT/scripts/reproduce_three_condition_table.py" \
  "$OUTPUT/checkpoints" "$OUTPUT/prepared_lmd" "$OUTPUT/three_condition" --device cuda
cp "$OUTPUT/three_condition/tables/three_condition_results.csv" "$OUTPUT/tables/"
cp "$OUTPUT/three_condition/tables/three_condition_results.md" "$OUTPUT/tables/"
cp "$OUTPUT/three_condition/tables/three_condition_results.json" "$OUTPUT/tables/"

echo
echo "Done. The three-condition table is in: $OUTPUT/tables"
