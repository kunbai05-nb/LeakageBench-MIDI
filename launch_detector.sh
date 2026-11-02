#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! python3 -c 'import tkinter' 2>/dev/null; then
    echo 'Tkinter is missing. On Ubuntu/WSL, run: sudo apt install python3-tk'
    exit 1
fi
exec python3 scripts/detector_gui.py
