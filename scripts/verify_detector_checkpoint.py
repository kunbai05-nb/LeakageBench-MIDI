#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from leakagebench_midi.detector import load_detector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("detector_dir", type=Path)
    args = parser.parse_args()
    config, models = load_detector(args.detector_dir)
    print(
        json.dumps(
            {
                "status": "PASS",
                "detector": config["detector_id"],
                "models": [type(model).__name__ for model in models],
                "ensemble_size": len(models),
                "features": config["feature_count"],
                "threshold": config["decision_threshold"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
