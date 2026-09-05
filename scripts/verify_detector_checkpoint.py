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
    maximum_component_size = config["component_max_size"]
    print(
        json.dumps(
            {
                "status": "PASS",
                "models": len(models),
                "features": len(config["feature_names"]),
                "threshold": config["threshold"],
                "maximum_component_size": maximum_component_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
