#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


FIT_NAMES = (
    "fit_00_14",
    "fit_14_28",
    "fit_28_42",
    "fit_42_56",
    "fit_56_70",
)
CALIBRATION = "calibration_70_85"


def run(command: list[str], log: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and train Same-Work Detector v1.9.")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--spec-dir", type=Path, required=True)
    parser.add_argument("--midi-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--workers-per-job", type=int, default=28)
    args = parser.parse_args()

    repo = args.repo.resolve()
    output = args.output.resolve()
    features = output / "candidate_features"
    logs = output / "logs"
    features.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    env.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )

    names = (*FIT_NAMES, CALIBRATION)

    def build(name: str) -> str:
        target = features / f"{name}.npz"
        if target.is_file() and target.with_suffix(".json").is_file():
            return f"{name}: cached"
        command = [
            sys.executable,
            str(repo / "scripts" / "build_detector_candidate_features.py"),
            str(args.spec_dir / f"{name}.csv.gz"),
            str(args.midi_root),
            str(target),
            "--workers",
            str(args.workers_per_job),
            "--backend",
            "exact",
        ]
        run(command, logs / f"{name}.log", env)
        return f"{name}: complete"

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(build, name): name for name in names}
        for future in as_completed(futures):
            print(future.result(), flush=True)

    detector = output / "same-work-detector-v1.9"
    train = [
        sys.executable,
            str(repo / "scripts" / "train_detector.py"),
        "--fit",
        *[str(features / f"{name}.npz") for name in FIT_NAMES],
        "--calibration",
        str(features / f"{CALIBRATION}.npz"),
        "--output",
        str(detector),
        "--precision-target",
        "0.97",
        "--negative-ratio",
        "50",
    ]
    run(train, logs / "train.log", env)
    summary = {
        "status": "TRAINED",
        "detector": str(detector),
        "fit_pools": list(FIT_NAMES),
        "calibration_pool": CALIBRATION,
        "parallel_jobs": args.parallel,
        "workers_per_job": args.workers_per_job,
    }
    (output / "PIPELINE_STATUS.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
