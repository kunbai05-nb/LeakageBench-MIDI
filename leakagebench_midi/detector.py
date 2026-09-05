from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from .local_evidence_inference import Components, LocalEvidenceDetector


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_detector(directory: str | Path) -> tuple[dict, list]:
    root = Path(directory).resolve()
    config_path = root / "MODEL_CONFIG.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text())
    models = []
    for artifact in config["ensemble_models"]:
        path = (root / artifact["file"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("detector artifact must be inside the model directory")
        expected = artifact.get("sha256")
        if expected and sha256(path) != expected:
            raise RuntimeError(f"model hash mismatch: {path.name}")
        models.append(joblib.load(path))
    return config, models


def _predict_probability(models, features: np.ndarray) -> np.ndarray:
    return np.stack([model.predict_proba(features)[:, 1] for model in models]).mean(
        axis=0
    )


def detect(
    paths: list[str | Path],
    detector_dir: str | Path,
    workers: int = 1,
    backend: str = "faiss",
    chunk_size: int = 20_000,
) -> dict:
    """Detect same-work relations and return accepted component labels.

    Candidate generation never uses family labels. The released detector uses
    nine structural retrieval views, local ordered evidence, and a calibrated
    five-model ensemble.
    """
    config, _ = load_detector(detector_dir)
    if "base_feature_names" not in config:
        raise ValueError("the public detector requires a v1.3 model bundle")
    runner = LocalEvidenceDetector(detector_dir)
    return runner.detect_pairs(paths, workers, backend, chunk_size)


__all__ = [
    "Components",
    "detect",
    "load_detector",
    "sha256",
]
