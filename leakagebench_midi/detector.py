from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np

from .alignment import (
    AlignmentConfig,
    alignment_feature_matrix,
    extract_alignment_bundle,
)
from .content import (
    compact_candidate_ranks,
    extract_feature_bundle,
    faiss_candidate_ranks,
    structural_pair_feature_matrix,
)


RETRIEVAL_VIEWS = ("bass", "harmony", "motif", "interval_hist", "duration_hist")
VERIFIER_RANK_VIEWS = ("bass", "harmony", "motif")


@dataclass(frozen=True)
class DetectorConfig:
    top_k: int = 100
    seed: int = 20260911
    alignment: AlignmentConfig = AlignmentConfig(zero_shift=True)


class Components:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int32)

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[item] != item:
            parent = int(self.parent[item])
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left

    def labels(self) -> np.ndarray:
        roots = np.asarray([self.find(i) for i in range(len(self.parent))])
        return np.unique(roots, return_inverse=True)[1].astype(np.int32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_detector(directory: str | Path) -> tuple[dict, object]:
    root = Path(directory).resolve()
    metadata = json.loads((root / "MODEL_CONFIG.json").read_text(encoding="utf-8"))
    if metadata.get("detector_id") != "same-work-detector-v1.7":
        raise ValueError("expected Same-Work Detector v1.7")
    model_path = (root / metadata["model_file"]).resolve()
    if not model_path.is_relative_to(root) or not model_path.is_file():
        raise ValueError("model file must be inside the detector directory")
    if sha256(model_path) != metadata["model_sha256"]:
        raise ValueError("model checksum mismatch")
    model = joblib.load(model_path)
    if getattr(model, "n_features_in_", None) != 57:
        raise ValueError("detector feature dimension mismatch")
    return metadata, model


def _rank_features(
    ranks: np.ndarray, signals: tuple[str, ...], k: int
) -> tuple[np.ndarray, list[str]]:
    indices = [signals.index(name) for name in VERIFIER_RANK_VIEWS]
    selected = np.asarray(ranks[:, indices], dtype=np.float32)
    clipped = np.minimum(selected, k + 1)
    reciprocal = 1.0 / np.min(clipped, axis=2)
    mutual = np.max(clipped, axis=2) <= k
    values = np.stack((reciprocal, mutual), axis=2).reshape(len(ranks), -1)
    names = [
        f"{view}_{suffix}"
        for view in VERIFIER_RANK_VIEWS
        for suffix in ("reciprocal_best_rank", "mutual")
    ]
    return values.astype(np.float32), names


def assemble_features(
    structural: np.ndarray,
    structural_names: list[str],
    ranks: np.ndarray,
    rank_signals: tuple[str, ...],
    aligned: np.ndarray,
    alignment_names: list[str],
    k: int = 100,
) -> tuple[np.ndarray, list[str]]:
    ranked, rank_names = _rank_features(ranks, rank_signals, k)
    values = np.column_stack((structural, ranked, aligned)).astype(np.float32)
    names = structural_names + rank_names + alignment_names
    at = {name: index for index, name in enumerate(names)}
    agreement = values[
        :,
        [
            at[f"align_{voice}_agreement"]
            for voice in ("melody", "bass", "rhythm", "harmony")
        ],
    ]
    best = values[:, at["align_best_score_per_match"]]
    engineered = np.column_stack(
        (
            agreement.min(axis=1),
            agreement.mean(axis=1),
            best * values[:, at["align_coverage_hmean"]],
            best * (1 - values[:, at["align_gap_fraction"]]),
            ranked[:, 1::2].sum(axis=1),
            ranked[:, ::2].max(axis=1),
        )
    )
    names += [
        "robust_alignment_agreement_min",
        "robust_alignment_agreement_mean",
        "robust_alignment_score_coverage",
        "robust_alignment_score_gap_adjusted",
        "retrieval_mutual_support",
        "retrieval_best_support",
    ]
    features = np.nan_to_num(
        np.column_stack((values, engineered)), nan=0.0, posinf=1.0, neginf=0.0
    ).astype(np.float32)
    if features.shape[1] != 57:
        raise RuntimeError(f"expected 57 features, got {features.shape[1]}")
    return features, names


def extract_pair_features(
    paths: list[str | Path],
    workers: int = 1,
    backend: str = "faiss",
    config: DetectorConfig = DetectorConfig(),
) -> dict:
    paths = [Path(path).resolve() for path in paths]
    bundle, content_failures = extract_feature_bundle(paths, workers=workers)
    sequences, alignment_failures = extract_alignment_bundle(
        paths, config.alignment, workers
    )
    valid = np.flatnonzero(bundle["valid"])
    matrices = {name: bundle[name] for name in RETRIEVAL_VIEWS}
    if backend == "exact":
        compact = compact_candidate_ranks(
            matrices, valid, config.top_k, workers=workers
        )
        diagnostics = {"backend": "exact", "candidate_pairs": len(compact["pairs"])}
    elif backend == "faiss":
        compact, diagnostics = faiss_candidate_ranks(
            matrices,
            valid,
            config.top_k,
            threads=max(1, workers // len(RETRIEVAL_VIEWS)),
            signal_workers=min(max(1, workers), len(RETRIEVAL_VIEWS)),
            seed=config.seed,
        )
    else:
        raise ValueError(f"unknown candidate backend: {backend}")
    pairs = compact["pairs"].astype(np.int64)
    structural, structural_names = structural_pair_feature_matrix(bundle, pairs)
    aligned, alignment_names = alignment_feature_matrix(
        sequences, pairs, config.alignment, workers
    )
    features, feature_names = assemble_features(
        structural,
        structural_names,
        compact["ranks"],
        tuple(compact["signals"]),
        aligned,
        alignment_names,
        config.top_k,
    )
    return {
        "pairs": pairs,
        "features": features,
        "feature_names": feature_names,
        "failures": content_failures + alignment_failures,
        "candidate_diagnostics": diagnostics,
    }


def detect(
    paths: list[str | Path],
    detector_dir: str | Path,
    workers: int = 1,
    backend: str = "faiss",
    threshold: float | None = None,
) -> dict:
    metadata, model = load_detector(detector_dir)
    extracted = extract_pair_features(paths, workers, backend)
    scores = model.predict_proba(extracted["features"])[:, 1]
    cutoff = float(metadata["decision_threshold"] if threshold is None else threshold)
    selected = np.flatnonzero(scores >= cutoff).astype(np.int64)
    components = Components(len(paths))
    for index in selected:
        components.union(*map(int, extracted["pairs"][index]))
    return {
        **extracted,
        "scores": scores,
        "threshold": cutoff,
        "selected": selected,
        "component_labels": components.labels(),
    }


def model_config(
    model_file: str, model_hash: str, threshold: float, feature_names: list[str]
) -> dict:
    return {
        "format_version": "1.7",
        "detector_id": "same-work-detector-v1.7",
        "model_type": "HistGradientBoostingClassifier",
        "model_file": model_file,
        "model_sha256": model_hash,
        "feature_count": 57,
        "feature_names": feature_names,
        "retrieval_views": list(RETRIEVAL_VIEWS),
        "verifier_rank_views": list(VERIFIER_RANK_VIEWS),
        "top_k": 100,
        "candidate_gate": "one-way support in at least one retained view",
        "alignment": asdict(AlignmentConfig(zero_shift=True)),
        "decision_threshold": threshold,
        "seed": 20260911,
    }


__all__ = [
    "Components",
    "DetectorConfig",
    "RETRIEVAL_VIEWS",
    "VERIFIER_RANK_VIEWS",
    "assemble_features",
    "detect",
    "extract_pair_features",
    "load_detector",
    "model_config",
    "sha256",
]
