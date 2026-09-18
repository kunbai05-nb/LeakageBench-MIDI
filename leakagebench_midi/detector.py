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
    top_k: int = 50
    rank_k: int = 100
    seed: int = 20260911
    component_soft_size: int = 8
    minimum_bridge_edges: int = 3
    component_max_size: int = 50
    alignment: AlignmentConfig = AlignmentConfig(zero_shift=True)


class Components:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int32)
        self.size = np.ones(size, dtype=np.int32)
        self.members = [{index} for index in range(size)]

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[item] != item:
            parent = int(self.parent[item])
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int, limit: int | None = None) -> bool:
        left, right = self.find(left), self.find(right)
        if left == right:
            return True
        if limit is not None and self.size[left] + self.size[right] > limit:
            return False
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]
        self.members[left].update(self.members[right])
        self.members[right].clear()
        return True

    def cross_edges(
        self, left: int, right: int, adjacency: list[set[int]]
    ) -> int:
        left, right = self.find(left), self.find(right)
        if left == right:
            return 0
        if len(self.members[left]) > len(self.members[right]):
            left, right = right, left
        target = self.members[right]
        return sum(
            neighbor in target
            for item in self.members[left]
            for neighbor in adjacency[item]
        )

    def labels(self) -> np.ndarray:
        roots = np.asarray([self.find(i) for i in range(len(self.parent))])
        return np.unique(roots, return_inverse=True)[1].astype(np.int32)


def build_components(
    pairs: np.ndarray,
    scores: np.ndarray,
    selected: np.ndarray,
    file_count: int,
    maximum_size: int = 50,
    soft_size: int = 8,
    minimum_bridge_edges: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build guarded components from score-ordered pair decisions."""
    if maximum_size < 2:
        raise ValueError("maximum component size must be at least two")
    if not 2 <= soft_size <= maximum_size:
        raise ValueError("soft component size must lie within [2, maximum_size]")
    if minimum_bridge_edges < 2:
        raise ValueError("minimum bridge-edge count must be at least two")
    adjacency = [set() for _ in range(file_count)]
    for index in selected:
        left, right = map(int, pairs[int(index)])
        adjacency[left].add(right)
        adjacency[right].add(left)
    order = selected[np.argsort(scores[selected], kind="mergesort")[::-1]]
    components = Components(file_count)
    accepted, rejected_by_size, rejected_by_bridge = [], [], []
    for index in order:
        left, right = map(int, pairs[int(index)])
        left_root, right_root = components.find(left), components.find(right)
        if left_root == right_root:
            accepted.append(int(index))
            continue
        merged_size = int(
            components.size[left_root] + components.size[right_root]
        )
        if merged_size > maximum_size:
            rejected_by_size.append(int(index))
            continue
        if (
            merged_size > soft_size
            and components.cross_edges(left_root, right_root, adjacency)
            < minimum_bridge_edges
        ):
            rejected_by_bridge.append(int(index))
            continue
        components.union(left_root, right_root)
        accepted.append(int(index))
    return (
        np.asarray(accepted, dtype=np.int64),
        np.asarray(rejected_by_size, dtype=np.int64),
        np.asarray(rejected_by_bridge, dtype=np.int64),
        components.labels(),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_detector(directory: str | Path) -> tuple[dict, list[object]]:
    root = Path(directory).resolve()
    metadata = json.loads((root / "MODEL_CONFIG.json").read_text(encoding="utf-8"))
    if metadata.get("detector_id") != "same-work-detector-v1.8":
        raise ValueError("expected Same-Work Detector v1.8")
    files = metadata.get("model_files")
    hashes = metadata.get("model_sha256")
    if not isinstance(files, list) or not isinstance(hashes, list):
        raise ValueError("detector ensemble metadata is invalid")
    if len(files) != 3 or len(hashes) != len(files):
        raise ValueError("Same-Work Detector v1.8 requires three classifiers")
    models = []
    for filename, expected_hash in zip(files, hashes):
        model_path = (root / filename).resolve()
        if not model_path.is_relative_to(root) or not model_path.is_file():
            raise ValueError("model file must be inside the detector directory")
        if sha256(model_path) != expected_hash:
            raise ValueError("model checksum mismatch")
        model = joblib.load(model_path)
        if getattr(model, "n_features_in_", None) != 57:
            raise ValueError("detector feature dimension mismatch")
        models.append(model)
    return metadata, models


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


def candidate_union_mask(ranks: np.ndarray, k: int) -> np.ndarray:
    if ranks.ndim != 3 or ranks.shape[2] != 2:
        raise ValueError("candidate ranks must have shape (pairs, views, directions)")
    return np.min(ranks, axis=(1, 2)) <= k


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
            matrices, valid, config.rank_k, workers=workers
        )
        diagnostics = {"backend": "exact", "candidate_pairs": len(compact["pairs"])}
    elif backend == "faiss":
        compact, diagnostics = faiss_candidate_ranks(
            matrices,
            valid,
            config.rank_k,
            threads=max(1, workers // len(RETRIEVAL_VIEWS)),
            signal_workers=min(max(1, workers), len(RETRIEVAL_VIEWS)),
            seed=config.seed,
        )
    else:
        raise ValueError(f"unknown candidate backend: {backend}")
    candidate_mask = candidate_union_mask(compact["ranks"], config.top_k)
    pairs = compact["pairs"][candidate_mask].astype(np.int64)
    ranks = compact["ranks"][candidate_mask]
    structural, structural_names = structural_pair_feature_matrix(bundle, pairs)
    aligned, alignment_names = alignment_feature_matrix(
        sequences, pairs, config.alignment, workers
    )
    features, feature_names = assemble_features(
        structural,
        structural_names,
        ranks,
        tuple(compact["signals"]),
        aligned,
        alignment_names,
        config.rank_k,
    )
    diagnostics = {
        **diagnostics,
        "candidate_pairs": len(pairs),
        "candidate_top_k": config.top_k,
        "rank_feature_top_k": config.rank_k,
        "candidate_rule": "one-way union across five views",
    }
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
    metadata, models = load_detector(detector_dir)
    config = DetectorConfig(
        top_k=int(metadata["top_k"]),
        rank_k=int(metadata["rank_k"]),
        seed=int(metadata["seeds"][0]),
        component_soft_size=int(metadata["component_soft_size"]),
        minimum_bridge_edges=int(metadata["minimum_bridge_edges"]),
        component_max_size=int(metadata["component_max_size"]),
        alignment=AlignmentConfig(**metadata["alignment"]),
    )
    extracted = extract_pair_features(paths, workers, backend, config)
    scores = np.mean(
        [model.predict_proba(extracted["features"])[:, 1] for model in models],
        axis=0,
    )
    cutoff = float(metadata["decision_threshold"] if threshold is None else threshold)
    selected = np.flatnonzero(scores >= cutoff).astype(np.int64)
    maximum_size = int(metadata.get("component_max_size", 50))
    soft_size = int(metadata.get("component_soft_size", 8))
    minimum_bridge_edges = int(metadata.get("minimum_bridge_edges", 3))
    accepted, rejected_by_size, rejected_by_bridge, labels = build_components(
        extracted["pairs"],
        scores,
        selected,
        len(paths),
        maximum_size,
        soft_size,
        minimum_bridge_edges,
    )
    return {
        **extracted,
        "scores": scores,
        "threshold": cutoff,
        "selected": selected,
        "accepted": accepted,
        "rejected_by_size": rejected_by_size,
        "rejected_by_bridge": rejected_by_bridge,
        "component_labels": labels,
    }


def model_config(
    model_files: list[str],
    model_hashes: list[str],
    threshold: float,
    feature_names: list[str],
) -> dict:
    return {
        "format_version": "1.8",
        "detector_id": "same-work-detector-v1.8",
        "model_type": "mean ensemble of HistGradientBoostingClassifier",
        "model_files": model_files,
        "model_sha256": model_hashes,
        "ensemble_size": 3,
        "feature_count": 57,
        "feature_names": feature_names,
        "retrieval_views": list(RETRIEVAL_VIEWS),
        "verifier_rank_views": list(VERIFIER_RANK_VIEWS),
        "top_k": 50,
        "rank_k": 100,
        "candidate_gate": "one-way Top-50 union across five retrieval views",
        "minimum_supporting_views": 1,
        "component_soft_size": 8,
        "minimum_bridge_edges": 3,
        "component_max_size": 50,
        "alignment": asdict(AlignmentConfig(zero_shift=True)),
        "decision_threshold": threshold,
        "seeds": [20260911, 20260912, 20260913],
    }


__all__ = [
    "Components",
    "DetectorConfig",
    "RETRIEVAL_VIEWS",
    "VERIFIER_RANK_VIEWS",
    "assemble_features",
    "build_components",
    "candidate_union_mask",
    "detect",
    "extract_pair_features",
    "load_detector",
    "model_config",
    "sha256",
]
