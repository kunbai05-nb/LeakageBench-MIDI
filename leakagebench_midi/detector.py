from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from .alignment import (
    AlignmentConfig,
    alignment_feature_matrix,
    extract_alignment_bundle,
)
from .content import (
    CANDIDATE_SIGNALS,
    canonical_chroma,
    compact_candidate_ranks,
    compact_pair_feature_matrix,
    extract_count_bundle,
    extract_feature_bundle,
    faiss_mutual_candidate_ranks,
    tfidf_count_bundle,
)

SIGNALS = CANDIDATE_SIGNALS
extract_features = extract_feature_bundle
extract_count_features = extract_count_bundle
apply_tfidf = tfidf_count_bundle


class Components:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int32)
        self.size = np.ones(size, dtype=np.int32)

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[item] != item:
            parent = int(self.parent[item])
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int, limit: int) -> bool:
        left, right = self.find(left), self.find(right)
        if left == right:
            return True
        if self.size[left] + self.size[right] > limit:
            return False
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]
        return True

    def labels(self) -> np.ndarray:
        roots = np.asarray([self.find(i) for i in range(len(self.parent))])
        return np.unique(roots, return_inverse=True)[1].astype(np.int32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_detector(directory: str | Path) -> tuple[dict, list]:
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    models = []
    for artifact in config["ensemble_models"]:
        path = directory / artifact["artifact"]
        if sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"model hash mismatch: {path.name}")
        models.append(joblib.load(path))
    return config, models


def _predict_probability(models, features: np.ndarray) -> np.ndarray:
    return np.stack([model.predict_proba(features)[:, 1] for model in models]).mean(
        axis=0
    )


def _enrich(values: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    position = {name: index for index, name in enumerate(names)}
    agreements = values[
        :,
        [
            position[f"align_{name}_agreement"]
            for name in ("melody", "bass", "rhythm", "harmony")
        ],
    ]
    best = values[:, position["align_best_score_per_match"]]
    coverage = values[:, position["align_coverage_hmean"]]
    gap = values[:, position["align_gap_fraction"]]
    signals = (
        "melody",
        "bass",
        "rhythm",
        "harmony",
        "motif",
        "interval_hist",
        "duration_hist",
        "ioi_hist",
        "chroma",
    )
    mutual = np.column_stack(
        [values[:, position[f"{name}_mutual"]] for name in signals]
    ).sum(axis=1)
    reciprocal = np.column_stack(
        [values[:, position[f"{name}_reciprocal_best_rank"]] for name in signals]
    ).max(axis=1)
    engineered_names = [
        "robust_alignment_agreement_min",
        "robust_alignment_agreement_mean",
        "robust_alignment_score_coverage",
        "robust_alignment_score_gap_adjusted",
        "robust_structural_mutual_support",
        "robust_structural_candidate_support",
    ]
    engineered = np.column_stack(
        [
            agreements.min(axis=1),
            agreements.mean(axis=1),
            best * coverage,
            best * (1.0 - gap),
            mutual,
            reciprocal,
        ]
    ).astype(np.float32)
    result = np.column_stack((values, engineered)).astype(np.float32)
    return (
        np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6),
        names + engineered_names,
    )


def _candidate_pairs(
    bundle: dict, valid: np.ndarray, config: dict, workers: int, backend: str
) -> tuple[dict, dict]:
    matrices = {
        "chroma": canonical_chroma(bundle["chroma"]),
        **{name: bundle[name] for name in SIGNALS if name != "chroma"},
    }
    k = int(config["candidate_k"])
    minimum = int(config["minimum_mutual_structural_signals"])
    if backend == "faiss":
        compact, diagnostics = faiss_mutual_candidate_ranks(
            matrices,
            valid,
            k,
            minimum_mutual_signals=minimum,
            threads=max(1, workers // len(SIGNALS)),
            signal_workers=min(workers, len(SIGNALS)),
            seed=int(config["seed"]),
        )
        return compact, diagnostics
    compact = compact_candidate_ranks(matrices, valid, k, workers=workers)
    support = (np.max(compact["ranks"], axis=2) <= k).sum(axis=1)
    keep = support >= minimum
    compact = {
        **compact,
        "pairs": compact["pairs"][keep],
        "ranks": compact["ranks"][keep],
    }
    return compact, {"backend": "exact", "candidate_pairs": len(compact["pairs"])}


def detect(
    paths: list[str | Path],
    detector_dir: str | Path,
    workers: int = 1,
    backend: str = "exact",
    chunk_size: int = 100_000,
) -> dict:
    paths = [Path(path) for path in paths]
    config, models = load_detector(detector_dir)
    bundle, failures = extract_feature_bundle(paths, workers=workers)
    valid = np.flatnonzero(bundle["valid"])
    compact, diagnostics = _candidate_pairs(bundle, valid, config, workers, backend)
    pairs = compact["pairs"]
    if not len(pairs):
        return {
            "pairs": pairs,
            "scores": np.empty(0),
            "selected": np.empty(0, dtype=int),
            "accepted": np.empty(0, dtype=int),
            "rejected_by_size": np.empty(0, dtype=int),
            "component_labels": np.arange(len(paths)),
            "failures": failures,
            "candidate_diagnostics": diagnostics,
        }

    needed = np.unique(pairs.ravel())
    selected_sequences, alignment_failures = extract_alignment_bundle(
        [paths[int(index)] for index in needed],
        AlignmentConfig(**config["alignment_config"]),
        workers=workers,
    )
    sequences = [None] * len(paths)
    for index, sequence in zip(needed, selected_sequences):
        sequences[int(index)] = sequence
    scores = np.empty(len(pairs), dtype=np.float64)
    for start in range(0, len(pairs), chunk_size):
        end = min(start + chunk_size, len(pairs))
        part = {
            **compact,
            "pairs": compact["pairs"][start:end],
            "ranks": compact["ranks"][start:end],
        }
        global_values, global_names, local_pairs = compact_pair_feature_matrix(
            bundle, part, workers=workers
        )
        aligned, alignment_names = alignment_feature_matrix(
            sequences,
            local_pairs,
            AlignmentConfig(**config["alignment_config"]),
            workers,
        )
        enriched, names = _enrich(
            np.column_stack((global_values, aligned)), global_names + alignment_names
        )
        columns = [names.index(name) for name in config["feature_names"]]
        scores[start:end] = _predict_probability(models, enriched[:, columns])

    selected = np.flatnonzero(scores >= float(config["threshold"]))
    order = selected[np.argsort(scores[selected], kind="mergesort")[::-1]]
    components = Components(len(paths))
    accepted, rejected = [], []
    for index in order:
        left, right = map(int, pairs[index])
        if components.find(left) == components.find(right):
            accepted.append(int(index))
        elif components.union(left, right, int(config["maximum_component_size"])):
            accepted.append(int(index))
        else:
            rejected.append(int(index))
    for failure in alignment_failures:
        failure["index"] = int(needed[int(failure["index"])])
    return {
        "pairs": pairs,
        "scores": scores,
        "selected": selected,
        "accepted": np.asarray(accepted, dtype=np.int64),
        "rejected_by_size": np.asarray(rejected, dtype=np.int64),
        "component_labels": components.labels(),
        "failures": failures + alignment_failures,
        "candidate_diagnostics": diagnostics,
    }
