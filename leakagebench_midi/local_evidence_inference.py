from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path

import joblib
import numpy as np

from .alignment import AlignmentConfig, alignment_feature_matrix, extract_alignment_bundle
from .content import (
    canonical_chroma,
    compact_candidate_ranks,
    extract_feature_bundle,
    faiss_mutual_candidate_ranks,
    structural_pair_feature_matrix,
)
from .local_evidence import (
    EVIDENCE_NAMES,
    SIGNALS,
    LocalEvidenceConfig,
    assemble_base_features,
    evidence_feature_matrix,
    select_candidates,
)
from .local_evidence_batch import sparse_local_candidates


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


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class LocalEvidenceDetector:
    def __init__(self, directory: str | Path):
        root = Path(directory).resolve()
        self.root = root
        config_path = root / "MODEL_CONFIG.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        self.metadata = json.loads(config_path.read_text())
        options = self.metadata["method"].copy()
        known = {field.name for field in fields(LocalEvidenceConfig)}
        if set(options) != known:
            raise ValueError("unrecognized local-evidence configuration schema")
        options["window_segments"] = tuple(options["window_segments"])
        self.config = LocalEvidenceConfig(**options)
        self.alignment = AlignmentConfig(**self.metadata["alignment_config"])
        self.models = []
        for artifact in self.metadata["ensemble_models"]:
            path = (root / artifact["file"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("model artifact must be inside the model directory")
            if digest(path) != artifact["sha256"]:
                raise ValueError(f"model checksum mismatch: {path.name}")
            self.models.append(joblib.load(path))
        if len(self.models) != 5:
            raise ValueError("expected five ensemble members")
        expected = self.metadata["base_feature_names"] + list(EVIDENCE_NAMES)
        if self.metadata["feature_names"] != expected:
            raise ValueError("feature schema differs from the trained model")
        if any(model.n_features_in_ != len(expected) for model in self.models):
            raise ValueError("model feature dimension mismatch")

    def score(self, features: np.ndarray, names: list[str]) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if names != self.metadata["feature_names"]:
            raise ValueError("feature order differs from the trained model")
        if features.ndim != 2 or features.shape[1] != len(names):
            raise ValueError("invalid feature matrix shape")
        if not np.isfinite(features).all():
            raise ValueError("feature matrix contains non-finite values")
        if not len(features):
            return np.zeros(0, dtype=float)
        return np.mean(
            [model.predict_proba(features)[:, 1] for model in self.models], axis=0
        )

    def classify(self, scores: np.ndarray) -> np.ndarray:
        threshold = self.metadata.get("threshold")
        if threshold is None or self.metadata.get("calibration_status") != "CALIBRATED":
            raise RuntimeError("precision target was infeasible; this artifact is score-only")
        return np.asarray(scores) >= float(threshold)

    def _global_candidates(
        self, bundle: dict, valid: np.ndarray, workers: int, backend: str
    ) -> tuple[dict, dict]:
        matrices = {
            name: canonical_chroma(bundle[name]) if name == "chroma" else bundle[name]
            for name in SIGNALS
        }
        if backend == "faiss":
            return faiss_mutual_candidate_ranks(
                matrices,
                valid,
                self.config.candidate_k,
                minimum_mutual_signals=self.config.minimum_mutual_views,
                threads=max(1, workers // len(SIGNALS)),
                signal_workers=min(max(1, workers), len(SIGNALS)),
                seed=int(self.metadata.get("candidate_seed", 20260824)),
            )
        if backend != "exact":
            raise ValueError(f"unknown candidate backend: {backend}")
        if len(valid) > 5000:
            raise ValueError("exact backend is limited to 5,000 valid files")
        compact = compact_candidate_ranks(
            matrices, valid, self.config.candidate_k, workers=workers
        )
        support = (np.max(compact["ranks"], axis=2) <= self.config.candidate_k).sum(axis=1)
        keep = support >= self.config.minimum_mutual_views
        return (
            {**compact, "pairs": compact["pairs"][keep], "ranks": compact["ranks"][keep]},
            {
                "backend": "exact",
                "candidate_pairs": int(keep.sum()),
                "minimum_mutual_signals": self.config.minimum_mutual_views,
            },
        )

    def detect_pairs(
        self,
        paths: list[str | Path],
        workers: int = 1,
        backend: str = "faiss",
        chunk_size: int = 20_000,
    ) -> dict:
        if workers < 1 or chunk_size < 1:
            raise ValueError("workers and chunk_size must be positive")
        paths = [Path(path).resolve() for path in paths]
        if len(set(paths)) != len(paths):
            raise ValueError("MIDI paths must be unique")

        bundle, feature_failures = extract_feature_bundle(paths, workers=workers)
        sequences, alignment_failures = extract_alignment_bundle(
            paths, self.alignment, workers
        )
        valid = np.flatnonzero(bundle["valid"])
        if len(valid) < 2:
            return {
                "pairs": np.empty((0, 2), dtype=np.int64),
                "scores": np.empty(0, dtype=float),
                "selected": np.empty(0, dtype=np.int64),
                "accepted": np.empty(0, dtype=np.int64),
                "rejected_by_size": np.empty(0, dtype=np.int64),
                "component_labels": np.arange(len(paths), dtype=np.int32),
                "failures": feature_failures + alignment_failures,
                "candidate_diagnostics": {"backend": backend, "candidate_pairs": 0},
            }

        compact, diagnostics = self._global_candidates(bundle, valid, workers, backend)
        global_pairs = compact["pairs"]
        global_ranks = compact["ranks"][:, [compact["signals"].index(name) for name in SIGNALS]]
        local_pairs, local_scores, local_stats = sparse_local_candidates(
            sequences, self.config, workers=workers
        )
        selection = select_candidates(
            global_pairs,
            global_ranks,
            len(paths),
            local_pairs,
            local_scores,
            self.config,
        )
        pairs = selection["proposed"]
        if not len(pairs):
            return {
                "pairs": pairs,
                "scores": np.empty(0, dtype=float),
                "selected": np.empty(0, dtype=np.int64),
                "accepted": np.empty(0, dtype=np.int64),
                "rejected_by_size": np.empty(0, dtype=np.int64),
                "component_labels": np.arange(len(paths), dtype=np.int32),
                "failures": feature_failures + alignment_failures,
                "candidate_diagnostics": {
                    **diagnostics,
                    **{key: value for key, value in selection.items() if key not in ("baseline", "proposed")},
                    "local_index": local_stats,
                },
            }

        universe = len(paths)
        lookup = {
            int(left) * universe + int(right): rank
            for (left, right), rank in zip(global_pairs, global_ranks)
        }
        fallback = np.full(
            (len(SIGNALS), 2), self.config.candidate_k + 1, dtype=np.int16
        )
        pair_ranks = np.asarray(
            [lookup.get(int(left) * universe + int(right), fallback) for left, right in pairs],
            dtype=np.int16,
        ).reshape(-1, len(SIGNALS), 2)
        scores = np.empty(len(pairs), dtype=float)
        for start in range(0, len(pairs), chunk_size):
            stop = min(start + chunk_size, len(pairs))
            block = pairs[start:stop]
            structural, structural_names = structural_pair_feature_matrix(bundle, block)
            aligned, alignment_names = alignment_feature_matrix(
                sequences, block, self.alignment, workers
            )
            base, base_names = assemble_base_features(
                structural,
                structural_names,
                pair_ranks[start:stop],
                aligned,
                self.config.candidate_k,
            )
            extra = evidence_feature_matrix(sequences, block, self.config, workers)
            scores[start:stop] = self.score(
                np.column_stack((base, extra)), base_names + list(EVIDENCE_NAMES)
            )

        selected = np.flatnonzero(self.classify(scores)).astype(np.int64)
        order = selected[np.argsort(scores[selected], kind="mergesort")[::-1]]
        components = Components(len(paths))
        limit = int(
            self.metadata.get(
                "component_max_size", self.metadata.get("maximum_component_size", 50)
            )
        )
        accepted, rejected = [], []
        for index in order:
            left, right = map(int, pairs[index])
            if components.find(left) == components.find(right):
                accepted.append(int(index))
            elif components.union(left, right, limit):
                accepted.append(int(index))
            else:
                rejected.append(int(index))
        return {
            "pairs": pairs,
            "scores": scores,
            "selected": selected,
            "accepted": np.asarray(accepted, dtype=np.int64),
            "rejected_by_size": np.asarray(rejected, dtype=np.int64),
            "component_labels": components.labels(),
            "failures": feature_failures + alignment_failures,
            "candidate_diagnostics": {
                **diagnostics,
                **{key: value for key, value in selection.items() if key not in ("baseline", "proposed")},
                "local_index": local_stats,
            },
        }
