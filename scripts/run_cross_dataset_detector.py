#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import resource
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leakagebench_midi.detector import (
    SIGNALS,
    _predict_probability,
    apply_tfidf,
    candidate_ranks,
    extract_count_features,
    extract_features,
    faiss_candidate_ranks,
    load_detector,
    pair_features,
)


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
        if left == right or self.size[left] + self.size[right] > limit:
            return False
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]
        return True

    def labels(self) -> np.ndarray:
        roots = np.asarray([self.find(index) for index in range(len(self.parent))])
        return np.unique(roots, return_inverse=True)[1].astype(np.int32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    def convert(item):
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.floating):
            return float(item)
        if isinstance(item, np.bool_):
            return bool(item)
        raise TypeError(type(item).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=convert) + "\n"
    )


def load_manifest(path: Path, midi_root: Path) -> tuple[list[dict], list[Path]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    paths = [midi_root / row["relative_path"] for row in rows]
    missing = next((path for path in paths if not path.is_file()), None)
    if missing is not None:
        raise FileNotFoundError(missing)
    if len({row["identity"] for row in rows}) != len(rows):
        raise ValueError("manifest identities are not unique")
    return rows, paths


def save_features(
    directory: Path, identities: list[str], bundle: dict, failures: list[dict]
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    dense = ("chroma", "interval_hist", "duration_hist", "ioi_hist", "scalars", "valid")
    np.savez_compressed(
        directory / "dense_features.npz", **{name: bundle[name] for name in dense}
    )
    for group in ("melody", "bass", "rhythm", "harmony", "motif"):
        sparse.save_npz(directory / f"{group}_tfidf.npz", bundle[group])
    write_json(directory / "identities.json", identities)
    write_json(directory / "parse_failures.json", failures)


def load_features(directory: Path) -> tuple[list[str], dict, list[dict]]:
    identities = json.loads((directory / "identities.json").read_text())
    with np.load(directory / "dense_features.npz") as dense:
        bundle = {name: dense[name] for name in dense.files}
    for group in ("melody", "bass", "rhythm", "harmony", "motif"):
        bundle[group] = sparse.load_npz(directory / f"{group}_tfidf.npz")
    failures = json.loads((directory / "parse_failures.json").read_text())
    return identities, bundle, failures


def prepare_features(
    rows: list[dict],
    paths: list[Path],
    cache: Path,
    workers: int,
    shard_size: int,
    keep_shards: bool,
) -> tuple[dict, list[dict]]:
    identities = [row["identity"] for row in rows]
    if (cache / "identities.json").is_file():
        cached, bundle, failures = load_features(cache)
        if cached != identities:
            raise RuntimeError("feature cache identity mismatch")
        return bundle, failures
    if len(paths) <= shard_size:
        bundle, failures = extract_features(paths, workers=workers)
        save_features(cache, identities, bundle, failures)
        return bundle, failures

    shard_root = cache / "count_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    dense_names = (
        "chroma",
        "interval_hist",
        "duration_hist",
        "ioi_hist",
        "scalars",
        "valid",
    )
    groups = ("melody", "bass", "rhythm", "harmony", "motif")
    shard_paths = []
    failures = []
    for start in range(0, len(paths), shard_size):
        stop = min(start + shard_size, len(paths))
        target = shard_root / f"{start:09d}_{stop:09d}"
        if not (target / "dense.npz").is_file():
            target.mkdir(parents=True, exist_ok=True)
            part, part_failures = extract_count_features(
                paths[start:stop], workers=workers
            )
            for item in part_failures:
                item["index"] += start
            np.savez_compressed(
                target / "dense.npz", **{name: part[name] for name in dense_names}
            )
            for group in groups:
                sparse.save_npz(target / f"{group}.npz", part[group])
            write_json(target / "failures.json", part_failures)
        shard_paths.append(target)
        failures.extend(json.loads((target / "failures.json").read_text()))

    counts = {name: [] for name in dense_names}
    for target in shard_paths:
        with np.load(target / "dense.npz") as part:
            for name in dense_names:
                counts[name].append(part[name])
    bundle = {name: np.concatenate(parts) for name, parts in counts.items()}
    for group in groups:
        bundle[group] = sparse.vstack(
            [sparse.load_npz(target / f"{group}.npz") for target in shard_paths],
            format="csr",
        )
    bundle = apply_tfidf(bundle)
    save_features(cache, identities, bundle, failures)
    if not keep_shards:
        shutil.rmtree(shard_root)
    return bundle, failures


def reference_labels(rows: list[dict]) -> np.ndarray | None:
    if not rows or not all("reference_work_id" in row for row in rows):
        return None
    values = [str(row["reference_work_id"]) for row in rows]
    mapping = {value: index for index, value in enumerate(sorted(set(values)))}
    return np.asarray([mapping[value] for value in values], dtype=np.int32)


def pair_metrics(
    left: np.ndarray, right: np.ndarray, reference: np.ndarray | None, complete: bool
) -> dict:
    if reference is None:
        return {"evaluated": False}
    counts = Counter(map(int, reference))
    total = sum(size * (size - 1) // 2 for size in counts.values())
    true_positive = int(np.count_nonzero(reference[left] == reference[right]))
    predicted = len(left)
    recall = true_positive / total if total else None
    result = {
        "evaluated": True,
        "reference_complete": complete,
        "predicted_pairs": predicted,
        "true_reference_pairs": true_positive,
        "reference_pairs": total,
        "reference_pair_recall": recall,
    }
    if complete:
        precision = true_positive / predicted if predicted else None
        result.update(
            {
                "precision": precision,
                "recall": recall,
                "f1": (
                    2 * precision * recall / (precision + recall)
                    if precision and recall
                    else 0.0
                ),
            }
        )
    else:
        result["predicted_pairs_outside_known_reference"] = predicted - true_positive
    return result


def component_metrics(
    labels: np.ndarray, reference: np.ndarray | None, complete: bool
) -> dict:
    groups = defaultdict(list)
    for index, label in enumerate(labels):
        groups[int(label)].append(index)
    sizes = np.asarray([len(members) for members in groups.values()], dtype=np.int64)
    result = {
        "components": len(groups),
        "multi_member_components": int(np.count_nonzero(sizes >= 2)),
        "files_in_multi_member_components": int(sizes[sizes >= 2].sum()),
        "largest_component": int(sizes.max(initial=0)),
        "component_size_median": float(np.median(sizes)),
        "component_size_p95": float(np.quantile(sizes, 0.95)),
    }
    if reference is None:
        result["reference_evaluated"] = False
        return result

    reference_counts = Counter(map(int, reference))
    total_reference = sum(size * (size - 1) // 2 for size in reference_counts.values())
    predicted = int(sum(size * (size - 1) // 2 for size in sizes))
    true_positive = 0
    recovered = defaultdict(int)
    maximum_reference_groups = 1
    for members in groups.values():
        local = Counter(map(int, reference[members]))
        maximum_reference_groups = max(maximum_reference_groups, len(local))
        for work, count in local.items():
            pairs = count * (count - 1) // 2
            true_positive += pairs
            recovered[work] += pairs
    recall = true_positive / total_reference if total_reference else None
    work_recall = [
        recovered[work] / (size * (size - 1) / 2)
        for work, size in reference_counts.items()
        if size >= 2
    ]
    result.update(
        {
            "reference_evaluated": True,
            "reference_complete": complete,
            "predicted_pairs": predicted,
            "true_reference_pairs": true_positive,
            "reference_pairs": total_reference,
            "reference_pair_recall": recall,
            "reference_work_macro_recall": (
                float(np.mean(work_recall)) if work_recall else None
            ),
            "reference_work_median_recall": (
                float(np.median(work_recall)) if work_recall else None
            ),
        }
    )
    if complete:
        precision = true_positive / predicted if predicted else None
        result.update(
            {
                "precision": precision,
                "recall": recall,
                "f1": (
                    2 * precision * recall / (precision + recall)
                    if precision and recall
                    else 0.0
                ),
                "maximum_reference_groups_in_component": maximum_reference_groups,
            }
        )
    else:
        result["predicted_pairs_outside_known_reference"] = predicted - true_positive
    return result


def split_simulation(labels: np.ndarray, seeds: int, seed_base: int) -> dict:
    size = len(labels)
    counts = np.bincount(labels)
    family_rates = []
    file_rates = []
    train_size, validation_size = int(size * 0.8), int(size * 0.1)
    for offset in range(seeds):
        order = np.random.default_rng(seed_base + offset).permutation(size)
        assignment = np.full(size, 2, dtype=np.int8)
        assignment[order[:train_size]] = 0
        assignment[order[train_size : train_size + validation_size]] = 1
        train = np.bincount(labels[assignment == 0], minlength=len(counts)) > 0
        test_counts = np.bincount(labels[assignment == 2], minlength=len(counts))
        test = test_counts > 0
        contaminated = train & test
        family_rates.append(contaminated.sum() / max(1, test.sum()))
        file_rates.append(test_counts[contaminated].sum() / max(1, test_counts.sum()))

    def summary(values: list[float]) -> dict:
        array = np.asarray(values)
        return {
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "empirical_95_interval": [
                float(np.quantile(array, 0.025)),
                float(np.quantile(array, 0.975)),
            ],
        }

    return {
        "seeds": seeds,
        "seed_base": seed_base,
        "contaminated_test_family_rate": summary(family_rates),
        "contaminated_test_file_rate": summary(file_rates),
    }


def official_split_metrics(rows: list[dict], labels: np.ndarray) -> dict:
    if not rows or not all("split" in row for row in rows):
        return {"evaluated": False}
    split_code = {"train": 0, "validation": 1, "test": 2}
    assignment = np.asarray([split_code[row["split"]] for row in rows], dtype=np.int8)
    families = int(labels.max(initial=-1)) + 1
    counts = np.bincount(labels * 3 + assignment, minlength=families * 3).reshape(-1, 3)
    presence = counts > 0
    test = presence[:, 2]
    contaminated = presence[:, 0] & test
    test_files = int(counts[:, 2].sum())
    return {
        "evaluated": True,
        "known_cross_split_family_count": int(np.count_nonzero(presence.sum(axis=1) > 1)),
        "known_train_test_family_count": int(contaminated.sum()),
        "test_family_count": int(test.sum()),
        "contaminated_test_family_rate": float(contaminated.sum() / max(1, test.sum())),
        "test_file_count": test_files,
        "contaminated_test_file_count": int(counts[contaminated, 2].sum()),
        "contaminated_test_file_rate": float(
            counts[contaminated, 2].sum() / max(1, test_files)
        ),
        "split_files": counts.sum(axis=0).tolist(),
    }


def score_candidates(
    bundle: dict, candidates: dict, config: dict, model, chunk_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kept_left, kept_right, kept_scores = [], [], []
    expected = config["selected_feature_names"]
    for start in range(0, len(candidates["pairs"]), chunk_size):
        stop = min(start + chunk_size, len(candidates["pairs"]))
        part = {
            "pairs": candidates["pairs"][start:stop],
            "ranks": candidates["ranks"][start:stop],
            "k": candidates["k"],
        }
        features, names = pair_features(bundle, part)
        if names != expected:
            raise RuntimeError("detector feature schema mismatch")
        scores = _predict_probability(model, features)
        keep = scores >= config["threshold"]
        kept_left.append(part["pairs"][keep, 0])
        kept_right.append(part["pairs"][keep, 1])
        kept_scores.append(scores[keep])
    return (
        np.concatenate(kept_left) if kept_left else np.empty(0, dtype=np.int32),
        np.concatenate(kept_right) if kept_right else np.empty(0, dtype=np.int32),
        np.concatenate(kept_scores) if kept_scores else np.empty(0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the structural detector on a complete MIDI collection."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("midi_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--detector", type=Path, default=ROOT / "artifacts/detector")
    parser.add_argument("--backend", choices=("exact", "faiss"), default="faiss")
    parser.add_argument(
        "--reference-mode", choices=("complete", "incomplete", "none"), default="none"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--faiss-threads", type=int, default=1)
    parser.add_argument("--faiss-signal-workers", type=int, default=1)
    parser.add_argument("--feature-shard-size", type=int, default=10_000)
    parser.add_argument("--score-chunk-size", type=int, default=200_000)
    parser.add_argument("--split-seeds", type=int, default=200)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--keep-count-shards", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, paths = load_manifest(args.manifest, args.midi_root)
    bundle, failures = prepare_features(
        rows,
        paths,
        args.cache,
        args.workers,
        args.feature_shard_size,
        args.keep_count_shards,
    )
    valid = np.flatnonzero(bundle["valid"])
    config, model = load_detector(args.detector)
    seed = args.seed if args.seed is not None else int(config["seed"])
    if args.backend == "exact":
        candidates = candidate_ranks(bundle, valid, int(config["candidate_k"]))
        mutual = np.sum(
            np.max(candidates["ranks"], axis=2) <= candidates["k"], axis=1
        )
        keep = mutual >= int(config["minimum_mutual_structural_signals"])
        candidates["pairs"] = candidates["pairs"][keep]
        candidates["ranks"] = candidates["ranks"][keep]
        diagnostics = {
            "backend": "exact_brute",
            "candidate_pairs": int(keep.sum()),
        }
    else:
        candidates, diagnostics = faiss_candidate_ranks(
            bundle,
            valid,
            int(config["candidate_k"]),
            minimum_mutual_signals=int(
                config["minimum_mutual_structural_signals"]
            ),
            threads=args.faiss_threads,
            signal_workers=args.faiss_signal_workers,
            seed=seed,
        )

    left, right, scores = score_candidates(
        bundle, candidates, config, model, args.score_chunk_size
    )
    order = np.argsort(scores, kind="mergesort")[::-1]
    components = Components(len(rows))
    accepted = rejected = 0
    for index in order:
        if components.union(
            int(left[index]),
            int(right[index]),
            int(config["maximum_component_size"]),
        ):
            accepted += 1
        else:
            rejected += 1
    labels = components.labels()

    reference = None if args.reference_mode == "none" else reference_labels(rows)
    if args.reference_mode != "none" and reference is None:
        raise ValueError("manifest has no reference_work_id values")
    complete = args.reference_mode == "complete"
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss /= 1024 * 1024 if sys.platform == "darwin" else 1024
    result = {
        "status": "PASS",
        "files": len(rows),
        "valid_files": len(valid),
        "parse_failures": failures,
        "backend": args.backend,
        "candidate_diagnostics": diagnostics,
        "candidate_reference_metrics": pair_metrics(
            candidates["pairs"][:, 0],
            candidates["pairs"][:, 1],
            reference,
            complete,
        ),
        "direct_edges": pair_metrics(left, right, reference, complete),
        "guarded_components": component_metrics(labels, reference, complete),
        "accepted_component_merges": accepted,
        "rejected_component_merges": rejected,
        "random_file_split": split_simulation(
            labels, args.split_seeds, seed * 1000
        ),
        "official_file_split": official_split_metrics(rows, labels),
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_mb": rss,
        "manifest_sha256": sha256(args.manifest),
        "model_sha256": config["model"]["sha256"],
    }
    write_json(args.output_dir / "results.json", result)
    with gzip.open(
        args.output_dir / "predicted_edges.csv.gz",
        "wt",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("left_identity", "right_identity", "score"))
        for left_index, right_index, score in zip(left, right, scores):
            writer.writerow(
                (
                    rows[int(left_index)]["identity"],
                    rows[int(right_index)]["identity"],
                    float(score),
                )
            )
    np.savez_compressed(args.output_dir / "component_labels.npz", labels=labels)
    print(f"STATUS={result['status']}")
    print(f"FILES={len(rows)}")
    print(f"PREDICTED_EDGES={len(left)}")
    print(f"RUNTIME_SECONDS={result['runtime_seconds']:.3f}")


if __name__ == "__main__":
    main()
