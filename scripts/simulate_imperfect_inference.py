#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import resource
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


RATIOS = np.asarray((0.8, 0.1, 0.1))
FN_LEVELS = (1.0, 0.95, 0.9, 0.8, 0.7, 0.5)
FP_LEVELS = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05)
COMBINED_LEVELS = tuple(
    (recall, false_positive)
    for recall in (0.9, 0.8, 0.7)
    for false_positive in (0.001, 0.005, 0.01)
)
SEED_BASE = 20260819000


class DSU:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, value: int) -> int:
        while value != self.parent[value]:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> int:
        left, right = self.find(left), self.find(right)
        if left == right:
            return left
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]
        return left


def stable_seed(*values: object) -> int:
    payload = "|".join(map(str, values)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def load_reference(path: Path) -> dict:
    with gzip.open(path / "family_map.json.gz", "rt") as handle:
        family_map = json.load(handle)
    ids = sorted(family_map)
    index = {identity: position for position, identity in enumerate(ids)}
    family_ids = {
        value: position
        for position, value in enumerate(sorted(set(family_map.values())))
    }
    reference_family = np.asarray(
        [family_ids[family_map[identity]] for identity in ids], dtype=np.int32
    )
    groups = defaultdict(list)
    for node, family in enumerate(reference_family):
        groups[int(family)].append(node)
    families = [np.asarray(groups[key], dtype=np.int32) for key in sorted(groups)]
    with gzip.open(path / "family_edges.csv.gz", "rt", newline="") as handle:
        edges = [
            (index[row["left"]], index[row["right"]]) for row in csv.DictReader(handle)
        ]
    sizes = np.asarray([len(group) for group in families], dtype=np.int32)
    multi = np.flatnonzero(sizes > 1)
    pair_count = int(np.sum(sizes[multi].astype(np.int64) * (sizes[multi] - 1) // 2))
    return {
        "files": len(ids),
        "family": reference_family,
        "families": families,
        "sizes": sizes,
        "multi": multi,
        "edges": edges,
        "pairs": pair_count,
    }


def target_counts(size: int) -> np.ndarray:
    counts = np.floor(RATIOS * size).astype(int)
    counts[0] += size - int(counts.sum())
    return counts


def file_split(size: int, seed: int) -> np.ndarray:
    output = np.empty(size, dtype=np.int8)
    order = np.random.default_rng(seed).permutation(size)
    train, validation, _ = target_counts(size)
    output[order[:train]] = 0
    output[order[train : train + validation]] = 1
    output[order[train + validation :]] = 2
    return output


def components(
    edges: list[tuple[int, int]], false_edges: list[tuple[int, int]]
) -> list[np.ndarray]:
    all_edges = edges + false_edges
    active = sorted({node for edge in all_edges for node in edge})
    if not active:
        return []
    positions = {node: position for position, node in enumerate(active)}
    dsu = DSU(len(active))
    for left, right in all_edges:
        dsu.union(positions[left], positions[right])
    groups = defaultdict(list)
    for node in active:
        groups[dsu.find(positions[node])].append(node)
    return [
        np.asarray(sorted(group), dtype=np.int32)
        for group in groups.values()
        if len(group) > 1
    ]


def component_split(size: int, groups: list[np.ndarray], seed: int) -> np.ndarray:
    split = np.full(size, -1, dtype=np.int8)
    target = target_counts(size)
    current = np.zeros(3, dtype=int)
    if groups:
        sizes = np.asarray([len(group) for group in groups], dtype=np.int32)
        tie = (np.arange(len(groups), dtype=np.int64) * 2654435761 + seed) & 0xFFFFFFFF
        order = np.lexsort((tie, -sizes))
        choices = np.searchsorted(
            np.cumsum(target), np.cumsum(sizes[order], dtype=np.int64), side="left"
        ).clip(0, 2)
        for position, group_index in enumerate(order):
            split[groups[int(group_index)]] = choices[position]
        current = np.bincount(choices, weights=sizes[order], minlength=3).astype(int)
    remaining = np.flatnonzero(split < 0)
    rng = np.random.default_rng(stable_seed("singleton-fill", seed, len(groups)))
    remaining = rng.permutation(remaining)
    cursor = 0
    for split_index, count in enumerate(np.maximum(target - current, 0)):
        take = min(int(count), len(remaining) - cursor)
        split[remaining[cursor : cursor + take]] = split_index
        cursor += take
    for node in remaining[cursor:]:
        choice = int(np.argmin(current / np.maximum(target, 1)))
        split[node] = choice
        current[choice] += 1
    return split


def relation_metrics(reference: dict, groups: list[np.ndarray]) -> dict:
    size = reference["files"]
    labels = np.arange(size, dtype=np.int32)
    for group_id, members in enumerate(groups):
        labels[members] = size + group_id
    width = int(labels.max()) + 1
    keys = np.unique(reference["family"].astype(np.int64) * width + labels)
    fragments = np.bincount(keys // width, minlength=len(reference["sizes"]))
    recovered = float(np.mean(fragments[reference["multi"]] == 1))
    fragmented = int(np.sum(fragments[reference["multi"]] > 1))
    predicted_pairs = true_pairs = overmerged = 0
    if groups:
        group_sizes = np.asarray([len(group) for group in groups], dtype=np.int64)
        members = np.concatenate(groups)
        group_ids = np.repeat(np.arange(len(groups), dtype=np.int64), group_sizes)
        predicted_pairs = int(np.sum(group_sizes * (group_sizes - 1) // 2))
        pair_keys, counts = np.unique(
            group_ids * len(reference["sizes"]) + reference["family"][members],
            return_counts=True,
        )
        true_pairs = int(np.sum(counts * (counts - 1) // 2))
        family_counts = np.bincount(
            pair_keys // len(reference["sizes"]), minlength=len(groups)
        )
        overmerged = int(np.sum(family_counts > 1))
    precision = true_pairs / predicted_pairs if predicted_pairs else 1.0
    recall = true_pairs / reference["pairs"] if reference["pairs"] else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "reference_family_recovery_rate": recovered,
        "pairwise_relation_precision": precision,
        "pairwise_same_family_recall": recall,
        "pairwise_relation_f1": f1,
        "over_merge_component_count": overmerged,
        "over_merge_component_rate": overmerged / max(1, len(groups)),
        "under_split_reference_family_count": fragmented,
        "under_split_rate": fragmented / max(1, len(reference["multi"])),
    }


def evaluate(
    reference: dict,
    groups: list[np.ndarray],
    split: np.ndarray,
    original: np.ndarray,
    condition: str,
    variant: str,
    seed: int,
    edge_recall: float | None,
    false_positive_rate: float | None,
    retained_edges: int,
    false_edges: int,
    runtime: float,
) -> dict:
    presence = np.zeros((len(reference["sizes"]), 3), dtype=bool)
    presence[reference["family"], split] = True
    contaminated = presence[:, 2] & np.any(presence[:, :2], axis=1)
    test = split == 2
    counts = np.bincount(split, minlength=3)
    group_sizes = np.asarray([len(group) for group in groups], dtype=np.int32)
    if len(group_sizes):
        median = float(np.median(group_sizes))
        p95 = float(np.quantile(group_sizes, 0.95))
        maximum = int(group_sizes.max())
    else:
        median = p95 = 1.0
        maximum = 1
    result = {
        "condition": condition,
        "variant": variant,
        "seed": seed,
        "edge_recall_target": edge_recall,
        "edge_recall_observed": retained_edges / len(reference["edges"]),
        "fp_injection_ratio_target": false_positive_rate,
        "fp_false_edge_count": false_edges,
        "fp_injection_ratio_observed": false_edges
        / max(1, retained_edges + false_edges),
        "fp_formula": "false_positive_edges / (reference_positive_edges + false_positive_edges)",
        "files_retained": reference["files"],
        "reference_files": reference["files"],
        "reference_families": len(reference["sizes"]),
        "reference_multi_member_families": len(reference["multi"]),
        "residual_known_cross_split_family_count": int(contaminated.sum()),
        "residual_contaminated_test_family_rate": float(
            contaminated.sum() / max(1, presence[:, 2].sum())
        ),
        "residual_contaminated_test_file_rate": float(
            np.sum(test & contaminated[reference["family"]]) / max(1, test.sum())
        ),
        "split_train_ratio": counts[0] / reference["files"],
        "split_validation_ratio": counts[1] / reference["files"],
        "split_test_ratio": counts[2] / reference["files"],
        "split_abs_ratio_error": float(
            np.max(np.abs(counts / reference["files"] - RATIOS))
        ),
        "split_ratio_l1_error": float(
            np.sum(np.abs(counts / reference["files"] - RATIOS))
        ),
        "reassignment_fraction_vs_file_split": float(np.mean(split != original)),
        "inferred_component_count": len(groups)
        + reference["files"]
        - int(group_sizes.sum()),
        "inferred_nontrivial_component_count": len(groups),
        "largest_inferred_component": maximum,
        "component_size_median": median,
        "component_size_p95": p95,
        "runtime_s": runtime,
        "peak_rss_mb_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
    }
    return result | relation_metrics(reference, groups)


def sample_false_edges(
    size: int,
    family: np.ndarray,
    count: int,
    seed: int,
    bounded: bool,
    maximum: int,
) -> list[tuple[int, int]]:
    rng = np.random.default_rng(stable_seed("fp", seed, bounded, count))
    dsu = DSU(size) if bounded else None
    output = []
    seen = set()
    while len(output) < count:
        left, right = map(int, rng.integers(0, size, size=2))
        if left == right or family[left] == family[right]:
            continue
        edge = tuple(sorted((left, right)))
        if edge in seen:
            continue
        if bounded:
            root_left, root_right = dsu.find(left), dsu.find(right)
            if (
                root_left != root_right
                and dsu.size[root_left] + dsu.size[root_right] > maximum
            ):
                continue
            dsu.union(left, right)
        seen.add(edge)
        output.append(edge)
    return output


def aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    excluded = set(keys) | {"condition", "variant", "seed", "fp_formula"}
    numeric = [
        key
        for key, value in rows[0].items()
        if key not in excluded and isinstance(value, (int, float)) and value is not None
    ]
    output = []
    for group_key, group in groups.items():
        item = dict(zip(keys, group_key)) | {"n_seeds": len(group)}
        for metric in numeric:
            values = np.asarray([row[metric] for row in group], dtype=float)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_median"] = float(np.median(values))
            item[f"{metric}_q025"] = float(np.quantile(values, 0.025))
            item[f"{metric}_q975"] = float(np.quantile(values, 0.975))
        output.append(item)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate component-aware splitting under graph noise."
    )
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")

    started = time.perf_counter()
    reference = load_reference(args.reference_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [SEED_BASE + index for index in range(args.seeds)]
    maximum = max(128, 2 * int(reference["sizes"].max()))
    rows = []

    def run(
        condition: str,
        variant: str,
        seed: int,
        kept: list[tuple[int, int]],
        false: list[tuple[int, int]],
        recall: float | None,
        false_positive: float | None,
    ) -> None:
        original = file_split(reference["files"], seed)
        begin = time.perf_counter()
        inferred = components(kept, false)
        split = (
            original
            if condition == "file_level_random"
            else component_split(reference["files"], inferred, seed)
        )
        rows.append(
            evaluate(
                reference,
                inferred,
                split,
                original,
                condition,
                variant,
                seed,
                recall,
                false_positive,
                len(kept),
                len(false),
                time.perf_counter() - begin,
            )
        )

    for seed in seeds:
        run("file_level_random", "baseline", seed, [], [], None, None)
        run("perfect_reference", "upper_bound", seed, reference["edges"], [], 1.0, 0.0)
    for recall in FN_LEVELS:
        keep_count = round(recall * len(reference["edges"]))
        for seed in seeds:
            rng = np.random.default_rng(stable_seed("fn", seed, recall))
            selected = rng.choice(len(reference["edges"]), keep_count, replace=False)
            kept = [reference["edges"][int(index)] for index in selected]
            run("false_negative", "edge_drop", seed, kept, [], recall, None)
    for false_positive in FP_LEVELS:
        count = round(
            false_positive * len(reference["edges"]) / max(1e-12, 1 - false_positive)
        )
        for variant, bounded in (("global_chain", False), ("bounded", True)):
            for seed in seeds:
                false = sample_false_edges(
                    reference["files"],
                    reference["family"],
                    count,
                    seed,
                    bounded,
                    maximum,
                )
                run(
                    "false_positive",
                    variant,
                    seed,
                    reference["edges"],
                    false,
                    1.0,
                    false_positive,
                )
    for recall, false_positive in COMBINED_LEVELS:
        keep_count = round(recall * len(reference["edges"]))
        false_count = round(
            false_positive * len(reference["edges"]) / (1 - false_positive)
        )
        for seed in seeds:
            rng = np.random.default_rng(
                stable_seed("combined-fn", seed, recall, false_positive)
            )
            selected = rng.choice(len(reference["edges"]), keep_count, replace=False)
            kept = [reference["edges"][int(index)] for index in selected]
            false = sample_false_edges(
                reference["files"],
                reference["family"],
                false_count,
                seed,
                False,
                maximum,
            )
            run("combined", "global_chain", seed, kept, false, recall, false_positive)

    summary_keys = (
        "condition",
        "variant",
        "edge_recall_target",
        "fp_injection_ratio_target",
    )
    summary = aggregate(rows, summary_keys)
    write_csv(args.output_dir / "per_run_results.csv", rows)
    write_csv(args.output_dir / "summary_by_condition.csv", summary)
    write_csv(
        args.output_dir / "fn_recall_curve.csv",
        aggregate(
            [row for row in rows if row["condition"] == "false_negative"],
            ("edge_recall_target",),
        ),
    )
    write_csv(
        args.output_dir / "fp_tradeoff_curve.csv",
        aggregate(
            [row for row in rows if row["condition"] == "false_positive"],
            ("variant", "fp_injection_ratio_target"),
        ),
    )
    write_csv(
        args.output_dir / "combined_noise_grid.csv",
        aggregate(
            [row for row in rows if row["condition"] == "combined"],
            ("edge_recall_target", "fp_injection_ratio_target"),
        ),
    )
    write_csv(
        args.output_dir / "component_distortion.csv",
        aggregate(
            [row for row in rows if row["condition"] == "false_positive"],
            ("variant", "fp_injection_ratio_target"),
        ),
    )
    (args.output_dir / "seed_registry.json").write_text(
        json.dumps({"seed_rule": "20260819000 + index", "seeds": seeds}, indent=2)
        + "\n"
    )
    (args.output_dir / "noise_conditions.json").write_text(
        json.dumps(
            {
                "false_negative_recall": FN_LEVELS,
                "false_positive_injection": FP_LEVELS,
                "combined": COMBINED_LEVELS,
                "bounded_component_size": maximum,
                "split_ratios": RATIOS.tolist(),
            },
            indent=2,
        )
        + "\n"
    )
    statistical_summary = {
        "runs": len(rows),
        "seeds_per_condition": args.seeds,
        "reference_files": reference["files"],
        "reference_families": len(reference["sizes"]),
        "reference_edges": len(reference["edges"]),
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "STATISTICAL_SUMMARY.json").write_text(
        json.dumps(statistical_summary, indent=2) + "\n"
    )
    print(json.dumps(statistical_summary, indent=2))


if __name__ == "__main__":
    main()
