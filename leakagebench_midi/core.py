from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def stable(*parts):
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def write_jsonl(path, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(
            json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n" for x in rows
        )
    )


def _required(row, fields, context):
    missing = sorted(set(fields) - set(row))
    if missing:
        raise ValueError(f"{context}: missing required columns: {', '.join(missing)}")


def _valid_id(value, field):
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(ch in value for ch in "\r\n\0")
    ):
        raise ValueError(f"malformed {field}: expected a non-empty single-line string")


def _validate_rows(rows, *, require_family=True, require_tokens=False, context="rows"):
    seen = set()
    for index, row in enumerate(rows):
        fields = (
            ["id"]
            + (["family_id"] if require_family else [])
            + (["tokens"] if require_tokens else [])
        )
        _required(row, fields, f"{context}[{index}]")
        _valid_id(row["id"], "file id")
        if row["id"] in seen:
            raise ValueError(f"duplicate file ID: {row['id']}")
        seen.add(row["id"])
        if require_family:
            _valid_id(row["family_id"], "family id")
        if require_tokens and (
            not isinstance(row["tokens"], (int, float))
            or not math.isfinite(row["tokens"])
            or row["tokens"] <= 0
        ):
            raise ValueError(f"non-positive token count for file {row['id']}")


def _validate_ratios(ratios, names):
    if len(ratios) != len(names):
        raise ValueError("ratio/name length mismatch")
    if not ratios:
        raise ValueError("at least one split ratio is required")
    for index, value in enumerate(ratios):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"split ratio {index} must be finite")
        if value < 0 or value > 1:
            raise ValueError(f"split ratio {index} must be between 0 and 1")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to one")


def build_family_map(ids, edges):
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate file IDs")
    for value in ids:
        _valid_id(value, "file id")
    parent = {x: x for x in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen_edges = set()
    for a, b in edges:
        if a not in parent or b not in parent:
            raise ValueError("edge references unknown id")
        edge = tuple(sorted((a, b)))
        if edge in seen_edges:
            raise ValueError(f"duplicate family assignment edge: {edge}")
        seen_edges.add(edge)
        x, y = find(a), find(b)
        if x != y:
            parent[max(x, y)] = min(x, y)
    groups = defaultdict(list)
    for x in sorted(parent):
        groups[find(x)].append(x)
    return {
        x: stable("family-component-v1", *members)
        for members in groups.values()
        for x in members
    }


def family_aware_split(
    rows, ratios=(0.8, 0.1, 0.1), seed=0, names=("train", "validation", "test")
):
    _validate_ratios(ratios, names)
    _validate_rows(
        rows, require_family=True, require_tokens=True, context="split input"
    )
    if not rows:
        raise ValueError("split input is empty")
    by = defaultdict(list)
    for row in rows:
        by[row["family_id"]].append(row)
    total_tokens = sum(row["tokens"] for row in rows)
    target = np.asarray(ratios, dtype=float) * total_tokens
    current = np.zeros(len(ratios))
    assigned = {}
    order = sorted(
        by,
        key=lambda family: (
            -sum(x["tokens"] for x in by[family]),
            stable(seed, family),
        ),
    )
    for family in order:
        weight = sum(x["tokens"] for x in by[family])
        choice = min(
            range(len(ratios)),
            key=lambda j: (
                sum(
                    ((current[k] + (weight if k == j else 0) - target[k]) / target[k])
                    ** 2
                    if target[k] > 0
                    else (
                        0 if current[k] + (weight if k == j else 0) == 0 else math.inf
                    )
                    for k in range(len(ratios))
                ),
                j,
            ),
        )
        assigned[family] = names[choice]
        current[choice] += weight
    output = [dict(row, split=assigned[row["family_id"]]) for row in rows]
    original = {row["id"]: row.get("split") for row in rows}
    changed = sum(original[row["id"]] not in (None, row["split"]) for row in output)
    audit = {
        "seed": seed,
        "family_components": len(by),
        "files": len(rows),
        "cross_split_family_count": 0,
        "file_counts": dict(Counter(row["split"] for row in output)),
        "token_counts": {
            name: sum(row["tokens"] for row in output if row["split"] == name)
            for name in names
        },
        "target_ratios": dict(zip(names, ratios)),
        "file_reassignment_count": changed,
        "file_reassignment_ratio": changed / len(rows),
    }
    return output, audit


def audit_split(rows, train="train", test="test"):
    _validate_rows(rows, require_family=True, context="split audit input")
    by = defaultdict(list)
    for row in rows:
        _required(row, ["split"], "split audit row")
        by[row["family_id"]].append(row)
    contaminated, test_families = [], []
    test_files = contaminated_files = 0
    for family, members in by.items():
        train_ids = [x["id"] for x in members if x["split"] == train]
        test_ids = [x["id"] for x in members if x["split"] == test]
        if test_ids:
            test_families.append(family)
            test_files += len(test_ids)
        if train_ids and test_ids:
            contaminated.append(
                {
                    "family_id": family,
                    "test_files": test_ids,
                    "train_siblings": train_ids,
                    "family_size": len(members),
                }
            )
            contaminated_files += len(test_ids)
    family_rate = len(contaminated) / len(test_families) if test_families else None
    file_rate = contaminated_files / test_files if test_files else None
    return {
        "test_families": len(test_families),
        "test_files": test_files,
        "contaminated_test_families": len(contaminated),
        "contaminated_test_files": contaminated_files,
        "test_family_contamination_rate": family_rate,
        "test_file_contamination_rate": file_rate,
        "undefined_rate_reason": "zero test denominator"
        if family_rate is None or file_rate is None
        else None,
        "family_size_statistics": dict(Counter(len(x) for x in by.values())),
        "contaminated": contaminated,
    }


def cross_probability(k, p_train, p_test):
    p_other = 1 - p_train - p_test
    return 1 - (1 - p_train) ** k - (1 - p_test) ** k + p_other**k


def conditional_sibling_probability(k, p_train):
    return 1 - (1 - p_train) ** (k - 1)


def census(family_sizes, ratios=(0.8, 0.1, 0.1), num_seeds=1000, seed=0):
    _validate_ratios(ratios, tuple(str(i) for i in range(len(ratios))))
    sizes = np.asarray(family_sizes, int)
    if sizes.size == 0 or np.any(sizes <= 0):
        raise ValueError("family sizes must be non-empty positive integers")
    probabilities = np.asarray(ratios, float)
    rng = np.random.default_rng(seed)
    rows = []
    undefined_seeds = 0
    for _ in range(num_seeds):
        contaminated_families = contaminated_files = test_families = test_files = 0
        for size in sizes:
            assignment = rng.choice(len(probabilities), int(size), p=probabilities)
            has_test, has_train = (
                np.any(assignment == len(probabilities) - 1),
                np.any(assignment == 0),
            )
            test_families += has_test
            test_files += np.sum(assignment == len(probabilities) - 1)
            if has_test and has_train:
                contaminated_families += 1
                contaminated_files += np.sum(assignment == len(probabilities) - 1)
        if test_families == 0 or test_files == 0:
            undefined_seeds += 1
            continue
        rows.append(
            (contaminated_families / test_families, contaminated_files / test_files)
        )
    if not rows:
        raise ValueError(
            "undefined census metric for every seed: zero test denominator"
        )
    values = np.asarray(rows)
    return {
        "num_seeds": num_seeds,
        "seed": seed,
        "undefined_zero_denominator_seeds": undefined_seeds,
        "test_family_contamination": {
            "mean": float(values[:, 0].mean()),
            "q025": float(np.quantile(values[:, 0], 0.025)),
            "q975": float(np.quantile(values[:, 0], 0.975)),
        },
        "test_file_contamination": {
            "mean": float(values[:, 1].mean()),
            "q025": float(np.quantile(values[:, 1], 0.025)),
            "q975": float(np.quantile(values[:, 1], 0.975)),
        },
        "analytical_by_size": {
            str(k): {
                "cross_probability": cross_probability(int(k), ratios[0], ratios[-1]),
                "test_member_has_train_sibling": conditional_sibling_probability(
                    int(k), ratios[0]
                ),
            }
            for k in sorted(set(sizes.tolist()))
        },
        "family_aware": {"known_cross_split_families": 0},
    }


def build_contamination(
    base,
    treated,
    control,
    validation,
    assignments,
    seed=0,
    *,
    max_total_token_rel_diff=None,
    max_pair_token_rel_diff=None,
    require_validation=True,
):
    _validate_rows(base, require_family=True, require_tokens=True, context="base")
    if not treated:
        raise ValueError("treated cohort is empty")
    if not control:
        raise ValueError("control cohort is empty")
    if require_validation and not validation:
        raise ValueError("required validation cohort is empty")
    for label, cohort in (
        ("treated", treated),
        ("control", control),
        ("validation", validation),
    ):
        family_ids = set()
        for index, family in enumerate(cohort):
            _required(family, ["family_id"], f"{label}[{index}]")
            _valid_id(family["family_id"], "family id")
            if family["family_id"] in family_ids:
                raise ValueError(
                    f"duplicate family assignment in {label}: {family['family_id']}"
                )
            family_ids.add(family["family_id"])
    treated_ids = {x["family_id"] for x in treated}
    control_ids = {x["family_id"] for x in control}
    validation_ids = {x["family_id"] for x in validation}
    if (
        treated_ids & control_ids
        or treated_ids & validation_ids
        or control_ids & validation_ids
    ):
        raise ValueError("probe family sets overlap")

    member_by_id, member_family = {}, {}
    for family in treated:
        members = family.get("members")
        if not isinstance(members, list) or len(members) < 2:
            raise ValueError(
                f"treated family {family['family_id']} must contain donor and receiver members"
            )
        for member in members:
            _required(
                member,
                ["id", "family_id", "tokens"],
                f"treated family {family['family_id']} member",
            )
            _valid_id(member["id"], "member id")
            if member["id"] in member_by_id:
                raise ValueError(f"duplicate file ID: {member['id']}")
            if member["family_id"] != family["family_id"]:
                raise ValueError(f"member {member['id']} belongs to wrong family")
            if (
                not isinstance(member["tokens"], (int, float))
                or not math.isfinite(member["tokens"])
                or member["tokens"] <= 0
            ):
                raise ValueError(f"non-positive token count for member {member['id']}")
            member_by_id[member["id"]] = member
            member_family[member["id"]] = family["family_id"]

    seen_assignments, receivers, selected = set(), set(), []
    for index, assignment in enumerate(assignments):
        _required(
            assignment, ["family_id", "donor_id", "receiver_id"], f"assignment[{index}]"
        )
        family, donor, receiver = (
            assignment["family_id"],
            assignment["donor_id"],
            assignment["receiver_id"],
        )
        for value, field in (
            (family, "family id"),
            (donor, "donor id"),
            (receiver, "receiver id"),
        ):
            _valid_id(value, field)
        identity = (family, donor, receiver)
        if identity in seen_assignments:
            raise ValueError(f"duplicate assignment row: {identity}")
        seen_assignments.add(identity)
        if family not in treated_ids:
            raise ValueError(f"assignment references missing family: {family}")
        if donor not in member_by_id:
            raise ValueError(f"donor does not exist: {donor}")
        if receiver not in member_by_id:
            raise ValueError(f"receiver does not exist: {receiver}")
        if member_family[donor] != family:
            raise ValueError(
                f"donor {donor} does not belong to declared family {family}"
            )
        if member_family[receiver] != family:
            raise ValueError(
                f"receiver {receiver} does not belong to declared family {family}"
            )
        if donor == receiver:
            raise ValueError(f"donor equals receiver in family {family}")
        if receiver in receivers:
            raise ValueError(f"duplicate receiver assignment: {receiver}")
        receivers.add(receiver)
        selected.append(dict(member_by_id[donor], source_role="designated_donor"))

    donors = {x["id"] for x in selected}
    clean = [
        row
        for row in base
        if row["family_id"] not in treated_ids | control_ids | validation_ids
        and row["id"] not in receivers | donors
    ]
    if len(selected) > len(clean):
        raise ValueError("donor count exceeds replaceable sample count")
    leak = list(clean)
    unused = set(range(len(clean)))
    replacements = []
    for donor in sorted(selected, key=lambda x: (x["family_id"], x["id"])):
        index = min(
            unused,
            key=lambda j: (
                abs(clean[j]["tokens"] - donor["tokens"]),
                stable(seed, clean[j]["id"]),
            ),
        )
        unused.remove(index)
        removed = clean[index]
        signed = donor["tokens"] - removed["tokens"]
        relative = abs(signed) / removed["tokens"]
        if max_pair_token_rel_diff is not None and relative > max_pair_token_rel_diff:
            raise ValueError(
                f"pair token relative difference {relative} exceeds configured maximum {max_pair_token_rel_diff}"
            )
        replacements.append(
            {
                "removed": removed["id"],
                "donor": donor["id"],
                "removed_tokens": removed["tokens"],
                "donor_tokens": donor["tokens"],
                "absolute_difference": abs(signed),
                "relative_difference": relative,
                "signed_difference": signed,
            }
        )
        leak[index] = donor
    clean_total = sum(x["tokens"] for x in clean)
    leak_total = sum(x["tokens"] for x in leak)
    absolute_total = abs(leak_total - clean_total)
    relative_total = absolute_total / clean_total if clean_total else None
    if relative_total is None:
        raise ValueError("undefined token budget: clean total is zero")
    if (
        max_total_token_rel_diff is not None
        and relative_total > max_total_token_rel_diff
    ):
        raise ValueError(
            f"total token relative difference {relative_total} exceeds configured maximum {max_total_token_rel_diff}"
        )
    return {
        "train_clean": clean,
        "train_family_leak": leak,
        "treated_probe": treated,
        "control_probe": control,
        "clean_validation": validation,
        "replacements": replacements,
        "token_budget": {
            "matching": "EXACT" if absolute_total == 0 else "APPROXIMATE",
            "clean_total_tokens": clean_total,
            "family_leak_total_tokens": leak_total,
            "absolute_total_difference": absolute_total,
            "relative_total_difference": relative_total,
            "configured_max_total_token_rel_diff": max_total_token_rel_diff,
            "configured_max_pair_token_rel_diff": max_pair_token_rel_diff,
        },
        "integrity": {
            "receiver_in_train": sum(x["id"] in receivers for x in clean + leak),
            "non_designated_treated_in_train": sum(
                x["family_id"] in treated_ids and x["id"] not in donors
                for x in clean + leak
            ),
            "family_sets_disjoint": True,
        },
    }


ANALYSIS_FIELDS = {
    "dataset",
    "architecture",
    "model_size",
    "seed",
    "condition",
    "split",
    "family_id",
    "nll",
}


def validate_family_manifest(rows, manifest):
    if isinstance(manifest, dict):
        families = manifest.get("families")
        if not isinstance(families, list):
            raise ValueError("family manifest must contain a families list")
        expected_hash = manifest.get("families_sha256")
        if (
            expected_hash
            and stable(
                "family-manifest-v1",
                json.dumps(families, sort_keys=True, separators=(",", ":")),
            )
            != expected_hash
        ):
            raise ValueError("family manifest hash mismatch")
    elif isinstance(manifest, list):
        families = manifest
    else:
        raise ValueError("malformed family manifest")
    manifest_ids = []
    manifest_splits = {}
    member_ids = set()
    for index, family in enumerate(families):
        _required(family, ["family_id", "split"], f"family manifest[{index}]")
        _valid_id(family["family_id"], "family id")
        _valid_id(family["split"], "family split")
        if family["split"] not in {"treated", "control", "clean_validation"}:
            raise ValueError(f"unsupported family manifest split: {family['split']}")
        if family["family_id"] in manifest_splits:
            previous = manifest_splits[family["family_id"]]
            if previous != family["split"]:
                raise ValueError(
                    f"family appears in multiple manifest splits: {family['family_id']}"
                )
            raise ValueError(
                f"duplicate family IDs in family manifest: {family['family_id']}"
            )
        manifest_ids.append(family["family_id"])
        manifest_splits[family["family_id"]] = family["split"]
        if "members" in family:
            if not isinstance(family["members"], list) or not family["members"]:
                raise ValueError(
                    f"family manifest[{index}] members must be a non-empty list"
                )
            for member in family["members"]:
                member_id = member.get("id") if isinstance(member, dict) else member
                _valid_id(member_id, "family member id")
                if (
                    isinstance(member, dict)
                    and member.get("family_id", family["family_id"])
                    != family["family_id"]
                ):
                    raise ValueError(
                        f"family manifest member {member_id} belongs to wrong family"
                    )
                if member_id in member_ids:
                    raise ValueError(
                        f"duplicate member ID in family manifest: {member_id}"
                    )
                member_ids.add(member_id)
    analysis_ids = {row["family_id"] for row in rows}
    if analysis_ids != set(manifest_ids):
        raise ValueError(
            f"analysis families do not match family manifest: analysis={len(analysis_ids)} manifest={len(set(manifest_ids))}"
        )
    analysis_splits = {}
    for row in rows:
        family_id = row["family_id"]
        analysis_splits.setdefault(family_id, set()).add(row["split"])
    collisions = sorted(
        family_id for family_id, splits in analysis_splits.items() if len(splits) != 1
    )
    if collisions:
        raise ValueError(
            f"family appears in multiple analysis cohorts: {collisions[0]}"
        )
    for family_id, splits in analysis_splits.items():
        analysis_split = next(iter(splits))
        if manifest_splits[family_id] != analysis_split:
            raise ValueError(
                f"family manifest split does not match analysis split for {family_id}: "
                f"manifest={manifest_splits[family_id]} analysis={analysis_split}"
            )
    return True


def analyze_effect(
    rows,
    bootstrap_samples=10000,
    bootstrap_seed=0,
    *,
    family_manifest=None,
    require_validation=True,
    bootstrap_draw_mode="alternating",
    family_order="sorted",
):
    if (
        not isinstance(bootstrap_samples, int)
        or isinstance(bootstrap_samples, bool)
        or bootstrap_samples <= 0
    ):
        raise ValueError("bootstrap_samples must be a positive integer")
    if not rows:
        raise ValueError("result rows are empty")
    key_counts = Counter()
    groups = set()
    for index, row in enumerate(rows):
        _required(row, ANALYSIS_FIELDS, f"result row[{index}]")
        for field in (
            "dataset",
            "architecture",
            "model_size",
            "condition",
            "split",
            "family_id",
        ):
            _valid_id(row[field], field)
        if not isinstance(row["seed"], (str, int)):
            raise ValueError("malformed seed")
        if not isinstance(row["nll"], (int, float)) or not math.isfinite(row["nll"]):
            raise ValueError(f"non-finite NLL for family {row['family_id']}")
        key = tuple(
            row[field]
            for field in (
                "dataset",
                "architecture",
                "model_size",
                "seed",
                "condition",
                "split",
                "family_id",
            )
        )
        key_counts[key] += 1
        groups.add(
            tuple(row[field] for field in ("dataset", "architecture", "model_size"))
        )
    duplicates = [key for key, count in key_counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate result key: {duplicates[0]}")
    if len(groups) != 1:
        raise ValueError(
            "analyze_effect requires exactly one dataset/architecture/model_size group"
        )
    if family_manifest is not None:
        validate_family_manifest(rows, family_manifest)

    conditions = {row["condition"] for row in rows}
    if conditions != {"clean", "family_leak"}:
        raise ValueError("both clean and family_leak conditions are required")
    seeds = {
        condition: {row["seed"] for row in rows if row["condition"] == condition}
        for condition in conditions
    }
    if seeds["clean"] != seeds["family_leak"]:
        raise ValueError("clean and family_leak seed sets differ")
    ordered_seeds = sorted(seeds["clean"], key=str)
    required_splits = ["treated", "control"] + (
        ["clean_validation"] if require_validation else []
    )
    by = {}
    for row in rows:
        by[row["seed"], row["condition"], row["split"], row["family_id"]] = float(
            row["nll"]
        )
    families = {}
    for split in required_splits:
        per = {}
        for seed in ordered_seeds:
            for condition in ("clean", "family_leak"):
                per[seed, condition] = {
                    row["family_id"]
                    for row in rows
                    if row["seed"] == seed
                    and row["condition"] == condition
                    and row["split"] == split
                }
            if per[seed, "clean"] != per[seed, "family_leak"]:
                missing_clean = per[seed, "family_leak"] - per[seed, "clean"]
                missing_leak = per[seed, "clean"] - per[seed, "family_leak"]
                raise ValueError(
                    f"missing result pair for split={split} seed={seed}: missing_clean={len(missing_clean)} missing_leak={len(missing_leak)}"
                )
        reference = per[ordered_seeds[0], "clean"]
        if not reference:
            raise ValueError(f"{split} cohort is empty")
        if any(value != reference for value in per.values()):
            raise ValueError(
                f"family sets differ across seeds/conditions for split={split}"
            )
        input_order = [
            row["family_id"]
            for row in rows
            if row["seed"] == ordered_seeds[0]
            and row["condition"] == "clean"
            and row["split"] == split
        ]
        if family_order == "sorted":
            families[split] = sorted(input_order)
        elif family_order == "input":
            families[split] = input_order
        else:
            raise ValueError(f"unsupported family order: {family_order}")

    def delta(seed, split, family):
        return by[seed, "family_leak", split, family] - by[seed, "clean", split, family]

    seed_effects = {
        str(seed): float(
            np.mean([delta(seed, "treated", f) for f in families["treated"]])
            - np.mean([delta(seed, "control", f) for f in families["control"]])
        )
        for seed in ordered_seeds
    }
    pooled = {
        split: {
            family: float(
                np.mean([delta(seed, split, family) for seed in ordered_seeds])
            )
            for family in families[split]
        }
        for split in required_splits
    }
    treated_values = np.asarray(list(pooled["treated"].values()))
    control_values = np.asarray(list(pooled["control"].values()))
    tau = float(treated_values.mean() - control_values.mean())
    rng = np.random.default_rng(bootstrap_seed)
    if bootstrap_draw_mode == "alternating":
        draws = np.asarray(
            [
                rng.choice(treated_values, len(treated_values), True).mean()
                - rng.choice(control_values, len(control_values), True).mean()
                for _ in range(bootstrap_samples)
            ]
        )
    elif bootstrap_draw_mode == "vectorized_blocks":
        draws = rng.choice(
            treated_values, size=(bootstrap_samples, len(treated_values)), replace=True
        ).mean(axis=1) - rng.choice(
            control_values, size=(bootstrap_samples, len(control_values)), replace=True
        ).mean(axis=1)
    else:
        raise ValueError(f"unsupported bootstrap draw mode: {bootstrap_draw_mode}")
    extreme_count = int(min(np.sum(draws <= 0), np.sum(draws >= 0)))
    validation_values = np.asarray(list(pooled.get("clean_validation", {}).values()))
    clean_treated_mean = float(
        np.mean(
            [
                by[seed, "clean", "treated", family]
                for seed in ordered_seeds
                for family in families["treated"]
            ]
        )
    )
    return {
        "tau": tau,
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "p_two_sided": min(1.0, 2.0 * (extreme_count + 1) / (bootstrap_samples + 1)),
        "p_correction": "two-sided add-one",
        "bootstrap_extreme_count": extreme_count,
        "seed_effects": seed_effects,
        "three_of_three_seed_negative": len(ordered_seeds) == 3
        and all(x < 0 for x in seed_effects.values()),
        "treated_negative_sign_rate": float((treated_values < 0).mean()),
        "treated_mean_delta": float(treated_values.mean()),
        "control_drift": float(control_values.mean()),
        "treated_relative_nll_improvement": float(
            -treated_values.mean() / clean_treated_mean
        ),
        "clean_validation_drift": float(validation_values.mean())
        if validation_values.size
        else None,
        "bootstrap_unit": "family",
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_draw_mode": bootstrap_draw_mode,
        "family_order": family_order,
        "seed_fixed_effect": True,
        "family_manifest_validated": family_manifest is not None,
    }
