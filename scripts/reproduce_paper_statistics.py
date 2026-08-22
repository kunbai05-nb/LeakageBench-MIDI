#!/usr/bin/env python3
"""Recompute paper statistics from the public analysis bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reproduction/data"
FROZEN = ROOT / "reproduction/frozen"
EXPECTED = ROOT / "results/manuscript_results_v2_public.json"
PROTOCOL = json.loads((ROOT / "configs/protocol_v2.json").read_text(encoding="utf-8"))
STATISTICS = PROTOCOL["statistics"]
PHASE2_SEEDS = tuple(PROTOCOL["controlled_exposure"]["formal_seeds"])
PHASE2_CONDITIONS = tuple(PROTOCOL["controlled_exposure"]["conditions"])
CROSS_PARADIGM_SEEDS = tuple(PROTOCOL["cross_paradigm"]["formal_seeds"])
CROSS_PARADIGM_CONDITIONS = tuple(PROTOCOL["cross_paradigm"]["conditions"])
BOOTSTRAP_SAMPLES = int(STATISTICS["bootstrap_replicates"])
PHASE2_BOOTSTRAP_SEED = int(STATISTICS["phase2_bootstrap_seed"])
CROSS_PARADIGM_BOOTSTRAP_SEED = int(STATISTICS["cross_paradigm_bootstrap_seed"])
CROSS_PARADIGM_SIGN_SEED = int(STATISTICS["cross_paradigm_sign_randomization_seed"])
LEGACY_SEEDS = range(3)
LEGACY_CONFIRMATORY_SEED = 20260804
CLEAN_VALIDATION_SEED = 20260808
CAPACITY_BOOTSTRAP_SEED = 20260812
PDMX_BOOTSTRAP_SEED = 20260813
RELATEDNESS_SPEARMAN_SEED = 20260823
RELATEDNESS_HUBER_SEED = 20260905
NUMERIC_FIELDS = (
    "mean_a",
    "mean_b",
    "effect",
    "relative_effect",
    "ci_low",
    "ci_high",
    "p_raw",
    "p_adjusted",
)
RECOMPUTED = "RECOMPUTED_FROM_PUBLIC_ROWS"
VERIFIED = "VERIFIED_FROM_FROZEN_SUMMARY"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value):
    if value in (None, ""):
        return None
    return float(value)


def mean(values) -> float:
    return float(np.mean(np.asarray(list(values), dtype=float)))


def is_finite(value) -> bool:
    value = f(value)
    return value is not None and math.isfinite(value)


def bootstrap_mean(
    values, seed: int, samples: int = BOOTSTRAP_SAMPLES, plus_one: bool = False
) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be non-empty and finite")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(
        axis=1
    )
    lower, upper = int(np.count_nonzero(draws <= 0)), int(np.count_nonzero(draws >= 0))
    if plus_one:
        p = min(1.0, 2.0 * (min(lower, upper) + 1) / (samples + 1))
    else:
        p = min(1.0, 2.0 * min(lower / samples, upper / samples))
    return {
        "effect": float(values.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "p_raw": float(p),
        "n_families": len(values),
        "bootstrap_B": samples,
    }


def bootstrap_mean_loop(
    values, seed: int, samples: int = BOOTSTRAP_SAMPLES, plus_one: bool = False
) -> dict:
    """Exact loop-based bootstrap used by the Phase-2 formal postprocessor."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be non-empty and finite")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [
            rng.choice(values, size=len(values), replace=True).mean()
            for _ in range(samples)
        ]
    )
    lower, upper = int(np.count_nonzero(draws <= 0)), int(np.count_nonzero(draws >= 0))
    p = min(
        1.0,
        2.0
        * (
            (min(lower, upper) + 1) / (samples + 1)
            if plus_one
            else min(lower, upper) / samples
        ),
    )
    return {
        "effect": float(values.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "p_raw": float(p),
        "n_families": len(values),
        "bootstrap_B": samples,
    }


def bootstrap_difference(
    left, right, seed: int, samples: int = BOOTSTRAP_SAMPLES
) -> dict:
    left, right = np.asarray(left, float), np.asarray(right, float)
    if (
        left.size == 0
        or right.size == 0
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("bootstrap groups must be non-empty and finite")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [
            rng.choice(left, len(left), True).mean()
            - rng.choice(right, len(right), True).mean()
            for _ in range(samples)
        ]
    )
    return {
        "effect": float(left.mean() - right.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "p_raw": float(min(1.0, 2 * min(np.mean(draws <= 0), np.mean(draws >= 0)))),
        "bootstrap_B": samples,
    }


class Audit:
    def __init__(self, expected: dict[str, dict]):
        self.expected = expected
        self.values: dict[str, dict[str, tuple[object, str, str, float]]] = defaultdict(
            dict
        )

    def add(
        self,
        result_id: str,
        method: str,
        provenance: str = RECOMPUTED,
        tolerance: float = 1e-10,
        **values,
    ) -> None:
        if provenance not in {RECOMPUTED, VERIFIED}:
            raise ValueError(f"unsupported audit provenance: {provenance}")
        for field, value in values.items():
            if value is not None:
                self.values[result_id][field] = (value, method, provenance, tolerance)

    def rows(self) -> list[dict]:
        rows = []
        for rid, expected in sorted(self.expected.items()):
            numeric = [
                field for field in NUMERIC_FIELDS if expected.get(field) is not None
            ]
            if not numeric:
                rows.append(
                    {
                        "result_id": rid,
                        "field": "not_applicable",
                        "expected": "",
                        "recomputed": "",
                        "absolute_difference": "",
                        "tolerance": "",
                        "method": "qualitative_or_intentionally_missing",
                        "status": "NOT_APPLICABLE",
                    }
                )
                continue
            for field in numeric:
                item = self.values.get(rid, {}).get(field)
                if item is None:
                    rows.append(
                        {
                            "result_id": rid,
                            "field": field,
                            "expected": expected[field],
                            "recomputed": "",
                            "absolute_difference": "",
                            "tolerance": "",
                            "method": "unavailable",
                            "status": "UNREPRODUCED",
                        }
                    )
                    continue
                value, method, provenance, tolerance = item
                # Display tables may be rounded; lockfile values retain tight tolerance.
                text = str(expected[field]).lower()
                if "e" not in text and "." in text:
                    decimals = len(text.split(".", 1)[1])
                    tolerance = max(tolerance, 0.5000001 * 10 ** (-decimals))
                delta = abs(float(value) - float(expected[field]))
                rows.append(
                    {
                        "result_id": rid,
                        "field": field,
                        "expected": expected[field],
                        "recomputed": value,
                        "absolute_difference": delta,
                        "tolerance": tolerance,
                        "method": method,
                        "provenance": provenance,
                        "status": "PASS" if delta <= tolerance else "MISMATCH",
                    }
                )
        return rows


def census(audit: Audit) -> None:
    dist = read_csv(DATA / "lmd_family_size_distribution.csv")
    family_count = sum(int(row["family_count"]) for row in dist)
    file_count = sum(int(row["file_count"]) for row in dist)
    multi_families = sum(
        int(row["family_count"]) for row in dist if int(row["component_size"]) > 1
    )
    multi_files = sum(
        int(row["file_count"]) for row in dist if int(row["component_size"]) > 1
    )
    largest = max(int(row["component_size"]) for row in dist)
    method = "recomputed_from_family_size_distribution"
    audit.add("lmd_census_frozen_downstream_identities", method, mean_a=file_count)
    audit.add("lmd_census_total_family_components", method, mean_a=family_count)
    audit.add("lmd_census_multi_member_families", method, mean_a=multi_families)
    audit.add("lmd_census_files_in_multi_member_families", method, mean_a=multi_files)
    audit.add("lmd_census_largest_family", method, mean_a=largest)

    monte = load_json(DATA / "lmd_monte_carlo_runs.json")["seed_level"]
    for protocol, token in (
        ("R1_80_10_10", "80_10_10"),
        ("R2_90_5_5", "90_5_5"),
        ("R3_90_10", "90_10"),
    ):
        rows = monte[protocol]["S0_FILE_SPLIT"]
        audit.add(
            f"lmd_{token}_test_family_contamination",
            "mean_of_1000_public_seed_level_simulations",
            mean_a=mean(r["test_family_contamination_rate"] for r in rows),
            n_seeds=len(rows),
        )
        rid = f"lmd_{token}_test_file_contamination"
        if audit.expected[rid].get("mean_a") is not None:
            audit.add(
                rid,
                "mean_of_1000_public_seed_level_simulations",
                mean_a=mean(r["test_file_contamination_rate"] for r in rows),
                n_seeds=len(rows),
            )
        else:
            audit.add(
                rid,
                "documented_simulation_count_for_intentionally_unreported_metric",
                n_seeds=len(rows),
            )


def pooled_nll(rows: list[dict], model_field="model"):
    by = defaultdict(dict)
    for row in rows:
        key = (
            row.get(model_field, ""),
            int(row["seed"]),
            row["condition"],
            row.get("cohort", "treated"),
        )
        by[key][row["family_id"]] = f(row.get("nll", row.get("metric")))
    return by


def legacy_and_capacity(audit: Audit) -> None:
    rows = read_csv(DATA / "capacity_nll_rows.csv")
    by = pooled_nll(rows)
    scale = PROTOCOL["architecture_checks"]["transformer_scale"]["models"]
    params = {
        "transformer_s": scale["S"]["parameter_count"],
        "transformer_m": scale["M"]["parameter_count"],
        "transformer_l": scale["L"]["parameter_count"],
        "tcn_384": PROTOCOL["architecture_checks"]["tcn"]["parameter_count"],
    }
    result_ids = {
        "transformer_s": "transformer_s_capacity",
        "transformer_m": "transformer_m_capacity",
        "transformer_l": "transformer_l_capacity",
        "tcn_384": "tcn_384_legacy",
    }
    family_values = {}
    for model in params:
        cohorts = {}
        for cohort in ("treated", "control", "clean_validation"):
            ids = list(by[model, 0, "clean", cohort])
            cohorts[cohort] = {
                family: mean(
                    by[model, seed, "family_leak", cohort][family]
                    - by[model, seed, "clean", cohort][family]
                    for seed in LEGACY_SEEDS
                )
                for family in ids
            }
        tv, cv = (
            np.asarray(list(cohorts["treated"].values())),
            np.asarray(list(cohorts["control"].values())),
        )
        family_values[model] = (tv, cv)
        stat = bootstrap_difference(tv, cv, CAPACITY_BOOTSTRAP_SEED + params[model])
        clean_nll = mean(
            by[model, seed, "clean", "treated"][family]
            for seed in LEGACY_SEEDS
            for family in by[model, seed, "clean", "treated"]
        )
        rid = result_ids[model]
        audit.add(
            rid,
            "recomputed_from_anonymous_family_seed_nll",
            tolerance=5e-5,
            **stat,
            relative_effect=float(-tv.mean() / clean_nll),
            n_families=len(tv),
        )

    legacy_rows = read_csv(DATA / "legacy_nll_rows.csv")
    legacy_by = pooled_nll(
        [row for row in legacy_rows if row["model"] == "transformer_l_legacy"]
    )
    # family_order preserves the original deterministic bootstrap order.
    legacy_order = {
        row["family_id"]: int(row["family_order"])
        for row in legacy_rows
        if row["model"] == "transformer_l_legacy"
    }
    legacy_treated_ids = sorted(
        legacy_by["transformer_l_legacy", 0, "clean", "treated"], key=legacy_order.get
    )
    legacy_control_ids = sorted(
        legacy_by["transformer_l_legacy", 0, "clean", "control"], key=legacy_order.get
    )
    legacy_treated = np.asarray(
        [
            mean(
                legacy_by["transformer_l_legacy", seed, "family_leak", "treated"][
                    family
                ]
                - legacy_by["transformer_l_legacy", seed, "clean", "treated"][family]
                for seed in LEGACY_SEEDS
            )
            for family in legacy_treated_ids
        ]
    )
    legacy_control = np.asarray(
        [
            mean(
                legacy_by["transformer_l_legacy", seed, "family_leak", "control"][
                    family
                ]
                - legacy_by["transformer_l_legacy", seed, "clean", "control"][family]
                for seed in LEGACY_SEEDS
            )
            for family in legacy_control_ids
        ]
    )
    legacy_clean_nll = mean(
        legacy_by["transformer_l_legacy", seed, "clean", "treated"][family]
        for seed in LEGACY_SEEDS
        for family in legacy_treated_ids
    )
    legacy_stat = bootstrap_difference(
        legacy_treated, legacy_control, LEGACY_CONFIRMATORY_SEED
    )
    audit.add(
        "lmd_transformer_l_legacy_confirmatory",
        "exact_preregistered_estimator_recomputed_from_public_family_seed_nll",
        effect=legacy_stat["effect"],
        relative_effect=float(-legacy_treated.mean() / legacy_clean_nll),
        ci_low=legacy_stat["ci_low"],
        ci_high=legacy_stat["ci_high"],
        p_raw=legacy_stat["p_raw"],
        n_families=len(legacy_treated),
        n_seeds=len(LEGACY_SEEDS),
        bootstrap_B=BOOTSTRAP_SAMPLES,
    )

    labels = {
        r["family_id"]: r["normalized_classification"]
        for r in read_csv(DATA / "normalized_subset_rows.csv")
    }
    lby = legacy_by
    selected = sorted(
        fam
        for fam, label in labels.items()
        if label == "NORMALIZED_STRUCTURAL_NONEXACT"
    )
    treated = [
        mean(
            lby["transformer_l_legacy", seed, "family_leak", "treated"][fam]
            - lby["transformer_l_legacy", seed, "clean", "treated"][fam]
            for seed in LEGACY_SEEDS
        )
        for fam in selected
    ]
    controls = [
        mean(
            lby["transformer_l_legacy", seed, "family_leak", "control"][fam]
            - lby["transformer_l_legacy", seed, "clean", "control"][fam]
            for seed in LEGACY_SEEDS
        )
        for fam in lby["transformer_l_legacy", 0, "clean", "control"]
    ]
    clean_subset = mean(
        lby["transformer_l_legacy", seed, "clean", "treated"][fam]
        for seed in LEGACY_SEEDS
        for fam in selected
    )
    normalized = load_json(FROZEN / "normalized_structural_summary.json")
    audit.add(
        "lmd_structurally_nonexact_normalized",
        "point_estimate_recomputed_from_joined_public_rows",
        effect=mean(treated) - mean(controls),
        relative_effect=-mean(treated) / clean_subset,
        n_families=len(selected),
    )
    audit.add(
        "lmd_structurally_nonexact_normalized",
        "verified_from_frozen_normalized_structural_bootstrap_summary",
        provenance=VERIFIED,
        ci_low=normalized["ci95"][0],
        ci_high=normalized["ci95"][1],
        n_families=len(selected),
        bootstrap_B=normalized["bootstrap_samples"],
    )

    clean_test = read_csv(DATA / "clean_test_nll_rows.csv")
    cby = defaultdict(dict)
    for row in clean_test:
        cby[int(row["seed"]), row["condition"]][row["family_id"]] = f(row["nll"])
    fams = [
        r["family_id"]
        for r in sorted(clean_test, key=lambda r: int(r["family_order"]))
        if int(r["seed"]) == 0 and r["condition"] == "clean"
    ]
    seen = set()
    fams = [x for x in fams if not (x in seen or seen.add(x))]
    validation = [
        mean(
            cby[seed, "family_leak"][fam] - cby[seed, "clean"][fam]
            for seed in LEGACY_SEEDS
        )
        for fam in fams
    ]
    stat = bootstrap_mean(validation, CLEAN_VALIDATION_SEED)
    audit.add(
        "lmd_clean_validation_generalization",
        "recomputed_from_family_disjoint_public_nll_rows",
        effect=mean(validation),
        ci_low=stat["ci_low"],
        ci_high=stat["ci_high"],
        n_families=len(validation),
        bootstrap_B=BOOTSTRAP_SAMPLES,
    )

    names = ("transformer_s", "transformer_m", "transformer_l")
    x = np.log([params[name] for name in names])
    observed = float(
        np.polyfit(
            x,
            [
                family_values[name][0].mean() - family_values[name][1].mean()
                for name in names
            ],
            1,
        )[0]
    )
    rng = np.random.default_rng(CAPACITY_BOOTSTRAP_SEED)
    slopes = []
    for _ in range(BOOTSTRAP_SAMPLES):
        taus = [
            rng.choice(family_values[name][0], len(family_values[name][0]), True).mean()
            - rng.choice(
                family_values[name][1], len(family_values[name][1]), True
            ).mean()
            for name in names
        ]
        slopes.append(np.polyfit(x, taus, 1)[0])
    slopes = np.asarray(slopes)
    audit.add(
        "transformer_capacity_trend_slope",
        "recomputed_from_anonymous_family_effect_rows",
        effect=observed,
        ci_low=float(np.quantile(slopes, 0.025)),
        ci_high=float(np.quantile(slopes, 0.975)),
        p_raw=float(min(1, 2 * min(np.mean(slopes <= 0), np.mean(slopes >= 0)))),
        n_families=sum(len(family_values[name][0]) for name in names),
        bootstrap_B=BOOTSTRAP_SAMPLES,
    )


def phase2(audit: Audit) -> None:
    rows = read_csv(DATA / "phase2_nll_rows.csv")
    maps = defaultdict(dict)
    for row in rows:
        maps[int(row["seed"]), row["condition"]][row["family_id"]] = f(row["nll"])
    order = {row["family_id"]: int(row["family_order"]) for row in rows}
    families = sorted(next(iter(maps.values())), key=order.get)
    pooled = {
        condition: {
            fam: mean(maps[seed, condition][fam] for seed in PHASE2_SEEDS)
            for fam in families
        }
        for condition in PHASE2_CONDITIONS
    }
    for condition, rid in (
        ("clean", "phase2_transformer_l_clean_nll"),
        ("unrelated_donor", "phase2_transformer_l_unrelated_donor_nll"),
        ("same_family_donor", "phase2_transformer_l_same_family_donor_nll"),
    ):
        audit.add(
            rid,
            "recomputed_from_anonymous_family_seed_rows",
            mean_a=mean(pooled[condition].values()),
            n_families=len(families),
            n_seeds=len(PHASE2_SEEDS),
        )
    contrasts = {
        "phase2_transformer_l_same_family_donor_vs_clean": (
            "same_family_donor",
            "clean",
        ),
        "phase2_transformer_l_unrelated_donor_vs_clean": ("unrelated_donor", "clean"),
        "phase2_transformer_l_same_family_donor_vs_unrelated_donor": (
            "same_family_donor",
            "unrelated_donor",
        ),
    }
    for rid, (left, right) in contrasts.items():
        diffs = np.asarray([pooled[left][fam] - pooled[right][fam] for fam in families])
        stat = bootstrap_mean_loop(diffs, PHASE2_BOOTSTRAP_SEED, plus_one=True)
        audit.add(
            rid,
            "recomputed_frozen_family_bootstrap",
            **stat,
            mean_a=mean(pooled[left].values()),
            mean_b=mean(pooled[right].values()),
            relative_effect=-mean(diffs) / mean(pooled[right].values()),
            n_seeds=len(PHASE2_SEEDS),
        )


def cross_paradigm(audit: Audit) -> None:
    rows = read_csv(DATA / "cross_paradigm_nll_rows.csv")
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["paradigm"], int(row["seed"]), row["condition"]][
            row["family_id"]
        ] = f(row["metric"])
    for paradigm, rid in (
        ("conditional_vae", "cross_paradigm_conditional_vae"),
        ("latent_diffusion", "cross_paradigm_latent_diffusion"),
    ):
        order = {
            row["family_id"]: int(row["family_order"])
            for row in rows
            if row["paradigm"] == paradigm
        }
        fams = sorted(
            grouped[paradigm, CROSS_PARADIGM_SEEDS[0], "clean"], key=order.get
        )
        pooled = {
            condition: {
                fam: mean(
                    grouped[paradigm, seed, condition][fam]
                    for seed in CROSS_PARADIGM_SEEDS
                )
                for fam in fams
            }
            for condition in CROSS_PARADIGM_CONDITIONS
        }
        diffs = np.asarray(
            [
                pooled["same_family_donor"][fam] - pooled["unrelated_donor"][fam]
                for fam in fams
            ]
        )
        stat = bootstrap_mean_loop(diffs, CROSS_PARADIGM_BOOTSTRAP_SEED)
        rng = np.random.default_rng(CROSS_PARADIGM_SIGN_SEED)
        null = (
            diffs * rng.choice((-1.0, 1.0), size=(BOOTSTRAP_SAMPLES, len(diffs)))
        ).mean(axis=1)
        p_sign = float(
            (np.count_nonzero(np.abs(null) >= abs(diffs.mean())) + 1)
            / (BOOTSTRAP_SAMPLES + 1)
        )
        audit.add(
            rid,
            "point_CI_and_sign_randomization_recomputed_from_public_family_rows",
            mean_a=mean(pooled["same_family_donor"].values()),
            mean_b=mean(pooled["unrelated_donor"].values()),
            effect=mean(diffs),
            ci_low=stat["ci_low"],
            ci_high=stat["ci_high"],
            p_raw=p_sign,
            n_families=len(fams),
            n_seeds=len(CROSS_PARADIGM_SEEDS),
            bootstrap_B=BOOTSTRAP_SAMPLES,
        )


def pdmx(audit: Audit) -> None:
    rows = read_csv(DATA / "pdmx_nll_rows.csv")
    by = pooled_nll([{"model": "pdmx", **r} for r in rows])
    deltas = read_csv(DATA / "pdmx_family_deltas.csv")
    values = {
        cohort: np.asarray(
            [
                f(row["mean_delta"])
                for row in sorted(
                    (x for x in deltas if x["split"] == cohort),
                    key=lambda x: int(x["family_order"]),
                )
            ]
        )
        for cohort in ("treated", "control")
    }
    stat = bootstrap_difference(
        values["treated"], values["control"], PDMX_BOOTSTRAP_SEED
    )
    clean = mean(
        by["pdmx", seed, "clean", "treated"][fam]
        for seed in LEGACY_SEEDS
        for fam in by["pdmx", seed, "clean", "treated"]
    )
    audit.add(
        "pdmx_reduced_external",
        "recomputed_from_anonymous_family_seed_nll",
        **stat,
        relative_effect=float(-values["treated"].mean() / clean),
        n_families=len(values["treated"]),
        n_seeds=len(LEGACY_SEEDS),
    )


def metric_rows(path: Path):
    out = defaultdict(dict)
    for row in read_csv(path):
        out[row["model"], row["condition"], row["metric"]][row["family_id"]] = f(
            row["value"]
        )
    return out


def musical_and_generation(audit: Audit) -> None:
    data = metric_rows(DATA / "musical_family_metrics.csv")
    canonical = {
        row["key"]: row
        for row in read_csv(FROZEN / "musical_canonical_holm_results.csv")
    }
    three_condition = {
        f"{row['model']}|{row['metric']}|{row['contrast']}": row
        for row in read_csv(FROZEN / "musical_three_condition_effects.csv")
    }
    mapping = {
        "musical_transformer_l_pitch_class_histogram_jsd_to_receiver": "pitch_class_histogram_jsd_to_receiver",
        "musical_transformer_l_pitch_interval_jsd_to_receiver": "pitch_interval_jsd_to_receiver",
        "musical_transformer_l_ioi_wasserstein": "ioi_wasserstein",
        "surface_null_note_density": "note_density",
        "surface_null_polyphony": "polyphony",
        "surface_null_pitch_range": "pitch_range",
        "surface_null_qualified_note_ratio": "qualified_note_ratio",
        "surface_null_onset_density": "onset_density",
    }
    for rid, metric in mapping.items():
        left = data["transformer_l", "same_family_donor", metric]
        right = data["transformer_l", "unrelated_donor", metric]
        fams = [x for x in left if x in right]
        diffs = np.asarray([left[x] - right[x] for x in fams])
        key = f"transformer_l|{metric}|same_minus_unrelated"
        c = canonical[key]
        t = three_condition.get(key, c)
        audit.add(
            rid,
            "point_estimate_recomputed_from_public_family_metrics",
            mean_a=mean(left[x] for x in fams),
            mean_b=mean(right[x] for x in fams),
            effect=mean(diffs),
        )
        audit.add(
            rid,
            "verified_from_frozen_canonical_bootstrap_and_multiplicity_audit",
            provenance=VERIFIED,
            ci_low=f(t["ci95_low"]),
            ci_high=f(t["ci95_high"]),
            p_raw=f(t["raw_p"]),
            p_adjusted=f(c["global_holm_p"]),
        )

    generation = metric_rows(DATA / "generation_family_metrics.csv")
    gsummary = load_json(FROZEN / "generation_statistics_summary.json")["results"][
        "transformer_l"
    ]["same_minus_unrelated"]
    gmap = {
        "generation_normalized_shared_span": "family_reference_longest_normalized_span",
        "generation_copy_tau": "family_reference_Copy_at_tau",
        "generation_exact_multi_bar_extraction": "exact_multi_bar_extraction",
    }
    for rid, metric in gmap.items():
        left = generation["transformer_l", "same_family_donor", metric]
        right = generation["transformer_l", "unrelated_donor", metric]
        fams = [x for x in left if x in right]
        diffs = [left[x] - right[x] for x in fams]
        meta = gsummary[metric]
        stat = bootstrap_mean(diffs, int(meta["bootstrap_seed"]), plus_one=True)
        audit.add(rid, "recomputed_from_public_generation_family_metrics", **stat)


def relatedness(audit: Audit) -> None:
    rows = read_csv(DATA / "relatedness_features.csv")
    field = "longest_shared_normalized_event_subsequence"
    x = np.asarray([f(r[field]) for r in rows])
    y = np.asarray([f(r["gain"]) for r in rows])
    point = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(RELATEDNESS_SPEARMAN_SEED)
    draws = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, len(x), len(x))
        value = spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(value):
            draws.append(value)
    audit.add(
        "relatedness_spearman_rho",
        "recomputed_family_bootstrap_spearman",
        effect=point,
        ci_low=float(np.quantile(draws, 0.025)),
        ci_high=float(np.quantile(draws, 0.975)),
        n_families=len(x),
        bootstrap_B=BOOTSTRAP_SAMPLES,
    )

    try:
        from sklearn.linear_model import HuberRegressor
    except ImportError as error:
        raise SystemExit(
            "scikit-learn is required for the Huber reproduction; install the locked reproduction extra"
        ) from error
    controls = np.asarray(
        [
            [
                f(r["receiver_length"]),
                f(r["donor_token_count"]),
                f(r["family_window_count_proxy"]),
            ]
            for r in rows
        ]
    )
    controls = (controls - controls.mean(0)) / controls.std(0)
    design = np.column_stack(((x - x.mean()) / x.std(), controls))
    model = HuberRegressor().fit(design, y)
    rng = np.random.default_rng(RELATEDNESS_HUBER_SEED)
    coefficients = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, len(x), len(x))
        try:
            coefficients.append(HuberRegressor().fit(design[idx], y[idx]).coef_[0])
        except ValueError:
            pass
    audit.add(
        "relatedness_robust_huber",
        "recomputed_HuberRegressor_and_family_bootstrap",
        effect=float(model.coef_[0]),
        ci_low=float(np.quantile(coefficients, 0.025)),
        ci_high=float(np.quantile(coefficients, 0.975)),
        n_families=len(x),
        bootstrap_B=BOOTSTRAP_SAMPLES,
    )


def localization(audit: Audit) -> None:
    rows = read_csv(DATA / "token_localization_family_rows.csv")
    summary = load_json(FROZEN / "localization_summary.json")["scales"]
    for scale in (4, 8, 16):
        block = [r for r in rows if int(r["scale"]) == scale]
        for kind in ("shared", "nonshared", "differential"):
            key = f"same_minus_unrelated_{kind}"
            values = [f(r[key]) for r in block if is_finite(r[key])]
            meta = summary[str(scale)][key]
            stat = bootstrap_mean(values, int(meta["bootstrap_seed"]), plus_one=True)
            rid = f"token_localization_{scale}_{'paired_differential' if kind == 'differential' else 'marginal_' + kind}"
            extra = {}
            if kind == "differential":
                common = [
                    r
                    for r in block
                    if is_finite(r["same_minus_unrelated_differential"])
                ]
                extra = {
                    "mean_a": mean(f(r["same_minus_unrelated_shared"]) for r in common),
                    "mean_b": mean(
                        f(r["same_minus_unrelated_nonshared"]) for r in common
                    ),
                }
            audit.add(
                rid, "recomputed_from_public_family_localization_rows", **stat, **extra
            )


def mitigation_and_imperfect(audit: Audit) -> None:
    mitigation = load_json(FROZEN / "mitigation_summary.json")
    cost = load_json(FROZEN / "mitigation_data_cost_summary.json")
    for key, rid in (
        ("S0", "mitigation_s0"),
        ("S1", "mitigation_s1"),
        ("S2", "mitigation_s2"),
    ):
        row = mitigation[key]
        effect = (
            cost["component_assignment"]["file_reassignment_ratio"]
            if key == "S2"
            else 0.0
        )
        audit.add(
            rid,
            "verified_from_public_nonidentifying_split_sufficient_statistics",
            provenance=VERIFIED,
            tolerance=5e-4,
            mean_a=row["family_overlap_count"],
            mean_b=row["discarded_token_ratio"],
            effect=effect,
            n_files=row["files_retained"],
        )
    delete = cost["delete_all_multimember_families_counterfactual"]
    audit.add(
        "mitigation_delete_all_multifamily",
        "verified_from_public_counterfactual_sufficient_statistics",
        provenance=VERIFIED,
        tolerance=5e-4,
        mean_a=delete["token_deletion_ratio"],
        n_files=cost["source_files"] - delete["files_deleted"],
    )

    rows = read_csv(DATA / "imperfect_inference_runs.csv")

    def select(condition, variant, recall=None, fp=None):
        out = [
            r for r in rows if r["condition"] == condition and r["variant"] == variant
        ]
        if recall is not None:
            out = [r for r in out if abs(f(r["edge_recall_target"]) - recall) < 1e-12]
        if fp is not None:
            out = [
                r for r in out if abs(f(r["fp_injection_ratio_target"]) - fp) < 1e-12
            ]
        return out

    baseline = select("file_level_random", "baseline")
    cross = np.asarray(
        [f(r["residual_known_cross_split_family_count"]) for r in baseline]
    )
    audit.add(
        "imperfect_inference_file_level_baseline_residual",
        "recomputed_from_100_public_simulation_runs",
        mean_a=float(cross.mean()),
        ci_low=float(np.quantile(cross, 0.025)),
        ci_high=float(np.quantile(cross, 0.975)),
    )
    audit.add(
        "imperfect_inference_file_level_baseline_test_family_contamination",
        "recomputed_from_100_public_simulation_runs",
        mean_a=mean(f(r["residual_contaminated_test_family_rate"]) for r in baseline),
    )
    perfect = select("perfect_reference", "upper_bound", 1.0, 0.0)
    audit.add(
        "imperfect_inference_perfect_reference_residual",
        "recomputed_from_public_simulation_runs",
        mean_a=mean(f(r["residual_known_cross_split_family_count"]) for r in perfect),
    )
    for recall in (0.95, 0.90, 0.50):
        group = select("false_negative", "edge_drop", recall=recall)
        token = f"{recall:.2f}".replace(".", "_")
        values = np.asarray(
            [f(r["residual_known_cross_split_family_count"]) for r in group]
        )
        audit.add(
            f"imperfect_inference_recall_{token}_residual",
            "recomputed_from_100_public_simulation_runs",
            mean_a=float(values.mean()),
            ci_low=float(np.quantile(values, 0.025)),
            ci_high=float(np.quantile(values, 0.975)),
        )
        audit.add(
            f"imperfect_inference_recall_{token}_pairwise_recall",
            "recomputed_from_100_public_simulation_runs",
            mean_a=mean(f(r["pairwise_same_family_recall"]) for r in group),
        )
    for fp in (0.01, 0.05):
        group = select("false_positive", "bounded", recall=1.0, fp=fp)
        token = f"{fp:.2f}".replace(".", "_")
        audit.add(
            f"imperfect_inference_fp_{token}_split_ratio_error",
            "recomputed_from_100_public_simulation_runs",
            mean_a=mean(f(r["split_abs_ratio_error"]) for r in group),
        )
        audit.add(
            f"imperfect_inference_fp_{token}_relation_precision",
            "recomputed_from_100_public_simulation_runs",
            mean_a=mean(f(r["pairwise_relation_precision"]) for r in group),
        )
        values = np.asarray([f(r["over_merge_component_count"]) for r in group])
        audit.add(
            f"imperfect_inference_fp_{token}_overmerge_components",
            "recomputed_from_100_public_simulation_runs",
            mean_a=float(values.mean()),
            ci_low=float(np.quantile(values, 0.025)),
            ci_high=float(np.quantile(values, 0.975)),
        )
    audit.add(
        "imperfect_inference_full_universe_files",
        "direct_count_recomputed_from_every_public_run",
        mean_a=int(rows[0]["reference_files"]),
    )
    audit.add(
        "imperfect_inference_full_universe_reference_families",
        "direct_count_recomputed_from_every_public_run",
        mean_a=int(rows[0]["reference_families"]),
    )
    audit.add(
        "imperfect_inference_simulation_run_count", "public_row_count", mean_a=len(rows)
    )


def write_outputs(output: Path, rows: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output / "REPRODUCED_FIELD_AUDIT.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = defaultdict(int)
    for row in rows:
        counts[row["status"]] += 1
    provenance_counts = defaultdict(int)
    for row in rows:
        if row.get("provenance"):
            provenance_counts[row["provenance"]] += 1
    payload = {
        "status": "PASS"
        if not counts["MISMATCH"] and not counts["UNREPRODUCED"]
        else "FAIL",
        "field_status_counts": dict(counts),
        "numeric_field_provenance_counts": dict(provenance_counts),
        "interpretation": {
            RECOMPUTED: "calculated from public analysis rows or public simulation rows",
            VERIFIED: "compared with a released frozen aggregate because public analysis rows are insufficient to reconstruct the exact display-chain statistic",
        },
        "formal_results_changed": False,
        "formal_protocol_changed": False,
        "gpu_required": False,
    }
    (output / "REPRODUCTION_STATUS.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    by_result = defaultdict(list)
    for row in rows:
        by_result[row["result_id"]].append(row)
    lines = [
        "# Paper-statistics audit",
        "",
        f"- Numerical fields recomputed from public rows: **{provenance_counts[RECOMPUTED]}**",
        f"- Numerical fields verified from frozen summaries: **{provenance_counts[VERIFIED]}**",
        "",
        "| Result | Fields | Status |",
        "|---|---:|---|",
    ]
    for rid, items in sorted(by_result.items()):
        status = (
            "PASS"
            if all(x["status"] in ("PASS", "NOT_APPLICABLE") for x in items)
            else "FAIL"
        )
        lines.append(f"| `{rid}` | {len(items)} | {status} |")
    (output / "REPRODUCED_RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))


def verify_manifest() -> None:
    manifest = load_json(ROOT / "reproduction/PUBLIC_REPRODUCTION_MANIFEST.json")
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if not path.is_file():
            raise SystemExit(f"reproduction manifest failure: missing {item['path']}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.stat().st_size != item["bytes"] or digest != item["sha256"]:
            raise SystemExit(f"reproduction manifest failure: {item['path']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "_reproduced_paper_statistics"
    )
    args = parser.parse_args()
    verify_manifest()
    expected_rows = load_json(EXPECTED)["results"]
    audit = Audit({row["result_id"]: row for row in expected_rows})
    census(audit)
    legacy_and_capacity(audit)
    phase2(audit)
    cross_paradigm(audit)
    pdmx(audit)
    musical_and_generation(audit)
    relatedness(audit)
    localization(audit)
    mitigation_and_imperfect(audit)
    write_outputs(args.output, audit.rows())


if __name__ == "__main__":
    main()
