from __future__ import annotations
import math
import pytest
from leakagebench_midi import (
    analyze_effect,
    audit_split,
    build_contamination,
    family_aware_split,
)


def sample():
    base = [{"id": "b", "family_id": "b", "tokens": 100}]
    treated = [
        {
            "family_id": "t",
            "members": [
                {"id": "r", "family_id": "t", "tokens": 100},
                {"id": "d", "family_id": "t", "tokens": 101},
            ],
        }
    ]
    control = [{"id": "c", "family_id": "c", "tokens": 100}]
    val = [{"id": "v", "family_id": "v", "tokens": 100}]
    assignment = [{"family_id": "t", "receiver_id": "r", "donor_id": "d"}]
    return base, treated, control, val, assignment


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda x: x[4][0].update(donor_id="c"), "donor"),
        (lambda x: x[4][0].update(receiver_id="missing"), "receiver"),
        (lambda x: x[1][0]["members"][0].update(family_id="wrong"), "family"),
        (lambda x: x[4][0].update(donor_id="r"), "donor"),
        (lambda x: x[4].append(dict(x[4][0])), "duplicate"),
    ],
)
def test_bad_assignments(mutation, message):
    x = list(sample())
    mutation(x)
    with pytest.raises(ValueError, match=message):
        build_contamination(*x, seed=1)


def test_duplicate_receiver_and_token_tolerance():
    x = list(sample())
    x[1].append(
        {
            "family_id": "u",
            "members": [
                {"id": "u1", "family_id": "u", "tokens": 20},
                {"id": "u2", "family_id": "u", "tokens": 21},
            ],
        }
    )
    x[4].append({"family_id": "u", "receiver_id": "r", "donor_id": "u2"})
    with pytest.raises(ValueError, match="receiver"):
        build_contamination(*x, seed=1)
    x = list(sample())
    x[1][0]["members"][1]["tokens"] = 10000
    with pytest.raises(ValueError, match="token"):
        build_contamination(*x, seed=1, max_pair_token_rel_diff=0.1)


@pytest.mark.parametrize(
    "ratios", [(-0.1, 0.1, 1.0), (math.nan, 0.5, 0.5), (math.inf, 0, 0), (1.1, 0, 0)]
)
def test_bad_ratios(ratios):
    with pytest.raises(ValueError):
        family_aware_split([{"id": "a", "family_id": "a", "tokens": 1}], ratios, 1)


def result_rows():
    out = []
    for seed in range(3):
        for split, ids in [
            ("treated", ["t"]),
            ("control", ["c"]),
            ("clean_validation", ["v"]),
        ]:
            for condition, nll in [
                ("clean", 2.0),
                ("family_leak", 1.9 if split == "treated" else 2.0),
            ]:
                for fid in ids:
                    out.append(
                        {
                            "dataset": "d",
                            "architecture": "a",
                            "model_size": "s",
                            "seed": seed,
                            "condition": condition,
                            "split": split,
                            "family_id": fid,
                            "nll": nll,
                        }
                    )
    return out


def family_manifest(rows=None):
    rows = result_rows() if rows is None else rows
    splits = {}
    for row in rows:
        splits.setdefault(row["family_id"], row["split"])
    return {
        "format_version": 1,
        "families": [
            {"family_id": family_id, "split": split}
            for family_id, split in sorted(splits.items())
        ],
    }


def test_family_manifest_split_corruption():
    rows = result_rows()
    manifest = family_manifest(rows)
    manifest["families"][0]["split"] = (
        "treated" if manifest["families"][0]["split"] != "treated" else "control"
    )
    with pytest.raises(
        ValueError, match="manifest split does not match analysis split"
    ):
        analyze_effect(rows, family_manifest=manifest)


def test_cross_cohort_family_collision():
    rows = result_rows()
    collision = dict(
        next(
            row
            for row in rows
            if row["family_id"] == "t"
            and row["seed"] == 0
            and row["condition"] == "clean"
        )
    )
    collision["split"] = "control"
    with pytest.raises(ValueError, match="multiple analysis cohorts"):
        analyze_effect(rows + [collision], family_manifest=family_manifest(rows))


def test_family_manifest_cross_split_collision():
    rows = result_rows()
    manifest = family_manifest(rows)
    duplicate = dict(manifest["families"][0])
    duplicate["split"] = "treated" if duplicate["split"] != "treated" else "control"
    manifest["families"].append(duplicate)
    with pytest.raises(ValueError, match="multiple manifest splits"):
        analyze_effect(rows, family_manifest=manifest)


def test_duplicate_identical_and_conflicting_rows():
    r = result_rows()
    with pytest.raises(ValueError, match="duplicate"):
        analyze_effect(r + [dict(r[0])])
    bad = dict(r[0], nll=9)
    with pytest.raises(ValueError, match="duplicate"):
        analyze_effect(r + [bad])


def test_missing_pairs_empty_groups_and_nonfinite():
    r = result_rows()
    with pytest.raises(ValueError, match="missing result pair"):
        analyze_effect(
            [
                x
                for x in r
                if not (
                    x["family_id"] == "t"
                    and x["seed"] == 0
                    and x["condition"] == "clean"
                )
            ]
        )
    with pytest.raises(ValueError, match="treated cohort is empty"):
        analyze_effect([x for x in r if x["split"] != "treated"])
    with pytest.raises(ValueError, match="control cohort is empty"):
        analyze_effect([x for x in r if x["split"] != "control"])
    for val in (math.nan, math.inf):
        q = [dict(x) for x in r]
        q[0]["nll"] = val
        with pytest.raises(ValueError, match="non-finite"):
            analyze_effect(q)


@pytest.mark.parametrize("samples", [0, -1, 1.5, True])
def test_bootstrap_samples_must_be_positive_integer(samples):
    with pytest.raises(ValueError, match="positive integer"):
        analyze_effect(result_rows(), bootstrap_samples=samples)


def test_zero_denominator_is_undefined():
    out = audit_split([{"id": "a", "family_id": "a", "tokens": 1, "split": "train"}])
    assert (
        out["test_file_contamination_rate"] is None
        and out["test_family_contamination_rate"] is None
    )
