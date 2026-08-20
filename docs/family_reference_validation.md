# Family-reference validation and its limits

The paper uses a conservative machine-built relation to define known
same-work families. The available evidence supports using that relation as a
high-precision reference, but it does not turn it into complete ground truth.

## Evidence already completed

| Evidence | Result | What it supports | What it does not support |
|---|---:|---|---|
| Upstream conservative-union benchmark | precision 0.9133, recall 0.2830, F1 0.4322 | The adopted high-threshold relation is comparatively precise on the upstream benchmark | Accuracy on the 100 experimental pairs or exhaustive LMD coverage |
| Outcome-blind symbolic check of all 100 experimental pairs | interval-4-gram same-family advantage 0.9208, 95% CI [0.8781, 0.9578], positive in 100% of families | The selected same-family pairs share substantially more symbolic musical structure than preselected matched cross-family controls | Human judgment, perfect negative labels, or detector recall |
| Structurally non-exact subset | 95 families; advantage 0.9166, 95% CI [0.8712, 0.9557] | The result is not explained only by byte/canonical duplicates | That every pair is a distinct human-perceived arrangement |
| Graph-noise sensitivity | 2,900 released simulation rows | How split behavior degrades under simulated missing or false edges | Real-world detector precision/recall |

The 100-pair symbolic audit was post-hoc but outcome-blind: it did not read a
checkpoint, NLL value, generation output, or Phase-2 effect. Each same-family
pair was compared with a feature-matched negative selected before model
outcomes. Its primary statistic used 10,000 paired-family bootstrap draws with
seed 20260818.

## Human annotation status

No completed independent human labels are available. Existing audit packages
contain blank forms only: a 39-pair semantic-relation package and a later
160-pair package covering all 100 predicted-positive pairs plus 60 boundary
negatives. Therefore this release does **not** report human precision, recall,
agreement, Cohen's kappa, or adjudicated arrangement labels.

No labels were filled in during this release repair, because doing so would be
a new study rather than documentation of an existing result.

## Reviewer-safe claim

Use this wording:

> Families are connected components of a conservative, high-threshold
> reference relation. Existing upstream benchmarking and an outcome-blind
> symbolic audit support its use as a high-precision operational reference,
> but coverage is incomplete and independent human validation of the study
> pairs remains unavailable. Consequently, “zero overlap” means zero known
> overlap under this relation.

Do not call the graph complete ground truth, do not claim global recall, and do
not interpret unknown cross-family pairs as certainly unrelated.
