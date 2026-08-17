# Frozen study protocol

## Family leakage

A same-work family contains files linked under the adopted family definition. Leakage occurs when at least one family member is in training while another is an evaluation receiver.

## Controlled intervention

For each frozen treated family, one receiver and one designated donor are selected without using model outcomes. In `clean`, the donor and all treated siblings are absent. In `family_leak`, only the designated donor is inserted. Receivers, controls, and clean-validation families stay outside training.

## Token-budget matching

Adding donor tokens changes optimization exposure. LeakageBench deterministically removes nearest-token family-disjoint base samples, producing approximately token-budget-matched clean and family-leak conditions. The formal runs' maximum total relative difference was `5.062852358702753e-07`. No preregistered token tolerance existed, so this observed value is disclosed rather than promoted to a post-hoc threshold. The toolkit records pairwise and total differences and optionally enforces user-specified tolerances.

## Controls and estimand

Controls measure run-wide drift unrelated to treated donor exposure. For family (f) and seed (s):

```text
Delta_treated = NLL_family_leak - NLL_clean  (treated families)
Delta_control = NLL_family_leak - NLL_clean  (control families)
tau = mean(Delta_treated) - mean(Delta_control)
```

Seed effects are removed through within-seed differences. Families receive equal weight.

## Bootstrap

Confidence intervals use a frozen-seed, family-cluster bootstrap. The family—not a window, note, or token—is the statistical unit. Treating windows as independent would understate uncertainty.
