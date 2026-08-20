# Reproduction-data dictionary

## Analysis-unit tables

| File | Unit | Purpose |
|---|---|---|
| `capacity_nll_rows.csv` | model × seed × condition × cohort × family | Legacy/capacity effects and capacity trend |
| `legacy_nll_rows.csv` | model × seed × condition × cohort × family | Confirmatory and normalized sensitivity |
| `phase2_nll_rows.csv` | seed × condition × family | Clean/unrelated/same-family controlled contrasts |
| `cross_paradigm_nll_rows.csv` | paradigm × seed × condition × family | Conditional-VAE and latent-diffusion contrasts |
| `pdmx_nll_rows.csv` | seed × condition × cohort × family | Reduced external analysis |
| `clean_test_nll_rows.csv` | seed × condition × family | Family-disjoint generalization control |
| `musical_family_metrics.csv` | model × condition × family × metric | Musical-property effects |
| `generation_family_metrics.csv` | model × condition × family × metric | Reproduction/copying indicators |
| `relatedness_features.csv` | family | Relatedness association and controls |
| `token_localization_family_rows.csv` | family index × scale | Shared/nonshared token-localization effects |
| `imperfect_inference_runs.csv` | noise condition × seed | Mitigation under graph noise |
| `lmd_family_size_distribution.csv` | family-size bin | Census reconstruction |
| `lmd_monte_carlo_runs.json` | split protocol × seed | Random-split leakage reconstruction |

## Common fields

- `family_id`: release-specific pseudonym used only for pairing.
- `family_order`: nonidentifying ordinal preserving the frozen estimator's
  deterministic family ordering.
- `condition`: model-training exposure condition.
- `cohort`: treated, control, or clean-validation group.
- `nll` / `metric` / `value`: scalar model objective or derived measurement.
- `token_count`: scalar weighting/denominator metadata; never token content.
- `seed`: frozen training or simulation seed.

## Frozen sufficient-statistics tables

Files under `frozen/` contain canonical multiplicity results, bootstrap
registries, and nonidentifying split summaries. The field audit labels uses of
these files explicitly; they are not silently substituted for recomputation
from analysis units.
