# Repository file guide

This page explains the public repository without repeating the Methods paper.

## Root

| File | Purpose |
|---|---|
| `.github/workflows/reproducibility.yml` | Runs the public reproduction and test workflow in GitHub Actions. |
| `.gitignore` | Excludes local environments, caches, data, logs, and checkpoints. |
| `.zenodo.json` | Metadata used for Zenodo software archiving. |
| `README.md` | Short project entry point. |
| `CITATION.cff` | Machine-readable citation metadata. |
| `LICENSE` | MIT license for code and documentation. |
| `pyproject.toml` | Python package metadata and dependencies. |
| `requirements-lock.txt` | Frozen versions used for release verification. |

## Configuration and documentation

| File | Purpose |
|---|---|
| `configs/protocol_v2.json` | Machine-readable public experimental protocol. |
| `configs/checkpoint_reference_outputs_v1_1.json` | Fixed outputs used to verify representative checkpoints. |
| `docs/METHODS_PUBLIC.md` | Public Methods description. |
| `docs/PROTOCOL_V2.md` | Human-readable detailed protocol. |
| `docs/REPRODUCIBILITY.md` | Reproduction instructions and expected outputs. |
| `docs/data_provenance.md` | Dataset-count and provenance scopes. |
| `docs/datasets.md` | Dataset acquisition and preparation notes. |
| `docs/family_reference_validation.md` | Validation and limitations of the reference family relation. |
| `docs/model_artifacts.md` | Checkpoint archives, loading, and verification. |
| `docs/REPOSITORY_GUIDE.md` | This file-by-file guide. |

## Python package

| File | Purpose |
|---|---|
| `leakagebench_midi/__init__.py` | Public package exports. |
| `leakagebench_midi/core.py` | Family graphs, splitting, leakage audits, census, and controlled-effect analysis. |
| `leakagebench_midi/structural.py` | MIDI canonicalization and structural comparison. |
| `leakagebench_midi/models/__init__.py` | Public model exports. |
| `leakagebench_midi/models/loader.py` | Safe checkpoint validation and model loading. |
| `leakagebench_midi/models/tokenizer.py` | Fixed REMI+ tokenizer. |
| `leakagebench_midi/models/transformer.py` | Transformer definition used by released weights. |
| `leakagebench_midi/models/tcn.py` | TCN definition used by released weights. |
| `leakagebench_midi/models/cross_paradigm.py` | Conditional VAE, latent diffusion, and neutral encoder definitions. |

## Commands

| File | Purpose |
|---|---|
| `scripts/build_family_map.py` | Builds family components from pairwise relations. |
| `scripts/build_family_split.py` | Creates component-atomic dataset splits. |
| `scripts/audit_split.py` | Measures known family leakage in a split. |
| `scripts/run_leakage_census.py` | Runs random-split leakage simulations. |
| `scripts/construct_contamination.py` | Builds controlled donor/receiver conditions. |
| `scripts/analyze_family_effects.py` | Analyzes paired family effects. |
| `scripts/classify_pair_structure.py` | Classifies MIDI pair structure. |
| `scripts/generate_synthetic_demo.py` | Generates the data-free synthetic example. |
| `scripts/run_synthetic_demo.sh` | Runs the synthetic workflow. |
| `scripts/reproduce_paper_statistics.py` | Audits published numerical results. |
| `scripts/reproduce_public_results.py` | Verifies and copies frozen tables and figures. |
| `scripts/reproduce_all.sh` | Runs the complete reviewer workflow. |
| `scripts/verify_model_checkpoints.py` | Verifies downloaded model files and representative outputs. |

## Synthetic example

| File | Purpose |
|---|---|
| `examples/synthetic_demo/README.md` | Explains the synthetic example. |
| `examples/synthetic_demo/spec.json` | Defines its artificial files and family relations. |

## Reproduction data

| File | Purpose |
|---|---|
| `reproduction/README.md` | Explains the public statistical reproduction bundle. |
| `reproduction/DATA_DICTIONARY.md` | Defines released columns and analysis units. |
| `reproduction/PUBLIC_REPRODUCTION_MANIFEST.json` | Lists reproduction files and hashes. |
| `reproduction/data/clean_test_nll_rows.csv` | Clean confirmatory test losses. |
| `reproduction/data/legacy_nll_rows.csv` | Legacy confirmatory loss rows. |
| `reproduction/data/capacity_nll_rows.csv` | Transformer capacity and TCN loss rows. |
| `reproduction/data/phase2_nll_rows.csv` | Phase-2 three-condition loss rows. |
| `reproduction/data/cross_paradigm_nll_rows.csv` | CVAE and diffusion endpoint rows. |
| `reproduction/data/pdmx_nll_rows.csv` | PDMX external-evaluation loss rows. |
| `reproduction/data/pdmx_family_deltas.csv` | PDMX family-level paired differences. |
| `reproduction/data/normalized_subset_rows.csv` | Structurally normalized subset results. |
| `reproduction/data/generation_family_metrics.csv` | Family-level generation metrics. |
| `reproduction/data/musical_family_metrics.csv` | Musical-property family metrics. |
| `reproduction/data/token_localization_family_rows.csv` | Localized shared-region metrics. |
| `reproduction/data/relatedness_features.csv` | Family-relatedness features and effects. |
| `reproduction/data/lmd_family_size_distribution.csv` | Reference family-size counts. |
| `reproduction/data/lmd_monte_carlo_runs.json` | Random file-split simulation runs. |
| `reproduction/data/imperfect_inference_runs.csv` | Noisy-family-inference simulation runs. |

## Frozen statistical summaries

| File | Purpose |
|---|---|
| `reproduction/frozen/census_summary.json` | Leakage census summary. |
| `reproduction/frozen/confirmatory_summary.json` | Confirmatory model summary. |
| `reproduction/frozen/transformer_scale_summary.json` | Transformer scaling summary. |
| `reproduction/frozen/capacity_trend_summary.json` | Capacity trend statistics. |
| `reproduction/frozen/cross_paradigm_summary.json` | CVAE and diffusion summary. |
| `reproduction/frozen/pdmx_summary.json` | PDMX summary. |
| `reproduction/frozen/phase2_summary.json` | Phase-2 summary. |
| `reproduction/frozen/normalized_structural_summary.json` | Normalized subset summary. |
| `reproduction/frozen/generation_statistics_summary.json` | Generation statistics summary. |
| `reproduction/frozen/localization_summary.json` | Token-localization summary. |
| `reproduction/frozen/relatedness_summary.json` | Relatedness analysis summary. |
| `reproduction/frozen/mitigation_summary.json` | Family-aware mitigation summary. |
| `reproduction/frozen/mitigation_data_cost_summary.json` | Mitigation data-cost summary. |
| `reproduction/frozen/imperfect_inference_summary.json` | Imperfect-inference robustness summary. |
| `reproduction/frozen/imperfect_inference_condition_summary.csv` | Condition-level noisy-inference results. |
| `reproduction/frozen/musical_bootstrap_summary.json` | Musical-metric bootstrap summary. |
| `reproduction/frozen/musical_condition_summary.csv` | Musical metrics by condition. |
| `reproduction/frozen/musical_paired_effects.csv` | Paired musical effects. |
| `reproduction/frozen/musical_three_condition_effects.csv` | Three-condition musical effects. |
| `reproduction/frozen/musical_canonical_holm_results.csv` | Holm-corrected canonical tests. |
| `reproduction/frozen/musical_holm_summary.json` | Holm correction summary. |

## Released results

| File | Purpose |
|---|---|
| `results/RESULTS_MANIFEST.json` | Hashes and sizes for released result files. |
| `results/manuscript_results_v2_public.json` | Public machine-readable result lock. |
| `results/manuscript_results_v2_public.csv` | Tabular form of the result lock. |
| `results/manuscript_results_v2_public.sha256` | Checksum for the JSON result lock. |
| `results/figures/fig1_leakage_landscape.svg` | Main leakage-prevalence figure. |
| `results/figures/fig2_phase2_core.svg` | Main Phase-2 figure. |
| `results/figures/fig3_model_dependence.svg` | Model-dependence figure. |
| `results/figures/fig4_musical_alignment.svg` | Musical-alignment figure. |
| `results/figures/fig5_mechanism_diagnostics.svg` | Mechanism-diagnostics figure. |
| `results/figures/fig6_imperfect_inference_robustness.svg` | Imperfect-inference robustness figure. |
| `results/figures/figure_prevalence.csv` | Plot data for prevalence. |
| `results/figures/figure_capacity.csv` | Plot data for capacity. |
| `results/figures/figure_external.csv` | Plot data for external evaluation. |
| `results/figures/figure_mitigation.csv` | Plot data for mitigation. |
| `results/tables/table1_final.svg`–`table6_final.svg` | Final rendered paper tables. |
| `results/tables/table_lmd_census.csv` | LMD census table values. |
| `results/tables/table_confirmatory.csv` | Confirmatory table values. |
| `results/tables/table_architecture_capacity.csv` | Architecture/capacity table values. |
| `results/tables/table_pdmx_external.csv` | PDMX table values. |
| `results/tables/table_mitigation.csv` | Mitigation table values. |

## Tests

| File | Purpose |
|---|---|
| `tests/test_validation_and_normalization.py` | Core validation and MIDI-normalization tests. |
| `tests/test_public_workflows.py` | Synthetic and command-workflow tests. |
| `tests/test_model_loader.py` | Checkpoint loader and model-forward tests. |
| `tests/test_release_safety.py` | Public-release content and checksum tests. |
| `tests/test_strong_reproduction_bundle.py` | Reproduction-data and result-field tests. |
