# Repository file guide

Each entry below states what the file contributes to reproduction.

## Project files

| File | Role |
|---|---|
| `.github/workflows/reproducibility.yml` | Runs the CPU reproduction suite on GitHub Actions. |
| `.gitignore` | Keeps datasets, prepared streams, checkpoints, runs, caches, and environments out of Git. |
| `.zenodo.json` | Supplies metadata for the Zenodo archive. |
| `CITATION.cff` | Machine-readable citation record. |
| `LICENSE` | MIT license for the software. |
| `README.md` | Introduces the project and the shortest reproduction paths. |
| `pyproject.toml` | Defines the Python package, dependencies, and test settings. |
| `requirements-lock.txt` | Pins the release verification environment. |

## Detector artifact

| File | Role |
|---|---|
| `artifacts/detector/README.md` | Lists the detector files and the inference command. |
| `artifacts/detector/config.json` | Stores the 47 selected features, score threshold, and component guards. |
| `artifacts/detector/shs_structural_detector.npz` | Portable gradient-boosted tree parameters used by the detector. |

## Configuration

| File | Role |
|---|---|
| `configs/checkpoint_reference_outputs_v1_1.json` | Reference forward-pass values for representative released weights. |
| `configs/protocol_v2.json` | Machine-readable settings used by the statistical reproduction. |
| `configs/training.json` | Formal seeds, model sizes, optimizer settings, and evaluation timesteps. |

## Documentation

| File | Role |
|---|---|
| `docs/METHODS_PUBLIC.md` | Compact Methods description matching the public implementation. |
| `docs/PROTOCOL_V2.md` | Detailed experimental protocol used by the frozen result set. |
| `docs/REPOSITORY_GUIDE.md` | Explains the role of every public file. |
| `docs/REPRODUCIBILITY.md` | Commands for statistics, LMD preparation, training, evaluation, and detector runs. |
| `docs/data_provenance.md` | Lists the datasets and source records behind each analysis. |
| `docs/datasets.md` | Links to LMD and explains the required local preparation. |
| `docs/family_reference_validation.md` | Documents how the reference relation was checked. |
| `docs/model_artifacts.md` | Describes checkpoint archives and their loader. |

## Python package

| File | Role |
|---|---|
| `leakagebench_midi/__init__.py` | Exports the public graph, split, census, and analysis API. |
| `leakagebench_midi/core.py` | Implements component construction, family-aware splitting, leakage audits, census simulation, and paired effects. |
| `leakagebench_midi/data.py` | Reads evaluation JSONL and memory-mapped packed training streams. |
| `leakagebench_midi/detector.py` | Extracts structural MIDI features, retrieves candidates, scores pairs, and builds guarded components. |
| `leakagebench_midi/models/__init__.py` | Exports model builders and the checkpoint loader. |
| `leakagebench_midi/models/cross_paradigm.py` | Defines the conditional VAE, neutral encoder, and latent diffusion model. |
| `leakagebench_midi/models/loader.py` | Validates checkpoint metadata and restores the matching architecture. |
| `leakagebench_midi/models/tcn.py` | Defines the causal dilated TCN baseline. |
| `leakagebench_midi/models/tokenizer.py` | Implements the fixed REMI+ tokenizer and four-bar windowing. |
| `leakagebench_midi/models/transformer.py` | Defines the causal Transformer used by the sequence experiments. |
| `leakagebench_midi/structural.py` | Canonicalizes note events and classifies structural pair relations. |

## Commands

| File | Role |
|---|---|
| `scripts/analyze_family_effects.py` | Computes paired family effects and family-level bootstrap intervals. |
| `scripts/audit_split.py` | Measures cross-split reference-family contamination. |
| `scripts/build_family_map.py` | Converts pairwise same-work edges into component IDs. |
| `scripts/build_family_split.py` | Assigns complete components to train, validation, and test. |
| `scripts/classify_pair_structure.py` | Compares two MIDI files after structural normalization. |
| `scripts/construct_contamination.py` | Creates token-matched clean and same-work training conditions. |
| `scripts/detect_same_work.py` | Runs the structural detector on a MIDI directory. |
| `scripts/evaluate_checkpoint.py` | Evaluates released sequence, VAE, or diffusion weights on rebuilt receiver windows. |
| `scripts/generate_synthetic_demo.py` | Creates a small artificial graph for data-free workflow tests. |
| `scripts/prepare_lmd.py` | Rebuilds and hash-checks the formal streams and evaluation windows from LMD-full. |
| `scripts/prepare_reference_graph.py` | Builds the 178,561-file reference graph from LMD-full and the pinned relation file. |
| `scripts/reproduce_all.sh` | Runs statistics, detector verification, artifact checks, and tests. |
| `scripts/reproduce_detector_statistics.py` | Recomputes ASAP detector metrics and the 10,000-sample composition bootstrap. |
| `scripts/reproduce_paper_statistics.py` | Recomputes the paper result fields from released analysis rows. |
| `scripts/reproduce_public_results.py` | Checks hashes and copies released tables and figures. |
| `scripts/run_leakage_census.py` | Simulates leakage under random file-level splits. |
| `scripts/run_synthetic_demo.sh` | Runs the data-free end-to-end graph and split example. |
| `scripts/simulate_imperfect_inference.py` | Runs 2,900 noisy-reference-graph split simulations with 100 fixed seeds. |
| `scripts/train_model.py` | Trains formal Transformer, conditional VAE, neutral encoder, and diffusion runs. |
| `scripts/verify_model_checkpoints.py` | Checks checkpoint hashes, metadata, loading, and reference outputs. |

## Reproduction bundle

| File | Role |
|---|---|
| `reproduction/DATA_DICTIONARY.md` | Defines each released dataset, row unit, and main columns. |
| `reproduction/PUBLIC_REPRODUCTION_MANIFEST.json` | Records every reproduction file, byte size, and SHA-256 hash. |
| `reproduction/README.md` | Introduces the public analysis and source-reconstruction bundle. |
| `reproduction/data/capacity_nll_rows.csv` | Family-level Transformer capacity and TCN NLL rows. |
| `reproduction/data/clean_test_nll_rows.csv` | Clean confirmatory test NLL rows. |
| `reproduction/data/cross_paradigm_nll_rows.csv` | Conditional VAE and latent diffusion family endpoints. |
| `reproduction/data/detector/asap_candidate_pairs.csv.gz` | Contains all 355,535 ASAP candidate-pair scores and labels. |
| `reproduction/data/detector/asap_files.csv` | Maps anonymous ASAP files to composition clusters for bootstrap resampling. |
| `reproduction/data/generation_family_metrics.csv` | Generated-sample similarity and extraction metrics by family. |
| `reproduction/data/imperfect_inference_runs.csv` | All 2,900 graph-noise simulation runs. |
| `reproduction/data/legacy_nll_rows.csv` | Legacy clean and family-exposed sequence-model NLL rows. |
| `reproduction/data/lmd_family_size_distribution.csv` | Reference component-size frequency table. |
| `reproduction/data/lmd_monte_carlo_runs.json` | Random file-split leakage census runs. |
| `reproduction/data/musical_family_metrics.csv` | Pitch, rhythm, density, and polyphony metrics by family and condition. |
| `reproduction/data/normalized_subset_rows.csv` | NLL rows for the structurally non-exact normalized subset. |
| `reproduction/data/pdmx_family_deltas.csv` | Paired PDMX family effects. |
| `reproduction/data/pdmx_nll_rows.csv` | PDMX external-dataset NLL rows. |
| `reproduction/data/phase2_nll_rows.csv` | Three-condition Transformer-L NLL rows. |
| `reproduction/data/relatedness_features.csv` | Reference-family relatedness features and effect sizes. |
| `reproduction/data/token_localization_family_rows.csv` | Shared-region and nonshared-region token NLL rows. |
| `reproduction/frozen/capacity_trend_summary.json` | Capacity trend estimates and interval. |
| `reproduction/frozen/census_summary.json` | Reference graph census and random-split contamination rates. |
| `reproduction/frozen/confirmatory_summary.json` | Confirmatory three-seed sequence-model effects. |
| `reproduction/frozen/cross_paradigm_summary.json` | Conditional VAE and diffusion contrasts. |
| `reproduction/frozen/detector_asap_summary.json` | Stores the expected detector point estimates and bootstrap intervals. |
| `reproduction/frozen/generation_statistics_summary.json` | Bootstrap tests for generation and extraction metrics. |
| `reproduction/frozen/imperfect_inference_condition_summary.csv` | Mean, median, and empirical intervals by graph-noise condition. |
| `reproduction/frozen/imperfect_inference_summary.json` | Compact noisy-graph simulation summary. |
| `reproduction/frozen/localization_summary.json` | Token-localization contrasts at 4, 8, and 16 events. |
| `reproduction/frozen/mitigation_data_cost_summary.json` | File retention and reassignment cost of mitigation baselines. |
| `reproduction/frozen/mitigation_summary.json` | Known-overlap results for file, deduplication, and component-aware splits. |
| `reproduction/frozen/musical_bootstrap_summary.json` | Bootstrap distributions for musical-property contrasts. |
| `reproduction/frozen/musical_canonical_holm_results.csv` | Canonical musical tests after Holm correction. |
| `reproduction/frozen/musical_condition_summary.csv` | Condition means and intervals for musical metrics. |
| `reproduction/frozen/musical_holm_summary.json` | Holm correction summary. |
| `reproduction/frozen/musical_paired_effects.csv` | Paired musical-property effect rows. |
| `reproduction/frozen/musical_three_condition_effects.csv` | Clean, unrelated, and same-work musical contrasts. |
| `reproduction/frozen/normalized_structural_summary.json` | Structurally normalized subset result. |
| `reproduction/frozen/pdmx_summary.json` | External PDMX replication summary. |
| `reproduction/frozen/phase2_summary.json` | Primary three-condition Transformer-L summary. |
| `reproduction/frozen/relatedness_summary.json` | Relatedness-to-effect analysis summary. |
| `reproduction/frozen/transformer_scale_summary.json` | Transformer-S/M/L scaling results. |
| `reproduction/source_specs/formal_data.json` | Records LMD archive identity, expected counts, and final stream hashes. |
| `reproduction/source_specs/formal_windows.csv.gz` | Identifies every clean, replacement, and probe window needed from LMD. |
| `reproduction/source_specs/phase2_slots.csv.gz` | Maps the 1,264 controlled replacement slots across the three conditions. |
| `reproduction/source_specs/probe_pieces.csv` | Lists the 700 evaluation pieces and splits. |

## Released results

| File | Role |
|---|---|
| `results/RESULTS_MANIFEST.json` | Records hashes for the frozen paper tables and figures. |
| `results/figures/fig1_leakage_landscape.svg` | Paper figure for prevalence and dataset structure. |
| `results/figures/fig2_phase2_core.svg` | Paper figure for the controlled three-condition result. |
| `results/figures/fig3_model_dependence.svg` | Paper figure comparing architectures and capacities. |
| `results/figures/fig4_musical_alignment.svg` | Paper figure for musical-property alignment. |
| `results/figures/fig5_mechanism_diagnostics.svg` | Paper figure for localization and extraction diagnostics. |
| `results/figures/fig6_imperfect_inference_robustness.svg` | Paper figure for noisy relation-graph robustness. |
| `results/figures/figure_capacity.csv` | Plot values for model-capacity effects. |
| `results/figures/figure_external.csv` | Plot values for external-dataset results. |
| `results/figures/figure_mitigation.csv` | Plot values for split mitigation. |
| `results/figures/figure_prevalence.csv` | Plot values for leakage prevalence. |
| `results/manuscript_results_v2_public.csv` | Flat table of the public result lock. |
| `results/manuscript_results_v2_public.json` | Machine-readable public result lock. |
| `results/manuscript_results_v2_public.sha256` | SHA-256 checksum for the JSON result lock. |
| `results/new_experiments/MANIFEST.json` | Records hashes for every new experiment table, figure, and data file. |
| `results/new_experiments/README.md` | Indexes the detector and mitigation experiments added after the v2 result lock. |
| `results/new_experiments/data/detector_evaluation.csv` | ASAP direct-edge and component-level detector results. |
| `results/new_experiments/data/detector_model_selection.csv` | Detector shortlist and selected operating point. |
| `results/new_experiments/data/false_negative_robustness.csv` | Residual leakage as reference edges are dropped. |
| `results/new_experiments/data/false_positive_robustness.csv` | Precision and component distortion under injected false edges. |
| `results/new_experiments/data/formal_assignment_impact.csv` | Effect of new detector edges on the frozen formal assignments. |
| `results/new_experiments/data/lmd_graph_completion.csv` | Full-LMD graph completion counts and component growth. |
| `results/new_experiments/data/primary_sensitivity.csv` | Primary conclusion under detector and assignment sensitivity checks. |
| `results/new_experiments/data/shs_mapping_error_audit.csv` | Audit of SHS-to-LMD mapping errors. |
| `results/new_experiments/figures/fig_combined_noise_grid.pdf` | Combined recall and false-positive trade-off grid. Print-ready PDF. |
| `results/new_experiments/figures/fig_combined_noise_grid.svg` | Combined recall and false-positive trade-off grid. Vector source. |
| `results/new_experiments/figures/fig_detector_evaluation.pdf` | ASAP detector evaluation figure. Print-ready PDF. |
| `results/new_experiments/figures/fig_detector_evaluation.svg` | ASAP detector evaluation figure. Vector source. |
| `results/new_experiments/figures/fig_imperfect_inference_tradeoffs.pdf` | False-negative and false-positive split trade-offs. Print-ready PDF. |
| `results/new_experiments/figures/fig_imperfect_inference_tradeoffs.svg` | False-negative and false-positive split trade-offs. Vector source. |
| `results/new_experiments/figures/fig_lmd_graph_completion.pdf` | Reference graph completion on full LMD. Print-ready PDF. |
| `results/new_experiments/figures/fig_lmd_graph_completion.svg` | Reference graph completion on full LMD. Vector source. |
| `results/new_experiments/tables/table_detector_evaluation.tex` | LaTeX table for detector evaluation. |
| `results/new_experiments/tables/table_detector_model_selection.tex` | LaTeX table for detector model selection. |
| `results/new_experiments/tables/table_false_negative_robustness.tex` | LaTeX table for false-negative robustness. |
| `results/new_experiments/tables/table_false_positive_robustness.tex` | LaTeX table for false-positive robustness. |
| `results/new_experiments/tables/table_formal_assignment_impact.tex` | LaTeX table for formal assignment impact. |
| `results/new_experiments/tables/table_lmd_graph_completion.tex` | LaTeX table for LMD graph completion. |
| `results/new_experiments/tables/table_primary_sensitivity.tex` | LaTeX table for primary-result sensitivity. |
| `results/new_experiments/tables/table_shs_mapping_error_audit.tex` | LaTeX table for SHS mapping audit. |
| `results/tables/table1_final.svg` | Rendered LMD census table. |
| `results/tables/table2_final.svg` | Rendered confirmatory result table. |
| `results/tables/table3_final.svg` | Rendered architecture and capacity table. |
| `results/tables/table4_final.svg` | Rendered musical-property table. |
| `results/tables/table5_final.svg` | Rendered mechanism table. |
| `results/tables/table6_final.svg` | Rendered mitigation table. |
| `results/tables/table_architecture_capacity.csv` | Source values for the architecture/capacity table. |
| `results/tables/table_confirmatory.csv` | Source values for the confirmatory table. |
| `results/tables/table_lmd_census.csv` | Source values for the LMD census table. |
| `results/tables/table_mitigation.csv` | Source values for the mitigation table. |
| `results/tables/table_pdmx_external.csv` | Source values for the PDMX table. |

## Tests

| File | Role |
|---|---|
| `tests/test_detector_and_source_reproduction.py` | Tests detector scores, source specifications, packed streams, schedules, and graph-noise metrics. |
| `tests/test_model_loader.py` | Tests all checkpoint architectures and loader validation. |
| `tests/test_public_workflows.py` | Exercises the command-line graph, split, census, and analysis workflows. |
| `tests/test_release_safety.py` | Checks public files, manifests, README workflows, and release safety. |
| `tests/test_strong_reproduction_bundle.py` | Checks row-level reproduction data and result coverage. |
| `tests/test_validation_and_normalization.py` | Tests input validation and structural normalization invariants. |
