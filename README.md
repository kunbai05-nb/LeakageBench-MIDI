# LeakageBench-MIDI

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22023100-blue.svg)](https://doi.org/10.5281/zenodo.22023100)

Code, checkpoints, and frozen specifications for the three-condition experiments and Same-Work Detector v1.9.

## 1. Setup

Requirements: Ubuntu/Linux and Python 3.10. Generation-model checkpoint evaluation (three-condition and ATEPP mitigation experiments) requires an NVIDIA GPU with BF16 support. Detector benchmarks, ablations, and deployment audit replay run on CPU. The pinned inference environment uses PyTorch 2.8.0 and scikit-learn 1.7.2. Reserve space for both the downloaded bundles and their extracted contents.

While the repository is private, authorized collaborators must authenticate before cloning or downloading release assets:

```bash
gh auth login
gh auth setup-git
```

Authentication is not required after a public release.

```bash
git clone https://github.com/kunbai05-nb/LeakageBench-MIDI.git
cd LeakageBench-MIDI
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e '.[scale]'
```

All commands below are run from the repository root with the virtual environment activated. Release bundles download automatically to `reproduction/` (three-condition weights to `reproduced/three_condition/`). No manual checkpoint paths are needed.

The Fast-versus-v1.9 comparison, seven-dataset audit replay, ATEPP mitigation, and default ablation commands do not require original MIDI downloads. The three-condition command requires `lmd_full.tar.gz`: download it from the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) and save it as `data/lmd_full.tar.gz`. The six-dataset benchmark and optional full-MIDI workflows require the original datasets, which are not distributed here.

For the six-dataset benchmark, place the original datasets in the following default directories. Each directory must contain the relative paths in its corresponding `reproduction/detector_benchmark/*.csv.gz` manifest, not merely any collection of MIDI files:

| Dataset | Default directory | Example path within that directory |
| --- | --- | --- |
| SHS (subset of LMD-full) | `data/shs` | `0/0207f2cc938c3691b3af6d8e359bedfd.mid` |
| ASAP | `data/asap` | `Bach/Fugue/bwv_846/Shi05M.mid` |
| ATEPP | `data/atepp` | `Sergei_Rachmaninoff/13_Preludes,_Op._32/No._10_in_B_Minor:_Lento/00046.mid` |
| LMD-clean | `data/lmd-clean` | `clean_midi/2 Brothers on the 4th Floor/Dreams.1.mid` |
| Vienna4x22 | `data/vienna4x22` | `Chopin_op10_no3_p01.mid` |
| PianoVAM | `data/pianovam` | `MIDI/2024-02-14_19-10-09.mid` |

Existing data can be linked into these directories instead of copied. Alternatively, set `LEAKAGEBENCH_DATA_DIR` once to a data directory with the same layout, or keep using the explicit-path forms shown by each script's `--help`. The scripts do not download licensed datasets automatically.

## 2. Reproduce the paper tables

### Multi-model three-condition experiment

The following command downloads and verifies the released checkpoints and source specifications, rebuilds only the frozen evaluation windows from MIDI (not the unused training streams), evaluates all models on the frozen test set, and creates the reported table:

```bash
bash scripts/reproduce_three_condition.sh
```

Interrupted downloads and completed three-condition evaluations are reused. Before a long run, check the environment with:

```bash
bash scripts/reproduce_three_condition.sh --check-only
```

Outputs:

```text
reproduced/three_condition/tables/three_condition_results.csv
reproduced/three_condition/tables/three_condition_results.md
reproduced/three_condition/tables/three_condition_results.json
```

### Six-dataset detector benchmark

With the default data layout above, first validate all six inputs without running inference, then reproduce the table. The command checks the frozen evaluation cohorts before inference, verifies the detector checkpoint and every MIDI checksum, reproduces the Ours v1.9 rows under both the fixed and oracle-optimal thresholds, computes 95% confidence intervals, and creates the detector-results table:

```bash
bash scripts/reproduce_six_dataset_table.sh --check-only
bash scripts/reproduce_six_dataset_table.sh
```

Outputs:

```text
reproduced/six_dataset/tables/six_dataset_results.csv
reproduced/six_dataset/tables/six_dataset_results.md
reproduced/six_dataset/tables/six_dataset_results.json
```

The fixed result uses the released decision threshold unchanged on all six datasets. The optimal result is a target-label oracle upper bound and is not used for deployment.

The paper's family-disjoint SHS evaluation cohort contains 306 checksum-verified MIDI files and 80 eligible reference pairs. The CAugBERT/CLaMP comparison rows are not recalculated by this command; they remain the separately frozen baseline results. This benchmark performs no detector retraining or threshold recalibration on the evaluation cohorts.

### Fast Detector versus v1.9

The following command downloads checksum-verified candidate features and both released detector checkpoints, runs fresh checkpoint inference, and generates the fixed-threshold comparison on ATEPP, LMD-clean, and ASAP:

```bash
bash scripts/reproduce_fast_vs_v19.sh
```

The generated CSV, Markdown, and JSON files are written to `reproduced/fast_vs_v19/`. The release asset contains checkpoints, candidate features, and manifests, but no predicted scores or preformatted result table.

### Seven-dataset deployment audit

The following command downloads frozen candidate features and the Precision-First Same-Work Auditor checkpoint, runs fresh pairwise inference, rebuilds protected families, replays all 200 splits, and generates the seven-dataset audit table:

```bash
bash scripts/reproduce_multidataset_deployment.sh
```

No manual data paths are needed. A CPU is sufficient. Outputs:

```text
reproduced/multidataset_deployment/multidataset_deployment.csv
reproduced/multidataset_deployment/multidataset_deployment.md
reproduced/multidataset_deployment/<dataset>_results.json
reproduced/multidataset_deployment/multidataset_verification.json
```

This fixed auditor checkpoint is independent of the v1.9 benchmark detector. Family/file exposure uses all test families/files as denominators; 95% EI denotes empirical quantiles across 200 splits.

The quick entry starts after retrieval and cheap-gate filtering. It recomputes classifier scores, decisions, families and exposure rather than loading frozen families. Valid-MIDI counts are the recorded parsing census; multi-version prevalence is recomputed from the rebuilt graph. The optional full-MIDI source and frozen file/hash manifests are included in the same asset under `source/` and `midi_hashes/` (see its `FULL_MIDI.md`).

### ATEPP mitigation experiment

The following command downloads the 12 checksum-verified final v1.9 model checkpoints and the frozen test windows, runs fresh BF16 inference for all five conditions and three seeds, and then regenerates the ATEPP mitigation table:

```bash
bash scripts/reproduce_atepp_mitigation.sh
```

Outputs:

```text
reproduction/atepp_mitigation_v1.9/reproduced_results/RESULTS.csv
reproduction/atepp_mitigation_v1.9/reproduced_results/RESULTS.json
```

The current experiment uses Same-Work Detector v1.9 and a detector-size, label-safe reference. All five conditions share the same held-out set and scored-token distribution; detector isolation, random removal, and the label reference each use 484 training files. A BF16-capable CUDA GPU is required; the released package contains no precomputed test-evaluation records or result table.

Simple deduplication and the uncorrected condition have identical frozen training streams, as verified by their matching SHA-256 digests. They therefore share three checkpoints: the five-condition table evaluates 15 condition/seed combinations from 12 distinct checkpoints.

The command automatically applies a small checksum-verified protocol correction after downloading the original checkpoint bundle. The scoring protocol lists the five reported conditions; the original frozen training protocol is retained separately. No duplicate checkpoint bundle is needed.

### Detector ablation and sensitivity experiments

The recommended reproduction downloads checksum-verified intermediate features, retrains every ablated classifier, reevaluates all settings, and regenerates Tables 1, 2, 4, and 5:

```bash
bash scripts/reproduce_detector_ablation.sh
```

This is the default reviewer workflow: it skips only the expensive deterministic MIDI feature extraction and does not download frozen result tables. For the strict end-to-end check, rerun feature extraction from the checksum-verified source MIDI files:

```bash
bash scripts/reproduce_detector_ablation.sh --from-midi
```

Both workflows generate the reported retrieval-view count, pairwise-verifier, Top-K, and classifier-count tables. No frozen result table or MIDI file is stored in this repository.

`verification.json` reports table generation separately from equality to the frozen paper values. `PASS` means an exact match of the reported values; `DIFFERENT` means new tables were generated but require comparison. Differences remain visible and are not treated as successful matching or automatically blamed on floating-point rounding. For a strict equality check, rerun `scripts/summarize_detector_ablation.py` with `--strict-match`.

## 3. Use the detector on another MIDI dataset

v1.9 retrieves the one-way Top-50 union of five complementary views and averages three independently seeded 57-feature classifiers. Scores are calibrated on a work-disjoint calibration split.

```bash
python scripts/verify_detector_checkpoint.py checkpoints/same-work-detector-v1.9
python scripts/detect_same_work.py \
  /path/to/midi \
  checkpoints/same-work-detector-v1.9 \
  ./detector_output \
  --workers 8 --backend faiss
```

## 4. Optional: retrain from scratch

Three-condition settings, seeds, and schedules are frozen in [configs/three_condition_models.json](configs/three_condition_models.json). For example:

```bash
python scripts/fetch_reproduction_assets.py three-condition-source-specs
python scripts/prepare_lmd.py /path/to/lmd_full.tar.gz ./prepared_lmd
python scripts/train_external_models.py midigpt clean 202608040 ./prepared_lmd ./runs/midigpt-clean-202608040
python scripts/train_model.py --help
```

Released model bundles from the [v1.3.0 release](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/tag/v1.3.0) can also be checked or evaluated individually:

```bash
python scripts/verify_model_checkpoints.py /path/to/checkpoint_bundle
python scripts/evaluate_checkpoint.py /path/to/final.pt ./prepared_lmd ./evaluation --device cuda
```

To retrain Same-Work Detector v1.9 from the frozen work-disjoint manifests:

```bash
python scripts/reproduce_detector_training.py \
  --repo . \
  --spec-dir reproduction/detector_training \
  --midi-root /path/to/lmd_matched \
  --output ./detector_retrained \
  --parallel 4 --workers-per-job 8
```

## 5. Tests and citation

```bash
bash scripts/reproduce_all.sh
```

See [CITATION.cff](CITATION.cff). Code is MIT licensed; released weights are CC BY 4.0.
