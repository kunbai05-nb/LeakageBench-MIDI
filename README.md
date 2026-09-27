# LeakageBench-MIDI

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22023100-blue.svg)](https://doi.org/10.5281/zenodo.22023100)

Code, checkpoints, and frozen specifications for the three-condition experiments and Same-Work Detector v1.9.

## 1. Setup

Requirements: Ubuntu/Linux, Python 3.10, and about 10 GB of free disk space. Checkpoint evaluation requires an NVIDIA GPU with BF16 support.

```bash
git clone https://github.com/kunbai05-nb/LeakageBench-MIDI.git
cd LeakageBench-MIDI
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e '.[scale]'
```

LMD is not distributed here. Download `lmd_full.tar.gz` from the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/).

While this repository is private, authorized collaborators should run `gh auth login` before downloading release assets. This is not required after a public release.

## 2. Reproduce the paper tables

### Multi-model three-condition experiment

The following command downloads and verifies the released checkpoints and source specifications, evaluates all models on the frozen test set, and creates the reported table:

```bash
bash scripts/reproduce_three_condition.sh /path/to/lmd_full.tar.gz
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

Supply the six prepared MIDI directories in the order shown below. The command verifies the detector checkpoint and every MIDI checksum, reproduces the Ours v1.9 rows under both the fixed and oracle-optimal thresholds, computes 95% confidence intervals, and creates the combined table:

```bash
bash scripts/reproduce_six_dataset_table.sh \
  /path/to/SHS \
  /path/to/ASAP \
  /path/to/ATEPP \
  /path/to/LMD-clean \
  /path/to/Vienna4x22 \
  /path/to/PianoVAM
```

Outputs:

```text
reproduced/six_dataset/tables/six_dataset_results.csv
reproduced/six_dataset/tables/six_dataset_results.md
reproduced/six_dataset/tables/six_dataset_results.json
```

The fixed result uses the released decision threshold unchanged on all six datasets. The optimal result is a target-label oracle upper bound and is not used for deployment.

The SHS cohort contains the final 733 checksum-verified MIDI files, 213 works, and 515 eligible reference pairs. No target dataset is used to retrain the detector or recalibrate the fixed threshold.

### Fast Detector versus v1.9

The following command downloads checksum-verified candidate features and both released detector checkpoints, runs fresh checkpoint inference, and generates the fixed-threshold comparison on ATEPP, LMD-clean, and ASAP:

```bash
bash scripts/reproduce_fast_vs_v19.sh
```

The generated CSV, Markdown, and JSON files are written to `reproduced/fast_vs_v19/`. The release asset contains checkpoints, candidate features, and manifests, but no predicted scores or preformatted result table.

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

### Detector ablation and sensitivity experiments

The recommended reproduction downloads checksum-verified intermediate features, retrains every ablated classifier, reevaluates all settings, and regenerates Tables 1, 2, 4, and 5:

```bash
bash scripts/reproduce_detector_ablation.sh
```

This is the default reviewer workflow: it skips only the expensive deterministic MIDI feature extraction and does not download frozen result tables. For the strict end-to-end check, rerun feature extraction from the checksum-verified source MIDI files:

```bash
bash scripts/reproduce_detector_ablation.sh --from-midi \
  /path/to/LMD-clean \
  /path/to/ATEPP \
  /path/to/ASAP
```

Both workflows generate the reported retrieval-view count, pairwise-verifier, Top-K, and classifier-count tables. No frozen result table or MIDI file is stored in this repository.

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
