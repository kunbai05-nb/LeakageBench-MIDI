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

## 2. Reproduce the paper tables

### Multi-model three-condition and ATEPP mitigation experiments

The following command downloads and verifies all released checkpoints, evaluates them on the frozen test sets, and creates both tables:

```bash
bash scripts/reproduce_paper_tables.sh /path/to/lmd_full.tar.gz
```

Interrupted downloads and completed three-condition evaluations are reused. Before a long run, check the environment with:

```bash
bash scripts/reproduce_paper_tables.sh --check-only
```

Outputs:

```text
reproduced/tables/three_condition_results.csv
reproduced/tables/three_condition_results.md
reproduced/tables/atepp_mitigation_results.csv
reproduced/tables/atepp_mitigation_results.json
```

The reported ATEPP numeric table comes from the frozen public v1.7.1 checkpoint bundle; the one-command workflow preserves that original protocol.

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

### Detector ablation and sensitivity experiments

For the quickest reviewer check, one command downloads the checksum-verified frozen experiment outputs, regenerates Tables 1, 2, 4, and 5, and fails if any reported value differs from the paper:

```bash
bash scripts/reproduce_detector_ablation.sh
```

Outputs:

```text
reproduced/detector_ablation/tables/table1_view_count.csv
reproduced/detector_ablation/tables/table2_pairwise_verifier.csv
reproduced/detector_ablation/tables/table4_top_k.csv
reproduced/detector_ablation/tables/table5_classifier_count.csv
reproduced/detector_ablation/tables/verification.json
```

`verification.json` reports `PASS` only when every generated paper value matches the frozen specification. No dataset download is required for this check. To rerun feature extraction, model fitting, and evaluation from the checksum-verified source MIDI files, run:

```bash
bash scripts/reproduce_detector_ablation_from_midi.sh \
  /path/to/LMD-clean \
  /path/to/ATEPP \
  /path/to/SHS \
  /path/to/ASAP
```

The public ablation workflow covers the reported retrieval-view count, pairwise-verifier, Top-K, and classifier-count experiments. The frozen bundle contains numeric experiment outputs only, not the paper tables or MIDI files. Family-merging safety guards are implementation safeguards and are not included as a reported ablation table.

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
