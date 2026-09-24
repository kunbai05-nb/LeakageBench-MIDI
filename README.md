# LeakageBench-MIDI

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22023100-blue.svg)](https://doi.org/10.5281/zenodo.22023100)

Code and frozen specifications for the three-condition experiments and Same-Work Detector v1.9.

## Reproduce the paper tables

This is the recommended path. It downloads and verifies the released checkpoints, evaluates them on the frozen test sets, and writes the two paper tables automatically.

Requirements: Ubuntu/Linux, Python 3.10, about 10 GB of free disk space, the original `lmd_full.tar.gz`, and an NVIDIA GPU with BF16 support.

```bash
git clone https://github.com/kunbai05-nb/LeakageBench-MIDI.git
cd LeakageBench-MIDI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e '.[scale]'
bash scripts/reproduce_paper_tables.sh /path/to/lmd_full.tar.gz
```

Interrupted downloads and completed checkpoint evaluations are reused. The final files are:

```text
reproduced/tables/three_condition_results.csv
reproduced/tables/three_condition_results.md
reproduced/tables/atepp_mitigation_results.csv
reproduced/tables/atepp_mitigation_results.json
```

Before a long run, check the environment with:

```bash
bash scripts/reproduce_paper_tables.sh --check-only
```

The ATEPP table is reproduced from its frozen public checkpoint bundle, whose protocol records Same-Work Detector v1.7.1. Ablation and six-dataset detector benchmarks are intentionally not included in this entry point.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e '.[scale]'
```

LMD is not distributed here. Download it from the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/).

## Three-condition experiments

Prepare the frozen LMD streams and train MIDI-GPT or LSTM:

```bash
python scripts/prepare_lmd.py /path/to/lmd_full.tar.gz ./prepared_lmd
python scripts/train_external_models.py midigpt clean 202608040 ./prepared_lmd ./runs/midigpt-clean-202608040
python scripts/train_external_models.py lstm clean 202608040 ./prepared_lmd ./runs/lstm-clean-202608040
```

Available conditions are clean, unrelated_donor, and same_family_donor. The conditions, seeds, schedules, and settings are frozen in [configs/three_condition_models.json](configs/three_condition_models.json).

Released checkpoints can be verified and evaluated without retraining:

```bash
python scripts/verify_model_checkpoints.py /path/to/checkpoint_bundle
python scripts/evaluate_checkpoint.py /path/to/final.pt ./prepared_lmd ./evaluation --device cpu
```

MIDI-GPT, LSTM, Transformer-S/M/L, TCN, VAE, and diffusion checkpoints are available in the [v1.3.0 release](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/tag/v1.3.0). The original architectures can be retrained with `scripts/train_model.py`.

## Same-Work Detector v1.9

v1.9 retrieves the one-way Top-50 union of five complementary views and averages three independently seeded 57-feature classifiers. Scores are calibrated on a work-disjoint calibration split. Components of up to eight files grow normally; larger merges require at least three independent cross-component edges, with 50 retained as a safety cap.

The released checkpoint is included in this repository. Verify it and detect same-work families with:

```bash
python scripts/verify_detector_checkpoint.py checkpoints/same-work-detector-v1.9
python scripts/detect_same_work.py /path/to/midi checkpoints/same-work-detector-v1.9 ./detector_output --workers 8 --backend faiss
```

To reproduce training from the original MIDI files, use the frozen work-disjoint manifests:

```bash
python scripts/reproduce_detector_training.py \
  --repo . \
  --spec-dir reproduction/detector_training \
  --midi-root /path/to/lmd_matched \
  --output ./detector_retrained \
  --parallel 4 --workers-per-job 8
```

## ATEPP mitigation experiment (manual)

The released final checkpoints reproduce the five-condition ATEPP comparison without retraining. Download and extract [atepp-v1.7.1-checkpoint-reproduction.tar.gz](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.7.1/atepp-v1.7.1-checkpoint-reproduction.tar.gz), then run the original server evaluation entry point:

```bash
pip install -r requirements.txt
for condition in uncorrected random_removal detector label_reference; do
  for seed in 202608040 202608041 202608042; do
    python evaluate_saved_checkpoint.py "$condition" "$seed" \
      --output "runs/$condition/seed_$seed/test_evaluation.json"
  done
done
python analyze.py
```

This generates RESULTS.csv, RESULTS.json, PER_WORK_RESULTS.json, and RESULTS_CN.md. The bundle contains the 12 physical checkpoints for three seeds, the fixed test set, frozen audit data, and the unchanged server evaluation and analysis code. Simple deduplication uses the same checkpoints as the uncorrected condition because it removes no additional training windows under the frozen split.

## Tests

```bash
bash scripts/reproduce_all.sh
```

See [CITATION.cff](CITATION.cff). Code is MIT licensed; released weights are CC BY 4.0.
