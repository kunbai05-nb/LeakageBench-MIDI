# LeakageBench-MIDI

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22023100-blue.svg)](https://doi.org/10.5281/zenodo.22023100)

Code and frozen specifications for the three-condition experiments and Same-Work Detector v1.8.

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

## Same-Work Detector v1.8

v1.8 retrieves the one-way Top-50 union of five complementary views. Three
independently seeded 57-feature classifiers are averaged. Components of
up to eight files grow normally; larger merges require at least three
independent cross-component edges, with 50 retained as a safety cap.

Download and extract [same-work-detector-v1.8.tar.gz](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.8/same-work-detector-v1.8.tar.gz), then run:

```bash
python scripts/verify_detector_checkpoint.py ./same-work-detector-v1.8
python scripts/detect_same_work.py /path/to/midi ./same-work-detector-v1.8 ./detector_output --workers 8 --backend faiss
```

Retrain the released classifier exactly from frozen features:

```bash
python scripts/train_detector.py reproduction/detector/training_features_v1_8.npz ./detector_retrained
```

The fixed-threshold and per-dataset optimal-threshold results use six frozen public registries. Run one dataset with:

```bash
python scripts/reproduce_detector_benchmark.py shs /path/to/shs ./same-work-detector-v1.8 ./benchmark/shs --workers 8
python scripts/reproduce_detector_benchmark.py asap /path/to/asap ./same-work-detector-v1.8 ./benchmark/asap --workers 8
python scripts/reproduce_detector_benchmark.py atepp /path/to/atepp ./same-work-detector-v1.8 ./benchmark/atepp --workers 8
python scripts/reproduce_detector_benchmark.py lmd-clean /path/to/lmd_clean ./same-work-detector-v1.8 ./benchmark/lmd-clean --workers 8
python scripts/reproduce_detector_benchmark.py vienna4x22 /path/to/vienna4x22 ./same-work-detector-v1.8 ./benchmark/vienna4x22 --workers 8
python scripts/reproduce_detector_benchmark.py pianovam /path/to/pianovam ./same-work-detector-v1.8 ./benchmark/pianovam --workers 8
```

Each command writes `results.json` containing both the released fixed-threshold
result and the dataset-specific optimal-threshold result. Every MIDI checksum is
verified before evaluation.

## ATEPP mitigation experiment

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
