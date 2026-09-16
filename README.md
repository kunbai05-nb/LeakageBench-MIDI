# LeakageBench-MIDI

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22023100-blue.svg)](https://doi.org/10.5281/zenodo.22023100)

Code and frozen specifications for the three-condition experiments and Same-Work Detector v1.7.1.

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

## Same-Work Detector v1.7.1

v1.7.1 reuses the v1.7 classifier weights and fixes deployment clustering: a
candidate must have reciprocal Top-100 support in at least two verifier views.
Components of up to eight files grow normally; larger merges require at least
three independent cross-component edges, with 50 retained only as a safety cap.

Download and extract [same-work-detector-v1.7.1.tar.gz](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.7.1/same-work-detector-v1.7.1.tar.gz), then run:

```bash
python scripts/verify_detector_checkpoint.py ./same-work-detector-v1.7.1
python scripts/detect_same_work.py /path/to/midi ./same-work-detector-v1.7.1 ./detector_output --workers 8 --backend faiss
```

Retrain the released classifier exactly from frozen features:

```bash
python scripts/train_detector.py reproduction/detector/training_features_v1_7.npz ./detector_retrained
```

To repeat feature extraction as well, provide the indexed LMD files:

```bash
python scripts/train_detector.py reproduction/source_specs/detector_training_index.csv ./detector_retrained --midi-root /path/to/lmd_matched --workers 8 --backend exact
```

The fixed-threshold and per-dataset optimal-threshold results use six frozen public registries. Run one dataset with:

```bash
python scripts/reproduce_detector_benchmark.py DATASET /path/to/dataset ./same-work-detector-v1.7.1 ./benchmark/DATASET --workers 8
```

Supported dataset names are shs, asap, atepp, lmd-clean, vienna4x22, and pianovam. Every MIDI checksum is verified before evaluation.

## Tests

```bash
bash scripts/reproduce_all.sh
```

See [CITATION.cff](CITATION.cff). Code is MIT licensed; released weights are CC BY 4.0.
