# LeakageBench-MIDI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22023100.svg)](https://doi.org/10.5281/zenodo.22023100)

Code and frozen specifications for measuring same-work leakage in symbolic-music generation.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test,scale]'
```

LMD is not distributed with this repository. Download it from the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) and provide the local archive or extracted directory to the preparation script.

## Detector

Download [same-work-detector-v1.3.0.tar.gz](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.3.0/same-work-detector-v1.3.0.tar.gz), extract it, and run:

```bash
python scripts/verify_detector_checkpoint.py ./same-work-detector-v1.3.0
python scripts/detect_same_work.py /path/to/midi ./same-work-detector-v1.3.0 ./detector_output --workers 8 --backend faiss
```

The detector combines nine structural views, reciprocal top-100 retrieval, local ordered evidence, and a calibrated five-model ensemble. It returns relation edges and component labels. The released family graph is a reference relation for this study, not a universal detector.

To retrain it from the public index:

```bash
python scripts/train_detector.py --midi-root /path/to/lmd_matched --index reproduction/source_specs/detector_training_index.csv --output ./detector_retrained --workers 8 --backend exact
```

## Detector experiments

The detector comparison is one cross-dataset experiment: CAugBERT/CLaMP Union is compared with Ours on SHS, ASAP, ATEPP, MAESTRO, and LMD-clean using precision, recall, F1, and false negatives (FN).

| Dataset | Method | Precision | Recall | F1 | FN |
| --- | --- | ---: | ---: | ---: | ---: |
| SHS | CAugBERT/CLaMP Union | 100.00 | 11.84 | 21.18 | 454 |
| SHS | Ours | **98.86** | **67.57** | **80.28** | **167** |
| ASAP | CAugBERT/CLaMP Union | 40.34 | 64.43 | 49.61 | 1,774 |
| ASAP | Ours | **99.82** | **99.52** | **99.67** | **24** |
| ATEPP | CAugBERT/CLaMP Union | 14.43 | 47.50 | 22.14 | 1,753 |
| ATEPP | Ours | **99.17** | **93.53** | **96.27** | **216** |
| MAESTRO | CAugBERT/CLaMP Union | — | 58.71 | — | 448 |
| MAESTRO | Ours | — | **80.00** | — | **217** |
| LMD-clean | CAugBERT/CLaMP Union | 89.83 | 39.31 | 54.68 | 18,574 |
| LMD-clean | Ours | **90.28** | **60.41** | **72.38** | **12,030** |

The frozen file order, checksums, work groups, and recording groups for the detector-side reruns are in `reproduction/detector_benchmark`. Run the reproducible SHS, ASAP, and LMD-clean rows after downloading those datasets:

```bash
bash scripts/reproduce_detector_benchmarks.sh \
  /path/to/shs /path/to/asap /path/to/lmd_clean \
  ./same-work-detector-v1.3.0 ./benchmark
```

Each output directory contains `results.json` and `predicted_pairs.csv.gz`. The script verifies every input MIDI checksum before detection. SHS and ASAP use pair-micro precision/recall/F1; LMD-clean uses query-macro precision/recall/F1. ATEPP and MAESTRO are included as the external-dataset rows of the same comparison experiment.

The same detector is also used in the ATEPP label-blind mitigation experiment. The final 20,000-step MIDI-GPT checkpoints for three seeds are available in [atepp-label-blind-mitigation-checkpoints.tar.gz](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.3.0/atepp-label-blind-mitigation-checkpoints.tar.gz).

## Three-condition models

The released weights are in [v1.3.0](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/tag/v1.3.0):

- [MIDI-GPT clean](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.3.0/midigpt-clean.tar.gz), [unrelated donor](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.3.0/midigpt-unrelated_donor.tar.gz), [same-family donor](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.3.0/midigpt-same_family_donor.tar.gz)
- [LSTM clean](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.3.0/lstm-clean.tar.gz), [unrelated donor](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.3.0/lstm-unrelated_donor.tar.gz), [same-family donor](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.3.0/lstm-same_family_donor.tar.gz)

Prepare the frozen streams:

```bash
python scripts/prepare_lmd.py /path/to/lmd_full.tar.gz ./prepared_lmd
```

Train or evaluate a run:

```bash
python scripts/train_external_models.py midigpt clean 202608040 ./prepared_lmd ./runs/midigpt-clean-202608040
python scripts/evaluate_checkpoint.py /path/to/final.pt ./prepared_lmd ./evaluation --device cpu
```

The exact conditions, seeds, batch schedules, and model settings are in `configs/three_condition_models.json`. Full training can use CUDA; released weights can be evaluated on CPU.

Check a downloaded model bundle with `python scripts/verify_model_checkpoints.py /path/to/bundle`.

## Capacity and architecture checkpoints

These checkpoints support the capacity and architecture comparisons and are in [v1.3.0](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/tag/v1.3.0):

- Transformer-S/M/L: [S](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.3.0/lmd-transformer-s.tar.gz), [M](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.3.0/lmd-transformer-m.tar.gz), [L](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.3.0/lmd-transformer-l.tar.gz)
- [TCN](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.3.0/lmd-tcn.tar.gz)
- [Conditional VAE](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.3.0/conditional-vae.tar.gz)
- [Latent Diffusion and neutral encoders](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.3.0/latent-diffusion-and-neutral-encoders.tar.gz)

## Other models

The original Transformer, TCN, VAE, and diffusion implementations remain available through `scripts/train_model.py` and `scripts/evaluate_checkpoint.py`.

## Tests

```bash
bash scripts/reproduce_all.sh
```

See [CITATION.cff](CITATION.cff) for citation information. Code is MIT licensed; released model weights are CC BY 4.0.
