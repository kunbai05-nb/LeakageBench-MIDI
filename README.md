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
