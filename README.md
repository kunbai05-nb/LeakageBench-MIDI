# LeakageBench-MIDI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22023100.svg)](https://zenodo.org/doi/10.5281/zenodo.22023100)

LeakageBench-MIDI detects same-work MIDI files, measures train–test leakage, and builds component-aware dataset splits. The repository contains the detector, model implementations, experiment configurations, training and evaluation scripts, tests, and exact source specifications used by the paper.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test,scale]'
```

## Same-work detector

Download [same-work-detector-v1.2.0.tar.gz](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.2.1/same-work-detector-v1.2.0.tar.gz), extract it, and run:

```bash
python scripts/verify_detector_checkpoint.py ./detector_v1.2.0
python scripts/detect_same_work.py \
  /path/to/midi ./detector_v1.2.0 ./detector_output \
  --workers 8 --backend faiss
```

The detector retrieves structurally similar candidates, aligns the three strongest transpositions, scores 69 pair features with a five-model ensemble, and groups accepted relations without allowing components larger than 50 files. Use `--backend exact` for small collections.

To retrain it, download LMD `match_scores.json`, `track_metadata.db`, and the corresponding MIDI files, then run:

```bash
python scripts/prepare_detector_manifest.py \
  --match-scores /path/to/match_scores.json \
  --track-metadata /path/to/track_metadata.db \
  --output ./detector_manifest.jsonl

python scripts/train_detector.py \
  --manifest ./detector_manifest.jsonl \
  --midi-root /path/to/lmd_matched \
  --template ./detector_v1.2.0/config.json \
  --output ./trained_detector --workers 8
```

## Reproduce the generation experiments

Download `lmd_full.tar.gz` from the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/), then build the exact training streams and evaluation windows:

```bash
python scripts/prepare_lmd.py /path/to/lmd_full.tar.gz ./prepared_lmd
```

Train one of the three experimental conditions:

```bash
python scripts/train_model.py \
  transformer clean 202608040 ./prepared_lmd ./runs/transformer-clean-202608040
```

Available models: `transformer`, `conditional_vae`, `neutral_encoder`, and `latent_diffusion`.

Available training conditions: `clean`, `unrelated_donor`, and `same_family_donor`.

Training the full 20,000-step experiment requires an NVIDIA GPU with CUDA. Released checkpoints can also be evaluated on CPU, although evaluation will be slower.

Evaluate a released checkpoint:

```bash
python scripts/evaluate_checkpoint.py \
  /path/to/checkpoint.pt ./prepared_lmd ./evaluation
```

Build a relation graph with the released detector and run the imperfect-inference split simulation:

```bash
python scripts/prepare_reference_graph.py \
  /path/to/extracted_lmd ./detector_v1.2.0 ./reference_graph \
  --workers 8 --backend faiss
python scripts/simulate_imperfect_inference.py ./reference_graph ./split_simulation
```

## Checkpoints

All weights are in the [v1.2.1 release](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/tag/v1.2.1):

The LMD Transformer and TCN bundles contain the final 20,000-step checkpoint for all three conditions and three seeds.

- [LMD Transformer-S](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.2.1/lmd-transformer-s.tar.gz)/[LMD Transformer-M](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.2.1/lmd-transformer-m.tar.gz)/[LMD Transformer-L](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.2.1/lmd-transformer-l.tar.gz)
- [Conditional VAE](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.2.1/conditional-vae.tar.gz)
- [Latent Diffusion and neutral encoders](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.2.1/latent-diffusion-and-neutral-encoders.tar.gz)
- [TCN](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.2.1/lmd-tcn.tar.gz)
- [PDMX Transformer-L](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.2.1/pdmx-transformer-l.tar.gz)

Model bundles can be checked with:

```bash
python scripts/verify_model_checkpoints.py /path/to/checkpoint_bundle --allow-partial
```

## Tests

```bash
bash scripts/reproduce_all.sh
```

## Citation

See [CITATION.cff](CITATION.cff). Code is released under MIT; model weights are CC BY 4.0.
