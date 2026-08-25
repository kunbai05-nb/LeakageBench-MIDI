# LeakageBench-MIDI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22023100.svg)](https://doi.org/10.5281/zenodo.22023100)

LeakageBench-MIDI measures same-work leakage in symbolic-music generation and
builds component-aware train/validation/test splits. It includes the paper
statistics, the structural detector, formal model code, checkpoints, and
reproduction scripts.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e '.[test]'
```

## Reproduce the paper statistics

```bash
bash scripts/reproduce_all.sh ./reproduced
```

This CPU workflow rebuilds the public statistics and checks every released
table, figure, and test.

Detector and mitigation additions are in
[`results/new_experiments`](results/new_experiments/README.md).

## Reproduce from LMD

Download `lmd_full.tar.gz` from the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/), then run:

```bash
python scripts/prepare_lmd.py /path/to/lmd_full.tar.gz ./prepared_lmd
```

The command recreates the three formal training streams and evaluation windows.
It verifies every MIDI file, token window, and final stream hash.

Build the reference relation graph and rerun the noisy-graph split experiment:

```bash
python scripts/prepare_reference_graph.py \
  /path/to/lmd_full.tar.gz ./reference_graph
python scripts/simulate_imperfect_inference.py \
  ./reference_graph ./imperfect_inference
```

Evaluate a released checkpoint:

```bash
python scripts/evaluate_checkpoint.py \
  checkpoints/lmd/phase2_transformer_l/clean/seed_202608040.pt \
  ./prepared_lmd ./evaluation/clean-202608040
```

Train a formal run from initialization:

```bash
python scripts/train_model.py \
  transformer clean 202608040 ./prepared_lmd ./runs/transformer-clean-202608040
```

The same command supports `conditional_vae`, `neutral_encoder`, and
`latent_diffusion`. Full training uses a CUDA GPU; preparation, statistics, and
the detector run on CPU.

## Same-work detector

```bash
python scripts/detect_same_work.py /path/to/midi ./detector_output --workers 8
```

The detector extracts 47 structural features, scores sparse candidate pairs,
and returns a guarded component graph. The paper's family IDs are connected
components of the reference relation. Its ASAP statistics reproduce from the
released candidate-pair rows with `scripts/reproduce_detector_statistics.py`.

For LMD- or PDMX-scale collections, install the CPU search backend and use the
sharded runner:

```bash
pip install -e '.[scale]'
python scripts/build_cross_dataset_manifest.py lmd /path/to/lmd ./lmd.jsonl
python scripts/run_cross_dataset_detector.py \
  ./lmd.jsonl /path/to/lmd ./lmd_results --cache ./lmd_cache
```

The matching POP909, ASAP, MAESTRO, PDMX, LMD, GigaMIDI, and Aria-MIDI results
are in [`results/new_experiments`](results/new_experiments/README.md).

## Checkpoints

The [v1.1.2 release](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/tag/v1.1.2)
contains 60 checkpoints.

- [Phase-2 Transformer-L](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/phase2-transformer-l.tar.gz)
- [Conditional VAE](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/conditional-vae.tar.gz)
- [Latent Diffusion and neutral encoders](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/latent-diffusion-and-neutral-encoders.tar.gz)
- [Transformer-S](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/lmd-transformer-s.tar.gz), [Transformer-M](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/lmd-transformer-m.tar.gz), [Transformer-L](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/lmd-transformer-l-legacy.tar.gz)
- [TCN](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/lmd-tcn.tar.gz) and [PDMX Transformer-L](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/pdmx-transformer-l.tar.gz)
- [Checksums](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/SHA256SUMS)

```bash
python scripts/verify_model_checkpoints.py /path/to/checkpoint_bundle
```

See [Reproduction](docs/REPRODUCIBILITY.md), [Datasets](docs/datasets.md),
[Methods](docs/METHODS_PUBLIC.md), and the
[repository guide](docs/REPOSITORY_GUIDE.md) for details.

## Citation

See [CITATION.cff](CITATION.cff). Code is MIT licensed; released model weights
are CC BY 4.0.
