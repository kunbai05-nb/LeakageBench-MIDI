# LeakageBench-MIDI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22023100.svg)](https://doi.org/10.5281/zenodo.22023100)

LeakageBench-MIDI measures same-work leakage in symbolic-music generation and
provides family-aware dataset splitting.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e '.[test]'
```

## Reproduce the paper results

```bash
bash scripts/reproduce_all.sh ./reproduced
```

This CPU-only command audits the paper statistics, verifies the released tables
and figures, and runs the tests.

For the synthetic demo:

```bash
bash scripts/run_synthetic_demo.sh ./demo
```

## Model checkpoints

The [v1.1.2 release](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/tag/v1.1.2)
contains 60 inference checkpoints covering Transformer, TCN, Conditional VAE,
Latent Diffusion, and the diffusion neutral encoders.

Downloads:

- [Phase-2 Transformer-L](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/phase2-transformer-l.tar.gz)
- [Conditional VAE](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/conditional-vae.tar.gz)
- [Latent Diffusion + neutral encoders](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/latent-diffusion-and-neutral-encoders.tar.gz)
- [Transformer-S](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/lmd-transformer-s.tar.gz) · [Transformer-M](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/lmd-transformer-m.tar.gz) · [Transformer-L](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/lmd-transformer-l-legacy.tar.gz)
- [TCN](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/lmd-tcn.tar.gz) · [PDMX Transformer-L](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/pdmx-transformer-l.tar.gz)
- [Manifest and model card](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/model-release-metadata.tar.gz) · [SHA-256 checksums](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/SHA256SUMS)

After extracting the metadata and model archives:

```bash
python scripts/verify_model_checkpoints.py /path/to/checkpoints
```

Load one checkpoint:

```python
from leakagebench_midi.models import load_checkpoint

model, metadata = load_checkpoint("checkpoint.pt", map_location="cpu")
```

See [model_artifacts.md](docs/model_artifacts.md) for the archive list.

## Documentation

- [Methods](docs/METHODS_PUBLIC.md)
- [Protocol](docs/PROTOCOL_V2.md)
- [Reproduction guide](docs/REPRODUCIBILITY.md)
- [Repository file guide](docs/REPOSITORY_GUIDE.md)

## Citation

See [CITATION.cff](CITATION.cff). Software DOI:
[`10.5281/zenodo.22023100`](https://doi.org/10.5281/zenodo.22023100).

Code is released under the [MIT License](LICENSE). Model weights are CC-BY-4.0.
