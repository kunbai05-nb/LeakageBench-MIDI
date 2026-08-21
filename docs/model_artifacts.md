# Model checkpoints

The [v1.1.2 release](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/tag/v1.1.2)
contains 60 inference checkpoints.

- [Phase-2 Transformer-L](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/phase2-transformer-l.tar.gz)
- [Conditional VAE](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/conditional-vae.tar.gz)
- [Latent Diffusion + neutral encoders](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/latent-diffusion-and-neutral-encoders.tar.gz)
- [Transformer-S](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/lmd-transformer-s.tar.gz)
- [Transformer-M](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/lmd-transformer-m.tar.gz)
- [Transformer-L](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/lmd-transformer-l-legacy.tar.gz)
- [TCN](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/lmd-tcn.tar.gz)
- [PDMX Transformer-L](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/pdmx-transformer-l.tar.gz)
- [Manifest and model card](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/model-release-metadata.tar.gz)
- [SHA-256 checksums](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/SHA256SUMS)

Verify the extracted bundle:

```bash
python scripts/verify_model_checkpoints.py /path/to/checkpoints
```

Load a checkpoint:

```python
from leakagebench_midi.models import load_checkpoint

model, metadata = load_checkpoint("checkpoint.pt", map_location="cpu")
```

The model archives are CC-BY-4.0.
