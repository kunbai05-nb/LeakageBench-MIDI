# Companion model artifacts

Model weights are not committed to the Git source tree. The unchanged
**LeakageBench-MIDI Model Checkpoints v1.1.0** companion archives are attached
to the [current consolidated v1.1.2 release](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/tag/v1.1.2).
Keeping weights as release assets avoids Git/LFS dependency and lets users
download only the architecture groups needed for a reproduction.

## Direct downloads

- [Checksums (`SHA256SUMS`)](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/SHA256SUMS)
- [Manifest, licenses, model card, and verification metadata](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/model-release-metadata.tar.gz)
- [Phase-2 Transformer-L](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/phase2-transformer-l.tar.gz)
- [Conditional VAE](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/conditional-vae.tar.gz)
- [Latent Diffusion and neutral encoders](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/latent-diffusion-and-neutral-encoders.tar.gz)
- [Legacy LMD Transformer-L](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/lmd-transformer-l-legacy.tar.gz)
- [LMD Transformer-M](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/lmd-transformer-m.tar.gz)
- [LMD Transformer-S](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/lmd-transformer-s.tar.gz)
- [LMD TCN](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/lmd-tcn.tar.gz)
- [PDMX Transformer-L](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/pdmx-transformer-l.tar.gz)

The companion archive contains 60 inference-only artifacts:

- 30 previously traced confirmatory, capacity/architecture, and PDMX external checkpoints;
- 9 Phase-2 Transformer-L checkpoints (three conditions by three formal seeds);
- 9 Conditional VAE checkpoints (three conditions by three formal seeds);
- 9 Latent Diffusion checkpoints (three conditions by three formal seeds); and
- 3 condition-invariant neutral encoders, one per formal seed, required to reproduce the diffusion endpoint.

Each `.pt` file contains model weights plus public metadata and configuration.
Optimizer state, scheduler state, RNG state, intermediate checkpoints, training
logs, raw MIDI, and token-bearing examples are excluded. `MODEL_MANIFEST.json`
maps each artifact to architecture, condition, seed, source checkpoint hash,
paper role, and—where applicable—the required neutral encoder.

## Integrity and compatibility check

After extracting the complete companion directory next to this repository,
run:

```bash
python scripts/verify_model_checkpoints.py \
  ../models/LeakageBench-MIDI-Model-Checkpoints-v1.1.0
```

The verifier performs no training. It checks the manifest byte sizes and
SHA-256 values, strictly loads every available state dictionary, validates
model IDs, tokenizer/model vocabulary and parameter counts, and compares fixed
CPU outputs for five representative architecture roles. The expected complete
result is 60 verified checkpoints, zero missing checkpoints, and five reference
output passes. Use `--allow-partial` only when intentionally checking a grouped
architecture download.

The v1.1.0 release audit obtained 60/60 strict loads, 60/60 matching parameter
counts, and five/five fixed-input reference signatures. The retained Phase-2
training implementation and public Transformer model also produced bitwise
equal logits and loss on a fixed input.

Load an extracted artifact with:

```python
from leakagebench_midi.models import load_checkpoint

model, metadata = load_checkpoint("path/to/checkpoint.pt", map_location="cpu")
```

Float16 Transformer artifacts preserve the released source tensors exactly. For
CPU inference, cast the loaded model to float32 before loss evaluation when the
local PyTorch build does not implement float16 cross-entropy.

The Phase-2 Transformer contains the historical RoPE tensor layout used by the
archived training source. It may look unusual compared with common textbook
implementations, but changing it would make the public definition incompatible
with the trained checkpoints. The loader therefore preserves it deliberately.

These are final, inference-only weights. They do not include optimizer,
scheduler, RNG state, intermediate checkpoints, or full training logs, so they
support integrity and inference checks but not exact training resumption.

Source code in this repository is licensed under MIT. Numerical model artifacts
are licensed under Creative Commons Attribution 4.0 International
(`CC-BY-4.0`), **only to the extent of rights held by the LeakageBench-MIDI
authors in those artifacts**.

That model-artifact license does not grant, transfer, replace, or modify rights
in the underlying third-party training datasets. LMD and PDMX remain subject to
their original terms and attribution requirements. No raw LMD or PDMX files,
MIDI, scores, lyrics, or training examples are included in the model bundles.
