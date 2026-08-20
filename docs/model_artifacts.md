# Companion model artifacts

Model weights are not committed to the Git source tree. They are released as the
separate [**LeakageBench-MIDI Model Checkpoints v1.1.0** companion release](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/tag/models-v1.1.0).
Keeping weights separate avoids Git/LFS dependency and
lets users download only the architecture groups needed for a reproduction.

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

Load an extracted artifact with:

```python
from leakagebench_midi.models import load_checkpoint

model, metadata = load_checkpoint("path/to/checkpoint.pt", map_location="cpu")
```

Float16 Transformer artifacts preserve the released source tensors exactly. For
CPU inference, cast the loaded model to float32 before loss evaluation when the
local PyTorch build does not implement float16 cross-entropy.

Source code in this repository is licensed under MIT. Numerical model artifacts
are licensed under Creative Commons Attribution 4.0 International
(`CC-BY-4.0`), **only to the extent of rights held by the LeakageBench-MIDI
authors in those artifacts**.

That model-artifact license does not grant, transfer, replace, or modify rights
in the underlying third-party training datasets. LMD and PDMX remain subject to
their original terms and attribution requirements. No raw LMD or PDMX files,
MIDI, scores, lyrics, or training examples are included in the model bundles.
