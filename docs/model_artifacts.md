# Model checkpoints

The [v1.1.2 release](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/tag/v1.1.2)
contains 60 inference checkpoints for the LMD and PDMX experiments. Archive
links are listed in the main [README](../README.md).

Verify a downloaded bundle:

```bash
python scripts/verify_model_checkpoints.py /path/to/checkpoints
```

Load one checkpoint:

```python
from leakagebench_midi.models import load_checkpoint

model, metadata = load_checkpoint("checkpoint.pt", map_location="cpu")
```

Evaluate it on the reconstructed LMD windows:

```bash
python scripts/evaluate_checkpoint.py checkpoint.pt prepared_lmd evaluation
```

The checkpoint archives are CC BY 4.0.
