# Model checkpoints

Download the archives linked in the [README](../README.md), then extract them
into one directory.

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
