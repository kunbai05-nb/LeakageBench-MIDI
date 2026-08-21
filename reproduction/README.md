# Paper-statistics bundle

Run:

```bash
python scripts/reproduce_paper_statistics.py --output ./statistics
```

`data/` contains anonymous analysis rows and simulations. `frozen/` contains
the aggregate statistics needed for fields that cannot be recalculated from
those rows. `PUBLIC_REPRODUCTION_MANIFEST.json` records every file size and
SHA-256 hash. Column definitions are in [DATA_DICTIONARY.md](DATA_DICTIONARY.md).
