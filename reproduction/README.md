# Public strong-reproduction bundle

This directory is the minimum public-safe dataset needed to recompute the
paper's numerical claims without redistributing music data.

Run from the repository root:

```bash
python scripts/reproduce_paper_statistics.py --output ./_reproduced_paper_statistics
```

`data/` contains anonymous analysis units and seed-level simulations.
`frozen/` contains nonidentifying canonical statistical-audit outputs needed
to verify multiplicity and a small number of preregistered display-chain
statistics. `PUBLIC_REPRODUCTION_MANIFEST.json` fixes every file by byte size
and SHA-256.

The public family identifiers are release-specific one-way pseudonyms. They
exist only to preserve pairing across conditions and seeds. They are not work
IDs, file hashes, dataset keys, or a universal family detector.

No file in this directory contains MIDI/audio, token sequences, local paths,
usernames, server locations, model weights, or training logs. `token_count`
columns are scalar denominators only.
