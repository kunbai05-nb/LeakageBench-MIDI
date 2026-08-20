# Public strong-reproduction bundle

This directory is the public-safe analysis bundle used to audit the paper's
numerical claims without redistributing music data.

Run from the repository root:

```bash
python scripts/reproduce_paper_statistics.py --output ./_reproduced_paper_statistics
```

`data/` contains anonymous analysis units and seed-level simulations.
`frozen/` contains nonidentifying canonical statistical-audit outputs needed
to verify multiplicity and display-chain statistics that cannot be rebuilt
from the public rows. The current field audit has 193 recalculated numerical
fields and 39 frozen-summary verifications; all 232 match. The audit labels
these categories separately. `PUBLIC_REPRODUCTION_MANIFEST.json` fixes every
file by byte size and SHA-256.

The public family identifiers are release-specific one-way pseudonyms. They
exist only to preserve pairing across conditions and seeds. They are not work
IDs, file hashes, dataset keys, or a universal family detector.

No file in this directory contains MIDI/audio, token sequences, local paths,
usernames, server locations, model weights, or training logs. `token_count`
columns are scalar denominators only.

## Boundaries

This is strong paper-statistics reproduction, not unrestricted raw-data
redistribution or exact retraining.

- Raw LMD/PDMX MIDI and token-bearing manifests are excluded for licensing and
  data-governance reasons.
- The release attachments contain final inference checkpoints, not optimizer,
  scheduler, RNG, intermediate-checkpoint, or training-log state.
- Anonymous model-output rows support exact re-analysis where their released
  units are sufficient, but they do not regenerate model outputs from MIDI.
- Of 232 numerical fields, 193 are recomputed from public rows and 39 are
  verified against frozen nonidentifying summaries; the latter are not
  independent recomputations.
- The adopted graph is a reference known-family relation, not complete ground
  truth or a universal detector. Zero overlap means zero known overlap under
  that relation.

End-to-end retraining additionally requires the original datasets and a full
training pipeline. These limits do not prevent CPU-only recomputation of the
released paper statistics.
