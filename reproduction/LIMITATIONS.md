# Reproduction boundaries

This is strong **paper-statistics reproduction**, not unrestricted raw-data
redistribution or zero-cost retraining.

- Raw LMD/PDMX MIDI and token-bearing manifests are excluded for licensing and
  data-governance reasons.
- Training checkpoints are prepared as a separately governed companion
  artifact but are not included in this GitHub tree.
- Published anonymous model-output rows support exact re-analysis where the
  released units are sufficient but do not regenerate those outputs from MIDI.
- Of 232 numerical fields, 193 are recomputed from public rows and 39 are
  verified against frozen nonidentifying summaries; the latter are not
  independent recomputations.
- The adopted graph is a reference known-family relation. It is neither
  complete ground truth nor a universal detector.
- Zero overlap means zero known overlap under that reference relation.

An end-to-end retraining attempt additionally requires the original datasets,
the companion checkpoints if inference-only verification is desired, and the
documented compute environment. None of those limitations prevents CPU-only
recomputation of the released paper statistics.
