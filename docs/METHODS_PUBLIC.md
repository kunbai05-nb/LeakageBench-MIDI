# Public Methods

## Scope

LeakageBench-MIDI studies same-work family leakage in symbolic-music generation evaluation. A file-level split is contaminated under the adopted relation when a training file and an evaluation file belong to the same connected reference family. The reference relation is deliberately treated as incomplete: it is an operational known-family graph, not a universal detector or exhaustive ground truth.

The exact frozen model, training, generation, and statistical settings are in
[PROTOCOL_V2.md](PROTOCOL_V2.md) and
[`configs/protocol_v2.json`](../configs/protocol_v2.json). This page is the
short conceptual overview.

## Family construction and audit

Files are represented as graph nodes and inferred same-work relations as undirected edges. Connected components define operational families. Split audits report known cross-split family count, test-family contamination, and test-file contamination. Unknown relations are outside the estimand and remain an explicit limitation.

## Family-aware splitting

Inferred connected components are assigned atomically to train, validation, or test. The splitter targets requested file or token ratios without deleting members. The bounded claim is zero known cross-split overlap under the adopted relation. When family inference is incomplete, residual known leakage is evaluated against the full frozen reference graph; false-positive simulations additionally track over-merging, largest-component growth, and ratio distortion.

## Controlled exposure design

For each treated family, a receiver is held out and a designated same-family donor is either excluded or admitted to training. Clean and unrelated-donor conditions separate family-specific exposure from generic added-data effects. Receivers, controls, and validation families remain outside training. Token budgets are matched by deterministic, family-disjoint removal from the base pool.

## Estimand and uncertainty

Within each seed, family-level changes are computed between conditions. The primary controlled estimand subtracts control-family drift from treated-family drift. Families receive equal weight. Confidence intervals use family-cluster resampling; windows, notes, and tokens are not treated as independent statistical units. Simulation seeds describe robustness and are summarized with empirical intervals rather than interpreted as independent real-world datasets.

## Structural and generation analyses

Structural comparison distinguishes byte identity, canonical event equivalence, track-order-invariant equivalence, normalized structural equivalence, and structurally non-exact family relations. Generation analyses compare receiver alignment, surface statistics, and local shared-span behavior. Likelihood advantage and symbolic similarity are not equated with perceptual quality or plagiarism.

## Public reproduction boundary

This GitHub release contains no research MIDI, token-bearing manifest, training checkpoint, or private evaluation item. It supports data-free unit tests, a synthetic end-to-end demonstration, and a field-level audit of final public results. Of 232 numerical fields, 193 are recalculated from anonymous public analysis rows and 39 are verified against frozen nonidentifying summaries because their exact display-chain inputs are not public. Re-running restricted-data training requires separately obtained datasets and is outside this software archive. Inference-only final weights are distributed separately.

## Result governance

`results/manuscript_results_v2_public.json` is a public-safe projection of the frozen v2 result lock. Numerical fields are retained; internal registry references, server paths, and source-artifact provenance are removed. The source formal result files and protocol are not modified by release construction.
