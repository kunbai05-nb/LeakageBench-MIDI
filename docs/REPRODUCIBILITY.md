# Reproducibility

Four public levels are supported. Reviewers can run all public-safe checks with
`bash scripts/reproduce_all.sh ./_reproduced_release`.

The repository CI executes that workflow in a clean Ubuntu/Python 3.10.12
environment and retains the machine-readable field audit as a workflow
artifact. This detects dependency, packaging, and numerical regressions before
a public tag is cut.

1. Run `python -m pytest -q` for data-free core and integration tests.
2. Run `bash scripts/run_synthetic_demo.sh ./_public_demo` for a synthetic end-to-end workflow.
3. Run `python scripts/reproduce_public_results.py --verify` to verify the sanitized v2 result export and every released table/figure artifact. Use `--output ./_reproduced_results` to materialize verified copies.
4. Run `python scripts/reproduce_paper_statistics.py --output ./_reproduced_paper_statistics` to audit every numerical manuscript-result field and record whether it was recomputed or verified.

Level 4 is the primary paper-statistics audit. It passes 232 numerical fields:
193 are recomputed from public analysis/simulation rows and 39 are verified
against released frozen summaries. The latter comprise inferential fields for
which the exact preregistered display-chain inputs are not public; they are not
called independent recomputations. The distinction is recorded per field in
`REPRODUCED_FIELD_AUDIT.csv`.

The release does not contain raw datasets, token-bearing manifests, or
checkpoints, so it cannot retrain the paper models by itself. Dataset
acquisition must follow the original providers' terms. The frozen analysis
design is documented in [PROTOCOL_V2.md](PROTOCOL_V2.md), and the public
analysis-unit schema is documented in
[`reproduction/DATA_DICTIONARY.md`](../reproduction/DATA_DICTIONARY.md).

Separately downloaded checkpoints can be checked with
`python scripts/verify_model_checkpoints.py CHECKPOINT_ROOT`. This verifies
weights and representative inference outputs; it does not resume training.
