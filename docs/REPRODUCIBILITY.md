# Reproduction

Run the complete CPU workflow from the repository root:

```bash
bash scripts/reproduce_all.sh ./reproduced
```

It audits all 232 numerical manuscript fields, verifies released tables and
figures, and runs the test suite. Outputs include a field-level CSV identifying
193 values recomputed from public rows and 39 checked against frozen summaries.

Useful individual commands:

```bash
bash scripts/run_synthetic_demo.sh ./demo
python scripts/reproduce_paper_statistics.py --output ./statistics
python scripts/reproduce_public_results.py --verify
python scripts/verify_model_checkpoints.py /path/to/checkpoints
```

See [PROTOCOL_V2.md](PROTOCOL_V2.md) for the experiment settings and
[`reproduction/DATA_DICTIONARY.md`](../reproduction/DATA_DICTIONARY.md) for the
released analysis columns.
