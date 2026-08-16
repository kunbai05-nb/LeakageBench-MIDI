# Reproducing frozen result tables

No model training or dataset access is needed:

```bash
python scripts/reproduce_paper_results.py
```

The script reads `metadata/result_registry.json` and writes deterministic CSVs to `reproducibility/tables/` and `reproducibility/figures/`:

- Table 1: Full-LMD census (`lmd_census.csv`)
- Table 2: Transformer-L confirmatory (`confirmatory_effect.csv`)
- Table 3: TCN and Transformer S/M/L (`architecture_capacity.csv`)
- Table 4: mitigation (`mitigation.csv`)
- Table 5: reduced PDMX external cohort (`pdmx_external.csv`)
- figure-ready prevalence, capacity, mitigation, and external comparison CSVs

The registry records source evidence paths and exact SHA-256 values. Reproduction validates presentation, not recomputation or replacement of frozen scientific estimates.
