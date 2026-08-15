# Reproducing frozen result tables

No model training or dataset access is needed:

```bash
bash scripts/reproduce_release_tables.sh
```

The script reads `assets/release_metadata/result_registry.json` and writes deterministic CSVs to `release_results/`:

- Table 1: Full-LMD census (`table_lmd_census.csv`)
- Table 2: Transformer-L confirmatory (`table_confirmatory.csv`)
- Table 3: TCN and Transformer S/M/L (`table_architecture_capacity.csv`)
- Table 4: mitigation (`table_mitigation.csv`)
- Table 5: reduced PDMX external cohort (`table_pdmx_external.csv`)
- figure-ready prevalence, capacity, mitigation, and external comparison CSVs

The registry records source audit paths and exact SHA-256 values. Reproduction validates presentation, not recomputation or replacement of frozen scientific estimates.

