# Reproducing frozen results

No model training or dataset access is needed:

```bash
python scripts/reproduce_paper_results.py
```

The command verifies the independent integrity manifest, checks every registry-to-evidence SHA-256 link, recomputes the five model result groups from public-safe per-family/per-seed/per-condition NLL rows, and fails if reconstructed effects or confidence intervals differ from the frozen registry. It then renders the paper-facing CSV tables.

The public raw analysis rows contain stable opaque family keys, formal seeds, conditions, splits, NLL, and token counts. They contain no MIDI, filenames, dataset paths, hostnames, usernames, or clean-test content. Formal mode in `analyze_family_effects.py` requires `--family_manifest`; omission is permitted only with the explicit exploratory flag.

Bootstrap p-values use the add-one correction `(extreme_count + 1) / (B + 1)`. With 10,000 draws and no opposing draws, the stored numeric value is `1/10001`, not zero. Primary effect sizes, confidence intervals, cohorts, and bootstrap draws remain frozen.

Seed-level results are a secondary sensitivity analysis. Only three training seeds are available, so they do not provide a precise estimate of population-level training-randomness variance.
