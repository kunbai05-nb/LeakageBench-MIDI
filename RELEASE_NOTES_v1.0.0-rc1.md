# LeakageBench-MIDI v1.0.0-rc1

This is the paper-grade release candidate for LeakageBench-MIDI.

## Included

- frozen, hash-linked result and claim registries;
- deterministic reproduction of release tables without training;
- a self-contained synthetic reproducibility workflow;
- family-map, leakage-audit, and family-aware split tools; and
- public documentation for protocol, data provenance, limitations, and redistribution policy.

## Evidence boundary

The release reports known family contamination under an adopted family definition, controlled leakage-induced likelihood advantage, a monotonic capacity-associated trend within the tested Transformer family, a smaller same-direction TCN effect whose practical replication threshold was rejected, and reduced protocol-eligible PDMX external evidence. NLL is an evaluation endpoint, not a perceptual-quality measure.

## Known limitations

Family labels have incomplete recall. The frozen 178,561-identity downstream LMD universe differs in scope from the official LMD v0.1 documentation's 176,581 MD5-distinct file count; this release does not independently reconstruct the transformation between them. PDMX evidence is limited to 67 eligible treated families. Canonical-event and track-order hashing is not full-corpus coverage.

## Data availability

No original LMD/PDMX MIDI, audio, scores, archives, model checkpoints, or closed evaluation data are redistributed. Users must obtain source datasets from their official locations and comply with their licenses.
