# LeakageBench-MIDI

> A reproducible framework for measuring and mitigating same-work family leakage in symbolic music generation evaluation.

Release candidate: **v1.0.0-rc1**.

## Why this matters

File identity is not musical-work identity. The same work can appear as duplicate exports, versions, transcriptions, or arrangements, so a random file split can place a training sibling beside an evaluation receiver. That can create leakage-induced likelihood advantage without establishing better perceptual music quality.

## What LeakageBench-MIDI studies

- prevalence of known same-work family contamination;
- controlled downstream evaluation inflation;
- robustness on structurally non-exact family relations;
- model susceptibility across tested architectures and Transformer capacities;
- reduced, protocol-eligible PDMX external evidence; and
- family-aware split mitigation.

Choi et al. (2025), *On the De-duplication of the Lakh MIDI Dataset*, is the duplicate/same-song identification and filtering precedent used for the adopted LMD family relations. LeakageBench-MIDI studies controlled downstream consequences, structural robustness, model susceptibility, and mitigation; it does not present that upstream identification resource as a project contribution.

## Key frozen findings

- **LMD 80/10/10:** 27.66% known test-family contamination and 29.77% known test-file contamination.
- **Transformer-L:** control-adjusted `tau = -0.115317`, corresponding to a 13.62% relative treated-family NLL improvement.
- **Structurally non-exact:** 95/100 treated LMD families retained a same-direction effect (`tau ≈ -0.113316`).
- **PDMX:** a 67-family reduced, protocol-eligible external cohort showed an 11.07% same-direction relative effect (`tau ≈ -0.067776`).
- **Mitigation:** family-aware assignment achieved zero known cross-split family overlap under the adopted family definition with 0% data deletion in the frozen 4,300-file pool.

All numerical claims are governed by [`result_registry.json`](assets/release_metadata/result_registry.json); permitted interpretation boundaries are governed by [`claim_registry.json`](assets/release_metadata/claim_registry.json).

## What this project does NOT claim

- It is not a new music-generation architecture.
- It is not a new duplicate detector; the adopted LMD relations build on Choi et al. (2025).
- Family membership or leakage evidence does not establish plagiarism.
- NLL differences do not establish perceptual music quality.
- The PDMX result is reduced, protocol-eligible external evidence rather than a corpus-wide replication.
- The Transformer result is a monotonic capacity-associated trend within the tested Transformer family, not a claim about every architecture or dataset.
- Zero known cross-split family overlap under the adopted family definition does not establish absence of latent, unlabeled relations.

The TCN result is a smaller but statistically consistent same-direction effect; its preregistered practical replication threshold was rejected.

## Quick Start

The first example is fully synthetic and does not require LMD, PDMX, or other restricted data.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[test]'
bash scripts/run_synthetic_mini_pipeline.sh /tmp/leakagebench-synthetic
```

## Reproducing Results

Frozen release tables can be regenerated without training or dataset access:

```bash
bash scripts/reproduce_release_tables.sh
```

See [`docs/REPRODUCING_RESULTS.md`](docs/REPRODUCING_RESULTS.md) for scope and [`docs/PROTOCOL.md`](docs/PROTOCOL.md) for the frozen estimand.

## Data Provenance

This repository does not redistribute LMD/PDMX MIDI, audio, scores, checkpoints, or the permanently closed clean-test. Users must legally obtain source datasets themselves. The official LMD v0.1 documentation reports 176,581 MD5-distinct MIDI files; this release uses a frozen 178,561-identity downstream analysis universe. These are different provenance scopes, and this release does not independently reconstruct a transformation between them. See [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md) and [`docs/DATASETS.md`](docs/DATASETS.md).

## Limitations

Family labels have incomplete recall, so unknown relations may remain. Full-LMD canonical-event and track-order hashing covers only 4,300 of 178,561 frozen identities. The TCN practical replication decision was rejected. PDMX evidence is limited to 67 eligible treated families biased toward 4/4, 1024-window-eligible works. Copy evidence is secondary. The frozen identity-count provenance distinction described above remains unresolved but disclosed and does not change the downstream calculations performed on the frozen universe.

## Citation

**Author:** Kun Bai  
**ORCID:** [0009-0006-4134-9796](https://orcid.org/0009-0006-4134-9796)  
**GitHub repository target:** <https://github.com/kunbai05-nb/LeakageBench-MIDI> (intended URL; not verified live at this release gate)

Software citation before archival DOI assignment: Bai, K. *LeakageBench-MIDI*, v1.0.0-rc1. See [`CITATION.cff`](CITATION.cff). A DOI will be added only after an archival release creates one.

## License

Project code and documentation are licensed under the [MIT License](LICENSE). This license does not grant redistribution rights for LMD, PDMX, or other third-party datasets.
