# LeakageBench-MIDI

> A reproducible framework for measuring and mitigating same-work family leakage in symbolic music generation evaluation.

## Overview

LeakageBench-MIDI provides data-agnostic Python APIs and command-line workflows for building same-work family maps, auditing split contamination, constructing family-atomic splits, designing controlled contamination datasets, estimating family-level effects, and reproducing the paper's frozen tables. The repository redistributes no LMD/PDMX music data, checkpoints, or closed evaluation material.

This candidate targets the stable software release **v1.0.0**.

Choi et al. (2025), *On the De-duplication of the Lakh MIDI Dataset*, provides the duplicate/same-song identification and filtering precedent used for the adopted LMD family relations. LeakageBench-MIDI studies downstream consequences and mitigation; it does not present that upstream resource as a project contribution.

## Why file split != work split

File identity is not musical-work identity. Duplicate exports, versions, transcriptions, and arrangements of the same work can cross a random file split, placing a training sibling beside an evaluation receiver. This can create leakage-induced likelihood advantage without establishing better perceptual music quality.

## Key findings

- **LMD 80/10/10:** 27.66% known test-family contamination and 29.77% known test-file contamination.
- **Transformer-L:** control-adjusted `tau = -0.115317`, corresponding to a 13.62% relative treated-family NLL improvement.
- **Normalized structural robustness:** after PPQ, serialization, event-order, metadata, and track-container normalization, 50/100 treated LMD families remained structurally non-exact. Within this stricter subset, the effect remained strong (`tau = -0.11485`, 95% CI `[-0.13946, -0.09208]`).
- **TCN:** a smaller 2.53% same-direction relative effect; the preregistered practical replication threshold was rejected.
- **PDMX:** a 67-family reduced, protocol-eligible cohort showed an 11.07% same-direction relative effect.
- **Mitigation:** family-aware assignment achieved zero known cross-split family overlap under the adopted family definition with 0% data deletion in the frozen 4,300-file pool.

Numerical results are governed by [`metadata/result_registry.json`](metadata/result_registry.json); interpretation boundaries are governed by [`metadata/claim_registry.json`](metadata/claim_registry.json).

## Installation

```bash
python -m venv .venv
.venv/bin/pip install -e '.[test]'
```

## Quick Start

The first workflow is fully synthetic and requires no LMD, PDMX, or other restricted data:

```bash
bash scripts/run_synthetic_demo.sh /tmp/leakagebench-synthetic
```

See [`examples/synthetic_demo/README.md`](examples/synthetic_demo/README.md) for the generated artifacts.

## Model checkpoint loading

The companion model artifact is loaded through the stable, inference-only API. After installing LeakageBench-MIDI v1.0.0 and downloading a checkpoint archive:

```python
from pathlib import Path
import torch
from leakagebench_midi.models import load_checkpoint

model, metadata = load_checkpoint(
    Path("lmd/transformer_l/clean/seed_202608040.pt")
)
token_ids = torch.zeros((1, 8), dtype=torch.long)
attention_mask = torch.ones_like(token_ids)
loss_mask = torch.ones_like(token_ids)
with torch.inference_mode():
    output = model(token_ids, attention_mask, loss_mask)
print(metadata["model_id"], output["logits"].shape)
```

The loader reads the embedded public configuration, instantiates the exact Transformer or TCN architecture, loads with `strict=True`, and returns an evaluation-mode model. It needs no training checkout, private module, source checkpoint, or dataset path.

## Core workflows

The stable task-oriented CLIs live in [`scripts/`](scripts/):

- `build_family_map.py` constructs connected family components from identifiers and edges;
- `audit_split.py` measures known family crossings;
- `build_family_split.py` assigns whole families atomically;
- `construct_contamination.py` creates approximately token-budget-matched controlled conditions and records every pairwise and total difference;
- `analyze_family_effects.py` estimates control-adjusted family effects;
- `run_leakage_census.py` evaluates contamination under random file splitting; and
- `classify_pair_structure.py` compares MIDI pairs under the frozen structural hierarchy.

Each CLI documents its arguments with `--help`. Dataset acquisition and redistribution boundaries are in [`docs/datasets.md`](docs/datasets.md) and [`docs/data_policy.md`](docs/data_policy.md).

## Reproducing paper results

Frozen paper tables and figure source data can be regenerated without training or dataset access:

```bash
python scripts/reproduce_paper_results.py
```

The script verifies [`reproducibility/INTEGRITY_MANIFEST.json`](reproducibility/INTEGRITY_MANIFEST.json), recomputes model effects from public-safe frozen raw-result rows, validates registry/evidence hashes, and writes deterministic CSVs. See [`docs/reproducing_results.md`](docs/reproducing_results.md) and [`docs/methodology.md`](docs/methodology.md).

## Data provenance

The official LMD v0.1 documentation reports **176,581 MD5-distinct files**. The frozen downstream LeakageBench-MIDI family universe contains **178,561 identities**. These are different provenance scopes; this project does not relabel 178,561 as the official LMD file count and does not claim to have independently reconstructed the transformation between them. See [`docs/data_provenance.md`](docs/data_provenance.md).

## Repository structure

- [`leakagebench_midi/`](leakagebench_midi/) — reusable, data-agnostic Python package;
- [`scripts/`](scripts/) — task-oriented command-line workflows;
- [`metadata/`](metadata/) — frozen result and claim registries;
- [`examples/synthetic_demo/`](examples/synthetic_demo/) — self-contained synthetic workflow;
- [`docs/`](docs/) — methodology, provenance, policy, and limitations context;
- [`reproducibility/`](reproducibility/) — paper protocol, aggregate evidence, tables, and figure source data; and
- [`tests/`](tests/) — lightweight public workflow tests.

## Limitations

Family labels have incomplete recall, so unknown relations may remain. Zero known cross-split family overlap is conditional on the adopted family definition. Canonical-event and track-order hashing covers only 4,300 of 178,561 frozen identities. NLL is an evaluation endpoint, not a perceptual-quality measure. PDMX evidence is limited to 67 eligible treated families, and the TCN practical replication decision was rejected. The tested Transformer capacity trend is not a universal scaling law. Family membership or leakage evidence does not establish plagiarism.

## Citation

**Author:** Kun Bai  
**ORCID:** [0009-0006-4134-9796](https://orcid.org/0009-0006-4134-9796)

Software citation metadata is provided in [`CITATION.cff`](CITATION.cff). DOI will be added after archival release.

## License

Project code and documentation are licensed under the [MIT License](LICENSE). This license does not grant redistribution rights for LMD, PDMX, or other third-party datasets.

The 30 traceable model checkpoints are released as a separate v1.0.0 companion archival artifact. The approved model-artifact policy is documented in [`docs/model_artifacts.md`](docs/model_artifacts.md); no model weights or dataset files are included in this software repository. Archival DOIs are assigned only during publication.
