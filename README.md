# LeakageBench-MIDI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22023100.svg)](https://doi.org/10.5281/zenodo.22023100)

LeakageBench-MIDI is a data-agnostic toolkit for measuring and mitigating same-work family leakage in symbolic-music generation evaluation. This directory is the public GitHub software release.

**Downloads:** [source code and public reproduction bundle (ZIP)](https://github.com/kunbai05-nb/LeakageBench-MIDI/archive/refs/tags/v1.1.2.zip) · [all v1.1.2 release files and model checkpoints](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/tag/v1.1.2) · [Zenodo archive](https://doi.org/10.5281/zenodo.22025589)

## Public-release boundaries

- **Data are not distributed.** No raw MIDI, audio, token-bearing training/evaluation manifest, private evaluation item, or third-party dataset is included.
- **The family graph is a reference relation, not a universal detector.** Reported zero overlap always means zero *known* overlap under the adopted reference relation; unknown same-work relations may remain.
- **Checkpoints are not included in this Git source tree; they are distributed separately.** The companion model release contains 60 inference-only artifacts: 30 confirmatory/capacity/external checkpoints, 9 Phase-2 Transformer-L checkpoints, 9 Conditional VAE checkpoints, 9 Latent Diffusion checkpoints, and 3 condition-invariant neutral encoders required by the diffusion endpoint. Intermediate checkpoints, optimizer/RNG state, and training logs are excluded. See [docs/model_artifacts.md](docs/model_artifacts.md).
- **Internal provenance is not distributed.** Server paths, synchronization records, training logs, private registries, and internal evidence trees are excluded.

## Installation

```bash
python -m venv .venv
.venv/bin/pip install -e '.[test]'
```

## Data-free quick start

```bash
bash scripts/run_synthetic_demo.sh ./_public_demo
```

The demo creates synthetic MIDI files locally in the selected output directory; no research data are downloaded or bundled.

## Strong paper-statistics reproduction

This release includes a public-safe analysis bundle: anonymous
per-family/per-seed model objectives, family-level generation and mechanism
metrics, family-size statistics, and all 2,900 mitigation-robustness simulation
runs. It contains scalar measurements only—no MIDI, token sequences, file
identities, original family hashes, or checkpoints.

Audit every numerical field used by `manuscript_results_v2`:

```bash
python scripts/reproduce_paper_statistics.py --output ./_reproduced_paper_statistics
```

The command verifies the reproduction-data manifest, reruns deterministic
estimators where the public rows are sufficient, and writes provenance for
every field. The expected release status is 232 numerical fields passed: 193
recomputed from public rows and 39 verified against frozen nonidentifying
summaries, with zero mismatches, zero unreproduced fields, and three
qualitative/intentionally-missing entries marked not applicable. It is
CPU-only and does not access the network. “Verified” is not presented as
independent recomputation.

For a single-command reviewer workflow that recomputes the paper statistics,
verifies and materializes the frozen typeset artifacts, and runs the public
test suite:

```bash
bash scripts/reproduce_all.sh ./_reproduced_release
```

The same command runs automatically in a clean Ubuntu/Python 3.10.12 GitHub
Actions environment on every push and pull request. The CI job uploads its
field-by-field audit as a build artifact, including on failed runs.

## Verify the typeset release artifacts

The public-safe `manuscript_results_v2` export retains frozen numerical results and removes internal source paths and registries. Verify its checksum and all released table/figure artifacts with:

```bash
python scripts/reproduce_public_results.py --verify
```

To materialize a verified copy of the released tables and figures:

```bash
python scripts/reproduce_public_results.py --output ./_reproduced_results
```

This verifies and materializes the frozen typeset artifacts after the paper
statistics have been audited as above. It does not retrain
models or reconstruct restricted datasets. Training reproduction requires
users to obtain the underlying datasets under their own terms and follow the
frozen method described in [METHODS_PUBLIC.md](docs/METHODS_PUBLIC.md) and the
detailed [public v2 protocol](docs/PROTOCOL_V2.md).

## Tests

```bash
python -m pytest -q
```

The tests cover family-component construction, leakage auditing, component-aware splitting, controlled contamination logic, structural normalization, synthetic integration, model-loader validation with temporary synthetic weights, and public-result integrity.

## Companion model checkpoints

The current **LeakageBench-MIDI v1.1.2** GitHub release includes the unchanged
**Model Checkpoints v1.1.0** companion assets. The weights are licensed under
CC-BY-4.0 to the extent of rights held by the authors and are split by
architecture so users can download only the model families they need. Every
checkpoint records its condition, seed, configuration, source-checkpoint hash,
and paper role in `MODEL_MANIFEST.json`.

### Direct checkpoint downloads

Download [`SHA256SUMS`](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/SHA256SUMS)
and [`model-release-metadata.tar.gz`](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/model-release-metadata.tar.gz)
first. The metadata archive contains `MODEL_MANIFEST.json`, licenses, model
card, and verification metadata.

| Models | Download | Approx. size |
|---|---|---:|
| Phase-2 Transformer-L (9 checkpoints) | [`phase2-transformer-l.tar.gz`](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/phase2-transformer-l.tar.gz) | 297 MB |
| Conditional VAE (9 checkpoints) | [`conditional-vae.tar.gz`](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/conditional-vae.tar.gz) | 40 MB |
| Latent Diffusion + neutral encoders (12 artifacts) | [`latent-diffusion-and-neutral-encoders.tar.gz`](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/latent-diffusion-and-neutral-encoders.tar.gz) | 31 MB |
| Legacy LMD Transformer-L | [`lmd-transformer-l-legacy.tar.gz`](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/lmd-transformer-l-legacy.tar.gz) | 166 MB |
| LMD Transformer-M | [`lmd-transformer-m.tar.gz`](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/lmd-transformer-m.tar.gz) | 127 MB |
| LMD Transformer-S | [`lmd-transformer-s.tar.gz`](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/lmd-transformer-s.tar.gz) | 102 MB |
| LMD TCN | [`lmd-tcn.tar.gz`](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/lmd-tcn.tar.gz) | 101 MB |
| PDMX Transformer-L | [`pdmx-transformer-l.tar.gz`](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/download/v1.1.2/pdmx-transformer-l.tar.gz) | 166 MB |

To download every release attachment with the GitHub CLI:

```bash
gh release download v1.1.2 \
  --repo kunbai05-nb/LeakageBench-MIDI
```

After downloading and extracting a model archive:

```python
from leakagebench_midi.models import load_checkpoint

model, metadata = load_checkpoint("path/to/seed_202608040.pt", map_location="cpu")
```

Verify a complete extracted companion directory without training:

```bash
python scripts/verify_model_checkpoints.py \
  ../models/LeakageBench-MIDI-Model-Checkpoints-v1.1.0
```

The expected result is 60/60 integrity and strict-load passes plus five/five
representative fixed-input output checks. These are final inference-only
weights; optimizer, scheduler, RNG state, and intermediate checkpoints are not
included.

Phase-2 covers `clean`, `unrelated_donor`, and `same_family_donor` for all three
formal seeds. Conditional VAE and Latent Diffusion use the same three-condition
coverage. Each diffusion checkpoint must be paired with the neutral encoder of
the same seed identified by `linked_encoder_model_id`. The checkpoint package is
not needed to recompute the released paper statistics, but it supports direct
inference and model-level inspection. Download the grouped assets from the
[current consolidated release](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/tag/v1.1.2).

## Repository structure

- `leakagebench_midi/`: reusable core code and model definitions.
- `scripts/`: public, data-agnostic command-line workflows.
- `configs/`: public protocol/configuration material.
- `tests/`: data-free public tests.
- `docs/`: public methods, data policy, provenance boundaries, and reproduction guidance.
- `results/`: final tables, figures, and sanitized `manuscript_results_v2` values.
- `reproduction/`: anonymous analysis units, frozen nonidentifying sufficient
  statistics, a data dictionary, and a hash manifest.
- `examples/`: self-contained synthetic example specification.

## Scientific scope

File identity is not necessarily musical-work identity. Multiple exports, arrangements, transcriptions, and versions can cross a random file split. LeakageBench-MIDI audits the resulting known family crossings and provides family/component-atomic splitting. Family labels have incomplete coverage, and detector quality must be reported through relation, pairwise, and component-level metrics. NLL effects do not by themselves establish perceptual quality or plagiarism.

The available family evidence and missing human-annotation boundary are
reported in [family_reference_validation.md](docs/family_reference_validation.md).
The official 176,581-file count and frozen 178,561-identity downstream count
are intentionally kept as separate provenance scopes in
[data_provenance.md](docs/data_provenance.md); their 1,980 difference is not
given an unsupported causal explanation.

## Citation and license

Citation metadata are in [CITATION.cff](CITATION.cff). Cite the current software
through the version-independent concept DOI
[`10.5281/zenodo.22023100`](https://doi.org/10.5281/zenodo.22023100), which
resolves to the latest retained Zenodo record.

Project code and documentation are released under the [MIT License](LICENSE);
that license does not grant redistribution rights for third-party music
datasets.
