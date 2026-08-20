# LeakageBench-MIDI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22023100.svg)](https://doi.org/10.5281/zenodo.22023100)

LeakageBench-MIDI is a data-agnostic toolkit for measuring and mitigating same-work family leakage in symbolic-music generation evaluation. This directory is the public GitHub software release.

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

This release includes a public-safe minimum sufficient dataset: anonymous
per-family/per-seed model objectives, family-level generation and mechanism
metrics, family-size statistics, and all 2,900 mitigation-robustness simulation
runs. It contains scalar measurements only—no MIDI, token sequences, file
identities, original family hashes, or checkpoints.

Recompute every numerical field used by `manuscript_results_v2`:

```bash
python scripts/reproduce_paper_statistics.py --output ./_reproduced_paper_statistics
```

The command verifies the reproduction-data manifest, reruns deterministic
estimators and bootstraps, and writes a field-by-field audit. The expected
release status is 232 numerical fields passed, zero mismatches, zero
unreproduced fields, and three qualitative/intentionally-missing entries marked
not applicable. It is CPU-only and does not access the network.

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
statistics can be independently recomputed as above. It does not retrain
models or reconstruct restricted datasets. Training reproduction requires
users to obtain the underlying datasets under their own terms and follow the
frozen method described in [METHODS_PUBLIC.md](docs/METHODS_PUBLIC.md).

## Tests

```bash
python -m pytest -q
```

The tests cover family-component construction, leakage auditing, component-aware splitting, controlled contamination logic, structural normalization, synthetic integration, model-loader validation with temporary synthetic weights, and public-result integrity.

## Companion model checkpoints

The separately archived **LeakageBench-MIDI Model Checkpoints v1.1.0** package is
licensed under CC-BY-4.0 to the extent of rights held by the authors. It is split
by architecture so users can download only the model families they need. Every
checkpoint records its condition, seed, configuration, source-checkpoint hash,
and paper role in `MODEL_MANIFEST.json`.

After downloading and extracting a model archive:

```python
from leakagebench_midi.models import load_checkpoint

model, metadata = load_checkpoint("path/to/seed_202608040.pt", map_location="cpu")
```

Phase-2 covers `clean`, `unrelated_donor`, and `same_family_donor` for all three
formal seeds. Conditional VAE and Latent Diffusion use the same three-condition
coverage. Each diffusion checkpoint must be paired with the neutral encoder of
the same seed identified by `linked_encoder_model_id`. The checkpoint package is
not needed to recompute the released paper statistics, but it supports direct
inference and model-level inspection. Download the grouped assets from the
[model checkpoint release](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/tag/models-v1.1.0).

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

A Chinese file-by-file guide to every public artifact is available in
[docs/PUBLIC_FILE_GUIDE_CN.md](docs/PUBLIC_FILE_GUIDE_CN.md).

## Scientific scope

File identity is not necessarily musical-work identity. Multiple exports, arrangements, transcriptions, and versions can cross a random file split. LeakageBench-MIDI audits the resulting known family crossings and provides family/component-atomic splitting. Family labels have incomplete coverage, and detector quality must be reported through relation, pairwise, and component-level metrics. NLL effects do not by themselves establish perceptual quality or plagiarism.

## Citation and license

Citation metadata are in [CITATION.cff](CITATION.cff). Cite the immutable v1.1.1
archive with DOI
[`10.5281/zenodo.22023101`](https://doi.org/10.5281/zenodo.22023101). Use the
concept DOI [`10.5281/zenodo.22023100`](https://doi.org/10.5281/zenodo.22023100)
when referring to the software project across versions.

Project code and documentation are released under the [MIT License](LICENSE);
that license does not grant redistribution rights for third-party music
datasets.
