# LeakageBench-MIDI

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22023100-blue.svg)](https://doi.org/10.5281/zenodo.22023100)

Code and checkpoints for studying work-level exposure in symbolic music generation: its effect on evaluation, same-work detection, cross-dataset auditing, and work-level isolation.

## Choose an experiment

Each experiment has a command that generates result tables. Start with the setup section, then run the command for the experiment you want.

| Experiment | Starting point | Original MIDI needed? | Hardware |
| --- | --- | --- | --- |
| [Three-condition experiment](#three-condition-experiment) | Generation-model checkpoints; evaluation windows rebuilt from MIDI | Yes: LMD-full | BF16-capable NVIDIA GPU |
| [Six-dataset detector benchmark](#six-dataset-detector-benchmark) | v1.9 checkpoint and frozen MIDI cohorts | Yes: six datasets | CPU |
| [Fast versus v1.9](#fast-versus-v19) | Candidate features and detector checkpoints | No | CPU |
| [Seven-dataset deployment audit](#seven-dataset-deployment-audit) | Candidate features and auditor checkpoint | No | CPU |
| [ATEPP mitigation](#atepp-mitigation) | Generation-model checkpoints and frozen test windows | No | BF16-capable NVIDIA GPU |
| [Detector ablation and sensitivity](#detector-ablation-and-sensitivity) | Intermediate features; classifiers retrained | No for the default command | CPU |

The feature-based workflows run new inference or training. They do not merely display saved result tables. Their starting points differ, as listed above.

## Setup

Use Ubuntu/Linux with Python 3.10. You need Git, `curl`, `tar`, and `sha256sum`. The pinned environment uses PyTorch 2.8.0 and scikit-learn 1.7.2.

**Private repository access:** authorized users must install GitHub CLI (`gh`) and sign in before cloning or downloading release assets:

```bash
gh auth login
gh auth setup-git
```

Skip this authentication step when the repository is public.

```bash
git clone https://github.com/kunbai05-nb/LeakageBench-MIDI.git
cd LeakageBench-MIDI
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e '.[scale]'
```

Run all remaining commands from this directory with the virtual environment activated. The scripts download and verify their release bundles automatically. Leave room for both the archives and their extracted files. No manual checkpoint paths are needed for the table-reproduction commands.

## Reproduce result tables

### Three-condition experiment

**Prepare:** download `lmd_full.tar.gz` from the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) and save it as `data/lmd_full.tar.gz`. Keep it compressed; the script reads the archive.

**Run:**

```bash
bash scripts/reproduce_three_condition.sh --check-only
bash scripts/reproduce_three_condition.sh
```

The first command checks the environment and entry points; it does not evaluate models or validate the MIDI archive. The second downloads the released model checkpoints, rebuilds the frozen evaluation windows, and evaluates the clean, unrelated-donor, and same-work-donor conditions. It does not retrain the models. Interrupted downloads and completed evaluations are reused.

**Result:** `reproduced/three_condition/tables/three_condition_results.csv`, with Markdown and JSON versions in the same folder.

### Six-dataset detector benchmark

**Prepare:** place the six datasets in the directories listed under [Original MIDI layout](#original-midi-layout).

**Run:**

```bash
bash scripts/reproduce_six_dataset_table.sh --check-only
bash scripts/reproduce_six_dataset_table.sh
```

The first command verifies the frozen cohorts, MIDI checksums, and detector checkpoint without inference. The second runs v1.9 from MIDI using exact retrieval and generates Precision, Recall, F1, and 95% confidence intervals for two threshold settings:

- **Fixed:** the released threshold is applied unchanged to every dataset.
- **Optimal:** the threshold maximizes F1 using that dataset's evaluation labels. This is an oracle comparison, not the deployment setting.

**Result:** `reproduced/six_dataset/tables/six_dataset_results.csv`, with Markdown and JSON versions in the same folder. Per-dataset inference outputs are in `reproduced/six_dataset/raw/`.

This command reproduces the v1.9 rows only; it does not rerun CAugBERT/CLaMP. The SHS cohort contains 306 MIDI files and 80 eligible reference pairs. There is no detector retraining or deployment-threshold recalibration on these evaluation cohorts.

### Fast versus v1.9

**Prepare:** no MIDI download is needed.

**Run:**

```bash
bash scripts/reproduce_fast_vs_v19.sh
```

The script downloads candidate features, manifests, and both detector checkpoints. It recomputes scores and fixed-threshold metrics on ATEPP, LMD-clean, and ASAP. It does not repeat MIDI feature extraction or candidate retrieval.

**Result:** `reproduced/fast_vs_v19/fast_vs_v19.csv`, `fast_vs_v19.md`, and `fast_vs_v19_results.json`.

### Seven-dataset deployment audit

**Prepare:** no MIDI download is needed.

**Run:**

```bash
bash scripts/reproduce_multidataset_deployment.sh
```

The script audits POP909, ASAP, PDMX, GigaMIDI, MAESTRO, LMD, and Aria-MIDI using the **Precision-First Same-Work Auditor**, a separate checkpoint from the v1.9 benchmark detector.

Starting from frozen candidate-pair features, it recomputes classifier scores, applies the decision threshold, rebuilds protected families, and repeats the 200 train/validation/test splits. It starts after candidate retrieval and cheap-gate filtering; neither of those stages is rerun.

**Result:**

```text
reproduced/multidataset_deployment/multidataset_deployment.csv
reproduced/multidataset_deployment/multidataset_deployment.md
reproduced/multidataset_deployment/<dataset>_results.json
reproduced/multidataset_deployment/multidataset_verification.json
```

Family-level and file-level exposure use all test families and all test files, respectively, as denominators. The 95% empirical interval (EI) is the 2.5%–97.5% quantile range across 200 splits. Available official-split results are also reported. Valid-MIDI counts come from the recorded parsing census; multi-version prevalence is recomputed from the rebuilt families.

For the full-MIDI workflow, the downloaded bundle also includes source code in `source/`, file/hash manifests in `midi_hashes/`, and instructions in `FULL_MIDI.md`.

### ATEPP mitigation

**Prepare:** no MIDI download is needed. Use a CUDA GPU with BF16 support.

**Run:**

```bash
bash scripts/reproduce_atepp_mitigation.sh
```

The script downloads the generation-model checkpoints trained under the v1.9 isolation protocol and the frozen test windows. It runs new BF16 evaluation for five conditions and three seeds: untreated, simple deduplication, detector isolation, equal-count random removal, and label-based reference.

**Result:**

```text
reproduction/atepp_mitigation_v1.9/reproduced_results/RESULTS.csv
reproduction/atepp_mitigation_v1.9/reproduced_results/RESULTS.json
```

All conditions use the same held-out evaluation windows and scored-token distribution. Detector isolation, random removal, and the label-based reference each retain 484 training files. Simple deduplication and the untreated condition have identical frozen training streams and share their three checkpoints. Thus, 12 distinct checkpoints cover the 15 condition/seed evaluations.

The script automatically applies the checksum-verified scoring-protocol update. It does not change model weights or require a second checkpoint bundle.

### Detector ablation and sensitivity

**Prepare:** no MIDI download is needed for the recommended workflow.

**Run:**

```bash
bash scripts/reproduce_detector_ablation.sh
```

The script downloads intermediate features, trains and calibrates the classifier variants, evaluates all settings, and generates four tables: view count (Table 1), pairwise-verifier ablation (Table 2), Top-K (Table 4), and classifier count (Table 5). The expensive MIDI feature-extraction stage is skipped; training and evaluation are rerun.

**Result:** CSV and Markdown tables in `reproduced/detector_ablation/tables/`, plus `generated_results.json` and `verification.json`. `PASS` means the reported values match the frozen paper values; `DIFFERENT` means the newly generated values differ and should be inspected.

To rebuild features from MIDI as well, prepare LMD-clean, ATEPP, and ASAP using the layout below, then run:

```bash
bash scripts/reproduce_detector_ablation.sh --from-midi
```

This alternative writes to `reproduced/detector_ablation_from_midi/`. The frozen MAESTRO view-selection input is downloaded automatically.

## Original MIDI layout

Only the three-condition, six-dataset, and optional full-MIDI workflows require original datasets. These datasets are not bundled with the repository or downloaded automatically.

For the six-dataset benchmark, each directory must preserve the relative paths in its corresponding manifest under `reproduction/detector_benchmark/`:

| Dataset | Default directory | Example relative path |
| --- | --- | --- |
| SHS (LMD-full subset) | `data/shs` | `0/0207f2cc938c3691b3af6d8e359bedfd.mid` |
| ASAP | `data/asap` | `Bach/Fugue/bwv_846/Shi05M.mid` |
| ATEPP | `data/atepp` | `Sergei_Rachmaninoff/13_Preludes,_Op._32/No._10_in_B_Minor:_Lento/00046.mid` |
| LMD-clean | `data/lmd-clean` | `clean_midi/2 Brothers on the 4th Floor/Dreams.1.mid` |
| Vienna4x22 | `data/vienna4x22` | `Chopin_op10_no3_p01.mid` |
| PianoVAM | `data/pianovam` | `MIDI/2024-02-14_19-10-09.mid` |

You can use symbolic links instead of copying existing datasets. To use another data location, set `LEAKAGEBENCH_DATA_DIR` once; it must contain the same dataset subdirectories. For example:

```bash
export LEAKAGEBENCH_DATA_DIR="$PWD/data"
```

The three-condition script expects `lmd_full.tar.gz` directly inside this data directory. Use a script's `--help` for explicit input and output paths.

## Detect same-work relations in your own dataset

Place your MIDI files in `data/my-midi/`, including subdirectories if needed, then run:

```bash
python scripts/verify_detector_checkpoint.py checkpoints/same-work-detector-v1.9
python scripts/detect_same_work.py \
  ./data/my-midi \
  ./checkpoints/same-work-detector-v1.9 \
  ./detector_output \
  --workers 8 --backend faiss
```

The output folder contains `components.csv` (file-to-family assignments), `relation_edges.csv` (scored relations and merge decisions), and `summary.json` (counts and parsing failures).

v1.9 uses five views, a one-way Top-50 candidate union, and three seeded classifiers with 57 pair features. The scores are calibrated on a work-disjoint calibration split. `--backend faiss` uses HNSW for retrieval; use `--backend exact` for the exact-search setting used by the six-dataset benchmark.

## Optional: train from scratch

Checkpoint reproduction above does not require these training commands.

### Generation models

The conditions, seeds, and training schedules are specified in [configs/three_condition_models.json](configs/three_condition_models.json). After placing the LMD archive in `data/`, this example prepares the data and trains one MIDI-GPT condition:

```bash
python scripts/fetch_reproduction_assets.py three-condition-source-specs
python scripts/prepare_lmd.py ./data/lmd_full.tar.gz ./prepared_lmd
python scripts/train_external_models.py midigpt clean 202608040 ./prepared_lmd ./runs/midigpt-clean-202608040
```

Use `unrelated_donor` or `same_family_donor` instead of `clean` for the other conditions. Run `python scripts/train_model.py --help` for the built-in generation-model training options. Released checkpoints are available in the [v1.3.0 release](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/tag/v1.3.0); `scripts/verify_model_checkpoints.py` checks a downloaded model bundle and `scripts/evaluate_checkpoint.py` evaluates an individual checkpoint.

### Same-Work Detector v1.9

With the MIDI paths required by `reproduction/detector_training/` available under `data/lmd-clean/`, run:

```bash
python scripts/reproduce_detector_training.py \
  --repo . \
  --spec-dir reproduction/detector_training \
  --midi-root ./data/lmd-clean \
  --output ./detector_retrained \
  --parallel 4 --workers-per-job 8
```

This rebuilds the training/calibration features and trains a new detector. It does not overwrite the released checkpoint.

## Tests and citation

To run the automated code tests:

```bash
bash scripts/reproduce_all.sh
```

Despite its filename, this command runs tests, not all paper experiments. Use the individual commands above to generate the experiment tables.

See [CITATION.cff](CITATION.cff) for citation details. Code is MIT licensed; released weights are CC BY 4.0.
