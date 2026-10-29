<div align="center">

<h1>🎼 LeakageBench-MIDI</h1>
<h3>Work-Level Exposure in Symbolic Music Generation</h3>

<p>Measure evaluation bias · Identify same-work versions · Audit datasets · Isolate works</p>

<a href="https://doi.org/10.5281/zenodo.22023100"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22023100-2563eb?style=flat-square" alt="DOI: Zenodo software archive"></a>
<a href="#reproduce-result-tables"><img src="https://img.shields.io/badge/Experiments-Reproduce-16a34a?style=flat-square" alt="Reproduce experiments"></a>
<a href="https://github.com/kunbai05-nb/LeakageBench-MIDI/releases"><img src="https://img.shields.io/badge/Downloads-Checkpoints-f59e0b?style=flat-square" alt="Download checkpoints and reproduction assets"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/Code-MIT-64748b?style=flat-square" alt="Code license: MIT"></a>

<p>
<a href="#overview">Overview</a> ·
<a href="#choose-an-experiment">Experiments</a> ·
<a href="#setup">Quick Start</a> ·
<a href="#detect-same-work-relations-in-your-own-dataset">Use the Detector</a> ·
<a href="#tests-and-citation">Citation</a>
</p>

</div>

---

<a id="overview"></a>

## 🔎 Overview

Code and checkpoints for studying work-level exposure in symbolic music generation: its effect on evaluation, same-work detection, cross-dataset auditing, and work-level isolation.

Different MIDI performances or arrangements of the same work can cross a file-level train/test split. This repository connects four parts of that evaluation problem:

| Measure | Identify | Audit | Mitigate |
| :--- | :--- | :--- | :--- |
| Compare controlled exposure conditions | Detect same-work MIDI versions | Measure exposure across datasets | Evaluate models after work-level isolation |

**Start here:** [set up the environment](#setup), then choose an experiment below. Each entry includes its inputs, a copyable command, and the generated result paths.

<a id="choose-an-experiment"></a>

## 🧭 Choose an Experiment

Choose the experiment you want to reproduce. After the one-time setup, run its command from the repository root.

| Experiment | Starting point | Original MIDI needed? | Hardware |
| --- | --- | --- | --- |
| [Three-condition experiment](#three-condition-experiment) | Generation-model checkpoints; evaluation windows rebuilt from MIDI | Yes: LMD-full | BF16-capable NVIDIA GPU |
| [Six-dataset detector benchmark](#six-dataset-detector-benchmark) | v1.9 checkpoint and frozen MIDI cohorts | Yes: six datasets | CPU |
| [Fast versus v1.9](#fast-versus-v19) | Candidate features and detector checkpoints | No | CPU |
| [Seven-dataset deployment audit](#seven-dataset-deployment-audit) | Candidate features and auditor checkpoint | No | CPU |
| [Five-condition cleaning study](#cleaning-mitigation) | 60 generation-model checkpoints and frozen test windows | No | BF16-capable NVIDIA GPU |
| [Detector ablation and sensitivity](#detector-ablation-and-sensitivity) | Intermediate features; classifiers retrained | No for the default command | CPU |

**Reproduction starting points:** the feature-based workflows run new inference or training. The table above identifies which stages are rerun and which inputs are frozen.

<a id="setup"></a>

## 🚀 Quick Start

Use Ubuntu/Linux with Python 3.10. You need Git, GitHub CLI (`gh`), `curl`, `tar`, and `sha256sum`. The pinned environment uses PyTorch 2.8.0 and scikit-learn 1.7.2.

<details>
<summary><b>Private repository? Authenticate once before cloning.</b></summary>

**Private repository access:** authorized users must install GitHub CLI (`gh`) and sign in before cloning or downloading release assets:

```bash
gh auth login
gh auth setup-git
```

Skip this authentication step when the repository is public.

</details>

### Install

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

<a id="reproduce-result-tables"></a>

## 📊 Reproduce Result Tables

<a id="three-condition-experiment"></a>

### 1. Three-Condition Experiment

**Prepare:** download `lmd_full.tar.gz` from the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) and save it as `data/lmd_full.tar.gz`. Keep it compressed; the script reads the archive.

**Run:**

```bash
bash scripts/reproduce_three_condition.sh --check-only
bash scripts/reproduce_three_condition.sh
```

The first command checks the environment and entry points; it does not evaluate models or validate the MIDI archive. The second downloads the released model checkpoints, rebuilds the frozen evaluation windows, and evaluates the clean, unrelated-donor, and same-work-donor conditions. It does not retrain the models. Interrupted downloads and completed evaluations are reused.

**Result:**

```text
reproduced/three_condition/tables/three_condition_results.csv
reproduced/three_condition/tables/three_condition_results.md
reproduced/three_condition/tables/three_condition_results.json
```

<a id="six-dataset-detector-benchmark"></a>

### 2. Six-Dataset Detector Benchmark

**Prepare:** place the six datasets in the directories listed under [Original MIDI layout](#original-midi-layout).

**Run:**

```bash
bash scripts/reproduce_six_dataset_table.sh --check-only
bash scripts/reproduce_six_dataset_table.sh
```

The first command verifies the frozen cohorts, MIDI checksums, and detector checkpoint without inference. The second runs v1.9 from MIDI using exact retrieval and generates Precision, Recall, F1, and 95% confidence intervals for two threshold settings:

- **Fixed:** the released threshold is applied unchanged to every dataset.
- **Optimal:** the threshold maximizes F1 using that dataset's evaluation labels. This is an oracle comparison, not the deployment setting.

**Result:**

```text
reproduced/six_dataset/tables/six_dataset_results.csv
reproduced/six_dataset/tables/six_dataset_results.md
reproduced/six_dataset/tables/six_dataset_results.json
```

Per-dataset inference outputs are saved in the raw subdirectory alongside the tables directory.

This command reproduces the v1.9 rows only; it does not rerun CAugBERT/CLaMP. The SHS cohort contains 306 MIDI files and 80 eligible reference pairs. There is no detector retraining or deployment-threshold recalibration on these evaluation cohorts.

<a id="fast-versus-v19"></a>

### 3. Fast versus v1.9

**Prepare:** no MIDI download is needed.

**Run:**

```bash
bash scripts/reproduce_fast_vs_v19.sh
```

The script downloads candidate features, manifests, and both detector checkpoints. It recomputes scores and fixed-threshold metrics on ATEPP, LMD-clean, and ASAP. It does not repeat MIDI feature extraction or candidate retrieval.

**Result:**

```text
reproduced/fast_vs_v19/fast_vs_v19.csv
reproduced/fast_vs_v19/fast_vs_v19.md
reproduced/fast_vs_v19/fast_vs_v19_results.json
```

<a id="seven-dataset-deployment-audit"></a>

### 4. Seven-Dataset Deployment Audit

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

<a id="atepp-mitigation"></a>
<a id="cleaning-mitigation"></a>

### 5. Five-Condition Cleaning Study

**Scope:** ATEPP and LMD-clean × MIDI-GPT and Transformer × five conditions × three paired seeds: **60 final checkpoints**. This is the current cleaning study, replacing the earlier ATEPP-only experiment.

**Prepare:** no MIDI download is needed. Use a BF16-capable CUDA GPU. The first run downloads approximately **4.4 GB** of checkpoints plus a small evaluation bundle; allow **10 GB of free disk space** for the downloaded and extracted files. The download archives are removed after extraction.

**Run all four result tables:**

```bash
bash scripts/reproduce_cleaning_mitigation.sh
```

The script automatically downloads and verifies the frozen checkpoints and evaluation arrays, runs fresh BF16 inference on the original test set and the independent clean test set, and generates the tables. It reuses completed, checksum-verified evaluations after interruption. It does not retrain models or rerun the upstream cleaning decisions.

**Five training conditions:** untreated, matched random removal, MVA-SWI isolation, CAugBERT+CLaMP joint baseline, and label-isolation reference. All models were trained for **3,000 updates and 49,152,000 scored tokens** with paired seeds `202608040`, `202608041`, and `202608042`. Each condition retains its natural training pool; training file and window counts are reported rather than forced to match. Within each dataset, all conditions and both models use identical evaluation arrays and scoring masks.

**Result:** one combined file containing the four dataset/model tables, plus fresh per-file evaluations and a numerical verification report:

```text
reproduced/cleaning_mitigation/tables/cleaning_results.csv
reproduced/cleaning_mitigation/tables/cleaning_results.md
reproduced/cleaning_mitigation/tables/cleaning_results.json
reproduced/cleaning_mitigation/verification.json
reproduced/cleaning_mitigation/raw/
```

Training-pool counts, label-checked exposure and clean-token retention come from the frozen preparation audit; the NLL values are recomputed from checkpoints. The tables report training-file counts, label-checked exposure, clean-token retention, work-macro NLL, seed standard deviations, signed and absolute relative deviations from label isolation, and independent-clean-set NLL. NLL is token-weighted within each file and file-averaged within each work. Three seeds are averaged within each work **before** the relative deviations are calculated; works receive equal weight. LMD-clean uses the frozen reference IDs.

`verification.json` reports whether the newly computed values agree with the frozen evaluations within the documented numerical tolerances. It compares fresh predictions, not saved table files. Scored-token and window counts must match exactly; per-file numerical differences are recorded separately. The final-table tolerances are 0.0001 NLL and 0.01 percentage points. `PASS` means all comparisons meet those tolerances; `DIFFERENT` preserves the newly computed values and lists the differences. Result generation and numerical agreement are reported separately. Add `--strict-match` to treat a numerical difference as an execution error.

<details>
<summary><b>Run one dataset/model, or adjust evaluation memory use</b></summary>

```bash
bash scripts/reproduce_cleaning_mitigation.sh --dataset atepp --model gpt
bash scripts/reproduce_cleaning_mitigation.sh --dataset lmd --model transformer
```

The default evaluation micro-batch is 16, matching the frozen runs. If GPU memory is insufficient, add `--micro-batch 4`; the scoring masks and statistical aggregation stay the same, and any numerical differences remain subject to verification. Add `--force-evaluate` to recompute completed evaluations.

</details>

<a id="detector-ablation-and-sensitivity"></a>

### 6. Detector Ablation and Sensitivity

**Prepare:** no MIDI download is needed for the recommended workflow.

**Run:**

```bash
bash scripts/reproduce_detector_ablation.sh
```

The script downloads intermediate features, trains and calibrates the classifier variants, evaluates all settings, and generates four tables: view count (Table 1), pairwise-verifier ablation (Table 2), Top-K (Table 4), and classifier count (Table 5). The expensive MIDI feature-extraction stage is skipped; training and evaluation are rerun.

**Result:** CSV and Markdown tables, generated_results.json, and verification.json are saved in:

```text
reproduced/detector_ablation/tables/
```

In verification.json, PASS means the reported values match the frozen paper values; DIFFERENT means the newly generated values differ and should be inspected.

To rebuild features from MIDI as well, prepare LMD-clean, ATEPP, and ASAP using the layout below, then run:

```bash
bash scripts/reproduce_detector_ablation.sh --from-midi
```

This alternative writes to `reproduced/detector_ablation_from_midi/`. The frozen MAESTRO view-selection input is downloaded automatically.

<a id="original-midi-layout"></a>

## 📁 Original MIDI Layout

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

<a id="detect-same-work-relations-in-your-own-dataset"></a>

## 🎹 Use the Detector on Your Own Dataset

Place your MIDI files in `data/my-midi/`, including subdirectories if needed, then run:

```bash
python scripts/verify_detector_checkpoint.py checkpoints/same-work-detector-v1.9
python scripts/detect_same_work.py \
  ./data/my-midi \
  ./checkpoints/same-work-detector-v1.9 \
  ./detector_output \
  --workers 8 --backend faiss
```

**Result:**

```text
detector_output/components.csv
detector_output/relation_edges.csv
detector_output/summary.json
```

These files contain file-to-family assignments, scored relations and merge decisions, and counts and parsing failures, respectively.

v1.9 uses five views, a one-way Top-50 candidate union, and three seeded classifiers with 57 pair features. The scores are calibrated on a work-disjoint calibration split. `--backend faiss` uses HNSW for retrieval; use `--backend exact` for the exact-search setting used by the six-dataset benchmark.

<a id="optional-train-from-scratch"></a>

## 🛠️ Optional: Train from Scratch

Checkpoint reproduction above does not require these training commands.

<details>
<summary><b>Generation models: data preparation and training</b></summary>

### Generation models

The conditions, seeds, and training schedules are specified in [configs/three_condition_models.json](configs/three_condition_models.json). After placing the LMD archive in `data/`, this example prepares the data and trains one MIDI-GPT condition:

```bash
python scripts/fetch_reproduction_assets.py three-condition-source-specs
python scripts/prepare_lmd.py ./data/lmd_full.tar.gz ./prepared_lmd
python scripts/train_external_models.py midigpt clean 202608040 ./prepared_lmd ./runs/midigpt-clean-202608040
```

The condition argument is one of:

```text
clean
unrelated_donor
same_family_donor
```

For the built-in generation-model training options, run:

```bash
python scripts/train_model.py --help
```

Released checkpoints are available in the [v1.3.0 release](https://github.com/kunbai05-nb/LeakageBench-MIDI/releases/tag/v1.3.0). The following scripts check a downloaded model bundle and evaluate an individual checkpoint, respectively; use their help messages for the required arguments:

```bash
python scripts/verify_model_checkpoints.py --help
python scripts/evaluate_checkpoint.py --help
```

</details>

<details>
<summary><b>Same-Work Detector v1.9: rebuild features and train</b></summary>

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

</details>

<a id="tests-and-citation"></a>

## 📖 Tests and Citation

To run the automated code tests:

```bash
bash scripts/reproduce_all.sh
```

Despite its filename, this command runs tests, not all paper experiments. Use the individual commands above to generate the experiment tables.

### Citation and License

Use [CITATION.cff](CITATION.cff) for citation details. The software archive is available on [Zenodo](https://doi.org/10.5281/zenodo.22023100).

- **Code:** [MIT License](LICENSE).
- **Released weights:** CC BY 4.0.
- **Third-party materials:** see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

<p align="right"><a href="#overview">↑ Back to overview</a></p>
