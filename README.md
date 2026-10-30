<div align="center">

<h1>🎼 LeakageBench-MIDI</h1>
<h3>Work-Level Exposure in Symbolic Music Generation</h3>

<p>Measure evaluation bias · Identify same-work versions · Audit datasets · Isolate works</p>

<p>
<a href="https://doi.org/10.5281/zenodo.22023100">Zenodo archive</a> ·
<a href="https://github.com/kunbai05-nb/LeakageBench-MIDI/releases">Download checkpoints</a> ·
<a href="LICENSE">MIT License</a>
</p>

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

## Overview

LeakageBench-MIDI studies how different MIDI versions of the same work affect generation evaluation. It provides same-work detectors, dataset audits, and experiments on work-level isolation.

**To reproduce a result table:** complete Step 1 once, then copy the command for your experiment in Step 2. Downloads use default locations. All result tables are saved under the following directory:

```text
reproduced/
```

<a id="setup"></a>

## Step 1 · Install

Use Ubuntu 22.04 (or another Linux system with Python 3.10). Install the required tools with the following commands:

```bash
sudo apt-get update
sudo apt-get install -y git gh curl tar python3.10 python3.10-venv
```

For access to this private repository, sign in with an authorized GitHub account:

```bash
gh auth login
gh auth setup-git
```

Clone and install:

```bash
git clone https://github.com/kunbai05-nb/LeakageBench-MIDI.git
cd LeakageBench-MIDI
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e '.[scale]'
```

Keep the terminal in the cloned repository with the virtual environment activated. Run all experiment commands there. Scripts download and check their required release bundles automatically. Keep space for both downloaded archives and extracted files.

<a id="choose-an-experiment"></a>
<a id="reproduce-result-tables"></a>

## Step 2 · Choose an Experiment

| Experiment | Starts from | Hardware | Original MIDI |
| --- | --- | --- | --- |
| [1. Controlled exposure](#four-condition-experiment) | 72 model checkpoints + evaluation windows | BF16-capable NVIDIA GPU | Not needed |
| [2. Six-dataset benchmark](#six-dataset-detector-benchmark) | v1.9 detector + frozen MIDI cohorts | CPU | Required |
| [3. Fast versus v1.9](#fast-versus-v19) | Candidate features + detector checkpoints | CPU | Not needed |
| [4. Seven-dataset audit](#seven-dataset-deployment-audit) | Candidate features + auditor checkpoint | CPU | Not needed |
| [5. Cleaning study](#cleaning-mitigation) | 60 model checkpoints + evaluation windows | BF16-capable NVIDIA GPU | Not needed |
| [6. Detector ablation](#detector-ablation-and-sensitivity) | Intermediate features; classifiers retrained | CPU | Not needed |

<a id="four-condition-experiment"></a>

### 1. Four-Condition Controlled Exposure

Six models × four conditions × three seeds. Conditions: clean, new unrelated donor, repeated unrelated content, and same-work donor. **Allow 8 GB free space**; the initial download is about 2.6 GB.

```bash
bash scripts/reproduce_four_condition.sh
```

**Result files:** open the Markdown table below. CSV and JSON versions are in the same table directory; fresh evaluations are in the raw directory.

```text
reproduced/four_condition/tables/four_condition_results.md
reproduced/four_condition/raw/
```

<details>
<summary>Protocol and optional commands</summary>

Models: Transformer-S/M/L, MIDI-GPT, LSTM, and TCN-384. The 72 final checkpoints use 20,000 steps and paired seeds 202608040/041/042. Evaluation uses the same 100 intervention groups and 1,299 receiver windows. The fourth condition replaces the same 1,264 positions in the 38,374-window training stream.

NLL is token-weighted within each receiver and averaged equally across groups. Tables include seed SDs, PPL, relative reductions, and same-work versus repeated-unrelated differences with paired group-bootstrap intervals. Seeds are averaged within groups before relative differences are computed; intervals resample groups conditional on the three fitted seeds.

Run one model, or additionally evaluate the 300 frozen control groups:

```bash
bash scripts/reproduce_four_condition.sh --model midigpt
bash scripts/reproduce_four_condition.sh --include-control
```

Completed, hash-verified evaluations are reused. --force-evaluate recomputes them; --strict-match requires numerical agreement with the frozen evaluations. Differences are listed in reproduced/four_condition/verification.json.

The repeated-unrelated LSTM checkpoint for seed 202608041 uses the recorded FP32 training repair and the same BF16 evaluation. Checkpoint exports retain their released dtypes.

The old scripts/reproduce_three_condition.sh command forwards to the four-condition workflow. The earlier MIDI-based workflow, including the VAE and diffusion three-condition supplement, remains available after placing data/lmd_full.tar.gz:

```bash
bash scripts/reproduce_three_condition_from_midi.sh
```

</details>

<a id="six-dataset-detector-benchmark"></a>

### 2. Six-Dataset Detector Benchmark

**First:** place the six MIDI datasets in the [default directories below](#original-midi-layout). Then check the inputs and run:

```bash
bash scripts/reproduce_six_dataset_table.sh --check-only
bash scripts/reproduce_six_dataset_table.sh
```

**Result table:** CSV and JSON versions are saved beside this file.

```text
reproduced/six_dataset/tables/six_dataset_results.md
```

This runs the v1.9 detector from MIDI with exact retrieval. It reports Precision, Recall, F1, and 95% confidence intervals at the fixed deployment threshold and the dataset-optimal F1 threshold. The latter uses evaluation labels. CAugBERT/CLaMP rows are not rerun. The frozen SHS cohort has 306 files and 80 eligible reference pairs.

<a id="fast-versus-v19"></a>

### 3. Fast versus v1.9

Compare fixed-threshold detection on ATEPP, LMD-clean, and ASAP. Scores are recomputed from candidate features and both checkpoints.

```bash
bash scripts/reproduce_fast_vs_v19.sh
```

**Result files:** open the Markdown table or use the CSV and JSON outputs.

```text
reproduced/fast_vs_v19/fast_vs_v19.md
reproduced/fast_vs_v19/fast_vs_v19.csv
reproduced/fast_vs_v19/fast_vs_v19_results.json
```

<a id="seven-dataset-deployment-audit"></a>

### 4. Seven-Dataset Deployment Audit

Audit POP909, ASAP, PDMX, GigaMIDI, MAESTRO, LMD, and Aria-MIDI with the **Precision-First Same-Work Auditor** checkpoint.

```bash
bash scripts/reproduce_multidataset_deployment.sh
```

**Result files:** open the Markdown table below. The same directory contains the CSV and per-dataset results.

```text
reproduced/multidataset_deployment/multidataset_deployment.md
reproduced/multidataset_deployment/multidataset_verification.json
```

The command starts from frozen candidate-pair features after retrieval and cheap-gate filtering. It recomputes scores, selects relations, rebuilds protected families, and runs 200 train/validation/test splits. Family/file exposure uses all test families/files as denominators. The 95% EI is the 2.5%–97.5% empirical quantile range across splits; available official-split results are also reported.

The downloaded bundle includes the full-MIDI source, manifests, and instructions in source/, midi_hashes/, and FULL_MIDI.md.

<a id="cleaning-mitigation"></a>

### 5. Five-Condition Cleaning Study

ATEPP and LMD-clean × MIDI-GPT and Transformer × five conditions × three seeds. Conditions: untreated, matched random removal, MVA-SWI isolation, CAugBERT+CLaMP joint baseline, and label isolation. **Allow 10 GB free space**; checkpoints download about 4.4 GB.

```bash
bash scripts/reproduce_cleaning_mitigation.sh
```

**Result files:** all four tables are in the Markdown file below. CSV and JSON versions are in the same table directory; fresh per-file evaluations are in the raw directory.

```text
reproduced/cleaning_mitigation/tables/cleaning_results.md
reproduced/cleaning_mitigation/raw/
```

<details>
<summary>Protocol, numerical verification, and smaller runs</summary>

The command evaluates all 60 final checkpoints on the original and independent clean test sets. Each was trained for 3,000 updates and 49,152,000 scored tokens with paired seeds 202608040/041/042. Conditions retain their natural training pools. Evaluation arrays and masks are identical within each dataset.

NLL is token-weighted within files, file-averaged within works, and work-averaged equally. Seeds are averaged within each work before signed and absolute relative deviations from label isolation are computed. LMD-clean uses frozen reference IDs. Training-pool counts, exposure, and clean-token retention come from the frozen preparation audit.

reproduced/cleaning_mitigation/verification.json compares the fresh results with frozen evaluations. Token/window counts must match exactly; final-table tolerances are 0.0001 NLL and 0.01 percentage points. PASS indicates agreement; DIFFERENT lists discrepancies and preserves fresh values. Add --strict-match to return an error on numerical differences.

Run one dataset/model:

```bash
bash scripts/reproduce_cleaning_mitigation.sh --dataset atepp --model gpt
bash scripts/reproduce_cleaning_mitigation.sh --dataset lmd --model transformer
```

The default micro-batch is 16. If GPU memory is insufficient, add --micro-batch 4. Completed, hash-verified evaluations are reused; add --force-evaluate to recompute them.

</details>

<a id="detector-ablation-and-sensitivity"></a>

### 6. Detector Ablation and Sensitivity

Download intermediate features, retrain/calibrate classifiers, and generate Tables 1, 2, 4, and 5: views, verifier features, Top-K, and classifier count.

```bash
bash scripts/reproduce_detector_ablation.sh
```

**Result directory:** contains the tables, generated results, and verification report. PASS means agreement with frozen values; DIFFERENT lists fresh differences.

```text
reproduced/detector_ablation/tables/
```

<details>
<summary>Rebuild the ablation features from MIDI</summary>

Prepare LMD-clean, ATEPP, and ASAP using the directories below, then run:

```bash
bash scripts/reproduce_detector_ablation.sh --from-midi
```

Outputs go to reproduced/detector_ablation_from_midi/. The frozen MAESTRO view-selection input is downloaded automatically.

</details>

<a id="original-midi-layout"></a>

## MIDI Directories · For Experiment 2 and Full-MIDI Workflows

Place datasets under data/, preserving the relative paths in the frozen manifests under reproduction/detector_benchmark/. Existing datasets can be linked here instead of copied.

| Dataset | Default directory | Example relative path |
| --- | --- | --- |
| SHS (LMD-full subset) | `data/shs` | `0/0207f2cc938c3691b3af6d8e359bedfd.mid` |
| ASAP | `data/asap` | `Bach/Fugue/bwv_846/Shi05M.mid` |
| ATEPP | `data/atepp` | `Sergei_Rachmaninoff/13_Preludes,_Op._32/No._10_in_B_Minor:_Lento/00046.mid` |
| LMD-clean | `data/lmd-clean` | `clean_midi/2 Brothers on the 4th Floor/Dreams.1.mid` |
| Vienna4x22 | `data/vienna4x22` | `Chopin_op10_no3_p01.mid` |
| PianoVAM | `data/pianovam` | `MIDI/2024-02-14_19-10-09.mid` |

The default is data/ inside the repository. For example, if your datasets are already under /mnt/d/music-data/, set that parent directory once:

```bash
export LEAKAGEBENCH_DATA_DIR="/mnt/d/music-data"
```

<a id="detect-same-work-relations-in-your-own-dataset"></a>

## Use the Detector on Your Own MIDI

Place MIDI files in data/my-midi/, then run:

```bash
python scripts/verify_detector_checkpoint.py checkpoints/same-work-detector-v1.9
python scripts/detect_same_work.py \
  ./data/my-midi \
  ./checkpoints/same-work-detector-v1.9 \
  ./detector_output \
  --workers 8 --backend faiss
```

**Result files**, in order: family assignments; scored relations and merge decisions; counts and parsing failures.

```text
detector_output/components.csv
detector_output/relation_edges.csv
detector_output/summary.json
```

v1.9 uses five views, a one-way Top-50 candidate union, 57 pair features, and three seeded classifiers with work-disjoint calibration. faiss uses HNSW; exact selects the benchmark retrieval setting.

<a id="optional-train-from-scratch"></a>

<details>
<summary><b>Optional: train models and the detector from scratch</b></summary>

Checkpoint reproduction above does not require these training commands.

<details>
<summary><b>Generation models: data preparation and training</b></summary>

### Generation models

The original three-condition model settings, seeds, and training schedules are specified in [configs/three_condition_models.json](configs/three_condition_models.json). These training commands cover the original conditions; the fourth-condition checkpoint workflow uses the frozen construction metadata included in its evaluation bundle. After placing the LMD archive in data/, this example prepares the data and trains one MIDI-GPT condition:

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

With the MIDI paths required by reproduction/detector_training/ available under data/lmd-clean/, run:

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


</details>

<a id="tests-and-citation"></a>

## Tests and Citation

Run code tests:

```bash
bash scripts/reproduce_all.sh
```

Experiment tables are generated by the individual commands above.

See [CITATION.cff](CITATION.cff) and the [Zenodo archive](https://doi.org/10.5281/zenodo.22023100). Code: [MIT](LICENSE). Released weights: CC BY 4.0. Third-party materials: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
