<div align="center">

<h1>🎼 LeakageBench-MIDI</h1>
<h3>Work-Level Exposure in Symbolic Music Generation</h3>

<p>Code and checkpoints to reproduce the experiments.</p>

<a href="https://doi.org/10.5281/zenodo.22023100"><img src="https://img.shields.io/badge/DOI-Zenodo-2563eb?style=flat-square" alt="Zenodo archive"></a>
<a href="https://github.com/kunbai05-nb/LeakageBench-MIDI/releases"><img src="https://img.shields.io/badge/Download-Checkpoints-f59e0b?style=flat-square" alt="Download checkpoints"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/Code-MIT-64748b?style=flat-square" alt="MIT license"></a>

<p><a href="#setup">Setup</a> · <a href="#experiments">Experiments</a> · <a href="#citation">Citation</a></p>

</div>

<a id="setup"></a>

## 🚀 Setup

Use Ubuntu/Linux and Python 3.10. Install Git, GitHub CLI (`gh`), `curl`, `tar`, and `sha256sum` first. The GPU experiments require an NVIDIA GPU with BF16 support.

<details>
<summary><b>Private repository: sign in before cloning</b></summary>

```bash
gh auth login
gh auth setup-git
```

</details>

```bash
git clone https://github.com/kunbai05-nb/LeakageBench-MIDI.git
cd LeakageBench-MIDI
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e '.[scale]'
```

Run the commands below from this folder, with the environment activated. Checkpoints and saved inputs download automatically. Allow space for both downloads and extracted files.

<a id="experiments"></a>

## 📊 Reproduce the Experiments

| Experiment | Hardware | Original MIDI needed? |
| --- | --- | --- |
| [Four-condition exposure experiment](#four-condition-experiment) | GPU | No |
| [Five-condition cleaning experiment](#cleaning-mitigation) | GPU | No |
| [Six-dataset detector comparison](#six-dataset-detector-benchmark) | CPU | Yes |
| [Seven-dataset exposure audit](#seven-dataset-deployment-audit) | CPU | No |
| [Fast versus v1.9](#fast-versus-v19) | CPU | No |
| [Detector ablation](#detector-ablation-and-sensitivity) | CPU | No |

<a id="four-condition-experiment"></a>

### 1. Four-Condition Exposure Experiment

Compare six models trained with clean data, new unrelated versions, repeated unrelated versions, or another version of a test work. Downloads about 2.6 GB; allow 8 GB of free space.

```bash
bash scripts/reproduce_four_condition.sh
```

Tables: `reproduced/four_condition/tables/four_condition_results.csv` (also Markdown and JSON).

<a id="cleaning-mitigation"></a>

### 2. Five-Condition Cleaning Experiment

Compare untreated data, random removal, MVA-SWI isolation, CAugBERT+CLaMP cleaning, and label isolation on ATEPP and LMD-clean, using MIDI-GPT and Transformer. Downloads about 4.4 GB; allow 10 GB of free space.

```bash
bash scripts/reproduce_cleaning_mitigation.sh
```

Tables: `reproduced/cleaning_mitigation/tables/cleaning_results.csv` (also Markdown and JSON).

Both GPU commands evaluate the released checkpoints again. They reuse completed evaluations after an interruption. Each output folder includes `verification.json`: `PASS` means values match within the stated tolerances; `DIFFERENT` lists the numerical differences and keeps the newly calculated results.

<a id="six-dataset-detector-benchmark"></a>

### 3. Six-Dataset Detector Comparison

First place the MIDI datasets in the folders below, preserving the relative paths in the [file lists](reproduction/detector_benchmark/). Existing datasets can be linked instead of copied.

<details>
<summary><b>Required MIDI folders and example file paths</b></summary>

| Dataset | Folder | Example path inside the folder |
| --- | --- | --- |
| SHS | `data/shs` | `0/0207f2cc938c3691b3af6d8e359bedfd.mid` |
| ASAP | `data/asap` | `Bach/Fugue/bwv_846/Shi05M.mid` |
| ATEPP | `data/atepp` | `Sergei_Rachmaninoff/13_Preludes,_Op._32/No._10_in_B_Minor:_Lento/00046.mid` |
| LMD-clean | `data/lmd-clean` | `clean_midi/2 Brothers on the 4th Floor/Dreams.1.mid` |
| Vienna4x22 | `data/vienna4x22` | `Chopin_op10_no3_p01.mid` |
| PianoVAM | `data/pianovam` | `MIDI/2024-02-14_19-10-09.mid` |

</details>

```bash
bash scripts/reproduce_six_dataset_table.sh --check-only
bash scripts/reproduce_six_dataset_table.sh
```

Tables: `reproduced/six_dataset/tables/six_dataset_results.csv` (also Markdown and JSON).

This runs v1.9 from MIDI and reports fixed-threshold and best-F1-threshold results. It does not rerun the CAugBERT/CLaMP baseline rows.

<a id="seven-dataset-deployment-audit"></a>

### 4. Seven-Dataset Exposure Audit

Recompute exposure on POP909, ASAP, PDMX, GigaMIDI, MAESTRO, LMD, and Aria-MIDI. This starts from saved candidate-pair features and the Precision-First Same-Work Auditor checkpoint; MIDI parsing and candidate search are skipped.

```bash
bash scripts/reproduce_multidataset_deployment.sh
```

Tables: `reproduced/multidataset_deployment/multidataset_deployment.csv` (also Markdown). Per-dataset JSON results are in the same folder.

<a id="fast-versus-v19"></a>

### 5. Fast versus v1.9

Compare the two detectors on ATEPP, LMD-clean, and ASAP by scoring the saved candidate-pair features again.

```bash
bash scripts/reproduce_fast_vs_v19.sh
```

Tables: `reproduced/fast_vs_v19/fast_vs_v19.csv` (also Markdown and JSON).

<a id="detector-ablation-and-sensitivity"></a>

### 6. Detector Ablation

Rebuild the view-count, feature-ablation, Top-K, and classifier-count tables. The command downloads saved features, then retrains and evaluates the classifier variants.

```bash
bash scripts/reproduce_detector_ablation.sh
```

Tables: `reproduced/detector_ablation/tables/`.

For extra options, add `--help` to an experiment command.

<a id="citation"></a>

## 📖 Citation and License

Citation details: [CITATION.cff](CITATION.cff). Code: [MIT](LICENSE). Released weights: CC BY 4.0. See [third-party notices](THIRD_PARTY_NOTICES.md) for other materials.
