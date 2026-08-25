# Datasets

## Lakh MIDI Dataset

Download `lmd_full.tar.gz` from the [official LMD page](https://colinraffel.com/projects/lmd/).
The expected SHA-256 is:

```text
6fcfe2ac49ca08f3f214cec86ab138d4fc4dabcd7f27f491a838dae6db45a12b
```

Prepare the formal streams with:

```bash
python scripts/prepare_lmd.py /path/to/lmd_full.tar.gz ./prepared_lmd
```

## ASAP

The detector evaluation uses the 1,067 performance MIDI files in
[ASAP](https://github.com/fosfrancesco/asap-dataset), commit
`afc815c75c42e83a79c03feb6da8a35e77d4c6b8`.

## MAESTRO

Download the MIDI-only
[MAESTRO v3.0.0 archive](https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip).

## POP909

Download [POP909](https://github.com/music-x-lab/POP909-Dataset) and include the
version MIDI files when reproducing the full detector experiment.

## PDMX

The external experiment uses [PDMX record 15571083](https://zenodo.org/records/15571083).
The expected `mid.tar.gz` MD5 is `d920a21b2fcd99a56d9c381b39debbb2`.

## GigaMIDI

Download GigaMIDI v2 from
[Metacreation/GigaMIDI](https://huggingface.co/datasets/Metacreation/GigaMIDI).

## Aria-MIDI

Download the
[Aria-MIDI Full archive](https://huggingface.co/datasets/loubb/aria-midi/blob/main/aria-midi-v1-ext.tar.gz).

## Detector manifests

After extraction, build the matching manifests with:

```bash
python scripts/build_cross_dataset_manifest.py asap ASAP_ROOT asap.jsonl \
  --metadata-csv ASAP_ROOT/metadata.csv
python scripts/build_cross_dataset_manifest.py maestro MAESTRO_ROOT maestro.jsonl \
  --metadata-csv MAESTRO_ROOT/maestro-v3.0.0.csv \
  --asap-metadata-csv ASAP_ROOT/metadata.csv
python scripts/build_cross_dataset_manifest.py gigamidi GIGAMIDI_ROOT gigamidi.jsonl \
  --metadata-csv GIGAMIDI_METADATA.csv
python scripts/build_cross_dataset_manifest.py aria ARIA_ROOT/data aria.jsonl \
  --metadata-json ARIA_ROOT/metadata.json
```

Pass each manifest and extraction root to `scripts/run_cross_dataset_detector.py`.
