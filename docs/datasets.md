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

## PDMX

The external experiment uses [PDMX record 15571083](https://zenodo.org/records/15571083).
The expected `mid.tar.gz` MD5 is `d920a21b2fcd99a56d9c381b39debbb2`.
