# Detector benchmark reproduction

The three compressed manifests in this directory freeze the file order, file checksums, work groups, and recording groups used by the benchmark:

- `shs.csv.gz`
- `asap.csv.gz`
- `lmd-clean.csv.gz`

Run one dataset with `scripts/reproduce_detector_benchmark.py`, or run all three with:

```bash
bash scripts/reproduce_detector_benchmarks.sh \
  /path/to/shs \
  /path/to/asap \
  /path/to/lmd_clean \
  ./same-work-detector-v1.3.0 \
  ./benchmark
```

Each output directory contains `results.json` and `predicted_pairs.csv.gz`. The script verifies every input MIDI checksum before detection. SHS and ASAP use pair-micro precision/recall/F1; LMD-clean uses query-macro precision/recall/F1, matching the evaluation protocol.
