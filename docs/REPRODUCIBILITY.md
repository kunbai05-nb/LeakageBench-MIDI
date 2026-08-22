# Reproduction

## Public statistics

```bash
bash scripts/reproduce_all.sh ./reproduced
```

This rebuilds the statistical summaries, verifies result artifacts, and runs the
tests. Detector statistics are recomputed from 355,535 candidate-pair rows with
the frozen 10,000-sample composition bootstrap.

## LMD preparation

```bash
python scripts/prepare_lmd.py /path/to/lmd_full.tar.gz ./prepared_lmd
```

Expected output:

```text
prepared_lmd/
  midi/
  streams/clean.tokens
  streams/clean_index.npz
  streams/unrelated_donor.tokens
  streams/unrelated_donor_index.npz
  streams/same_family_donor.tokens
  streams/same_family_donor_index.npz
  evaluation_windows.jsonl
  PREPARATION_SUMMARY.json
```

`PREPARATION_SUMMARY.json` records the row count, token count, and identity hash
of each stream.

## Reference graph and split robustness

```bash
python scripts/prepare_reference_graph.py /path/to/lmd_full.tar.gz ./reference_graph
python scripts/simulate_imperfect_inference.py \
  ./reference_graph ./imperfect_inference --seeds 100
```

The first command rebuilds the 178,561-file reference graph. The second runs
the false-negative, false-positive, and combined-noise split simulations.

## Checkpoint evaluation

```bash
python scripts/evaluate_checkpoint.py CHECKPOINT PREPARED_DIR OUTPUT_DIR
```

For latent diffusion, add the seed-matched neutral encoder:

```bash
python scripts/evaluate_checkpoint.py DIFFUSION_CHECKPOINT PREPARED_DIR OUTPUT_DIR \
  --encoder NEUTRAL_ENCODER_CHECKPOINT --device cuda
```

## Training

```bash
python scripts/train_model.py MODEL CONDITION SEED PREPARED_DIR OUTPUT_DIR
```

Models: `transformer`, `conditional_vae`, `neutral_encoder`, and
`latent_diffusion`. Conditions: `clean`, `unrelated_donor`, and
`same_family_donor`. Add `--resume` to continue from `last.pt`.

## Detector

```bash
python scripts/detect_same_work.py MIDI_DIR OUTPUT_DIR --workers 8
python scripts/reproduce_detector_statistics.py --verify
```
