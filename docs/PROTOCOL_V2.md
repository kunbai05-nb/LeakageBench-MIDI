# Frozen experiment protocol (public v2 disclosure)

This document states what was actually frozen and run. The matching
machine-readable record is [`configs/protocol_v2.json`](../configs/protocol_v2.json).
It does not authorize a new experiment and does not change any formal result.

## Data and family definition

The official LMD v0.1 page reports 176,581 MD5-distinct files. The downstream
archive inventory frozen for this project contains 178,561 archive-header
identities. These are different provenance scopes; the retained records do not
reconstruct a one-to-one transformation between them. All reported LMD census
results are therefore conditional on the frozen 178,561-identity universe.

Same-work edges use the conservative CAugBERT 0.99 / CLaMP 0.99 union. The
undirected connected components are the operational families, and uncovered
identities are treated as singletons. This is a known-family reference relation,
not exhaustive ground truth. See [data_provenance.md](data_provenance.md) and
[family_reference_validation.md](family_reference_validation.md).

## Controlled exposure experiment

The main Phase-2 design contains 100 receiver families and three conditions:

- `clean`: neither the designated same-family donor nor its matched unrelated
  donor is added;
- `unrelated_donor`: the matched cross-family donor is added; and
- `same_family_donor`: the designated same-family donor is added.

Receiver families stay outside training. Added exposure is offset by
deterministic family-disjoint removal from the base pool so the training budget
is matched. The formal seeds are 202608040, 202608041, and 202608042. The
primary contrast is same-family donor minus unrelated donor; clean is a
context/control condition rather than the primary reference.

## Tokenizer and Transformer-L

The frozen tokenizer uses a 237-event REMI-style vocabulary and context length
1024. Transformer-L has 10 pre-layer-normalized blocks, width 384, six heads,
FFN width 1536, dropout 0.1, tied embeddings, and 17,821,056 parameters.

The public Transformer reproduces the historical RoPE tensor layout found in
the archived training source. That layout is intentionally preserved for
checkpoint compatibility; replacing it with a conventional RoPE
implementation would define a different model.

Each run used 20,000 steps, micro-batch 8, gradient accumulation 4, effective
batch 32, AdamW at peak learning rate 2e-4, betas 0.9/0.95, weight decay 0.1,
gradient clipping 1.0, 1,000 warm-up steps, then cosine decay. The selected
checkpoint was the preregistered final step 20,000. Receiver results were not
used for checkpoint selection.

## Architecture and paradigm checks

The capacity analysis used three tested Transformer sizes under the same
20,000-step budget: S (10,942,512 parameters), M (13,658,064), and L
(17,821,056). Its slope is descriptive within this Transformer family; TCN,
CVAE, and diffusion are not mixed into that regression.

The TCN check used nine causal dilated blocks, 384 channels, kernel size 5,
dilations 1 through 256, FFN width 1536, dropout 0.1, and 10,753,536 parameters.
It used three seeds, clean/family-leak conditions, and the same 20,000-step
budget.

The conditional VAE and latent diffusion checks use the same 100 families,
three conditions, and three formal seeds. CVAE's endpoint is prompt-prior-mean
teacher-forced continuation NLL. Diffusion's endpoint is paired fixed-noise
receiver-latent denoising MSE at timesteps 49, 199, 499, and 799, with eight
noise replicates per timestep. One neutral encoder per seed is shared across
all three diffusion conditions. Raw objective magnitudes are not compared
between paradigms.

## Generation protocol

Generation was run for Transformer-L and conditional VAE, with 100 families,
three training seeds, five generation seeds, and three conditions: 9,000
samples total. The prompt is the end of the lexicographically first frozen
receiver window, capped at 896 tokens. Each output has 128 generated tokens,
temperature 1.0, top-p 0.95, and no EOS stopping. Conditions are paired on
receiver, training seed, generation seed, prompt, output length, and decoding
policy.

## Statistics

The primary sampling unit is the family, not a token, window, or generated
sample. The three paired training seeds are averaged within family before
resampling. Unless a named legacy analysis says otherwise, confidence
intervals use 10,000 family-bootstrap draws and two-sided plus-one tail
probabilities. Phase-2 uses bootstrap seed 0. Cross-paradigm analysis uses seed
20260818 for the confidence interval and 20260819 for paired sign
randomization. Generation uses seed 20260818. Holm correction is applied to
declared metric families; the cross-paradigm Holm calculation is a post-hoc
sensitivity analysis, not the primary gate.

## What the public release can reproduce

The Git tree contains anonymous scalar analysis units, not MIDI, token-bearing
manifests, or training states. The field audit checks all 232 numerical result
fields: 193 are recalculated from public rows and 39 are verified against
released frozen nonidentifying summaries because the exact preregistered
display-chain inputs are not public. All 60 separately distributed final
checkpoints support inference inspection, but they do not contain optimizer,
scheduler, or RNG state and therefore are not a full training-resume archive.
