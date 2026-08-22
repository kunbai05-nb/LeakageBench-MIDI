# Methods

## Same-work relations

MIDI files are nodes in an undirected graph. Same-work relations form edges and
connected components form families. A split is contaminated when a reference
family has members in both training and evaluation.

The learned detector starts from nine sparse retrieval signals: melody, bass,
rhythm, harmony, motif, interval, duration, onset interval, and transposition-
normalized chroma. Candidate pairs are represented by 47 structural and rank
features. A gradient-boosted classifier scores each pair; the released threshold
is 0.9910136. Predicted edges require support from at least three mutual signals,
and component growth is capped at 20 files.

## Dataset splitting

The random baseline assigns files independently. The mitigation assigns whole
components to train, validation, or test while minimizing split-ratio error. The
80/10/10 split is the main setting. Leakage is measured by cross-split family
count, contaminated test-family rate, and contaminated test-file rate.

The imperfect-inference experiment drops reference edges, injects cross-family
edges, and combines both noise types. Each inferred graph is split without using
the evaluation graph. Results are summarized across fixed seeds with means,
medians, and empirical 95% intervals.

## Controlled training experiment

Each treated family contributes a held-out receiver. The three training
conditions are clean, an unrelated matched donor, and a same-work donor. All
conditions use 38,374 windows and the same batch order within each seed. The
formal matrix contains three seeds for Transformer-L, Conditional VAE, and
Latent Diffusion. The diffusion model uses a seed-matched neutral encoder trained
on the 37,110 windows shared by all conditions.

Models are trained for 20,000 steps with batch size 32, micro-batch size 8,
AdamW, 1,000 warmup steps, cosine decay, and gradient clipping at 1.0. The
primary endpoint is receiver continuation NLL for sequence models and fixed-
noise latent denoising MSE for diffusion.

## Statistical analysis

Families receive equal weight. Condition differences are paired within family
and averaged across the three seeds. Confidence intervals use 10,000 family-
level bootstrap samples. Families, rather than windows or tokens, are the
analysis units. Multiple musical-property comparisons use Holm correction.

Exact settings are available in [`configs/protocol_v2.json`](../configs/protocol_v2.json)
and [`configs/training.json`](../configs/training.json).
