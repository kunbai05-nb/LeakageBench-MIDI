# Reproducibility and determinism

Public manifest operations use stable SHA-256 ordering and explicit seeds. Formal paired runs shared initialization, Python/NumPy/Torch/CUDA RNG state, batch schedules, and optimizer schedules within each seed. Fresh runs restored frozen RNG before the first stochastic forward; resumes restored checkpoint runtime RNG. A pre-result implementation correction fixed an earlier fresh-run RNG restoration bug, and affected partial evidence was invalidated. Formal confirmatory results came only from corrected runs.

Statistical outputs use frozen bootstrap seeds and family-cluster sampling. Release tables are registry-derived and byte-identical across repeated runs. Every formal registry entry includes its source artifact SHA-256; release files receive a separate integrity manifest.

