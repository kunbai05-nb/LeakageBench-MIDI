# Dataset acquisition

## LMD

LeakageBench-MIDI does not distribute LMD MIDI. Obtain it legally from its official source and configure `LMD_ROOT=/path/to/lmd`. Public manifests should contain source-relative identifiers or hashes, never MIDI bytes or token sequences capable of reconstructing restricted music. The permanently closed clean-test is not required for public tooling or table reproduction.

The project adopts duplicate/same-song identification and filtering relations from Eunjin Choi, Hyerin Kim, Jiwoo Ryu, Juhan Nam, and Dasaem Jeong, “On the De-duplication of the Lakh MIDI Dataset,” ISMIR 2025 ([arXiv:2509.16662](https://arxiv.org/abs/2509.16662)). Those relations are upstream precedent, not a detector contribution of LeakageBench-MIDI. The release-level identity-count distinction and frozen resource revisions are documented in [data_provenance.md](data_provenance.md).

## PDMX

The frozen study used PDMX record 15571083, DOI `10.5281/zenodo.15571083` (concept DOI `10.5281/zenodo.13763755`), published 2025-06-01 under CC BY 4.0. The MIDI archive `mid.tar.gz` expected MD5 is `d920a21b2fcd99a56d9c381b39debbb2`; the metadata CSV expected SHA-256 is `fc2187e7e09185f4b28be57b6478a96a1243a037f8fd13624f17de2d8bfd44bd`. Obtain files from the [official Zenodo record](https://zenodo.org/records/15571083) and prefer `no_license_conflict` plus `all_valid` rows. This repository does not redistribute the archive.

Dataset licenses are separate from the MIT software license. Users are responsible for compliance.
