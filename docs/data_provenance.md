# Data provenance

The current public software archive is registered under the
version-independent project identifier
[10.5281/zenodo.22023100](https://doi.org/10.5281/zenodo.22023100), which
resolves to the latest retained Zenodo record.

## LMD v0.1 official count

The [official Lakh MIDI Dataset page](https://colinraffel.com/projects/lmd/) describes LMD v0.1 as containing **176,581 MD5-distinct MIDI files**. LeakageBench-MIDI treats that number as the original dataset documentation's count. It does not relabel that count as the size of the downstream analysis universe.

## Frozen LeakageBench-MIDI identity universe

The frozen full-LMD census uses **178,561 identities**. The earliest retained local record of this count is the 2026-08-09 census inventory. The formal census records:

- archive SHA-256 `6fcfe2ac49ca08f3f214cec86ab138d4fc4dabcd7f27f491a838dae6db45a12b`;
- 178,561 distinct archive-header MIDI identities;
- 58,931 identities covered by multi-member components; and
- 119,630 archive identities outside those multi-member components.

The corresponding frozen Monte Carlo result artifact has SHA-256 `9607daf226a6e51f281f6c52eb06c5a7c701309ca2f2e51b6febc61cfedfc84b`. This release does not redistribute the archive, its MIDI bytes, or a private identity manifest.

## Count ledger

| Count | Meaning | Role in the paper |
|---:|---|---|
| 176,581 | Official LMD v0.1 MD5-distinct count | External dataset-documentation reference; not asserted to be the direct input row count of the frozen census |
| 178,561 | Distinct archive-header identities in the retained downstream inventory | Denominator for the frozen census and split simulations |
| 58,931 | Downstream identities in adopted multi-member components | Known multi-member family coverage |
| 119,630 | Downstream identities outside adopted multi-member components | Treated as singleton components in the frozen census |
| 1,980 | Arithmetic difference between 178,561 and 176,581 | Unresolved provenance difference; not interpreted as added, duplicated, or recovered files |

The two top-line counts are parallel provenance facts, not consecutive rows in
a demonstrated processing flow. No file-level bridge table is retained or
published, so presenting 176,581 → 178,561 as a known transformation would be
unsupported.

## Adopted family resource

Family edges come from the conservative `CAugBERT_0.99_with_CLaMP_0.99.json` filtering resource associated with:

> Eunjin Choi, Hyerin Kim, Jiwoo Ryu, Juhan Nam, and Dasaem Jeong. “On the De-duplication of the Lakh MIDI Dataset.” ISMIR 2025. [arXiv:2509.16662](https://arxiv.org/abs/2509.16662).

The frozen upstream repository revision is `c42fa1c3f881261b92c0cf0d58dba5b0e5955d26`. The locally recorded SHA-256 of the conservative filtering JSON is `8b0989f76ea4dcf13f9333c027110949bae4ff0e1b23202224bb0d48fa6a8751`. Choi et al. (2025) is the duplicate/same-song identification and filtering precedent. LeakageBench-MIDI contributes the downstream controlled consequence, robustness, model-susceptibility, and mitigation analyses.

The upstream GitHub filtering-list revision had no observed license file.
Consequently, this public release does not redistribute that filtering JSON or
derived pair-level/private manifests. It publishes aggregate family-size
statistics and release-specific anonymous analysis units sufficient to
recompute paper statistics; those rows contain no original file/family hash,
MIDI content, or token sequence. The separately hosted LMD de-duplication
supplements were recorded at revision
`7565f0d1d814f2ce3915439a302a767d9109aa2b` with a CC BY 4.0 dataset card.

## What is known

- The official LMD v0.1 documentation reports 176,581 MD5-distinct MIDI files.
- The frozen LeakageBench-MIDI census inventory records 178,561 identities for the archive scope actually used downstream.
- The Choi et al. conservative filtering resource supplies the adopted multi-member family edges; archive identities outside that graph are treated as singleton components for the frozen census.
- Release statistics and registries consistently use the frozen 178,561-identity universe.

## What is not independently reconstructed

The exact transformation from the official 176,581 count to the frozen 178,561 identity inventory is **not independently reconstructed by this release**. Existing records do not prove a one-to-one conversion, enumerate a 1,980-item difference, or justify a causal explanation for that difference. The counts refer to different processing/provenance scopes. This release does not claim that 178,561 is the original LMD v0.1 file count and does not claim that the number itself comes directly from Choi et al.'s processing universe.

This means that a reviewer can reproduce the reported statistics conditional
on the frozen downstream inventory, but cannot use this public release to audit
the identity-level transition from the provider's count to that inventory.

## Why frozen downstream statistics do not change

This disclosure changes neither the frozen analysis universe nor any registered result. Prevalence, controlled experiments, robustness analyses, and mitigation statistics were computed against their already frozen manifests and identity definitions. The release preserves those values and narrows only the public provenance language: results are conditional on the frozen 178,561-identity downstream universe and the adopted family definition.
