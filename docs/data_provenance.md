# Data provenance

| Count | Meaning |
|---:|---|
| 176,581 | LMD v0.1 MD5-distinct files reported by the [official dataset page](https://colinraffel.com/projects/lmd/) |
| 178,561 | Archive-header identities in the frozen LeakageBench-MIDI census inventory |
| 58,931 | Identities in adopted multi-member components |
| 119,630 | Remaining identities treated as singleton components |

The frozen census uses the 178,561-identity inventory. Its archive SHA-256 is
`6fcfe2ac49ca08f3f214cec86ab138d4fc4dabcd7f27f491a838dae6db45a12b`.
The Monte Carlo result artifact SHA-256 is
`9607daf226a6e51f281f6c52eb06c5a7c701309ca2f2e51b6febc61cfedfc84b`.

The official count and the archive-header inventory use different collection
rules; all census calculations use the 178,561-file inventory.
