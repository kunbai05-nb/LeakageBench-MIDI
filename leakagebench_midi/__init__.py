"""Public, data-agnostic APIs for LeakageBench-MIDI."""

from .core import (
    analyze_effect,
    audit_split,
    build_contamination,
    build_family_map,
    census,
    conditional_sibling_probability,
    cross_probability,
    family_aware_split,
    read_jsonl,
    stable,
    write_jsonl,
)
from .structural import (
    NORMALIZATION_VERSION,
    classify_pair,
    classify_pair_normalized,
    midi_hashes,
    normalize_midi_structure,
    normalized_structural_hash,
)

__all__ = [
    "analyze_effect",
    "audit_split",
    "build_contamination",
    "build_family_map",
    "census",
    "classify_pair",
    "classify_pair_normalized",
    "conditional_sibling_probability",
    "cross_probability",
    "family_aware_split",
    "midi_hashes",
    "normalize_midi_structure",
    "normalized_structural_hash",
    "NORMALIZATION_VERSION",
    "read_jsonl",
    "stable",
    "write_jsonl",
]
