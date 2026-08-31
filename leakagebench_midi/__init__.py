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
from .structural import normalize_midi_structure, normalized_structural_hash

__all__ = [
    "analyze_effect",
    "audit_split",
    "build_contamination",
    "build_family_map",
    "census",
    "conditional_sibling_probability",
    "cross_probability",
    "family_aware_split",
    "normalize_midi_structure",
    "normalized_structural_hash",
    "read_jsonl",
    "stable",
    "write_jsonl",
]
