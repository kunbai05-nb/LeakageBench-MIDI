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
from .structural import classify_pair, midi_hashes

__all__ = [
    "analyze_effect",
    "audit_split",
    "build_contamination",
    "build_family_map",
    "census",
    "classify_pair",
    "conditional_sibling_probability",
    "cross_probability",
    "family_aware_split",
    "midi_hashes",
    "read_jsonl",
    "stable",
    "write_jsonl",
]
