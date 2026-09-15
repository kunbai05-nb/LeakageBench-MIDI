"""Public, data-agnostic APIs for LeakageBench-MIDI."""

__version__ = "1.7.0"

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
__all__ = [
    "__version__",
    "analyze_effect",
    "audit_split",
    "build_contamination",
    "build_family_map",
    "census",
    "conditional_sibling_probability",
    "cross_probability",
    "family_aware_split",
    "read_jsonl",
    "stable",
    "write_jsonl",
]
