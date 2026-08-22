from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import mido


NORMALIZATION_VERSION = "normalized-midi-structure-v1"


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def midi_hashes(path):
    """Historical rc2 comparator retained for backward-compatible classification."""
    target = Path(path)
    midi = mido.MidiFile(target, clip=False)
    tracks = []
    for track in midi.tracks:
        tick = 0
        events = []
        for message in track:
            tick += message.time
            if message.is_meta and message.type not in {"time_signature", "set_tempo"}:
                continue
            payload = message.dict()
            payload.pop("time", None)
            events.append((tick, tuple(sorted(payload.items()))))
        tracks.append(tuple(events))
    return {
        "byte_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "canonical_event_sha256": _digest(tracks),
        "track_order_invariant_sha256": _digest(sorted(tracks)),
    }


def classify_pair(a, b):
    """Historical rc2 classification; use classify_pair_normalized for v1 robustness."""
    x, y = midi_hashes(a), midi_hashes(b)
    if x["byte_sha256"] == y["byte_sha256"]:
        classification = "BYTE_EXACT"
    elif x["canonical_event_sha256"] == y["canonical_event_sha256"]:
        classification = "CANONICAL_EQUIVALENT"
    elif x["track_order_invariant_sha256"] == y["track_order_invariant_sha256"]:
        classification = "TRACK_ORDER_EQUIVALENT"
    else:
        classification = "STRUCTURALLY_NONEXACT"
    return {
        "classification": classification,
        "a": x,
        "b": y,
        "frozen_hierarchy": [
            "BYTE_EXACT",
            "CANONICAL_EQUIVALENT",
            "TRACK_ORDER_EQUIVALENT",
            "STRUCTURALLY_NONEXACT",
        ],
    }


def _fraction_pair(value):
    value = Fraction(value)
    return [value.numerator, value.denominator]


def normalize_midi_structure(path):
    """Return a deterministic note-semantic representation in rational beat units.

    Track containers, track order, melodic channel allocation, tempo, text and other
    metadata are excluded. MIDI channel 10 is retained only as a drum-role boolean.
    Pitch, onset, duration and attack velocity remain part of the identity.
    """
    midi = mido.MidiFile(Path(path), clip=False)
    ppq = midi.ticks_per_beat
    if not isinstance(ppq, int) or ppq <= 0:
        raise ValueError("MIDI ticks_per_beat must be positive")
    notes = []
    for track in midi.tracks:
        tick = 0
        active = defaultdict(list)
        for message in track:
            tick += message.time
            if message.type == "note_on" and message.velocity > 0:
                active[message.channel, message.note].append((tick, message.velocity))
            elif message.type == "note_off" or (
                message.type == "note_on" and message.velocity == 0
            ):
                key = (message.channel, message.note)
                if active[key]:
                    onset, velocity = active[key].pop(0)
                    notes.append(
                        (
                            Fraction(onset, ppq),
                            Fraction(tick - onset, ppq),
                            int(message.note),
                            int(velocity),
                            message.channel == 9,
                        )
                    )
        unclosed = [
            (channel, pitch, len(starts))
            for (channel, pitch), starts in active.items()
            if starts
        ]
        if unclosed:
            channel, pitch, count = sorted(unclosed)[0]
            raise ValueError(
                f"unclosed active MIDI note: channel={channel} pitch={pitch} count={count}"
            )
    # Counter semantics make same-timestamp event order and track containers irrelevant.
    counter = Counter(notes)
    events = []
    for (onset, duration, pitch, velocity, is_drum), count in sorted(counter.items()):
        events.append(
            {
                "onset_beats": _fraction_pair(onset),
                "duration_beats": _fraction_pair(duration),
                "pitch": pitch,
                "velocity": velocity,
                "is_drum": is_drum,
                "multiplicity": count,
            }
        )
    return {
        "normalization_version": NORMALIZATION_VERSION,
        "event_type": "note",
        "events": events,
    }


def normalized_structural_hash(path):
    return _digest(normalize_midi_structure(path))


def classify_pair_normalized(a, b):
    a_hash, b_hash = normalized_structural_hash(a), normalized_structural_hash(b)
    return {
        "classification": "NORMALIZED_STRUCTURAL_EXACT"
        if a_hash == b_hash
        else "NORMALIZED_STRUCTURAL_NONEXACT",
        "normalization_version": NORMALIZATION_VERSION,
        "a_normalized_sha256": a_hash,
        "b_normalized_sha256": b_hash,
    }
