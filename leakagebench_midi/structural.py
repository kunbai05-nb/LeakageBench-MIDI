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
