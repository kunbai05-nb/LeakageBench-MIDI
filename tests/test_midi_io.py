from __future__ import annotations

import struct

import mido
import pytest

from leakagebench_midi.alignment import extract_alignment_sequence
from leakagebench_midi.content import parse_midi
from leakagebench_midi.midi_io import TruncatedMidiError, load_midi


def _write_drum_midi(path):
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    track.extend(
        [
            mido.Message("note_on", channel=9, note=36, velocity=100, time=0),
            mido.Message("note_off", channel=9, note=36, velocity=0, time=480),
        ]
    )
    midi.tracks.append(track)
    midi.save(path)


def test_invalid_key_signature_is_ignored(tmp_path):
    path = tmp_path / "invalid-key.mid"
    track = b"\x00\xff\x59\x02\x0e\x4e\x00\xff\x2f\x00"
    path.write_bytes(
        b"MThd" + struct.pack(">IHHH", 6, 0, 1, 480)
        + b"MTrk" + struct.pack(">I", len(track)) + track
    )
    assert load_midi(path).tracks[0][0].key == "C"


def test_truncated_track_is_rejected(tmp_path):
    path = tmp_path / "truncated.mid"
    _write_drum_midi(path)
    data = bytearray(path.read_bytes())
    size = struct.unpack(">I", data[18:22])[0]
    data[18:22] = struct.pack(">I", size + 5)
    path.write_bytes(data)
    with pytest.raises(TruncatedMidiError):
        load_midi(path)


def test_drum_only_midi_uses_fallback_features(tmp_path):
    path = tmp_path / "drums.mid"
    _write_drum_midi(path)
    assert parse_midi(path)["scalars"][0] == 1
    assert extract_alignment_sequence(path)["density"].sum() > 0
