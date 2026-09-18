from __future__ import annotations

from pathlib import Path

import mido
from mido.midifiles import meta


class TruncatedMidiError(Exception):
    pass


_KEY_SIGNATURE_SPEC = meta._META_SPECS[0x59]
_DECODE_KEY_SIGNATURE = _KEY_SIGNATURE_SPEC.decode


def _decode_key_signature(message, data):
    try:
        _DECODE_KEY_SIGNATURE(message, data)
    except meta.KeySignatureError:
        message.key = "C"


_KEY_SIGNATURE_SPEC.decode = _decode_key_signature


def load_midi(path: Path) -> mido.MidiFile:
    try:
        return mido.MidiFile(path, clip=True)
    except EOFError as error:
        raise TruncatedMidiError("truncated MIDI stream") from error
