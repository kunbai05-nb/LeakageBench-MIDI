"""Fixed public REMI+ tokenizer used to identify the formal model vocabulary."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import mido
TOKENIZER_CONFIG = {'schema_version': 1, 'type': 'fixed_REMI_plus', 'positions_per_bar': 16, 'duration_positions': [1, 64], 'velocity_bins': 16, 'instrument_families': 8, 'drums_separate': True, 'token_types': ['BAR', 'POSITION', 'PROGRAM', 'PITCH', 'DURATION', 'VELOCITY'], 'context_length': 1024, 'bpe': False}

class MidiTokenizer:
    PAD = 0
    BOS = 1
    EOS = 2
    BAR = 3
    POSITION = 4
    PROGRAM = 20
    PITCH = 29
    DURATION = 157
    VELOCITY = 221
    VOCAB_SIZE = 237

    def __init__(self, config=None):
        self.config = dict(TOKENIZER_CONFIG if config is None else config)

    @staticmethod
    def config_sha256(config):
        return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def vocabulary(self):
        out = ['PAD', 'BOS', 'EOS', 'BAR'] + [f'POSITION_{i}' for i in range(16)] + [f'PROGRAM_{i}' for i in range(9)] + [f'PITCH_{i}' for i in range(128)] + [f'DURATION_{i}' for i in range(1, 65)] + [f'VELOCITY_{i}' for i in range(16)]
        assert len(out) == self.VOCAB_SIZE
        return out

    def encode_file(self, path):
        midi = mido.MidiFile(Path(path), clip=False)
        if midi.ticks_per_beat <= 0:
            raise ValueError('INVALID_TICKS_PER_BEAT')
        absolute = 0
        programs = {c: 0 for c in range(16)}
        active = {}
        notes = []
        signatures = []
        for msg in mido.merge_tracks(midi.tracks):
            absolute += msg.time
            if msg.type == 'time_signature':
                signatures.append((absolute, msg.numerator, msg.denominator))
            elif msg.type == 'program_change':
                programs[msg.channel] = msg.program
            elif msg.type == 'note_on' and msg.velocity > 0:
                active.setdefault((msg.channel, msg.note), []).append((absolute, msg.velocity, programs[msg.channel]))
            elif msg.type in {'note_off', 'note_on'}:
                stack = active.get((msg.channel, msg.note))
                if stack:
                    (start, velocity, program) = stack.pop(0)
                    if absolute > start:
                        notes.append((start, absolute, msg.channel, program, msg.note, velocity))
        if not notes:
            raise ValueError('NO_NOTES')
        if any(((n, d) != (4, 4) for (_, n, d) in signatures)):
            raise ValueError('NON_4_4_METER')
        step = midi.ticks_per_beat / 4.0
        quantized = []
        for (start, end, channel, program, pitch, velocity) in notes:
            pos = max(0, int(round(start / step)))
            duration = min(64, max(1, int(round((end - start) / step))))
            family = 8 if channel == 9 else min(7, program // 16)
            quantized.append((pos // 16, pos % 16, family, pitch, duration, min(15, velocity * 16 // 128)))
        quantized.sort()
        encoded = []
        current = None
        offsets = {}
        for (bar, position, family, pitch, duration, velocity) in quantized:
            if bar != current:
                current = bar
                offsets[bar] = len(encoded)
                encoded.append(self.BAR)
            encoded += [self.POSITION + position, self.PROGRAM + family, self.PITCH + pitch, self.DURATION + duration - 1, self.VELOCITY + velocity]
        return (encoded, {'note_count': len(notes), 'track_count': len(midi.tracks), 'total_bars': max((x[0] for x in quantized)) + 1, 'ticks_per_beat': midi.ticks_per_beat, 'bar_token_offsets': offsets})
PilotTokenizer = MidiTokenizer
