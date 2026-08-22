from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mido


TOKENIZER_CONFIG = {
    "schema_version": 1,
    "type": "fixed_REMI_plus",
    "positions_per_bar": 16,
    "duration_positions": [1, 64],
    "velocity_bins": 16,
    "instrument_families": 8,
    "drums_separate": True,
    "token_types": ["BAR", "POSITION", "PROGRAM", "PITCH", "DURATION", "VELOCITY"],
    "context_length": 1024,
    "bpe": False,
}


class MidiTokenizer:
    PAD = 0
    BOS = 1
    EOS = 2
    BAR = 3
    POSITION = 4
    PROGRAM = POSITION + 16
    PITCH = PROGRAM + 9
    DURATION = PITCH + 128
    VELOCITY = DURATION + 64
    VOCAB_SIZE = VELOCITY + 16

    def __init__(self, config: dict | None = None):
        self.config = dict(TOKENIZER_CONFIG if config is None else config)

    @staticmethod
    def config_sha256(config: dict) -> str:
        payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def vocabulary(self) -> list[str]:
        output = ["PAD", "BOS", "EOS", "BAR"]
        output += [f"POSITION_{index}" for index in range(16)]
        output += [f"PROGRAM_{index}" for index in range(9)]
        output += [f"PITCH_{index}" for index in range(128)]
        output += [f"DURATION_{index}" for index in range(1, 65)]
        output += [f"VELOCITY_{index}" for index in range(16)]
        return output

    def encode_file(self, path: str | Path) -> tuple[list[int], dict]:
        midi = mido.MidiFile(Path(path), clip=False)
        if midi.ticks_per_beat <= 0:
            raise ValueError("INVALID_TICKS_PER_BEAT")

        absolute = 0
        programs = {channel: 0 for channel in range(16)}
        active = {}
        notes = []
        signatures = []
        for message in mido.merge_tracks(midi.tracks):
            absolute += message.time
            if message.type == "time_signature":
                signatures.append((absolute, message.numerator, message.denominator))
            elif message.type == "program_change":
                programs[message.channel] = message.program
            elif message.type == "note_on" and message.velocity > 0:
                active.setdefault((message.channel, message.note), []).append(
                    (absolute, message.velocity, programs[message.channel])
                )
            elif message.type in {"note_off", "note_on"}:
                pending = active.get((message.channel, message.note))
                if pending:
                    start, velocity, program = pending.pop(0)
                    if absolute > start:
                        notes.append(
                            (
                                start,
                                absolute,
                                message.channel,
                                program,
                                message.note,
                                velocity,
                            )
                        )

        if not notes:
            raise ValueError("NO_NOTES")
        if any(
            (numerator, denominator) != (4, 4)
            for _, numerator, denominator in signatures
        ):
            raise ValueError("NON_4_4_METER")

        step = midi.ticks_per_beat / 4.0
        quantized = []
        for start, end, channel, program, pitch, velocity in notes:
            position = max(0, int(round(start / step)))
            duration = min(64, max(1, int(round((end - start) / step))))
            family = 8 if channel == 9 else min(7, program // 16)
            quantized.append(
                (
                    position // 16,
                    position % 16,
                    family,
                    pitch,
                    duration,
                    min(15, velocity * 16 // 128),
                )
            )
        quantized.sort()

        encoded = []
        current_bar = None
        offsets = {}
        for bar, position, family, pitch, duration, velocity in quantized:
            if bar != current_bar:
                current_bar = bar
                offsets[bar] = len(encoded)
                encoded.append(self.BAR)
            encoded.extend(
                (
                    self.POSITION + position,
                    self.PROGRAM + family,
                    self.PITCH + pitch,
                    self.DURATION + duration - 1,
                    self.VELOCITY + velocity,
                )
            )
        return encoded, {
            "note_count": len(notes),
            "track_count": len(midi.tracks),
            "total_bars": max(row[0] for row in quantized) + 1,
            "ticks_per_beat": midi.ticks_per_beat,
            "bar_token_offsets": offsets,
        }

    def windows(
        self,
        tokens: list[int],
        metadata: dict,
        max_bars: int = 16,
        prompt_bars: int = 4,
    ) -> list[dict]:
        offsets = metadata["bar_token_offsets"]
        bars = sorted(offsets)
        output = []
        for start_bar in range(0, metadata["total_bars"] - prompt_bars, prompt_bars):
            if start_bar not in offsets:
                continue
            begin = offsets[start_bar]
            continuation_bar = start_bar + prompt_bars
            if continuation_bar not in offsets:
                continue
            prompt_end = offsets[continuation_bar] - begin
            maximum = min(metadata["total_bars"], start_bar + max_bars)
            for end_bar in range(maximum, continuation_bar, -1):
                end = next(
                    (offsets[bar] for bar in bars if bar >= end_bar), len(tokens)
                )
                window = [self.BOS, *tokens[begin:end], self.EOS]
                prompt_tokens = 1 + prompt_end
                if (
                    len(window) <= self.config["context_length"]
                    and prompt_tokens < len(window) - 1
                ):
                    output.append(
                        {
                            "start_bar": start_bar,
                            "end_bar": end_bar,
                            "prompt_bars": prompt_bars,
                            "token_ids": window,
                            "token_count": len(window),
                            "prompt_token_count": prompt_tokens,
                        }
                    )
                    break
        return output


PilotTokenizer = MidiTokenizer
