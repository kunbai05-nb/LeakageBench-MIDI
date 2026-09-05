from __future__ import annotations

import json
import math
import os
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import asdict, dataclass
from multiprocessing import get_all_start_methods, get_context
from pathlib import Path

import mido
import numpy as np

try:
    from numba import njit
except ImportError:

    def njit(*args, **kwargs):
        def decorate(function):
            return function

        return decorate


ALIGNMENT_FEATURE_NAMES = (
    "align_best_score_per_match",
    "align_mean_score_per_match",
    "align_best_matches",
    "align_total_matches",
    "align_path_count",
    "align_short_coverage",
    "align_long_coverage",
    "align_coverage_hmean",
    "align_gap_fraction",
    "align_melody_agreement",
    "align_bass_agreement",
    "align_rhythm_agreement",
    "align_harmony_agreement",
    "align_density_agreement",
    "align_transposition_confidence",
    "align_transposition_margin",
)

_WORKER_SEQUENCES: list[dict | None] | None = None
_WORKER_CONFIG: AlignmentConfig | None = None


@dataclass(frozen=True)
class AlignmentConfig:
    segment_beats: float = 2.0
    onset_bins: int = 8
    max_segments: int = 512
    max_paths: int = 3
    min_path_matches: int = 4
    transposition_candidates: int = 3
    match_baseline: float = 0.525
    gap_open: float = 0.35
    gap_extend: float = 0.08


def _unit_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).tiny)


def extract_alignment_sequence(
    path: Path, config: AlignmentConfig = AlignmentConfig()
) -> dict:
    midi = mido.MidiFile(path, clip=True)
    ticks_per_beat = max(1, midi.ticks_per_beat)
    notes = []
    for track in midi.tracks:
        tick = 0
        active: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for message in track:
            tick += int(message.time)
            if (
                message.type == "note_on"
                and message.velocity > 0
                and int(message.channel) != 9
            ):
                key = (int(message.channel), int(message.note))
                active.setdefault(key, []).append((tick, int(message.velocity)))
            elif message.type in {"note_off", "note_on"} and int(message.channel) != 9:
                key = (int(message.channel), int(message.note))
                pending = active.get(key)
                if pending:
                    onset, velocity = pending.pop(0)
                    notes.append(
                        (
                            onset / ticks_per_beat,
                            max(1, tick - onset) / ticks_per_beat,
                            int(message.note),
                            velocity,
                        )
                    )
        for (_, pitch), pending in active.items():
            for onset, velocity in pending:
                notes.append((onset / ticks_per_beat, 0.25, pitch, velocity))
    if not notes:
        raise ValueError("no non-drum notes")

    notes.sort(key=lambda item: (item[0], item[2]))
    origin = notes[0][0]
    notes = [
        (onset - origin, duration, pitch, velocity)
        for onset, duration, pitch, velocity in notes
    ]
    end_beat = max(onset + duration for onset, duration, _, _ in notes)
    segments = min(
        config.max_segments, max(1, math.ceil(end_beat / config.segment_beats))
    )
    shape = (segments, config.onset_bins)
    melody = np.full(shape, np.nan, dtype=np.float32)
    bass = np.full(shape, np.nan, dtype=np.float32)
    rhythm = np.zeros(shape, dtype=np.float32)
    chroma = np.zeros((segments, 12), dtype=np.float32)
    density = np.zeros(segments, dtype=np.float32)

    for onset, duration, pitch, velocity in notes:
        segment = int(onset // config.segment_beats)
        if segment >= segments:
            continue
        phase = (onset - segment * config.segment_beats) / config.segment_beats
        onset_bin = min(config.onset_bins - 1, int(phase * config.onset_bins))
        melody[segment, onset_bin] = np.nanmax((melody[segment, onset_bin], pitch))
        bass[segment, onset_bin] = np.nanmin((bass[segment, onset_bin], pitch))
        rhythm[segment, onset_bin] += 1
        chroma[segment, pitch % 12] += min(duration, config.segment_beats) * max(
            velocity, 1
        )
        density[segment] += 1

    return {
        "melody": melody,
        "bass": bass,
        "rhythm": _unit_rows(rhythm),
        "chroma": _unit_rows(chroma),
        "density": np.log1p(density).astype(np.float32),
    }


def _extract_one(
    task: tuple[int, str, AlignmentConfig],
) -> tuple[int, dict | None, dict | None]:
    index, raw_path, config = task
    path = Path(raw_path)
    try:
        return index, extract_alignment_sequence(path, config), None
    except Exception as error:
        return (
            index,
            None,
            {
                "index": index,
                "path": path.name,
                "error": f"{type(error).__name__}: {error}",
            },
        )


def extract_alignment_bundle(
    paths: list[Path], config: AlignmentConfig = AlignmentConfig(), workers: int = 1
) -> tuple[list[dict | None], list[dict]]:
    sequences: list[dict | None] = [None] * len(paths)
    failures = []
    tasks = [(index, str(path), config) for index, path in enumerate(paths)]
    if workers > 1:
        executor_class = (
            ProcessPoolExecutor if "fork" in get_all_start_methods() else ThreadPoolExecutor
        )
        kwargs = {"max_workers": workers}
        if executor_class is ProcessPoolExecutor:
            kwargs["mp_context"] = get_context("fork")
        with executor_class(**kwargs) as executor:
            task_iter = iter(tasks)
            pending = set()
            for _ in range(min(max(1, workers * 2), len(tasks))):
                try:
                    pending.add(executor.submit(_extract_one, next(task_iter)))
                except StopIteration:
                    break
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    index, sequence, failure = future.result()
                    sequences[index] = sequence
                    if failure:
                        failures.append(failure)
                    try:
                        pending.add(executor.submit(_extract_one, next(task_iter)))
                    except StopIteration:
                        pass
    else:
        for task in tasks:
            index, sequence, failure = _extract_one(task)
            sequences[index] = sequence
            if failure:
                failures.append(failure)
    return sequences, failures


def save_alignment_bundle(
    directory: Path,
    identities: list[str],
    sequences: list[dict | None],
    failures: list[dict],
    config: AlignmentConfig,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    lengths = np.asarray(
        [0 if sequence is None else len(sequence["density"]) for sequence in sequences],
        dtype=np.int32,
    )
    offsets = np.r_[0, np.cumsum(lengths, dtype=np.int64)]

    def joined(name: str, width: int | None = None) -> np.ndarray:
        valid = [sequence[name] for sequence in sequences if sequence is not None]
        if valid:
            return np.concatenate(valid, axis=0)
        shape = (0,) if width is None else (0, width)
        return np.empty(shape, dtype=np.float32)

    np.savez_compressed(
        directory / "alignment_sequences.npz",
        offsets=offsets,
        melody=joined("melody", config.onset_bins),
        bass=joined("bass", config.onset_bins),
        rhythm=joined("rhythm", config.onset_bins),
        chroma=joined("chroma", 12),
        density=joined("density"),
    )
    (directory / "identities.json").write_text(json.dumps(identities, indent=2) + "\n")
    (directory / "parse_failures.json").write_text(
        json.dumps(failures, indent=2) + "\n"
    )
    (directory / "alignment_config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n"
    )


def load_alignment_bundle(
    directory: Path,
) -> tuple[list[str], list[dict | None], list[dict], AlignmentConfig]:
    identities = json.loads((directory / "identities.json").read_text())
    failures = json.loads((directory / "parse_failures.json").read_text())
    config = AlignmentConfig(
        **json.loads((directory / "alignment_config.json").read_text())
    )
    stored = np.load(directory / "alignment_sequences.npz")
    offsets = stored["offsets"]
    arrays = {
        name: stored[name] for name in ("melody", "bass", "rhythm", "chroma", "density")
    }
    sequences = []
    for index in range(len(identities)):
        start, end = int(offsets[index]), int(offsets[index + 1])
        if start == end:
            sequences.append(None)
            continue
        sequences.append(
            {name: values[start:end].copy() for name, values in arrays.items()}
        )
    return identities, sequences, failures, config


def _pitch_similarity(left: np.ndarray, right: np.ndarray, shift: int) -> np.ndarray:
    result = np.zeros((len(left), len(right)), dtype=np.float32)
    counts = np.zeros_like(result)
    for column in range(left.shape[1]):
        a = left[:, column, None]
        b = right[None, :, column] + shift
        valid = np.isfinite(a) & np.isfinite(b)
        result += np.where(valid, np.exp(-np.abs(a - b) / 2.5), 0).astype(np.float32)
        counts += valid
    return result / np.maximum(counts, 1)


def _view_matrices(left: dict, right: dict, shift: int) -> dict[str, np.ndarray]:
    melody = _pitch_similarity(left["melody"], right["melody"], shift)
    bass = _pitch_similarity(left["bass"], right["bass"], shift)
    rhythm = left["rhythm"] @ right["rhythm"].T
    harmony = left["chroma"] @ np.roll(right["chroma"], shift, axis=1).T
    density = np.exp(-np.abs(left["density"][:, None] - right["density"][None, :]))
    combined = (
        0.30 * melody + 0.15 * bass + 0.25 * harmony + 0.20 * rhythm + 0.10 * density
    )
    return {
        "combined": combined.astype(np.float32),
        "melody": melody,
        "bass": bass,
        "rhythm": rhythm,
        "harmony": harmony,
        "density": density.astype(np.float32),
    }


def _transposition_scores(left: dict, right: dict) -> np.ndarray:
    a = left["chroma"].sum(axis=0)
    b = right["chroma"].sum(axis=0)
    a /= max(float(np.linalg.norm(a)), np.finfo(np.float32).tiny)
    b /= max(float(np.linalg.norm(b)), np.finfo(np.float32).tiny)
    return np.asarray(
        [np.dot(a, np.roll(b, shift)) for shift in range(12)], dtype=np.float32
    )


@njit(cache=True, nogil=True)
def _smith_waterman_affine(score: np.ndarray, gap_open: float, gap_extend: float):
    rows, columns = score.shape
    match = np.zeros((rows + 1, columns + 1), dtype=np.float32)
    delete = np.zeros_like(match)
    insert = np.zeros_like(match)
    state = np.zeros((rows + 1, columns + 1), dtype=np.uint8)
    delete_from = np.zeros_like(state)
    insert_from = np.zeros_like(state)
    best_score = 0.0
    best_i = 0
    best_j = 0

    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            from_match = match[i - 1, j] - gap_open
            from_delete = delete[i - 1, j] - gap_extend
            if from_match >= from_delete and from_match > 0:
                delete[i, j] = from_match
                delete_from[i, j] = 1
            elif from_delete > 0:
                delete[i, j] = from_delete
                delete_from[i, j] = 2

            from_match = match[i, j - 1] - gap_open
            from_insert = insert[i, j - 1] - gap_extend
            if from_match >= from_insert and from_match > 0:
                insert[i, j] = from_match
                insert_from[i, j] = 1
            elif from_insert > 0:
                insert[i, j] = from_insert
                insert_from[i, j] = 3

            diagonal = match[i - 1, j - 1] + score[i - 1, j - 1]
            if diagonal > 0:
                match[i, j] = diagonal
                state[i, j] = 1
            if delete[i, j] > match[i, j]:
                match[i, j] = delete[i, j]
                state[i, j] = 2
            if insert[i, j] > match[i, j]:
                match[i, j] = insert[i, j]
                state[i, j] = 3
            if match[i, j] > best_score:
                best_score = match[i, j]
                best_i = i
                best_j = j

    path_i = np.empty(min(rows, columns), dtype=np.int32)
    path_j = np.empty(min(rows, columns), dtype=np.int32)
    count = 0
    i = best_i
    j = best_j
    current = state[i, j]
    while i > 0 and j > 0 and current != 0:
        if current == 1:
            path_i[count] = i - 1
            path_j[count] = j - 1
            count += 1
            i -= 1
            j -= 1
            current = state[i, j]
        elif current == 2:
            source = delete_from[i, j]
            i -= 1
            current = state[i, j] if source == 1 else 2
        else:
            source = insert_from[i, j]
            j -= 1
            current = state[i, j] if source == 1 else 3
    return best_score, path_i[:count][::-1], path_j[:count][::-1]


def _local_paths(similarity: np.ndarray, config: AlignmentConfig) -> list[dict]:
    score = 2 * similarity - 2 * config.match_baseline
    available = score.copy()
    paths = []
    for _ in range(config.max_paths):
        raw_score, left, right = _smith_waterman_affine(
            available, config.gap_open, config.gap_extend
        )
        if len(left) < config.min_path_matches:
            break
        paths.append({"score": float(raw_score), "left": left, "right": right})
        available[left, :] = -1e6
        available[:, right] = -1e6
    return paths


def alignment_pair_features(
    left: dict | None, right: dict | None, config: AlignmentConfig = AlignmentConfig()
) -> np.ndarray:
    if left is None or right is None:
        return np.zeros(len(ALIGNMENT_FEATURE_NAMES), dtype=np.float32)
    transposition = _transposition_scores(left, right)
    shift_order = np.argsort(transposition, kind="mergesort")[::-1]
    candidates = []
    for shift in shift_order[: config.transposition_candidates]:
        signed_shift = int(shift if shift <= 6 else shift - 12)
        views = _view_matrices(left, right, signed_shift)
        paths = _local_paths(views["combined"], config)
        total_score = sum(path["score"] for path in paths)
        total_matches = sum(len(path["left"]) for path in paths)
        candidates.append((total_score, total_matches, int(shift), views, paths))
    _, _, shift, views, paths = max(candidates, key=lambda item: (item[0], item[1]))
    alternatives = np.delete(transposition, shift)
    shift_margin = transposition[shift] - alternatives.max()

    if not paths:
        values = np.zeros(len(ALIGNMENT_FEATURE_NAMES), dtype=np.float32)
        values[-2] = transposition[shift]
        values[-1] = shift_margin
        return values

    left_indices = np.concatenate([path["left"] for path in paths])
    right_indices = np.concatenate([path["right"] for path in paths])
    path_scores = np.asarray([path["score"] for path in paths], dtype=np.float32)
    path_lengths = np.asarray([len(path["left"]) for path in paths], dtype=np.float32)
    coverage_left = len(np.unique(left_indices)) / len(left["density"])
    coverage_right = len(np.unique(right_indices)) / len(right["density"])
    if len(left["density"]) <= len(right["density"]):
        short_coverage, long_coverage = coverage_left, coverage_right
    else:
        short_coverage, long_coverage = coverage_right, coverage_left
    coverage_hmean = (
        2 * coverage_left * coverage_right / max(coverage_left + coverage_right, 1e-8)
    )
    gap_parts = []
    for path in paths:
        span = int(path["left"].max() - path["left"].min() + 1) + int(
            path["right"].max() - path["right"].min() + 1
        )
        gap_parts.append(max(0, span - 2 * len(path["left"])) / max(span, 1))
    aligned = (left_indices, right_indices)
    return np.asarray(
        [
            np.max(path_scores / path_lengths),
            path_scores.sum() / path_lengths.sum(),
            path_lengths.max(),
            path_lengths.sum(),
            len(paths),
            short_coverage,
            long_coverage,
            coverage_hmean,
            np.average(gap_parts, weights=path_lengths),
            views["melody"][aligned].mean(),
            views["bass"][aligned].mean(),
            views["rhythm"][aligned].mean(),
            views["harmony"][aligned].mean(),
            views["density"][aligned].mean(),
            transposition[shift],
            shift_margin,
        ],
        dtype=np.float32,
    )


def _alignment_worker(pair: np.ndarray) -> np.ndarray:
    if _WORKER_SEQUENCES is None or _WORKER_CONFIG is None:
        raise RuntimeError("alignment worker is not initialized")
    return alignment_pair_features(
        _WORKER_SEQUENCES[int(pair[0])],
        _WORKER_SEQUENCES[int(pair[1])],
        _WORKER_CONFIG,
    )


def alignment_feature_matrix(
    sequences: list[dict | None],
    pairs: np.ndarray,
    config: AlignmentConfig = AlignmentConfig(),
    workers: int = 1,
) -> tuple[np.ndarray, list[str]]:
    pairs = np.asarray(pairs, dtype=np.int64)

    def compute(pair: np.ndarray) -> np.ndarray:
        return alignment_pair_features(
            sequences[int(pair[0])], sequences[int(pair[1])], config
        )

    if workers > 1 and os.name == "posix" and "fork" in get_all_start_methods():
        global _WORKER_SEQUENCES, _WORKER_CONFIG
        _WORKER_SEQUENCES = sequences
        _WORKER_CONFIG = config
        try:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=get_context("fork")
            ) as executor:
                rows = list(executor.map(_alignment_worker, pairs, chunksize=128))
        finally:
            _WORKER_SEQUENCES = None
            _WORKER_CONFIG = None
    elif workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(compute, pairs, chunksize=16))
    else:
        rows = [compute(pair) for pair in pairs]
    if not rows:
        shape = (0, len(ALIGNMENT_FEATURE_NAMES))
        return np.empty(shape, dtype=np.float32), list(ALIGNMENT_FEATURE_NAMES)
    return np.stack(rows), list(ALIGNMENT_FEATURE_NAMES)
