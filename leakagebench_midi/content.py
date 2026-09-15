from __future__ import annotations

import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_all_start_methods, get_context
from pathlib import Path

import mido
import numpy as np
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors


TOKEN_GROUPS = ("melody", "bass", "rhythm", "harmony", "motif")
CANDIDATE_SIGNALS = (
    "bass",
    "harmony",
    "motif",
    "interval_hist",
    "duration_hist",
)


def _quantized(value: float, resolution: int = 12, limit: int = 192) -> int:
    return int(np.clip(round(value * resolution), 0, limit))


def _sampled_ngrams(values: list[int], n: int, maximum: int = 2048):
    count = len(values) - n + 1
    if count <= 0:
        return
    step = max(1, math.ceil(count / maximum))
    for index in range(0, count, step):
        yield tuple(values[index : index + n])


def _rotate_mask(mask: int, shift: int) -> int:
    shift %= 12
    if not shift:
        return mask
    return ((mask << shift) | (mask >> (12 - shift))) & 0xFFF


def _canonical_chroma_window(values: tuple[int, ...]) -> tuple[int, ...]:
    return min(
        tuple(_MASK_ROTATIONS[value][shift] for value in values) for shift in range(12)
    )


_MASK_ROTATIONS = tuple(
    tuple(_rotate_mask(mask, shift) for shift in range(12)) for mask in range(4096)
)


def _cosine_hist(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values)
    return values / norm if norm else values


def parse_midi(path: Path) -> dict:
    midi = mido.MidiFile(path, clip=True)
    ticks_per_beat = max(1, midi.ticks_per_beat)
    notes = []
    melodic_tracks = 0
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        programs = defaultdict(int)
        active: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
        track_notes = 0
        for message in track:
            tick += int(message.time)
            if message.type == "program_change":
                programs[int(message.channel)] = int(message.program)
            elif message.type == "note_on" and message.velocity > 0:
                channel = int(message.channel)
                if channel == 9:
                    continue
                key = (channel, int(message.note))
                active[key].append((tick, int(message.velocity), programs[channel]))
                track_notes += 1
            elif message.type in {"note_off", "note_on"}:
                channel = int(message.channel)
                if channel == 9:
                    continue
                key = (channel, int(message.note))
                if active[key]:
                    onset, velocity, program = active[key].pop(0)
                    duration = max(1, tick - onset)
                    notes.append(
                        (
                            onset / ticks_per_beat,
                            duration / ticks_per_beat,
                            int(message.note),
                            velocity,
                            program,
                            track_index,
                        )
                    )
        if track_notes:
            melodic_tracks += 1
        for (channel, pitch), pending in active.items():
            for onset, velocity, program in pending:
                notes.append(
                    (
                        onset / ticks_per_beat,
                        0.25,
                        pitch,
                        velocity,
                        program,
                        track_index,
                    )
                )
    notes.sort(key=lambda item: (item[0], item[2], item[5]))
    if not notes:
        raise ValueError("no non-drum notes")

    grouped: dict[int, list[tuple]] = defaultdict(list)
    for note in notes:
        grouped[round(note[0] * 4)].append(note)
    onset_keys = sorted(grouped)
    melody = [max(note[2] for note in grouped[key]) for key in onset_keys]
    bass = [min(note[2] for note in grouped[key]) for key in onset_keys]
    onset_beats = [key / 4 for key in onset_keys]
    melody_intervals = [
        int(np.clip(b - a, -36, 36)) for a, b in zip(melody, melody[1:])
    ]
    bass_intervals = [int(np.clip(b - a, -36, 36)) for a, b in zip(bass, bass[1:])]
    rhythm = [_quantized(b - a) for a, b in zip(onset_beats, onset_beats[1:])]

    tokens = {name: Counter() for name in TOKEN_GROUPS}
    for n in (2, 3, 4, 5):
        for value in _sampled_ngrams(melody_intervals, n):
            tokens["melody"][f"mi{n}:{','.join(map(str, value))}"] += 1
        for value in _sampled_ngrams(bass_intervals, n):
            tokens["bass"][f"bi{n}:{','.join(map(str, value))}"] += 1
    for n in (2, 3, 4):
        for value in _sampled_ngrams(rhythm, n):
            tokens["rhythm"][f"r{n}:{','.join(map(str, value))}"] += 1
    motif_count = min(len(melody_intervals), len(rhythm)) - 2
    motif_step = max(1, math.ceil(max(0, motif_count) / 2048))
    for index in range(0, max(0, motif_count), motif_step):
        value = tuple(
            (melody_intervals[index + offset], rhythm[index + offset])
            for offset in range(3)
        )
        tokens["motif"][f"mr3:{value}"] += 1

    half_beat: dict[int, int] = defaultdict(int)
    for onset, _, pitch, _, _, _ in notes:
        half_beat[round(onset * 2)] |= 1 << (pitch % 12)
    if half_beat:
        first, last = min(half_beat), min(max(half_beat), min(half_beat) + 2047)
        sequence = [half_beat.get(index, 0) for index in range(first, last + 1)]
        for n in (2, 4, 8):
            for value in _sampled_ngrams(sequence, n, maximum=512):
                canonical = _canonical_chroma_window(value)
                tokens["harmony"][f"h{n}:{','.join(map(str, canonical))}"] += 1

    chroma = np.zeros(12, dtype=np.float32)
    interval_hist = np.zeros(75, dtype=np.float32)
    duration_hist = np.zeros(16, dtype=np.float32)
    ioi_hist = np.zeros(16, dtype=np.float32)
    for _, duration, pitch, velocity, _, _ in notes:
        chroma[pitch % 12] += max(duration, 0.125) * max(velocity, 1)
        bucket = int(np.clip(round(math.log2(max(duration, 1 / 32)) * 2) + 8, 0, 15))
        duration_hist[bucket] += 1
    for interval in melody_intervals:
        interval_hist[int(np.clip(interval, -37, 37)) + 37] += 1
    for value in np.diff(onset_beats):
        bucket = int(np.clip(round(math.log2(max(value, 1 / 32)) * 2) + 8, 0, 15))
        ioi_hist[bucket] += 1

    pitches = np.asarray([note[2] for note in notes], dtype=float)
    end_beat = max(note[0] + note[1] for note in notes)
    return {
        "tokens": tokens,
        "chroma": _cosine_hist(chroma),
        "interval_hist": _cosine_hist(interval_hist),
        "duration_hist": _cosine_hist(duration_hist),
        "ioi_hist": _cosine_hist(ioi_hist),
        "scalars": np.asarray(
            [
                len(notes),
                len(onset_keys),
                end_beat,
                melodic_tracks,
                pitches.min(),
                pitches.max(),
                pitches.mean(),
                pitches.std(),
                len(set(note[4] for note in notes)),
                len(half_beat),
            ],
            dtype=np.float32,
        ),
    }


def _parse_one(task: tuple[int, str]) -> tuple[int, dict | None, dict | None]:
    index, raw_path = task
    path = Path(raw_path)
    try:
        return index, parse_midi(path), None
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


def extract_feature_bundle(
    paths: list[Path], hash_width: int = 8192, workers: int = 1
) -> tuple[dict, list[dict]]:
    parsed = [None] * len(paths)
    failures = []
    tasks = [(index, str(path)) for index, path in enumerate(paths)]
    if workers > 1:
        executor_class = (
            ProcessPoolExecutor
            if "fork" in get_all_start_methods()
            else ThreadPoolExecutor
        )
        kwargs = {"max_workers": workers}
        if executor_class is ProcessPoolExecutor:
            kwargs["mp_context"] = get_context("fork")
        with executor_class(**kwargs) as executor:
            outputs = executor.map(_parse_one, tasks, chunksize=4)
            for index, item, failure in outputs:
                parsed[index] = item
                if failure:
                    failures.append(failure)
    else:
        for task in tasks:
            index, item, failure = _parse_one(task)
            parsed[index] = item
            if failure:
                failures.append(failure)
    valid = np.asarray([item is not None for item in parsed], dtype=bool)
    dense = {}
    widths = {
        "chroma": 12,
        "interval_hist": 75,
        "duration_hist": 16,
        "ioi_hist": 16,
        "scalars": 10,
    }
    for name, width in widths.items():
        matrix = np.zeros((len(parsed), width), dtype=np.float32)
        for index, item in enumerate(parsed):
            if item is not None:
                matrix[index] = item[name]
        dense[name] = matrix
    hashed = {}
    for group in TOKEN_GROUPS:
        dictionaries = [
            item["tokens"][group] if item is not None else {} for item in parsed
        ]
        matrix = FeatureHasher(
            n_features=hash_width, input_type="dict", alternate_sign=False
        ).transform(dictionaries)
        hashed[group] = (
            TfidfTransformer(norm="l2", sublinear_tf=True)
            .fit_transform(matrix)
            .astype(np.float32)
        )
    return {**dense, **hashed, "valid": valid}, failures


def compact_candidate_ranks(
    matrices: dict[str, np.ndarray | sparse.spmatrix],
    indices: np.ndarray,
    k: int,
    workers: int = 1,
) -> dict:
    signals = tuple(name for name in CANDIDATE_SIGNALS if name in matrices)
    universe_size = next(iter(matrices.values())).shape[0]
    items = list(enumerate(signals))
    signal_workers = min(max(1, workers), len(items))
    threads = max(1, workers // signal_workers)
    global _EXACT_SIGNAL_STATE
    _EXACT_SIGNAL_STATE = {
        "matrices": matrices,
        "indices": np.asarray(indices, dtype=np.int64),
        "k": k,
        "threads": threads,
        "universe_size": universe_size,
    }
    try:
        if signal_workers > 1 and "fork" in get_all_start_methods():
            with ProcessPoolExecutor(
                max_workers=signal_workers, mp_context=get_context("fork")
            ) as executor:
                batches = list(executor.map(_exact_signal_job, items))
        else:
            batches = [_exact_signal_job(item) for item in items]
    finally:
        _EXACT_SIGNAL_STATE = None
    batches.sort(key=lambda item: item[0])
    unique_keys = np.unique(np.concatenate([batch[1] for batch in batches]))
    rank_values = np.full((len(unique_keys), len(signals), 2), k + 1, dtype=np.int16)
    for signal_index, keys, direction, local_ranks in batches:
        positions = np.searchsorted(unique_keys, keys)
        for value in (0, 1):
            mask = direction == value
            np.minimum.at(
                rank_values[:, signal_index, value], positions[mask], local_ranks[mask]
            )
    pairs = np.column_stack(
        (unique_keys // universe_size, unique_keys % universe_size)
    ).astype(np.int32)
    return {"pairs": pairs, "ranks": rank_values, "signals": signals, "k": k}


_FAISS_SIGNAL_STATE = None
_EXACT_SIGNAL_STATE = None


def _exact_signal_job(item):
    signal_index, signal = item
    state = _EXACT_SIGNAL_STATE
    indices = state["indices"]
    matrix = state["matrices"][signal][indices]
    neighbours = NearestNeighbors(
        n_neighbors=min(state["k"] + 1, len(indices)),
        metric="cosine",
        algorithm="brute",
        n_jobs=state["threads"],
    ).fit(matrix)
    distances, positions = neighbours.kneighbors(matrix)
    left = np.repeat(indices, positions.shape[1])
    right = indices[positions.ravel()]
    distance = distances.ravel()
    keep = (left != right) & np.isfinite(distance)
    left, right = left[keep], right[keep]
    self_positions = positions == np.arange(len(indices))[:, None]
    ranks = np.cumsum(~self_positions, axis=1, dtype=np.int16).ravel()[keep]
    low, high = np.minimum(left, right), np.maximum(left, right)
    keys = low.astype(np.int64) * state["universe_size"] + high.astype(np.int64)
    direction = (left > right).astype(np.int8)
    return signal_index, keys, direction, ranks


def _faiss_signal_job(item):
    import faiss
    from sklearn.random_projection import SparseRandomProjection

    signal_index, signal = item
    state = _FAISS_SIGNAL_STATE
    matrices = state["matrices"]
    indices = state["indices"]
    k = state["k"]
    matrix = matrices[signal][indices]
    if sparse.issparse(matrix):
        width = min(state["projection_dim"], max(8, len(indices) - 1))
        matrix = SparseRandomProjection(
            n_components=width,
            dense_output=True,
            random_state=state["seed"] + signal_index,
        ).fit_transform(matrix)
    vectors = np.ascontiguousarray(matrix, dtype=np.float32)
    faiss.omp_set_num_threads(state["threads"])
    faiss.normalize_L2(vectors)
    index = faiss.IndexHNSWFlat(
        vectors.shape[1], state["hnsw_connections"], faiss.METRIC_INNER_PRODUCT
    )
    index.hnsw.efConstruction = state["ef_construction"]
    index.hnsw.efSearch = state["ef_search"]
    index.add(vectors)
    _, neighbours = index.search(vectors, min(k + 1, len(indices)))

    local_left = np.repeat(np.arange(len(indices), dtype=np.int64), neighbours.shape[1])
    local_right = neighbours.ravel().astype(np.int64)
    valid = (local_right >= 0) & (local_left != local_right)
    rank_grid = np.cumsum(
        (neighbours >= 0) & (neighbours != np.arange(len(indices))[:, None]),
        axis=1,
        dtype=np.int16,
    ).ravel()
    left = indices[local_left[valid]]
    right = indices[local_right[valid]]
    ranks = rank_grid[valid]
    low, high = np.minimum(left, right), np.maximum(left, right)
    keys = low * state["universe_size"] + high
    direction = left > right
    unique_keys, positions = np.unique(keys, return_inverse=True)
    low_to_high = np.full(len(unique_keys), k + 1, dtype=np.int16)
    high_to_low = np.full(len(unique_keys), k + 1, dtype=np.int16)
    np.minimum.at(low_to_high, positions[~direction], ranks[~direction])
    np.minimum.at(high_to_low, positions[direction], ranks[direction])
    mutual = (low_to_high <= k) & (high_to_low <= k)
    result = (unique_keys, low_to_high, high_to_low, unique_keys[mutual])
    row = {
        "signal": signal,
        "dimensions": int(vectors.shape[1]),
        "directed_neighbours": int(valid.sum()),
        "mutual_pairs": int(mutual.sum()),
    }
    return signal_index, result, row


def faiss_candidate_ranks(
    matrices: dict[str, np.ndarray | sparse.spmatrix],
    indices: np.ndarray,
    k: int,
    projection_dim: int = 256,
    hnsw_connections: int = 32,
    ef_construction: int = 160,
    ef_search: int = 200,
    threads: int = 1,
    signal_workers: int = 1,
    seed: int = 20260911,
) -> tuple[dict, dict]:
    """Return the union of directed HNSW top-k lists with both ranks retained."""
    indices = np.asarray(indices, dtype=np.int64)
    signals = tuple(name for name in CANDIDATE_SIGNALS if name in matrices)
    universe_size = next(iter(matrices.values())).shape[0]
    global _FAISS_SIGNAL_STATE
    _FAISS_SIGNAL_STATE = {
        "matrices": matrices,
        "indices": indices,
        "k": k,
        "projection_dim": projection_dim,
        "hnsw_connections": hnsw_connections,
        "ef_construction": ef_construction,
        "ef_search": ef_search,
        "threads": max(1, threads),
        "seed": seed,
        "universe_size": universe_size,
    }
    items = list(enumerate(signals))
    if signal_workers > 1:
        executor_class = (
            ProcessPoolExecutor
            if "fork" in get_all_start_methods()
            else ThreadPoolExecutor
        )
        kwargs = {"max_workers": min(signal_workers, len(items))}
        if executor_class is ProcessPoolExecutor:
            kwargs["mp_context"] = get_context("fork")
        with executor_class(**kwargs) as executor:
            outputs = list(executor.map(_faiss_signal_job, items))
    else:
        outputs = [_faiss_signal_job(item) for item in items]
    _FAISS_SIGNAL_STATE = None
    outputs.sort(key=lambda item: item[0])
    directional = [item[1] for item in outputs]
    selected_keys = np.unique(np.concatenate([item[0] for item in directional]))
    rank_values = np.full((len(selected_keys), len(signals), 2), k + 1, dtype=np.int16)
    for signal_index, (keys, low_to_high, high_to_low, _) in enumerate(directional):
        if not len(keys):
            continue
        positions = np.searchsorted(keys, selected_keys)
        found = (positions < len(keys)) & (
            keys[np.minimum(positions, len(keys) - 1)] == selected_keys
        )
        rank_values[found, signal_index, 0] = low_to_high[positions[found]]
        rank_values[found, signal_index, 1] = high_to_low[positions[found]]
    pairs = np.column_stack(
        (selected_keys // universe_size, selected_keys % universe_size)
    ).astype(np.int32)
    return (
        {"pairs": pairs, "ranks": rank_values, "signals": signals, "k": k},
        {
            "backend": "faiss_hnsw",
            "candidate_pairs": len(pairs),
            "top_k": k,
            "signals": [item[2] for item in outputs],
        },
    )


def _sparse_pair_cosine(
    matrix: sparse.spmatrix,
    left: np.ndarray,
    right: np.ndarray,
    block_size: int = 20_000,
) -> np.ndarray:
    result = np.empty(len(left), dtype=np.float32)
    for start in range(0, len(left), block_size):
        end = min(len(left), start + block_size)
        result[start:end] = np.asarray(
            matrix[left[start:end]].multiply(matrix[right[start:end]]).sum(axis=1)
        ).ravel()
    return result


def _sparse_pair_set_overlap(
    matrix: sparse.spmatrix,
    left: np.ndarray,
    right: np.ndarray,
    block_size: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    binary = matrix.copy().tocsr()
    binary.data.fill(1)
    counts = np.diff(binary.indptr).astype(np.float32)
    jaccard = np.empty(len(left), dtype=np.float32)
    containment = np.empty(len(left), dtype=np.float32)
    for start in range(0, len(left), block_size):
        end = min(len(left), start + block_size)
        intersection = (
            np.asarray(
                binary[left[start:end]].multiply(binary[right[start:end]]).sum(axis=1)
            )
            .ravel()
            .astype(np.float32)
        )
        left_count, right_count = counts[left[start:end]], counts[right[start:end]]
        union = left_count + right_count - intersection
        jaccard[start:end] = intersection / np.maximum(union, 1)
        containment[start:end] = intersection / np.maximum(
            np.minimum(left_count, right_count), 1
        )
    return jaccard, containment


def _dense_pair_cosine(
    matrix: np.ndarray, left: np.ndarray, right: np.ndarray, block_size: int = 20_000
) -> np.ndarray:
    result = np.empty(len(left), dtype=np.float32)
    for start in range(0, len(left), block_size):
        end = min(len(left), start + block_size)
        result[start:end] = np.einsum(
            "ij,ij->i", matrix[left[start:end]], matrix[right[start:end]]
        )
    return result


def structural_pair_feature_matrix(
    bundle: dict, pairs: np.ndarray, block_size: int = 20_000
) -> tuple[np.ndarray, list[str]]:
    """Compute the 29 rank-free structural pair features used by the shadow audit."""
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape (n, 2)")
    left, right = pairs[:, 0], pairs[:, 1]
    names = [
        name
        for group in TOKEN_GROUPS
        for name in (
            f"{group}_tfidf_cosine",
            f"{group}_set_jaccard",
            f"{group}_set_containment",
        )
    ]
    names.extend(
        (
            "interval_hist_cosine",
            "duration_hist_cosine",
            "ioi_hist_cosine",
            "chroma_transposition_cosine",
        )
    )
    scalar_names = (
        "notes",
        "onsets",
        "beats",
        "tracks",
        "pitch_min",
        "pitch_max",
        "pitch_mean",
        "pitch_sd",
        "programs",
        "half_beats",
    )
    names.extend(
        (
            f"{name}_absolute_difference"
            if name in {"pitch_min", "pitch_max", "pitch_mean", "pitch_sd"}
            else f"{name}_ratio"
        )
        for name in scalar_names
    )
    output = np.empty((len(pairs), len(names)), dtype=np.float32)
    column = 0
    for group in TOKEN_GROUPS:
        output[:, column] = _sparse_pair_cosine(bundle[group], left, right, block_size)
        jaccard, containment = _sparse_pair_set_overlap(
            bundle[group], left, right, block_size
        )
        output[:, column + 1] = jaccard
        output[:, column + 2] = containment
        column += 3
    for group in ("interval_hist", "duration_hist", "ioi_hist"):
        output[:, column] = _dense_pair_cosine(bundle[group], left, right, block_size)
        column += 1
    for start in range(0, len(left), block_size):
        end = min(len(left), start + block_size)
        a, b = bundle["chroma"][left[start:end]], bundle["chroma"][right[start:end]]
        output[start:end, column] = np.stack(
            [np.einsum("ij,ij->i", a, np.roll(b, shift, axis=1)) for shift in range(12)]
        ).max(axis=0)
    column += 1
    scalar_left, scalar_right = bundle["scalars"][left], bundle["scalars"][right]
    for index, name in enumerate(scalar_names):
        a, b = scalar_left[:, index], scalar_right[:, index]
        if name in {"pitch_min", "pitch_max", "pitch_mean", "pitch_sd"}:
            output[:, column] = np.abs(a - b)
        else:
            output[:, column] = np.minimum(a, b) / np.maximum(np.maximum(a, b), 1e-6)
        column += 1
    return output, names
