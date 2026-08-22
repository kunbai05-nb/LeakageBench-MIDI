from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mido
import numpy as np
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors


TOKEN_GROUPS = ("melody", "bass", "rhythm", "harmony", "motif")
SIGNALS = (*TOKEN_GROUPS, "interval_hist", "duration_hist", "ioi_hist", "chroma")
SCALARS = (
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


def _sampled_ngrams(values: list[int], n: int, limit: int = 2048):
    count = len(values) - n + 1
    if count <= 0:
        return
    step = max(1, math.ceil(count / limit))
    for index in range(0, count, step):
        yield tuple(values[index : index + n])


def _quantize(value: float) -> int:
    return int(np.clip(round(value * 12), 0, 192))


def _rotate(mask: int, shift: int) -> int:
    shift %= 12
    if shift == 0:
        return mask
    return ((mask << shift) | (mask >> (12 - shift))) & 0xFFF


_MASK_ROTATIONS = tuple(
    tuple(_rotate(mask, shift) for shift in range(12)) for mask in range(4096)
)


def _canonical_chroma(values: tuple[int, ...]) -> tuple[int, ...]:
    return min(
        tuple(_MASK_ROTATIONS[value][shift] for value in values) for shift in range(12)
    )


def _unit(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values)
    return values / norm if norm else values


def parse_midi(path: str | Path) -> dict:
    midi = mido.MidiFile(Path(path), clip=True)
    ticks_per_beat = max(1, midi.ticks_per_beat)
    notes = []
    melodic_tracks = 0

    for track_index, track in enumerate(midi.tracks):
        tick = 0
        programs = defaultdict(int)
        active = defaultdict(list)
        track_notes = 0
        for message in track:
            tick += int(message.time)
            if message.type == "program_change":
                programs[int(message.channel)] = int(message.program)
            elif message.type == "note_on" and message.velocity > 0:
                channel = int(message.channel)
                if channel == 9:
                    continue
                active[(channel, int(message.note))].append(
                    (tick, int(message.velocity), programs[channel])
                )
                track_notes += 1
            elif message.type in {"note_off", "note_on"}:
                channel = int(message.channel)
                if channel == 9:
                    continue
                pending = active[(channel, int(message.note))]
                if pending:
                    onset, velocity, program = pending.pop(0)
                    notes.append(
                        (
                            onset / ticks_per_beat,
                            max(1, tick - onset) / ticks_per_beat,
                            int(message.note),
                            velocity,
                            program,
                            track_index,
                        )
                    )
        melodic_tracks += bool(track_notes)
        for (_, pitch), pending in active.items():
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

    grouped = defaultdict(list)
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
    rhythm = [_quantize(b - a) for a, b in zip(onset_beats, onset_beats[1:])]

    tokens = {name: Counter() for name in TOKEN_GROUPS}
    for n in (2, 3, 4, 5):
        for values in _sampled_ngrams(melody_intervals, n):
            tokens["melody"][f"mi{n}:{','.join(map(str, values))}"] += 1
        for values in _sampled_ngrams(bass_intervals, n):
            tokens["bass"][f"bi{n}:{','.join(map(str, values))}"] += 1
    for n in (2, 3, 4):
        for values in _sampled_ngrams(rhythm, n):
            tokens["rhythm"][f"r{n}:{','.join(map(str, values))}"] += 1

    motif_count = min(len(melody_intervals), len(rhythm)) - 2
    motif_step = max(1, math.ceil(max(0, motif_count) / 2048))
    for index in range(0, max(0, motif_count), motif_step):
        values = tuple(
            (melody_intervals[index + offset], rhythm[index + offset])
            for offset in range(3)
        )
        tokens["motif"][f"mr3:{values}"] += 1

    half_beats = defaultdict(int)
    for onset, _, pitch, _, _, _ in notes:
        half_beats[round(onset * 2)] |= 1 << (pitch % 12)
    first = min(half_beats)
    last = min(max(half_beats), first + 2047)
    harmony = [half_beats.get(index, 0) for index in range(first, last + 1)]
    for n in (2, 4, 8):
        for values in _sampled_ngrams(harmony, n, limit=512):
            canonical = _canonical_chroma(values)
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
        "chroma": _unit(chroma),
        "interval_hist": _unit(interval_hist),
        "duration_hist": _unit(duration_hist),
        "ioi_hist": _unit(ioi_hist),
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
                len({note[4] for note in notes}),
                len(half_beats),
            ],
            dtype=np.float32,
        ),
    }


def _parse_one(task: tuple[int, str]):
    index, path = task
    try:
        return index, parse_midi(path), None
    except Exception as error:
        return index, None, f"{type(error).__name__}: {error}"


def extract_features(paths: list[Path], workers: int = 1, hash_width: int = 8192):
    parsed = [None] * len(paths)
    failures = []
    tasks = [(index, str(path)) for index, path in enumerate(paths)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            outputs = executor.map(_parse_one, tasks, chunksize=4)
            for index, item, error in outputs:
                parsed[index] = item
                if error:
                    failures.append(
                        {"index": index, "file": paths[index].name, "error": error}
                    )
    else:
        for task in tasks:
            index, item, error = _parse_one(task)
            parsed[index] = item
            if error:
                failures.append(
                    {"index": index, "file": paths[index].name, "error": error}
                )

    valid = np.asarray([item is not None for item in parsed])
    bundle = {"valid": valid}
    for name, width in (
        ("chroma", 12),
        ("interval_hist", 75),
        ("duration_hist", 16),
        ("ioi_hist", 16),
        ("scalars", 10),
    ):
        values = np.zeros((len(paths), width), dtype=np.float32)
        for index, item in enumerate(parsed):
            if item is not None:
                values[index] = item[name]
        bundle[name] = values

    hasher = FeatureHasher(
        n_features=hash_width, input_type="dict", alternate_sign=False
    )
    for group in TOKEN_GROUPS:
        counts = hasher.transform(
            [item["tokens"][group] if item is not None else {} for item in parsed]
        )
        bundle[group] = (
            TfidfTransformer(norm="l2", sublinear_tf=True)
            .fit_transform(counts)
            .astype(np.float32)
        )
    return bundle, failures


def candidate_ranks(bundle: dict, indices: np.ndarray, k: int) -> dict:
    matrices = {signal: bundle[signal] for signal in SIGNALS}
    matrices["chroma"] = np.stack(
        [
            np.roll(row, -int(np.argmax(row))) if np.any(row) else row
            for row in bundle["chroma"]
        ]
    ).astype(np.float32)
    batches = []
    for signal_index, signal in enumerate(SIGNALS):
        matrix = matrices[signal][indices]
        search = NearestNeighbors(
            n_neighbors=min(k + 1, len(indices)),
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        ).fit(matrix)
        distances, positions = search.kneighbors(matrix)
        left = np.repeat(indices, positions.shape[1])
        right = indices[positions.ravel()]
        distance = distances.ravel()
        keep = (left != right) & np.isfinite(distance)
        left, right = left[keep], right[keep]
        self_positions = positions == np.arange(len(indices))[:, None]
        ranks = np.cumsum(~self_positions, axis=1, dtype=np.int16).ravel()[keep]
        low, high = np.minimum(left, right), np.maximum(left, right)
        keys = low.astype(np.int64) * len(bundle["valid"]) + high.astype(np.int64)
        batches.append((signal_index, keys, (left > right).astype(np.int8), ranks))

    keys = np.unique(np.concatenate([batch[1] for batch in batches]))
    rank_values = np.full((len(keys), len(SIGNALS), 2), k + 1, dtype=np.int16)
    for signal_index, batch_keys, direction, ranks in batches:
        positions = np.searchsorted(keys, batch_keys)
        for side in (0, 1):
            mask = direction == side
            np.minimum.at(
                rank_values[:, signal_index, side], positions[mask], ranks[mask]
            )
    pairs = np.column_stack((keys // len(bundle["valid"]), keys % len(bundle["valid"])))
    return {"pairs": pairs.astype(np.int32), "ranks": rank_values, "k": k}


def _sparse_cosine(
    matrix: sparse.spmatrix, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    output = np.empty(len(left), dtype=np.float32)
    for start in range(0, len(left), 20_000):
        stop = min(start + 20_000, len(left))
        output[start:stop] = np.asarray(
            matrix[left[start:stop]].multiply(matrix[right[start:stop]]).sum(axis=1)
        ).ravel()
    return output


def _set_overlap(matrix: sparse.spmatrix, left: np.ndarray, right: np.ndarray):
    binary = matrix.copy().tocsr()
    binary.data.fill(1)
    counts = np.diff(binary.indptr).astype(np.float32)
    jaccard = np.empty(len(left), dtype=np.float32)
    containment = np.empty(len(left), dtype=np.float32)
    for start in range(0, len(left), 20_000):
        stop = min(start + 20_000, len(left))
        intersection = np.asarray(
            binary[left[start:stop]].multiply(binary[right[start:stop]]).sum(axis=1)
        ).ravel()
        left_count, right_count = counts[left[start:stop]], counts[right[start:stop]]
        union = left_count + right_count - intersection
        jaccard[start:stop] = intersection / np.maximum(union, 1)
        containment[start:stop] = intersection / np.maximum(
            np.minimum(left_count, right_count), 1
        )
    return jaccard, containment


def pair_features(bundle: dict, candidates: dict):
    pairs = candidates["pairs"]
    left, right = pairs[:, 0], pairs[:, 1]
    columns = []
    names = []

    def add(name: str, values):
        names.append(name)
        columns.append(np.asarray(values, dtype=np.float32))

    for group in TOKEN_GROUPS:
        add(f"{group}_tfidf_cosine", _sparse_cosine(bundle[group], left, right))
        jaccard, containment = _set_overlap(bundle[group], left, right)
        add(f"{group}_set_jaccard", jaccard)
        add(f"{group}_set_containment", containment)
    for group in ("interval_hist", "duration_hist", "ioi_hist"):
        add(
            f"{group}_cosine",
            np.einsum("ij,ij->i", bundle[group][left], bundle[group][right]),
        )

    chroma_left, chroma_right = bundle["chroma"][left], bundle["chroma"][right]
    add(
        "chroma_transposition_cosine",
        np.stack(
            [
                np.einsum("ij,ij->i", chroma_left, np.roll(chroma_right, shift, axis=1))
                for shift in range(12)
            ]
        ).max(axis=0),
    )

    scalar_left, scalar_right = bundle["scalars"][left], bundle["scalars"][right]
    for index, name in enumerate(SCALARS):
        a, b = scalar_left[:, index], scalar_right[:, index]
        if name.startswith("pitch_"):
            add(f"{name}_absolute_difference", np.abs(a - b))
        else:
            add(f"{name}_ratio", np.minimum(a, b) / np.maximum(np.maximum(a, b), 1e-6))

    ranks = candidates["ranks"]
    for signal_index, signal in enumerate(SIGNALS):
        left_rank, right_rank = ranks[:, signal_index, 0], ranks[:, signal_index, 1]
        add(f"{signal}_reciprocal_best_rank", 1 / np.minimum(left_rank, right_rank))
        add(f"{signal}_mutual", np.maximum(left_rank, right_rank) <= candidates["k"])
    return np.column_stack(columns), names


class _Components:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int32)
        self.size = np.ones(size, dtype=np.int32)

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[item] != item:
            parent = int(self.parent[item])
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]

    def labels(self) -> np.ndarray:
        roots = np.asarray([self.find(index) for index in range(len(self.parent))])
        return np.unique(roots, return_inverse=True)[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_detector(artifact_dir: str | Path):
    artifact_dir = Path(artifact_dir)
    config = json.loads((artifact_dir / "config.json").read_text())
    model_path = artifact_dir / config["model"]["artifact"]
    if _sha256(model_path) != config["model"]["sha256"]:
        raise ValueError("detector model hash mismatch")
    return config, np.load(model_path)


def _predict_probability(model, features: np.ndarray) -> np.ndarray:
    raw = np.full(len(features), model["baseline"][0], dtype=np.float64)
    for start, end in zip(model["offsets"][:-1], model["offsets"][1:]):
        node = np.zeros(len(features), dtype=np.int16)
        while True:
            absolute = start + node
            active = ~model["leaf"][absolute]
            if not np.any(active):
                break
            rows = np.flatnonzero(active)
            absolute = absolute[active]
            values = features[rows, model["feature"][absolute]]
            go_left = np.where(
                np.isnan(values),
                model["missing_left"][absolute],
                values <= model["threshold"][absolute],
            )
            node[rows] = np.where(
                go_left, model["left"][absolute], model["right"][absolute]
            )
        raw += model["value"][start + node]
    return 1 / (1 + np.exp(-raw))


def detect(paths: list[str | Path], artifact_dir: str | Path, workers: int = 1) -> dict:
    paths = [Path(path) for path in paths]
    config, model = load_detector(artifact_dir)
    bundle, failures = extract_features(paths, workers=workers)
    valid = np.flatnonzero(bundle["valid"])
    if len(valid) < 2:
        raise ValueError("at least two valid MIDI files are required")
    candidates = candidate_ranks(bundle, valid, config["candidate_k"])
    features, names = pair_features(bundle, candidates)
    if names != config["selected_feature_names"]:
        raise ValueError("detector feature schema mismatch")

    scores = _predict_probability(model, features)
    mutual_columns = [names.index(f"{signal}_mutual") for signal in SIGNALS]
    mutual = features[:, mutual_columns].sum(axis=1).astype(np.int16)
    selected = np.flatnonzero(
        (scores >= config["threshold"])
        & (mutual >= config["minimum_mutual_structural_signals"])
    )
    selected = selected[np.argsort(scores[selected], kind="mergesort")[::-1]]

    components = _Components(len(paths))
    accepted = []
    merged = []
    rejected = []
    for index in selected:
        left, right = map(int, candidates["pairs"][index])
        left_root, right_root = components.find(left), components.find(right)
        if left_root == right_root:
            accepted.append(index)
            continue
        if (
            components.size[left_root] + components.size[right_root]
            > config["maximum_component_size"]
        ):
            rejected.append(index)
            continue
        components.union(left, right)
        accepted.append(index)
        merged.append(index)

    return {
        "paths": paths,
        "pairs": candidates["pairs"],
        "scores": scores,
        "mutual_support": mutual,
        "selected": np.asarray(selected, dtype=np.int64),
        "accepted": np.asarray(accepted, dtype=np.int64),
        "merged": np.asarray(merged, dtype=np.int64),
        "rejected_by_size": np.asarray(rejected, dtype=np.int64),
        "component_labels": components.labels(),
        "failures": failures,
    }
