from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_all_start_methods, get_context

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from .local_evidence import _tokens

_STATE = None


def encode_token(token):
    voice = 0 if token[0] == 'bass' else 1
    kind = 0 if token[1] == 'joint' else 1
    pitch = (int(token[2]) + 24) * 49 ** 2 + (int(token[3]) + 24) * 49 + int(token[4]) + 24
    rhythm = sum((int(value) + 6) * 13 ** (2 - i) for i, value in enumerate(token[5:]))
    return (voice * 2 + kind) * 49 ** 3 * 13 ** 3 + pitch * 13 ** 3 + rhythm


def _local_query(owner):
    matrix, owners, boundaries, config = _STATE
    start, end = boundaries[owner:owner + 2]
    if start == end:
        return []
    matches = (matrix[start:end] @ matrix.T).tocoo()
    keep = (owners[matches.col] != owner) & (matches.data >= config.min_window_similarity - 1e-12)
    docs, values = owners[matches.col[keep]], matches.data[keep]
    best = np.zeros(len(boundaries) - 1)
    np.maximum.at(best, docs, values)
    candidates = np.flatnonzero(best >= config.min_window_similarity - 1e-12)
    order = np.lexsort((candidates, -np.round(best[candidates], 12)))[:config.neighbours_per_file]
    return [(min(owner, int(other)), max(owner, int(other)), float(best[other])) for other in candidates[order]]


def sparse_local_candidates(sequences, config, workers=4, progress=None):
    token_arrays, owners, boundaries = [], [], [0]
    per_scale = max(1, config.windows_per_file // len(config.window_segments))
    for owner, sequence in enumerate(sequences):
        if sequence is not None:
            for width in config.window_segments:
                length = len(sequence['density'])
                starts = np.arange(0, max(0, length - width + 1), max(1, width // 2))
                if len(starts) > per_scale:
                    starts = starts[np.linspace(0, len(starts) - 1, per_scale).astype(int)]
                for start in starts:
                    token_arrays.append(np.array(sorted(encode_token(token) for token in _tokens(sequence, start, start + width)), dtype=np.uint32))
                    owners.append(owner)
        boundaries.append(len(owners))
        if progress is not None and (owner + 1) % 256 == 0:
            progress('local_windows', owner + 1, len(sequences))
    owners = np.array(owners, dtype=np.int32)
    boundaries = np.array(boundaries, dtype=np.int32)
    lengths = np.array([len(values) for values in token_arrays], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths)]
    flat = np.concatenate(token_arrays) if token_arrays else np.empty(0, np.uint32)
    del token_arrays
    vocabulary, inverse, occurrences = np.unique(flat, return_inverse=True, return_counts=True)
    del flat
    file_tokens = inverse.astype(np.int64) * max(1, len(sequences)) + np.repeat(owners, lengths)
    unique_file_tokens = np.unique(file_tokens)
    df = np.bincount(unique_file_tokens // max(1, len(sequences)), minlength=len(vocabulary))
    del file_tokens, unique_file_tokens
    allowed = (df >= 2) & (df <= max(2, int(len(sequences) * config.max_document_frequency))) & (occurrences <= config.max_token_postings)
    weights = (np.log((1 + len(sequences)) / (1 + df)) + 1) ** 2
    selected_columns, values, rowptr = [], [], [0]
    for start, end in zip(offsets[:-1], offsets[1:]):
        columns = inverse[start:end]
        columns = columns[allowed[columns]]
        order = np.lexsort((vocabulary[columns], -weights[columns]))[:config.tokens_per_window]
        columns = columns[order]
        local_weights = np.sqrt(weights[columns])
        norm = np.linalg.norm(local_weights)
        selected_columns.extend(columns)
        values.extend(local_weights / max(norm, 1e-12))
        rowptr.append(len(values))
    del inverse, offsets, lengths
    matrix = sparse.csr_matrix((np.asarray(values), np.asarray(selected_columns, np.int32), np.asarray(rowptr, np.int64)), shape=(len(owners), len(vocabulary)))
    del selected_columns, values, rowptr, weights, df, occurrences, vocabulary
    matrix.sort_indices()
    global _STATE
    _STATE = matrix, owners, boundaries, config
    pairs = {}
    try:
        if workers > 1 and 'fork' in get_all_start_methods():
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=get_context('fork'))
        else:
            pool = ThreadPoolExecutor(max_workers=max(1, workers))
        with pool:
            for i, result in enumerate(pool.map(_local_query, range(len(sequences)), chunksize=16)):
                for a, b, score in result:
                    pairs[a, b] = max(pairs.get((a, b), 0), score)
                if progress is not None and (i + 1) % 256 == 0:
                    progress('local_retrieval', i + 1, len(sequences))
    finally:
        _STATE = None
    keys = sorted(pairs)
    return np.asarray(keys, dtype=np.int64).reshape(-1, 2), np.asarray([pairs[key] for key in keys], np.float32), {
        'windows': len(owners), 'sparse_coordinates': matrix.nnz, 'pairs': len(keys),
        'tie_rounding_decimals': 12}


def _exact_signal(name):
    matrices, valid, k, batch = _STATE
    matrix = matrices[name][valid]
    index = NearestNeighbors(n_neighbors=min(k + 1, len(valid)), metric='cosine', algorithm='brute', n_jobs=1).fit(matrix)
    neighbours = np.full((len(valid), k), -1, dtype=np.int32)
    for start in range(0, len(valid), batch):
        distances, positions = index.kneighbors(matrix[start:start + batch])
        for offset, (distance, row) in enumerate(zip(distances, positions)):
            row = row[(row != start + offset) & np.isfinite(distance)][:k]
            neighbours[start + offset, :len(row)] = valid[row]
    return name, neighbours


def blocked_exact_ranks(matrices, valid, k, workers=3, batch=128, progress=None):
    global _STATE
    _STATE = matrices, valid, k, batch
    output = {}
    try:
        if workers > 1 and 'fork' in get_all_start_methods():
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=get_context('fork'))
        else:
            pool = ThreadPoolExecutor(max_workers=max(1, workers))
        with pool:
            for name, values in pool.map(_exact_signal, list(matrices)):
                output[name] = values
                if progress is not None:
                    progress('global_views', len(output), len(matrices))
    finally:
        _STATE = None
    return output


def ranks_for_pairs(pairs, neighbours, valid, size, signals, k):
    wanted = pairs[:, 0].astype(np.int64) * size + pairs[:, 1]
    reverse = pairs[:, 1].astype(np.int64) * size + pairs[:, 0]
    ranks = np.full((len(pairs), len(signals), 2), k + 1, dtype=np.int16)
    for signal_index, name in enumerate(signals):
        array = neighbours[name]
        left = np.repeat(valid, array.shape[1])
        right = array.ravel()
        present = right >= 0
        keys = left[present].astype(np.int64) * size + right[present]
        values = np.tile(np.arange(1, array.shape[1] + 1, dtype=np.int16), len(valid))[present]
        order = np.argsort(keys)
        keys, values = keys[order], values[order]
        for direction, query in enumerate((wanted, reverse)):
            positions = np.searchsorted(keys, query)
            found = positions < len(keys)
            found[found] &= keys[positions[found]] == query[found]
            ranks[found, signal_index, direction] = values[positions[found]]
    return ranks
