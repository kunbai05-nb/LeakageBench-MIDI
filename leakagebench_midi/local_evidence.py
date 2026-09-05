from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice

import numpy as np

from .alignment import (
    ALIGNMENT_FEATURE_NAMES, AlignmentConfig, _smith_waterman_affine,
    _transposition_scores, _unit_rows, _view_matrices,
)


SIGNALS = ('melody', 'bass', 'rhythm', 'harmony', 'motif',
           'interval_hist', 'duration_hist', 'ioi_hist', 'chroma')
EVIDENCE_NAMES = (
    'local_contrast_score', 'local_contrast_per_match', 'local_coverage_short',
    'local_coverage_long', 'local_coverage_hmean', 'local_path_count',
    'local_gap_fraction', 'local_mean_similarity', 'local_mean_background',
    'local_melody_agreement', 'local_rhythm_agreement', 'local_harmony_agreement',
    'local_background_fraction', 'local_ordered_support', 'local_shift_margin',
    'local_shift_rank', 'local_extra_shift_selected', 'local_missing_sequence',
)


@dataclass(frozen=True)
class LocalEvidenceConfig:
    window_segments: tuple[int, ...] = (4, 8)
    windows_per_file: int = 24
    tokens_per_window: int = 32
    max_token_postings: int = 96
    max_document_frequency: float = 0.12
    neighbours_per_file: int = 32
    min_window_similarity: float = 0.20
    pairs_per_file: int = 24
    rescue_fraction: float = 0.25
    candidate_k: int = 100
    minimum_mutual_views: int = 3
    evidence_max_segments: int = 96
    background_quantile: float = 0.75
    context_weight: float = 0.5
    initial_shifts: int = 3
    maximum_shifts: int = 6
    shift_boundary_margin: float = 0.015

    def __post_init__(self):
        if not self.window_segments or min(self.window_segments) < 2:
            raise ValueError('window sizes must be at least two segments')
        if self.windows_per_file < len(self.window_segments):
            raise ValueError('window budget must cover every scale')
        for name in ('windows_per_file', 'tokens_per_window', 'max_token_postings',
                     'neighbours_per_file', 'pairs_per_file', 'candidate_k'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        if self.evidence_max_segments < 4:
            raise ValueError('evidence_max_segments must be at least four')
        if not 1 <= self.initial_shifts <= self.maximum_shifts <= 12:
            raise ValueError('shift limits must lie in [1, 12]')
        if not 1 <= self.minimum_mutual_views <= len(SIGNALS):
            raise ValueError('invalid mutual-view requirement')
        for name in ('max_document_frequency', 'min_window_similarity',
                     'rescue_fraction', 'background_quantile', 'context_weight'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must lie in [0, 1]')
        if self.shift_boundary_margin < 0:
            raise ValueError('shift_boundary_margin must be nonnegative')


def _tokens(sequence: dict, start: int, end: int) -> set[tuple]:
    tokens = set()
    for voice in ('melody', 'bass'):
        values = sequence[voice][start:end].ravel()
        positions = np.flatnonzero(np.isfinite(values))
        pitches = values[positions]
        if len(pitches) < 4:
            continue
        keep = np.r_[True, np.diff(pitches) != 0]
        positions, pitches = positions[keep], pitches[keep]
        intervals = np.clip(np.diff(pitches), -24, 24).astype(int)
        gaps = np.diff(positions).astype(float)
        for i in range(len(intervals) - 2):
            pattern = tuple(intervals[i:i + 3])
            tokens.add((voice, 'pitch', *pattern))
            relative = gaps[i:i + 3] / max(float(np.median(gaps[i:i + 3])), 1)
            rhythm = tuple(np.clip(np.rint(2 * np.log2(relative)), -6, 6).astype(int))
            tokens.add((voice, 'joint', *pattern, *rhythm))
    return tokens


def local_candidate_pairs(sequences: list[dict | None], indices: np.ndarray,
                          config: LocalEvidenceConfig = LocalEvidenceConfig()
                          ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sparse window search; labels never enter the index or candidate budget."""
    indices = np.unique(np.asarray(indices, dtype=np.int64))
    if len(indices) and (indices[0] < 0 or indices[-1] >= len(sequences)):
        raise ValueError('sequence index out of bounds')
    windows, owners = [], []
    per_scale = max(1, config.windows_per_file // len(config.window_segments))
    for index in indices:
        sequence = sequences[int(index)]
        if sequence is None:
            continue
        length = len(sequence['density'])
        for width in config.window_segments:
            if length < width:
                continue
            starts = np.arange(0, length - width + 1, max(1, width // 2))
            if len(starts) > per_scale:
                starts = starts[np.linspace(0, len(starts) - 1, per_scale).astype(int)]
            for start in starts:
                windows.append(_tokens(sequence, int(start), int(start + width)))
                owners.append(int(index))
    documents = defaultdict(set)
    occurrences = Counter()
    for owner, tokens in zip(owners, windows):
        for token in tokens:
            documents[token].add(owner)
            occurrences[token] += 1
    max_df = max(2, int(len(indices) * config.max_document_frequency))
    weights = {token: float(np.log((1 + len(indices)) / (1 + len(docs))) + 1) ** 2
               for token, docs in documents.items()
               if 2 <= len(docs) <= max_df and occurrences[token] <= config.max_token_postings}
    postings = defaultdict(list)
    norms = np.zeros(len(windows), dtype=float)
    for window, tokens in enumerate(windows):
        selected = sorted((token for token in tokens if token in weights),
                          key=lambda token: (-weights[token], token))[:config.tokens_per_window]
        windows[window] = selected
        norms[window] = np.sqrt(sum(weights[token] for token in selected))
        for token in selected:
            postings[token].append(window)
    by_owner = defaultdict(list)
    for window, owner in enumerate(owners):
        by_owner[owner].append(window)
    best_pairs = {}
    for owner in sorted(by_owner):
        best_documents = {}
        for window in by_owner[owner]:
            scores = defaultdict(float)
            for token in windows[window]:
                for other in postings[token]:
                    if owners[other] != owner:
                        scores[other] += weights[token]
            for other, overlap in scores.items():
                similarity = overlap / max(norms[window] * norms[other], 1e-12)
                candidate = owners[other]
                if similarity >= config.min_window_similarity:
                    best_documents[candidate] = max(best_documents.get(candidate, 0), similarity)
        ranked = sorted(best_documents.items(), key=lambda item: (-item[1], item[0]))
        for candidate, score in islice(ranked, config.neighbours_per_file):
            pair = (min(owner, candidate), max(owner, candidate))
            best_pairs[pair] = max(best_pairs.get(pair, 0), score)
    pairs = np.asarray(sorted(best_pairs), dtype=np.int64).reshape(-1, 2)
    scores = np.asarray([best_pairs[tuple(pair)] for pair in pairs], dtype=np.float32)
    stats = {'files': len(indices), 'windows': len(windows), 'retained_tokens': len(postings),
             'candidate_pairs': len(pairs), 'maximum_postings': max(map(len, postings.values()), default=0)}
    return pairs, scores, stats


def rank_features(ranks: np.ndarray, k: int) -> np.ndarray:
    ranks = np.asarray(ranks)
    if k < 1 or ranks.ndim != 3 or ranks.shape[1:] != (len(SIGNALS), 2) or not np.isfinite(ranks).all() or np.any(ranks < 1):
        raise ValueError('ranks must be positive and have shape (pairs, 9, 2)')
    clipped = np.minimum(ranks, k + 1)
    reciprocal = 1 / np.min(clipped, axis=2)
    mutual = np.max(clipped, axis=2) <= k
    return np.stack((reciprocal, mutual), axis=2).reshape(len(ranks), 18).astype(np.float32)


def select_candidates(pairs: np.ndarray, ranks: np.ndarray, files: int,
                      local_pairs: np.ndarray, local_scores: np.ndarray,
                      config: LocalEvidenceConfig = LocalEvidenceConfig()) -> dict:
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(pairs) != len(ranks) or len(local_pairs) != len(local_scores):
        raise ValueError('candidate arrays have inconsistent lengths')
    if files < 0 or np.any(pairs[:, 0] >= pairs[:, 1]) or len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError('global candidates must be unique ordered non-self pairs')
    if not np.isfinite(local_scores).all():
        raise ValueError('local candidate scores must be finite')
    features = rank_features(ranks, config.candidate_k)
    support = features[:, 1::2].sum(axis=1)
    priority = support + features[:, ::2].mean(axis=1)
    eligible = np.flatnonzero(support >= config.minimum_mutual_views)
    order = sorted(eligible, key=lambda i: (-float(priority[i]), int(pairs[i, 0]), int(pairs[i, 1])))
    budget = min(len(order), files * config.pairs_per_file)
    baseline = [tuple(map(int, pairs[i])) for i in order[:budget]]
    reserve = int(budget * config.rescue_fraction)
    selected = dict.fromkeys(baseline[:budget - reserve])
    additions = 0
    local_order = sorted(range(len(local_pairs)),
                         key=lambda i: (-float(local_scores[i]), *map(int, local_pairs[i])))
    for i in local_order:
        if additions >= reserve or len(selected) >= budget:
            break
        pair = tuple(sorted(map(int, local_pairs[i])))
        if pair[0] != pair[1] and pair not in selected:
            selected[pair] = None
            additions += 1
    for pair in baseline:
        if len(selected) >= budget:
            break
        selected[pair] = None
    proposed = sorted(selected)
    return {
        'baseline': np.asarray(sorted(baseline), dtype=np.int64).reshape(-1, 2),
        'proposed': np.asarray(proposed, dtype=np.int64).reshape(-1, 2),
        'eligible_pairs': len(eligible), 'budget': budget,
        'new_pairs': len(set(proposed) - set(baseline)),
    }


def _pool(sequence: dict, factor: int) -> dict:
    if factor == 1:
        return sequence
    output = {}
    for name in ('melody', 'bass', 'rhythm', 'chroma', 'density'):
        values = sequence[name]
        starts = np.arange(0, len(values), factor)
        finite = np.isfinite(values)
        count = np.add.reduceat(finite.astype(np.int32), starts, axis=0)
        sums = np.add.reduceat(np.where(finite, values, 0), starts, axis=0)
        output[name] = (sums / np.maximum(count, 1)).astype(np.float32)
        if name in ('melody', 'bass'):
            output[name] = np.where(count, output[name], np.nan)
        if name in ('rhythm', 'chroma'):
            output[name] = _unit_rows(output[name])
    return output


def contrast_scores(similarity: np.ndarray, config: LocalEvidenceConfig
                    ) -> tuple[np.ndarray, np.ndarray]:
    similarity = np.asarray(similarity, dtype=np.float32)
    if similarity.ndim != 2 or not np.isfinite(similarity).all():
        raise ValueError('similarity must be a finite matrix')
    if not similarity.size:
        return similarity.copy(), similarity.copy()
    context = similarity.copy()
    counts = np.ones_like(similarity)
    context[1:, 1:] += similarity[:-1, :-1]
    counts[1:, 1:] += 1
    context[:-1, :-1] += similarity[1:, 1:]
    counts[:-1, :-1] += 1
    local = ((1 - config.context_weight) * similarity
             + config.context_weight * context / counts)
    row = np.quantile(local, config.background_quantile, axis=1, keepdims=True)
    column = np.quantile(local, config.background_quantile, axis=0, keepdims=True)
    background = np.maximum(AlignmentConfig().match_baseline, np.maximum(row, column))
    return (2 * (local - background)).astype(np.float32), background.astype(np.float32)


def _paths(scores: np.ndarray, config: AlignmentConfig) -> list[dict]:
    available = scores.copy()
    paths = []
    for _ in range(config.max_paths):
        gain, left, right = _smith_waterman_affine(available, config.gap_open, config.gap_extend)
        if len(left) < config.min_path_matches or gain <= 0:
            break
        paths.append({'score': float(gain), 'left': left, 'right': right})
        available[left, :] = -1e6
        available[:, right] = -1e6
    return paths


def local_evidence_features(left: dict | None, right: dict | None,
                            config: LocalEvidenceConfig = LocalEvidenceConfig()) -> np.ndarray:
    empty = np.zeros(len(EVIDENCE_NAMES), dtype=np.float32)
    if left is None or right is None or not len(left['density']) or not len(right['density']):
        empty[-1] = 1
        return empty
    factor = max(1, int(np.ceil(max(len(left['density']), len(right['density']))
                               / config.evidence_max_segments)))
    left, right = _pool(left, factor), _pool(right, factor)
    transpose = _transposition_scores(left, right)
    order = np.argsort(-transpose, kind='stable')
    count = config.initial_shifts
    if count < config.maximum_shifts and transpose[order[count - 1]] - transpose[order[count]] <= config.shift_boundary_margin:
        count = config.maximum_shifts
    candidates = []
    alignment = AlignmentConfig()
    for rank, shift in enumerate(order[:count]):
        views = _view_matrices(left, right, int(shift if shift <= 6 else shift - 12))
        scores, background = contrast_scores(views['combined'], config)
        paths = _paths(scores, alignment)
        gain = sum(path['score'] for path in paths)
        candidates.append((gain, -rank, views, background, paths))
    best = max(candidates, key=lambda item: item[:2])
    gain, negative_rank, views, background, paths = best
    if not paths:
        return empty
    a = np.concatenate([path['left'] for path in paths])
    b = np.concatenate([path['right'] for path in paths])
    lengths = np.asarray([len(path['left']) for path in paths], dtype=float)
    coverage = (len(np.unique(a)) / len(left['density']), len(np.unique(b)) / len(right['density']))
    short, long = coverage if len(left['density']) <= len(right['density']) else coverage[::-1]
    gaps = []
    for path in paths:
        span = np.ptp(path['left']) + np.ptp(path['right']) + 2
        gaps.append(max(0, span - 2 * len(path['left'])) / span)
    comparison = sorted((item[0] for item in candidates), reverse=True)
    raw, bg = views['combined'][a, b], background[a, b]
    return np.asarray([
        gain / max(1, min(len(left['density']), len(right['density']))),
        gain / len(a), short, long,
        2 * coverage[0] * coverage[1] / max(sum(coverage), 1e-8),
        len(paths), np.average(gaps, weights=lengths), raw.mean(), bg.mean(),
        views['melody'][a, b].mean(), views['rhythm'][a, b].mean(),
        views['harmony'][a, b].mean(), (background > alignment.match_baseline + .05).mean(),
        np.mean(raw > bg), (gain - comparison[1]) / max(gain, 1e-8) if len(comparison) > 1 else 1,
        -negative_rank + 1, int(-negative_rank >= config.initial_shifts), 0,
    ], dtype=np.float32)


def evidence_feature_matrix(sequences: list[dict | None], pairs: np.ndarray,
                             config: LocalEvidenceConfig = LocalEvidenceConfig(), workers: int = 1,
                             progress=None) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(pairs) and (pairs.min() < 0 or pairs.max() >= len(sequences)):
        raise ValueError('pair index out of bounds')
    output = np.empty((len(pairs), len(EVIDENCE_NAMES)), dtype=np.float32)
    def compute(pair):
        return local_evidence_features(sequences[int(pair[0])], sequences[int(pair[1])], config)
    if workers <= 1:
        for i, pair in enumerate(pairs):
            output[i] = compute(pair)
            if progress is not None and (i + 1) % 256 == 0:
                progress(i + 1, len(pairs))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for start in range(0, len(pairs), 256):
                stop = min(start + 256, len(pairs))
                output[start:stop] = list(executor.map(compute, pairs[start:stop]))
                if progress is not None:
                    progress(stop, len(pairs))
    return output


def assemble_base_features(structural: np.ndarray, structural_names: list[str],
                            ranks: np.ndarray, aligned: np.ndarray, k: int = 100
                            ) -> tuple[np.ndarray, list[str]]:
    ranked = rank_features(ranks, k)
    rank_names = [f'{signal}_{suffix}' for signal in SIGNALS
                  for suffix in ('reciprocal_best_rank', 'mutual')]
    values = np.column_stack((structural, ranked, aligned)).astype(np.float32)
    names = structural_names + rank_names + list(ALIGNMENT_FEATURE_NAMES)
    at = {name: i for i, name in enumerate(names)}
    agreement = values[:, [at[f'align_{voice}_agreement']
                           for voice in ('melody', 'bass', 'rhythm', 'harmony')]]
    best = values[:, at['align_best_score_per_match']]
    engineered = np.column_stack((agreement.min(axis=1), agreement.mean(axis=1),
                                  best * values[:, at['align_coverage_hmean']],
                                  best * (1 - values[:, at['align_gap_fraction']]),
                                  ranked[:, 1::2].sum(axis=1), ranked[:, ::2].max(axis=1)))
    names += ['robust_alignment_agreement_min', 'robust_alignment_agreement_mean',
              'robust_alignment_score_coverage', 'robust_alignment_score_gap_adjusted',
              'robust_structural_mutual_support', 'robust_structural_candidate_support']
    return np.nan_to_num(np.column_stack((values, engineered)), nan=0., posinf=1., neginf=0.).astype(np.float32), names
