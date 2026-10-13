"""Objective-only rankings and explicit random/oracle evaluation references."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.experiments.amiga_exp.decision_baselines import (
    normalized_objective_badness_matrix, weighted_sum_scores_from_badness,
    ideal_l2_scores_from_badness, topsis_scores_from_badness,
    vikor_scores_from_badness, augmented_tchebycheff_scores_from_badness,
)

HEURISTICS = ('objective_mean_rank', 'objective_normalized_mean', 'objective_ideal_l2',
              'objective_topsis', 'objective_vikor', 'objective_augmented_tchebycheff',
              'objective_tchebycheff', 'objective_knee', 'random_uniform')


def baseline_ids(objectives):
    return [*(f'objective__{o}' for o in objectives), *HEURISTICS, 'oracle']


def knee_scores(badness: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Rank global trade-off worthiness; encode infinite values as finite ranks.

    This monotone encoding preserves all score ties and avoids passing infinity
    to metrics. Identical vectors share a score; completely flat fronts tie.
    """
    matrix = badness.to_numpy(dtype=float)
    if matrix.ndim != 2 or len(matrix) == 0 or not np.isfinite(matrix).all():
        raise ValueError('Knee scoring requires a finite nonempty objective matrix')
    variable = np.ptp(matrix, axis=0) > 0
    matrix = matrix[:, variable]
    worth = np.full(len(matrix), np.inf)
    dominated = 0
    for i, point in enumerate(matrix):
        delta = matrix - point
        improvement = np.maximum(-delta, 0).sum(axis=1)
        deterioration = np.maximum(delta, 0).sum(axis=1)
        usable = improvement > 0
        if usable.any():
            worth[i] = np.min(deterioration[usable] / improvement[usable])
        dominated += int(np.any(usable & (deterioration == 0)))
    score = np.searchsorted(np.unique(worth), worth).astype(float)
    return score, {'constant_objectives': int((~variable).sum()),
                   'duplicate_vectors': int(len(matrix) - len(np.unique(matrix, axis=0))),
                   'dominated_candidates': dominated, 'infinite_worthiness': int(np.isinf(worth).sum()),
                   'score_encoding': 'ascending dense ranks of global trade-off worthiness'}


def score_front(frame: pd.DataFrame, objectives: list[str], directions: dict, *, include_oracle=False):
    """Return aligned scores for exactly one front; labels are optional."""
    if frame.empty or frame['front_id'].nunique() != 1:
        raise ValueError('Baseline scoring expects exactly one nonempty front')
    if not np.isfinite(frame[objectives].to_numpy(dtype=float)).all():
        raise ValueError('Objective values must be finite')
    badness = normalized_objective_badness_matrix(frame, objective_columns=objectives,
                                                 objective_directions=directions, front_col='front_id')
    scores = {f'objective__{o}': frame[o].to_numpy(dtype=float) * (-1 if directions[o] == 'minimize' else 1)
              for o in objectives}
    ranks = [frame[o].rank(method='average', ascending=directions[o] == 'minimize') for o in objectives]
    scores['objective_mean_rank'] = -pd.concat(ranks, axis=1).mean(axis=1).to_numpy()
    for name, function in (
        ('objective_normalized_mean', weighted_sum_scores_from_badness),
        ('objective_ideal_l2', ideal_l2_scores_from_badness),
        ('objective_topsis', topsis_scores_from_badness),
        ('objective_vikor', vikor_scores_from_badness),
        ('objective_augmented_tchebycheff', augmented_tchebycheff_scores_from_badness),
    ):
        scores[name] = function(badness).to_numpy()
    scores['objective_tchebycheff'] = -badness.max(axis=1).to_numpy()
    scores['objective_knee'], diagnostic = knee_scores(badness)
    scores['random_uniform'] = np.zeros(len(frame))
    if include_oracle:
        scores['oracle'] = frame['AUPR'].to_numpy(dtype=float)
    if any(not np.isfinite(values).all() for values in scores.values()):
        raise ValueError('A baseline produced invalid scores')
    return scores, diagnostic
