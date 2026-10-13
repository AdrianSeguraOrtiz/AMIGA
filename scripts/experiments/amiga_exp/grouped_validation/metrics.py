"""Front metrics and topology aggregation for the grouped evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from amiga.selection.learn2rank import _tie_aware_topk_best_and_hit

METRICS = tuple(f'{name}@{k}' for name in ('Regret', 'BestAUPR', 'Hit') for k in (1, 3, 5, 10))


def front_metrics(frame: pd.DataFrame, scores, *, ks=(1, 3, 5, 10)) -> pd.DataFrame:
    """Evaluate uniform score ties, including k greater than the front size."""
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(frame),) or not np.isfinite(scores).all():
        raise ValueError('Scores must be finite and aligned with candidate rows')
    if frame.empty or frame.duplicated(['front_id', 'item_id']).any():
        raise ValueError('Evaluation needs nonempty fronts with unique candidate IDs')
    if any(not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError('Evaluation cutoffs must be positive integers')
    data = frame[['front_id', 'item_id', 'AUPR']].copy()
    data['score'] = scores
    if not np.isfinite(data['AUPR'].to_numpy(dtype=float)).all():
        raise ValueError('Evaluation targets must be finite')
    rows = []
    for front_id, group in data.groupby('front_id', sort=True):
        target = group['AUPR'].to_numpy(dtype=float)
        score = group['score'].to_numpy(dtype=float)
        best = float(target.max())
        row = dict(front_id=int(front_id), n_items=len(group))
        for k in ks:
            expected, hit = _tie_aware_topk_best_and_hit(target, score, k=min(k, len(group)), best_true=best)
            row.update({f'Regret@{k}': max(0.0, best - expected),
                        f'BestAUPR@{k}': expected, f'Hit@{k}': hit})
        rows.append(row)
    return pd.DataFrame(rows)


def topology_metrics(fronts: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Average conditions within topology after callers average training seeds."""
    if fronts.empty or fronts['front_id'].duplicated().any():
        raise ValueError('Exactly one metric row per front is required')
    data = fronts.copy()
    data['topology_id'] = data['front_id'].map(lambda f: mapping.get(str(int(f))))
    if data['topology_id'].isna().any():
        raise ValueError('Evaluation contains unmapped topologies')
    columns = [m for m in METRICS if m in data]
    if not columns or not np.isfinite(data[columns].to_numpy()).all():
        raise ValueError('Metric values must be present and finite')
    return data.groupby('topology_id', sort=True)[columns].mean().reset_index()


def select_configuration(candidates: dict[str, pd.DataFrame], mapping: dict) -> tuple[str, list[dict]]:
    """Choose solely from inner predictions, with equal weight per topology."""
    evidence, reference = [], None
    for identifier, frame in sorted(candidates.items()):
        fronts = set(frame['front_id'])
        if reference is not None and fronts != reference:
            raise ValueError('Configurations must cover the same inner validation fronts')
        reference = fronts
        means = topology_metrics(frame, mapping)[['Regret@5', 'Regret@1']].mean()
        evidence.append({'config_id': identifier, **{m: float(v) for m, v in means.items()}})
    if not evidence:
        raise ValueError('No complete configurations are available')
    winner = min(evidence, key=lambda r: (round(r['Regret@5'], 12), round(r['Regret@1'], 12), r['config_id']))
    return winner['config_id'], evidence
