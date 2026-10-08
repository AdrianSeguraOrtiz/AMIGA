"""Change the binary target while retaining the existing native tree adapters."""
from __future__ import annotations

import numpy as np

from scripts.experiments.amiga_exp.sequential_selection import models as original

ARM = 'clf_top05'
ARMS = (ARM,)
FAMILIES = original.FAMILIES
FRACTIONS = original.FRACTIONS
TARGET = dict(positive_fraction=.05, requested_top_k='ceil(n / 20)',
              include_threshold_ties=True,
              minimum_threshold_fallback='positive iff AUPR strictly exceeds minimum',
              scope='current training fronts only', score='positive-class probability')


def binary_labels(quality, groups):
    quality, groups = np.asarray(quality, dtype=float), np.asarray(groups)
    if quality.ndim != 1 or quality.shape != groups.shape or not np.isfinite(quality).all():
        raise ValueError('Invalid classification quality/groups')
    labels, diagnostics = np.zeros(len(quality), dtype=np.int64), []
    for front in np.unique(groups):
        indices = np.flatnonzero(groups == front)
        values = quality[indices]
        k = max(1, (len(values) + 19) // 20)
        threshold = float(np.partition(values, len(values) - k)[len(values) - k])
        fallback = threshold == values.min()
        positive = values > values.min() if fallback else values >= threshold
        if not positive.any() or positive.all():
            raise ValueError('Classification needs two classes on each informative front')
        labels[indices] = positive.astype(np.int64)
        diagnostics.append(dict(front_id=int(front), n_candidates=len(values), requested_top_k=k,
                                threshold=threshold, n_positive=int(positive.sum()),
                                positive_fraction=float(positive.mean()), minimum_threshold_fallback=bool(fallback)))
    return labels, diagnostics


def prepare(data, case, mapping, features, label='rank_dense', seed=1101):
    result = original.prepare(data, case, mapping, features, label, seed)
    labels, report = binary_labels(result['labels']['reg_aupr'], result['group_id'])
    result['labels'][ARM] = labels
    del result['labels']['clf_top20']
    result['report']['class_fractions'] = report
    result['report']['classification_policy'] = dict(TARGET)
    return result


def _adapter(data, arm):
    if arm != ARM:
        raise ValueError('This experiment only fits top-five-percent classification')
    # The existing classifier adapter's internal key is retained locally only.
    # Every public artifact identifies the actual top-5% target and its policy.
    return dict(data, labels=dict(data['labels'], clf_top20=data['labels'][ARM]))


def fit(data, family, arm, params, *, seed=1101, threads=8):
    model, report = original.fit(_adapter(data, arm), family, 'clf_top20', params, seed=seed, threads=threads)
    return model, dict(report, classification_policy=dict(TARGET))


def scores(model, family, arm, X, threads=8):
    if arm != ARM:
        raise ValueError('Unknown classification arm')
    return original.scores(model, family, 'clf_top20', X, threads)


def feature_path(data, family, arm, params, *, seed=1101, threads=8, fractions=FRACTIONS, callback=None):
    paths, importance = original.feature_path(_adapter(data, arm), family, 'clf_top20', params,
                                              seed=seed, threads=threads, fractions=fractions, callback=callback)
    for path in paths:
        path['classification_policy'] = dict(TARGET)
    return paths, importance


feature_importance = original.feature_importance
