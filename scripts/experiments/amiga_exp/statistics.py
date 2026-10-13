"""Paired topology summaries and Holm multiplicity adjustment."""
import numpy as np

def paired_difference(reference, alternative, *, seed=1401, samples=10000):
    """Negative mean differences favor the reference (both metrics are regret)."""
    delta = np.asarray(reference, dtype=float) - np.asarray(alternative, dtype=float)
    if delta.ndim != 1 or not len(delta) or not np.isfinite(delta).all():
        raise ValueError('Paired differences must be finite and nonempty')
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), size=(samples, len(delta)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return dict(mean_difference=float(delta.mean()), ci_low=float(low), ci_high=float(high),
                improved=int((delta < 0).sum()), tied=int((delta == 0).sum()),
                worsened=int((delta > 0).sum()), n_topologies=len(delta))


def holm(pvalues):
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values, kind='stable')
    corrected = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    result = np.empty_like(values)
    result[order] = corrected
    return result
