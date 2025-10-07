"""
Feature extraction from a gene expression matrix (genes × conditions).
- Rows = genes (index), columns = conditions/timepoints.
- Metric families can be toggled on/off by the caller.
"""

from __future__ import annotations

import random
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

EPS = 1e-12


# ----------------------------- Utils -----------------------------

def _q(arr: np.ndarray, q: float) -> float:
    """Safe percentile with NaN handling; returns NaN if the array is empty."""
    if arr.size == 0:
        return float("nan")
    return float(np.nanpercentile(arr, q))


def _safe_mean(arr: np.ndarray) -> float:
    """Mean that tolerates NaNs and empty arrays."""
    return float(np.nanmean(arr)) if arr.size else float("nan")


def _safe_std(arr: np.ndarray) -> float:
    """Std that tolerates NaNs and empty arrays."""
    return float(np.nanstd(arr)) if arr.size else float("nan")


def _linear_trend(y: np.ndarray) -> Tuple[float, float]:
    """
    Return (slope, r2) of the OLS fit y ~ a + b*t with t = 0..n-1, ignoring NaNs.

    Notes
    -----
    - Time axis is centered for numerical stability.
    - R^2 is computed against the mean of y and protected against division by zero.
    """
    idx = ~np.isnan(y)
    y = y[idx]
    if y.size < 2:
        return float("nan"), float("nan")
    t = np.arange(y.size, dtype=float)

    # Center both axes for numeric stability
    t_c = t - t.mean()
    y_c = y - y.mean()

    denom = (t_c @ t_c)
    if denom <= EPS:
        return float("nan"), float("nan")

    b = float((t_c @ y_c) / denom)             # slope
    a = float(y.mean() - b * t.mean())         # intercept
    y_pred = a + b * t

    ss_res = float(np.nansum((y - y_pred) ** 2))
    ss_tot = float(np.nansum((y - y.mean()) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + EPS))
    return b, r2


def _lag_autocorr(y: np.ndarray, lag: int = 1) -> float:
    """
    Pearson autocorrelation at a given lag, ignoring NaNs.
    Returns NaN if insufficient length after removing NaNs.
    """
    idx = ~np.isnan(y)
    y = y[idx]
    if y.size <= lag:
        return float("nan")
    y1 = y[:-lag]
    y2 = y[lag:]
    if y1.size < 2:
        return float("nan")

    # Pearson correlation
    y1m, y2m = y1 - y1.mean(), y2 - y2.mean()
    num = float((y1m * y2m).sum())
    den = float(np.sqrt((y1m**2).sum() * (y2m**2).sum()) + EPS)
    return num / den


def _effective_rank(singular_values: np.ndarray) -> float:
    """
    Effective rank = exp(Shannon entropy of the normalized energy distribution).

    Let s be singular values; energy ~ s^2; p = energy / sum(energy).
    Effective rank is exp(-Σ p log p), ignoring zero-probability entries.
    """
    if singular_values.size == 0:
        return float("nan")
    energy = singular_values**2
    p = energy / (energy.sum() + EPS)
    p = p[p > 0]
    H = float(-(p * np.log(p)).sum())
    return float(np.exp(H))


# --------------------------- Feature sets -------------------------

def features_exp_global(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Global matrix descriptors:
    - Size (n_genes, n_conditions)
    - Mean/std/min/max
    - Missing/zero proportions
    - Skewness and kurtosis (Fisher)
    """
    vals = df.values.astype(float)
    feats = {
        "n_genes": int(df.shape[0]),
        "n_conditions": int(df.shape[1]),
        "global_mean": _safe_mean(vals),
        "global_std": _safe_std(vals),
        "global_min": float(np.nanmin(vals)) if vals.size else float("nan"),
        "global_max": float(np.nanmax(vals)) if vals.size else float("nan"),
        "prop_missing": float(np.isnan(vals).mean()) if vals.size else 0.0,
        "prop_zeros": float((vals == 0).mean()) if vals.size else 0.0,
    }
    # Skewness and kurtosis via pandas (Fisher=True by default)
    s = pd.Series(vals.ravel())
    feats["global_skew"] = float(s.skew(skipna=True))
    feats["global_kurtosis"] = float(s.kurtosis(skipna=True))
    return feats


def features_gene_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Per-gene statistics aggregated across conditions:
    - Means, stds and CVs (with zero-protection)
    - Frac. of zero and missing entries per gene (averaged across genes)
    """
    gmean = df.mean(axis=1)
    gstd = df.std(axis=1, ddof=0)
    gcv = gstd / (gmean.replace(0, np.nan) + EPS)
    feats = {
        "gene_mean_mean": _safe_mean(gmean.values),
        "gene_mean_std": _safe_std(gmean.values),
        "gene_std_mean": _safe_mean(gstd.values),
        "gene_std_std": _safe_std(gstd.values),
        "gene_cv_mean": _safe_mean(gcv.values),
        "gene_cv_p10": _q(gcv.values, 10),
        "gene_cv_p50": _q(gcv.values, 50),
        "gene_cv_p90": _q(gcv.values, 90),
        "gene_zero_frac_mean": _safe_mean((df == 0).mean(axis=1).values),
        "gene_missing_frac_mean": _safe_mean(df.isna().mean(axis=1).values),
    }
    return feats


def features_condition_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Per-condition statistics aggregated across genes:
    - Means, stds and CVs
    - Fraction of zeros/missing per condition
    - Max absolute z-score for condition means (simple outlier flag)
    """
    cmean = df.mean(axis=0)
    cstd = df.std(axis=0, ddof=0)
    ccv = cstd / (cmean.replace(0, np.nan) + EPS)
    feats = {
        "cond_mean_mean": _safe_mean(cmean.values),
        "cond_mean_std": _safe_std(cmean.values),
        "cond_std_mean": _safe_mean(cstd.values),
        "cond_std_std": _safe_std(cstd.values),
        "cond_cv_mean": _safe_mean(ccv.values),
        "cond_cv_p10": _q(ccv.values, 10),
        "cond_cv_p50": _q(ccv.values, 50),
        "cond_cv_p90": _q(ccv.values, 90),
        "cond_zero_frac_mean": _safe_mean((df == 0).mean(axis=0).values),
        "cond_missing_frac_mean": _safe_mean(df.isna().mean(axis=0).values),
    }
    # Simple condition outlier detection via z-score of the mean
    z = (cmean - cmean.mean()) / (cmean.std(ddof=0) + EPS)
    feats["cond_mean_max_abs_z"] = float(np.nanmax(np.abs(z.values))) if z.size else float("nan")
    return feats


def features_correlations(
    df: pd.DataFrame,
    max_pairs: Optional[int] = None,
    seed: int = 13
) -> Dict[str, Any]:
    """
    Pairwise correlations:
    - gene–gene (across conditions): computed on df.T
    - condition–condition (across genes): computed on df

    For large matrices, you can subsample unique pairs up to `max_pairs`.
    Uses Pearson correlation with pairwise NaN handling.
    """
    rng = random.Random(seed)
    feats: Dict[str, Any] = {}

    # --- gene-gene ---
    # pandas corr uses pairwise complete observations; transpose to correlate genes
    gg_corr = df.T.corr(method="pearson", min_periods=2)  # genes × genes
    if gg_corr.shape[0] >= 2:
        iu = np.triu_indices(gg_corr.shape[0], k=1)
        vals = gg_corr.values[iu]
        vals = vals[~np.isnan(vals)]
        if max_pairs and vals.size > max_pairs:
            idx = rng.sample(range(vals.size), max_pairs)
            vals = vals[idx]
        feats.update({
            "gene_gene_corr_mean": _safe_mean(vals),
            "gene_gene_corr_std": _safe_std(vals),
            "gene_gene_abs_corr_mean": _safe_mean(np.abs(vals)),
            "gene_gene_abs_corr_p90": _q(np.abs(vals), 90),
        })
    else:
        feats.update({
            "gene_gene_corr_mean": float("nan"),
            "gene_gene_corr_std": float("nan"),
            "gene_gene_abs_corr_mean": float("nan"),
            "gene_gene_abs_corr_p90": float("nan"),
        })

    # --- condition-condition ---
    cc_corr = df.corr(method="pearson", min_periods=2)  # conditions × conditions
    if cc_corr.shape[0] >= 2:
        iu = np.triu_indices(cc_corr.shape[0], k=1)
        vals = cc_corr.values[iu]
        vals = vals[~np.isnan(vals)]
        if max_pairs and vals.size > max_pairs:
            idx = rng.sample(range(vals.size), max_pairs)
            vals = vals[idx]
        feats.update({
            "cond_cond_corr_mean": _safe_mean(vals),
            "cond_cond_corr_std": _safe_std(vals),
            "cond_cond_abs_corr_mean": _safe_mean(np.abs(vals)),
            "cond_cond_abs_corr_p90": _q(np.abs(vals), 90),
        })
    else:
        feats.update({
            "cond_cond_corr_mean": float("nan"),
            "cond_cond_corr_std": float("nan"),
            "cond_cond_abs_corr_mean": float("nan"),
            "cond_cond_abs_corr_p90": float("nan"),
        })

    return feats


def features_pca(df: pd.DataFrame, top_k: int = 5, center_by: str = "condition") -> Dict[str, Any]:
    """
    PCA via SVD:

    - center_by='condition': center each column (variation across genes).
    - center_by='gene': center each row (variation across conditions).

    Returns
    -------
    dict
        {
          "pca_var_exp": list of top-k explained-variance ratios,
          "pca_eff_rank": effective rank,
          "pca_cond_number": spectral condition number
        }
    """
    X = df.values.astype(float)
    if X.size == 0 or min(X.shape) < 2:
        return {"pca_var_exp": [], "pca_eff_rank": float("nan"), "pca_cond_number": float("nan")}

    if center_by == "condition":
        X = X - np.nanmean(X, axis=0, keepdims=True)
    elif center_by == "gene":
        X = X - np.nanmean(X, axis=1, keepdims=True)

    # Replace NaNs with the centered mean (0 after centering)
    X = np.nan_to_num(X, nan=0.0)

    # Compact SVD: X = U S V^T; singular values = S
    try:
        s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError:
        return {"pca_var_exp": [], "pca_eff_rank": float("nan"), "pca_cond_number": float("nan")}

    energy = s**2
    total = float(energy.sum() + EPS)
    var_exp = (energy / total).tolist()
    k = min(top_k, len(var_exp))
    var_topk = var_exp[:k]

    cond_number = float((s.max() / (s.min() + EPS))) if s.size else float("nan")
    eff_rank = _effective_rank(s)

    return {
        "pca_var_exp": var_topk,          # list
        "pca_eff_rank": eff_rank,         # scalar
        "pca_cond_number": cond_number,   # scalar
    }


def features_timeseries(df: pd.DataFrame, max_lag: int = 1) -> Dict[str, Any]:
    """
    Per-gene time-series metrics, assuming columns are ordered in time:
    - Lag-1 autocorrelation (and optionally >1 if extended)
    - Linear trend slope and R^2

    The function aggregates per-gene metrics into summary statistics
    (mean/median/p90) for each magnitude.
    """
    vals = df.values.astype(float)
    G = vals.shape[0]
    if G == 0 or df.shape[1] < 2:
        return {
            "ts_lag1_autocorr_mean": float("nan"),
            "ts_lag1_autocorr_median": float("nan"),
            "ts_slope_mean": float("nan"),
            "ts_slope_p90": float("nan"),
            "ts_r2_mean": float("nan"),
            "ts_pos_slope_frac": float("nan"),
        }

    # Per-gene autocorrelation at lag 1
    lag1 = np.array([_lag_autocorr(vals[i, :], lag=1) for i in range(G)], dtype=float)

    # Per-gene linear trend (slope, R^2)
    slopes = np.empty(G, dtype=float)
    r2s = np.empty(G, dtype=float)
    for i in range(G):
        b, r2 = _linear_trend(vals[i, :])
        slopes[i] = b
        r2s[i] = r2

    pos_frac = float(np.nanmean(slopes > 0))

    return {
        "ts_lag1_autocorr_mean": _safe_mean(lag1),
        "ts_lag1_autocorr_median": float(np.nanmedian(lag1)),
        "ts_slope_mean": _safe_mean(slopes),
        "ts_slope_p90": _q(slopes, 90),
        "ts_r2_mean": _safe_mean(r2s),
        "ts_pos_slope_frac": pos_frac,
    }
