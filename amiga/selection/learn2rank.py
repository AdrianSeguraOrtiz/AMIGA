"""
Learning-to-Rank (LTR) over Pareto fronts.

This module implements utilities to train and evaluate ranking models on fronts
(e.g., solutions from multi-objective algorithms), as well as to construct
intra-front labels and compute ranking metrics.

Includes:
- Enums for model types (LightGBM / XGBoost / CatBoost) and label modes.
- Label construction (dense ranks, average ranks, quantiles, continuous).
- Stable reordering and per-group sizes for rankers that require contiguous blocks.
- Per-group and aggregated metrics (NDCG@k, P@1, Regret@k, Spearman, KendallTau).
- Per-fold training and validation predictions for the supported models.

Notes
-----
* This is a pure LTR logic module; it does NOT persist files.
* LightGBM/XGBoost require samples of each group (front) to be presented in
  contiguous blocks; `order_by_group_and_sizes` is provided for this.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import ndcg_score


# -----------------------------------------------------------------------------
# Enums & config
# -----------------------------------------------------------------------------

class ModelType(str, enum.Enum):
    """
    Supported ranking model types.

    - LGBMRanker: LightGBM LambdaRank.
    - XGBRanker: XGBoost 'rank:ndcg'.
    - CatBoostRanker: CatBoost YetiRank.
    """
    LGBMRanker = "LGBMRanker"
    XGBRanker = "XGBRanker"
    CatBoostRanker = "CatBoostRanker"


class LabelMode(str, enum.Enum):
    """
    Intra-front label construction modes.

    RANK_DENSE
        Dense ranks 0..L-1 (best = L-1).
    RANK_AVG
        Average ranks 0..L-1 (ties averaged, then discretized).
    QUANTILES
        Quantile bins 0..Q-1 (best = Q-1).
    CONTINUOUS
        Continuous relevance 0..1 via per-front min-max scaling (best = 1).
    """
    RANK_DENSE = "rank_dense"
    RANK_AVG = "rank_avg"
    QUANTILES = "quantiles"
    CONTINUOUS = "continuous"


@dataclass
class DatasetSpec:
    """
    Dataset column specification and exclusions.

    Attributes
    ----------
    front_col : str
        Name of the group/front column.
    target_col : str
        Name of the target column (e.g., AUPR).
    id_col : str
        Name of the individual identifier column.
    drop_cols : Sequence[str]
        Extra columns to exclude from features.
    """
    front_col: str
    target_col: str
    id_col: str
    drop_cols: Sequence[str]


# -----------------------------------------------------------------------------
# I/O helpers (filenames used by upper/CLI layer)
# -----------------------------------------------------------------------------

FEATURES_META = "feature_columns.json"
CV_REPORT = "cv_report.json"
MODEL_PREFIX = "model_fold"


def ensure_dependencies(model_type: ModelType) -> None:
    """
    Check that optional dependencies for the chosen model are importable.

    Parameters
    ----------
    model_type : ModelType
        Model type to verify.

    Raises
    ------
    Exception
        If the corresponding library is not installed.
    """
    try:
        if model_type == ModelType.LGBMRanker:
            import lightgbm as _  # noqa: F401
        elif model_type == ModelType.XGBRanker:
            import xgboost as _  # noqa: F401
        elif model_type == ModelType.CatBoostRanker:
            import catboost as _  # noqa: F401
    except Exception as exc:  # pragma: no cover
        print(f"Missing dependency for {model_type}: {exc}")
        raise


# -----------------------------------------------------------------------------
# Label & group preparation
# -----------------------------------------------------------------------------

def build_labels(
    df: pd.DataFrame,
    front_col: str,
    target_col: str,
    mode: LabelMode,
    n_quantiles: int = 20,
) -> np.ndarray:
    """
    Build intra-front labels/relevances from a continuous target (e.g., AUPR).

    Parameters
    ----------
    df : pd.DataFrame
        Data containing `front_col` and `target_col`, along with features.
    front_col : str
        Group/front column name.
    target_col : str
        Continuous target to be converted to intra-front relevances.
    mode : LabelMode
        Label construction mode (see `LabelMode`).
    n_quantiles : int, optional
        Number of quantiles if `mode == QUANTILES`.

    Returns
    -------
    np.ndarray
        Vector of relevances/labels aligned with the order of `df`.

    Notes
    -----
    * RANK_DENSE / RANK_AVG return integer labels 0..L-1.
    * QUANTILES returns integers 0..Q-1; uses `duplicates="drop"` and robust fallback on failure.
    * CONTINUOUS returns floats 0..1 (per-front min-max).
    """
    if mode == LabelMode.RANK_DENSE:
        # r: 1..L (1 is best if ascending=False)
        r = df.groupby(front_col)[target_col].rank(method="dense", ascending=False).astype(int)
        L = df.groupby(front_col)[target_col].transform("count").astype(int)
        # label: 0..L-1 with best = L-1
        return (L - r).astype(int).to_numpy()

    if mode == LabelMode.RANK_AVG:
        r = df.groupby(front_col)[target_col].rank(method="average", ascending=False)
        L = df.groupby(front_col)[target_col].transform("count")
        # (L - r) makes higher targets map to larger labels; discretize to nearest int
        return (L - r).round(0).astype(int).to_numpy()

    if mode == LabelMode.QUANTILES:
        # 0..Q-1 per group; qcut assigns 0 to lowest and Q-1 to highest
        def qbin(g: pd.Series) -> pd.Series:
            g = g.astype(float)
            # Limit Q to number of unique values; duplicates="drop" handles ties
            q = min(n_quantiles, max(1, g.nunique()))
            if q <= 1:
                return pd.Series(0, index=g.index, dtype=int)
            try:
                return pd.qcut(g, q=q, labels=False, duplicates="drop")
            except Exception:
                # Fallback: dense rank 0..L-1 (higher is better)
                rd = g.rank(method="dense", ascending=True).astype(int) - 1
                # ascending=True → smaller values rank first; invert to make higher better
                Lloc = rd.max() + 1
                return (Lloc - 1 - rd).astype(int)

        qlabels = df.groupby(front_col)[target_col].transform(qbin).astype(int)
        return qlabels.to_numpy()

    if mode == LabelMode.CONTINUOUS:
        # Per-front min-max scaling to [0, 1]
        def mm(g: pd.Series) -> pd.Series:
            g = g.astype(float)
            mn, mx = g.min(), g.max()
            if mx > mn:
                return (g - mn) / (mx - mn)
            else:
                return pd.Series(0.0, index=g.index)
        rel = df.groupby(front_col)[target_col].transform(mm).astype(float)
        return rel.to_numpy()

    raise ValueError(f"Unsupported label mode: {mode}")


def order_by_group_and_sizes(group_ids: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Compute a stable index that groups elements by group id, and the block sizes.

    Several rankers (LightGBM/XGBoost) require samples for each group to be
    contiguous blocks and to know the group sizes.

    Parameters
    ----------
    group_ids : np.ndarray
        Group id vector aligned with the rows of X/y.

    Returns
    -------
    order : np.ndarray
        Stable (mergesort) indices that group by `group_ids`.
    sizes : list[int]
        Sizes of each contiguous block (in the order they appear after sorting).
    """
    order = np.argsort(group_ids, kind="mergesort")  # stable
    ids_sorted = group_ids[order]

    sizes: List[int] = []
    last = None
    count = 0
    for g in ids_sorted:
        if (last is None) or (g != last):
            if last is not None:
                sizes.append(count)
            last = g
            count = 1
        else:
            count += 1
    if last is not None:
        sizes.append(count)

    return order, sizes


# -----------------------------------------------------------------------------
# Ranking metrics
# -----------------------------------------------------------------------------

def compute_ranking_metrics(
    df_fold: pd.DataFrame,
    front_col: str,
    target_col: str,
    score_col: str,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Compute per-group ranking metrics and a simple aggregate (mean over groups).

    Parameters
    ----------
    df_fold : pd.DataFrame
        Subset (e.g., validation of one fold) with columns `front_col`, `target_col`, `score_col`.
    front_col : str
        Group/front column.
    target_col : str
        Column with the ground-truth value / “relevance”.
    score_col : str
        Column with model-predicted scores.
    ks : Sequence[int], optional
        Cutoffs for metrics@k (NDCG@k, Regret@k; P@1 is reported separately).

    Returns
    -------
    agg : dict[str, float]
        Aggregated metrics (mean over groups) for all found keys.
    per_group : list[dict[str, float]]
        Per-group metrics including: NDCG@k, Regret@k, P@1, Spearman, KendallTau, n_items.

    Notes
    -----
    * P@1 checks whether the argmax of y_true equals the top-1 by `score`.
    * Regret@k = max(y_true) - max(y_true@topk_by_score)
    """
    per_group: List[Dict[str, float]] = []

    for _, group in df_fold.groupby(front_col):
        y_true = group[target_col].to_numpy().reshape(1, -1)
        y_score = group[score_col].to_numpy().reshape(1, -1)

        # Order by predicted score for top-k dependent metrics
        order = np.argsort(-y_score.ravel())
        y_true_sorted = y_true.ravel()[order]

        ndcgs = {f"NDCG@{k}": float(ndcg_score(y_true, y_score, k=k)) for k in ks if k <= group.shape[0]}
        p_at_1 = float(np.argmax(y_true.ravel()) == order[0]) if group.shape[0] > 0 else 0.0
        regret_at_k = {
            f"Regret@{k}": float(np.max(y_true.ravel()) - (np.max(y_true_sorted[:k]) if k <= len(y_true_sorted) else 0.0))
            for k in ks
            if k <= group.shape[0]
        }

        # Rank correlations
        rho = spearmanr(y_true.ravel(), y_score.ravel()).statistic
        ktau = kendalltau(y_true.ravel(), y_score.ravel()).correlation

        per_group.append(
            {
                **ndcgs,
                **regret_at_k,
                "P@1": float(p_at_1),
                "Spearman": float(rho if np.isfinite(rho) else 0.0),
                "KendallTau": float(ktau if np.isfinite(ktau) else 0.0),
                "n_items": int(group.shape[0]),
            }
        )

    # Aggregate as simple mean across groups
    agg: Dict[str, float] = {}
    if per_group:
        keys = set().union(*per_group)
        for k in keys:
            vals = [d[k] for d in per_group if k in d]
            if vals:
                agg[k] = float(np.mean(vals))

    return agg, per_group


# -----------------------------------------------------------------------------
# Training & prediction
# -----------------------------------------------------------------------------

def fit_ranker(
    model_type: ModelType,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    gid_train: np.ndarray,
    *,
    random_state: int = 42,
    # Optional validation set (if provided, used for early stopping/monitoring)
    X_valid: Optional[pd.DataFrame] = None,
    y_valid: Optional[np.ndarray] = None,
    gid_valid: Optional[np.ndarray] = None,
) -> Tuple[object, Optional[np.ndarray]]:
    """
    Train an LTR ranker and, if validation is provided, also return validation
    scores **in the original order of X_valid**.

    Usage
    -----
        # Cross-validation:
        model, scores_va = fit_ranker(model_type, X_tr, y_tr, gid_tr,
                                      random_state=seed,
                                      X_valid=X_va, y_valid=y_va, gid_valid=gid_va)

        # Full-train (production):
        model, _ = fit_ranker(model_type, X, y, gid, random_state=seed)

    Returns
    -------
    model : object
        Trained model, ready for `.predict(X)`.
    scores_valid : np.ndarray | None
        If validation was passed, array of scores aligned with the original order of `X_valid`.
    """
    ensure_dependencies(model_type)

    has_valid = (X_valid is not None) and (y_valid is not None) and (gid_valid is not None)

    # Models that require contiguous blocks per group (train and, if present, valid)
    if model_type in {ModelType.LGBMRanker, ModelType.XGBRanker}:
        tr_order, sizes_tr = order_by_group_and_sizes(gid_train)
        X_tr, y_tr = X_train.iloc[tr_order], y_train[tr_order]
        if has_valid:
            va_order, sizes_va = order_by_group_and_sizes(gid_valid)  # for eval_group
            X_va, y_va = X_valid.iloc[va_order], y_valid[va_order]
        else:
            X_va = y_va = sizes_va = va_order = None
    else:
        # CatBoost: no reordering needed; uses group_id directly
        X_tr, y_tr = X_train, y_train
        sizes_tr = None
        if has_valid:
            X_va, y_va, sizes_va, va_order = X_valid, y_valid, None, None
        else:
            X_va = y_va = sizes_va = va_order = None

    # -------------------------------------------------------------------------
    # LightGBM
    # -------------------------------------------------------------------------
    if model_type == ModelType.LGBMRanker:
        from lightgbm import LGBMRanker, early_stopping

        # Quantize labels if they are not non-negative integers (consistent with CV)
        n_bins = 256
        if not (np.issubdtype(y_tr.dtype, np.integer) and y_tr.min() >= 0):
            y_tr_q = np.floor(np.clip(y_tr, 0, 1) * (n_bins - 1)).astype(int)
            label_gain = list(range(n_bins))
            if has_valid:
                y_va_q = np.floor(np.clip(y_va, 0, 1) * (n_bins - 1)).astype(int)
        else:
            y_tr_q = y_tr.astype(int)
            label_gain = list(range(int(y_tr_q.max()) + 1))
            if has_valid:
                y_va_q = y_va.astype(int)

        params = dict(
            objective="lambdarank",
            metric="ndcg",
            num_leaves=63,
            learning_rate=0.05,
            n_estimators=2000,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,
            random_state=random_state,
            label_gain=label_gain,
        )
        model = LGBMRanker(**params)

        if has_valid:
            model.fit(
                X_tr, y_tr_q,
                group=sizes_tr,
                eval_set=[(X_va, y_va_q)],
                eval_group=[sizes_va],
                eval_at=[1, 3, 5, 10],
                callbacks=[early_stopping(stopping_rounds=100, verbose=False)],
            )
            scores_va = model.predict(
                X_va, num_iteration=getattr(model, "best_iteration_", None)
            )
            # Return in the original user order:
            if va_order is not None:
                inv = np.empty_like(va_order)
                inv[va_order] = np.arange(len(va_order))
                scores_va = scores_va[inv]
            return model, scores_va
        else:
            model.fit(X_tr, y_tr_q, group=sizes_tr)
            return model, None

    # -------------------------------------------------------------------------
    # XGBoost
    # -------------------------------------------------------------------------
    if model_type == ModelType.XGBRanker:
        from xgboost import XGBRanker

        model = XGBRanker(
            objective="rank:ndcg",
            eval_metric="ndcg@10",
            learning_rate=0.05,
            n_estimators=2000,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            ndcg_exp_gain=False,
        )

        if has_valid:
            model.fit(
                X_tr, y_tr,
                group=sizes_tr,
                eval_set=[(X_va, y_va)],
                eval_group=[sizes_va],
                verbose=False,
            )
            scores_va = model.predict(
                X_va,
                iteration_range=(0, getattr(model, "best_iteration", None) + 1)
                if hasattr(model, "best_iteration") and model.best_iteration is not None
                else None
            )
            if va_order is not None:
                inv = np.empty_like(va_order)
                inv[va_order] = np.arange(len(va_order))
                scores_va = scores_va[inv]
            return model, scores_va
        else:
            model.fit(X_tr, y_tr, group=sizes_tr, verbose=False)
            return model, None

    # -------------------------------------------------------------------------
    # CatBoost
    # -------------------------------------------------------------------------
    if model_type == ModelType.CatBoostRanker:
        from catboost import CatBoostRanker, Pool
        train_pool = Pool(X_tr, y_tr, group_id=gid_train)
        model = CatBoostRanker(
            loss_function="YetiRank",
            eval_metric="NDCG:top=10",
            learning_rate=0.05,
            depth=6,
            iterations=2000,
            random_seed=random_state,
            verbose=False,
            od_type="Iter" if has_valid else "IncToDec",  # OD only if validation is present
            od_wait=100 if has_valid else None,
        )
        if has_valid:
            valid_pool = Pool(X_va, y_va, group_id=gid_valid)
            model.fit(train_pool, eval_set=valid_pool)
            scores_va = np.asarray(model.predict(valid_pool))
            return model, scores_va
        else:
            model.fit(train_pool)
            return model, None

    raise ValueError(f"Unsupported model type: {model_type}")
