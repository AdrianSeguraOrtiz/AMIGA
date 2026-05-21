"""
Learning-to-Rank (LTR) over Pareto fronts.

This module implements utilities to train and evaluate ranking models on fronts
(e.g., solutions from multi-objective algorithms), as well as to construct
intra-front labels and compute ranking metrics.

Includes:
- Enums for model types (LightGBM / XGBoost / CatBoost) and label modes.
- Label construction (dense ranks, average ranks, quantiles, continuous).
- Stable reordering and per-group sizes for rankers that require contiguous blocks.
- Per-group and aggregated metrics with decision-first emphasis.
- Per-fold training and validation predictions for the supported models.

Notes
-----
* This is a pure LTR logic module; it does NOT persist files.
* LightGBM/XGBoost require samples of each group (front) to be presented in
  contiguous blocks; `order_by_group_and_sizes` is provided for this.
"""

from __future__ import annotations

import enum
import math
import zlib
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, kendalltau, spearmanr
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
    REVERSED
        Negative control: inverted per-front continuous relevance (best = 0).
    SHUFFLED
        Negative control: shuffled per-front continuous relevance.
    """
    RANK_DENSE = "rank_dense"
    RANK_AVG = "rank_avg"
    QUANTILES = "quantiles"
    CONTINUOUS = "continuous"
    REVERSED = "reversed"
    SHUFFLED = "shuffled"


# -----------------------------------------------------------------------------
# I/O helpers (filenames used by upper/CLI layer)
# -----------------------------------------------------------------------------

FEATURES_META = "feature_columns.json"
CV_REPORT = "cv_report.json"
MODEL_PREFIX = "model_fold"

# Canonical display order for reports, plots, and logs.
METRIC_REPORT_ORDER: Dict[str, int] = {
    "Regret@1": 10,
    "Regret@3": 11,
    "Regret@5": 12,
    "Regret@10": 13,
    "BestAUPR@1": 20,
    "BestAUPR@3": 21,
    "BestAUPR@5": 22,
    "BestAUPR@10": 23,
    "Hit@1": 30,
    "Hit@3": 31,
    "Hit@5": 32,
    "Hit@10": 33,
    "NDCG@1": 40,
    "NDCG@3": 41,
    "NDCG@5": 42,
    "NDCG@10": 43,
    "Spearman": 50,
    "KendallTau": 51,
    "n_items": 99,
}


def metric_report_order(metric: str) -> int:
    """Return a stable display-order key for known ranking metrics."""
    return METRIC_REPORT_ORDER.get(metric, 1000)


def stable_tie_breaker(front_id: Any, item_id: Any, *, seed: int = 0) -> int:
    """
    Return a deterministic pseudo-random tie-break value for a front/item pair.

    The value is intentionally independent of dataframe row order. It is used
    only to make exported ranked CSVs deterministic when predicted scores tie;
    tie-aware metrics below do not depend on this value.
    """
    key = f"{int(seed)}|{front_id!s}|{item_id!s}"
    return zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF


def assign_rank_in_front(
    df: pd.DataFrame,
    *,
    front_col: str,
    score_col: str = "score",
    id_col: str | None = "item_id",
    rank_col: str = "rank_in_front",
    tie_seed: int = 0,
) -> pd.DataFrame:
    """
    Add `rank_col` using score descending and row-order-independent tie breaks.

    If `id_col` is present, ties are resolved by a stable hash of
    `(front_col, id_col, tie_seed)`. If no identifier column is available, the
    helper falls back to a stable fingerprint of row values. Duplicate IDs
    inside a front are rejected because they would make exported tie ordering
    ambiguous.
    """
    missing = [column for column in (front_col, score_col) if column not in df.columns]
    if missing:
        raise ValueError(f"ranked dataframe is missing required column(s): {missing}")

    working = df.copy()
    if id_col is not None and id_col in working.columns:
        duplicated = working.duplicated([front_col, id_col], keep=False)
        if duplicated.any():
            pairs = (
                working.loc[duplicated, [front_col, id_col]]
                .drop_duplicates()
                .head(5)
                .to_dict(orient="records")
            )
            raise ValueError(f"duplicate '{id_col}' values within fronts: {pairs}")
        tie_values = [
            stable_tie_breaker(front_id, item_id, seed=tie_seed)
            for front_id, item_id in zip(working[front_col], working[id_col], strict=False)
        ]
    else:
        tie_columns = [column for column in working.columns if column != rank_col]
        tie_values = [
            zlib.crc32(
                "|".join(_stable_value(row[column]) for column in tie_columns).encode("utf-8")
            )
            & 0xFFFFFFFF
            for _, row in working[tie_columns].iterrows()
        ]

    tie_col = "__amiga_tie_breaker"
    working[tie_col] = tie_values
    working = working.sort_values(
        [front_col, score_col, tie_col],
        ascending=[True, False, True],
        kind="mergesort",
    )
    working[rank_col] = working.groupby(front_col, sort=False).cumcount() + 1
    return working.drop(columns=[tie_col])


def _stable_value(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, float | np.floating):
        return repr(float(value))
    return str(value)


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
    random_state: int = 42,
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
    random_state : int, optional
        Seed used by stochastic negative controls such as `SHUFFLED`.

    Returns
    -------
    np.ndarray
        Vector of relevances/labels aligned with the order of `df`.

    Notes
    -----
    * RANK_DENSE / RANK_AVG return integer labels 0..L-1.
    * QUANTILES returns integers 0..Q-1; uses `duplicates="drop"` and robust fallback on failure.
    * CONTINUOUS returns floats 0..1 (per-front min-max).
    * REVERSED inverts CONTINUOUS within each front.
    * SHUFFLED permutes CONTINUOUS within each front using a deterministic seed.
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

    if mode in {LabelMode.REVERSED, LabelMode.SHUFFLED}:
        def mm(g: pd.Series) -> pd.Series:
            g = g.astype(float)
            mn, mx = g.min(), g.max()
            if mx > mn:
                return (g - mn) / (mx - mn)
            return pd.Series(0.0, index=g.index)

        rel = df.groupby(front_col)[target_col].transform(mm).astype(float)
        if mode == LabelMode.REVERSED:
            return (1.0 - rel).to_numpy()

        shuffled = np.empty(len(df), dtype=float)
        for front_id, idx in df.groupby(front_col, sort=False).groups.items():
            values = rel.loc[idx].to_numpy(copy=True)
            seed = (zlib.crc32(str(front_id).encode("utf-8")) ^ int(random_state)) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            rng.shuffle(values)
            shuffled[np.asarray(idx, dtype=int)] = values
        return shuffled

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

def _comb_ratio(available: int, selected: int, total: int) -> float:
    """Return C(available, selected) / C(total, selected) without huge integers."""
    if selected < 0 or total < selected or available < selected:
        return 0.0
    if selected == 0:
        return 1.0
    if available == total:
        return 1.0

    ratio = 1.0
    for offset in range(selected):
        ratio *= (available - offset) / (total - offset)
    return float(ratio)


def _expected_max_with_partial_tie(
    block_targets: np.ndarray,
    sample_size: int,
    base_best: float,
) -> float:
    """Expected max target after uniformly sampling from a tied-score block."""
    block_targets = np.asarray(block_targets, dtype=float)
    block_size = int(block_targets.size)
    if sample_size <= 0:
        return float(base_best)
    if sample_size >= block_size:
        return float(max(base_best, float(np.max(block_targets))))

    support_values = [block_targets]
    if np.isfinite(base_best):
        support_values.append(np.array([base_best], dtype=float))
    support = np.unique(np.concatenate(support_values))
    expected = 0.0
    previous_cdf = 0.0
    for value in support:
        if value < base_best:
            continue
        count_leq = int(np.sum(block_targets <= value))
        cdf = _comb_ratio(count_leq, sample_size, block_size)
        probability = max(0.0, cdf - previous_cdf)
        expected += float(value) * probability
        previous_cdf = cdf

    if not math.isclose(previous_cdf, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        expected += float(max(base_best, float(np.max(block_targets)))) * max(0.0, 1.0 - previous_cdf)
    return float(expected)


def _hit_probability_with_partial_tie(
    block_best_mask: np.ndarray,
    sample_size: int,
    deterministic_hit: bool,
) -> float:
    """Probability that a top-k with random tie-breaking includes a best item."""
    if deterministic_hit:
        return 1.0
    block_best_mask = np.asarray(block_best_mask, dtype=bool)
    block_size = int(block_best_mask.size)
    if sample_size <= 0:
        return 0.0
    best_count = int(block_best_mask.sum())
    if best_count == 0:
        return 0.0
    if sample_size >= block_size:
        return 1.0
    return float(1.0 - _comb_ratio(block_size - best_count, sample_size, block_size))


def _tie_aware_topk_best_and_hit(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    k: int,
    best_true: float,
) -> tuple[float, float]:
    """
    Expected Best@k and Hit@k under uniform random ordering of score ties.

    Higher score blocks are always selected first. If the cutoff falls inside a
    tied-score block, every subset of the remaining tied candidates is treated
    as equally likely. This makes top-k metrics invariant to dataframe row order.
    """
    selected = 0
    deterministic_best = -np.inf
    deterministic_hit = False

    for score in sorted(np.unique(y_score), reverse=True):
        block_mask = y_score == score
        block_targets = y_true[block_mask]
        block_size = int(block_targets.size)
        remaining = k - selected
        if remaining <= 0:
            break
        if remaining >= block_size:
            deterministic_best = max(deterministic_best, float(np.max(block_targets)))
            deterministic_hit = deterministic_hit or bool(np.any(np.isclose(block_targets, best_true)))
            selected += block_size
            continue

        expected_best = _expected_max_with_partial_tie(
            block_targets,
            sample_size=remaining,
            base_best=deterministic_best,
        )
        hit_probability = _hit_probability_with_partial_tie(
            np.isclose(block_targets, best_true),
            sample_size=remaining,
            deterministic_hit=deterministic_hit,
        )
        return expected_best, hit_probability

    return float(deterministic_best), float(deterministic_hit)

def compute_ranking_metrics_by_front(
    df_fold: pd.DataFrame,
    front_col: str,
    target_col: str,
    score_col: str,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> pd.DataFrame:
    """
    Compute ranking metrics at front/group granularity.

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
        Cutoffs for metrics@k.

    Returns
    -------
    pd.DataFrame
        One row per front with ranking metrics computed from that front only.

    Notes
    -----
    * Regret@k = max(y_true) - E[max(y_true@topk_by_score)]
    * BestAUPR@k = E[max(y_true@topk_by_score)]
    * Hit@k = probability that a best-AUPR item appears in the top-k.
    * If the predicted-score cutoff falls inside a tie, every ordering of the
      tied items is treated uniformly; metrics never depend on dataframe row
      order.
    """
    per_group: List[Dict[str, float]] = []

    for front_id, group in df_fold.groupby(front_col):
        y_true = group[target_col].to_numpy().reshape(1, -1)
        y_score = group[score_col].to_numpy().reshape(1, -1)

        y_true_flat = y_true.ravel().astype(float)
        y_score_flat = y_score.ravel().astype(float)
        ndcgs = {
            f"NDCG@{k}": float(ndcg_score(y_true, y_score, k=k, ignore_ties=False))
            for k in ks
            if k <= group.shape[0]
        }
        best_true = float(np.max(y_true_flat)) if group.shape[0] > 0 else 0.0
        regret_at_k = {}
        best_aupr_at_k = {}
        hit_at_k = {}
        for k in ks:
            if k > group.shape[0]:
                continue
            expected_best, hit_probability = _tie_aware_topk_best_and_hit(
                y_true_flat,
                y_score_flat,
                k=int(k),
                best_true=best_true,
            )
            regret_at_k[f"Regret@{k}"] = float(max(0.0, best_true - expected_best))
            best_aupr_at_k[f"BestAUPR@{k}"] = float(expected_best)
            hit_at_k[f"Hit@{k}"] = float(hit_probability)

        # Rank correlations
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConstantInputWarning)
            rho = spearmanr(y_true.ravel(), y_score.ravel()).statistic
        rho = 0.0 if not np.isfinite(rho) else float(rho)
        ktau = kendalltau(y_true.ravel(), y_score.ravel()).correlation

        per_group.append(
            {
                front_col: front_id,
                **regret_at_k,
                **best_aupr_at_k,
                **hit_at_k,
                **ndcgs,
                "Spearman": float(rho if np.isfinite(rho) else 0.0),
                "KendallTau": float(ktau if np.isfinite(ktau) else 0.0),
                "n_items": int(group.shape[0]),
            }
        )

    return pd.DataFrame(per_group)


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
        Cutoffs for metrics@k.

    Returns
    -------
    agg : dict[str, float]
        Aggregated metrics (mean over groups) for all found keys.
    per_group : list[dict[str, float]]
        Per-group metrics including decision-first top-k metrics and
        lower-priority global rank correlations.
    """
    per_group_df = compute_ranking_metrics_by_front(
        df_fold=df_fold,
        front_col=front_col,
        target_col=target_col,
        score_col=score_col,
        ks=ks,
    )

    # Aggregate as simple mean across groups
    agg: Dict[str, float] = {}
    per_group = per_group_df.to_dict(orient="records")
    if per_group:
        keys = sorted((set().union(*per_group)) - {front_col}, key=metric_report_order)
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
    model_params: Optional[Dict[str, Any]] = None,
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
    model_params = dict(model_params or {})

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
            if has_valid:
                y_va_q = y_va.astype(int)
                max_label = int(max(y_tr_q.max(), y_va_q.max()))
            else:
                max_label = int(y_tr_q.max())
            label_gain = list(range(max_label + 1))

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
        params.update(model_params)
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

        params = dict(
            objective="rank:ndcg",
            eval_metric="ndcg@10",
            learning_rate=0.05,
            n_estimators=2000,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            random_state=random_state,
            ndcg_exp_gain=False,
            early_stopping_rounds=100 if has_valid else None,
        )
        params.update(model_params)
        model = XGBRanker(**params)

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
        params = dict(
            loss_function="YetiRank",
            eval_metric="NDCG:top=10",
            learning_rate=0.05,
            depth=6,
            iterations=2000,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
            od_type="Iter" if has_valid else "IncToDec",  # OD only if validation is present
            od_wait=100 if has_valid else None,
        )
        params.update(model_params)
        model = CatBoostRanker(**params)
        if has_valid:
            valid_pool = Pool(X_va, y_va, group_id=gid_valid)
            model.fit(train_pool, eval_set=valid_pool)
            scores_va = np.asarray(model.predict(valid_pool))
            return model, scores_va
        else:
            model.fit(train_pool)
            return model, None

    raise ValueError(f"Unsupported model type: {model_type}")
