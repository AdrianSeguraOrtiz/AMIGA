# amiga/core/main.py
"""
Pure logic (no disk I/O) for:
- Training Learn-to-Rank models with group-based cross-validation.
- Generating rankings with a trained model.
- Extracting features from gene expression matrices and Gene Regulatory Networks (GRNs).

Notes
-----
- This module DOES NOT read or write files to disk.
- The CLI layer is responsible for parsing arguments and persisting results.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from amiga.utils import row_weights_from_front, weighted_confidence

from amiga.features.expression import (
    features_condition_stats,
    features_correlations,
    features_gene_stats,
    features_exp_global,
    features_pca,
    features_timeseries,
)
from amiga.features.grn import (
    build_digraph,
    features_advanced,
    features_assortativity,
    features_clustering,
    features_communities,
    features_grn_global,
    features_paths,
    features_reciprocity,
    features_strength,
)
from amiga.selection.learn2rank import (
    LabelMode,
    ModelType,
    build_labels,
    compute_ranking_metrics,
    fit_ranker,
)

# ---------------------------------------------------------------------
# Typed result containers
# ---------------------------------------------------------------------

@dataclass
class TrainFoldReport:
    """
    Per-fold validation report.

    Attributes
    ----------
    fold : int
        Fold index (1..n_splits).
    agg : Dict[str, float]
        Aggregated ranking metrics on the validation set (e.g., NDCG@k, P@k, Regret@k).
    groups : List[Dict[str, Any]]
        Per-group/front metrics (one entry per value of `front_col` present in validation).
    label_mode : str
        Label construction mode used (value of `LabelMode`).
    label_quantiles : Optional[int]
        Number of quantiles Q if `label_mode == QUANTILES`; `None` otherwise.
    """
    fold: int
    agg: Dict[str, float]
    groups: List[Dict[str, Any]]
    label_mode: str
    label_quantiles: Optional[int]


@dataclass
class TrainResult:
    """
    Complete LTR training result.

    Attributes
    ----------
    models : List[object]
        List of trained models (one per fold).
    feature_columns : List[str]
        Ordered list of feature column names used for training.
    fold_reports : List[TrainFoldReport]
        Per-fold reports including aggregated and per-group metrics.
    valid_folds : List[pd.DataFrame]
        List of validation DataFrames (one per fold) with original columns + 'score'.
    """
    models: List[object]
    feature_columns: List[str]
    fold_reports: List[TrainFoldReport]
    valid_folds: List[pd.DataFrame]  # each df includes original columns + 'score'


@dataclass
class FitResult:
    """
    Result of fitting a single model.

    Attributes
    ----------
    model : object
        Trained model instance.
    feature_columns : List[str]
        Names of the feature columns used for training.
    label_mode : str
        Label mode used for training (value of `LabelMode`).
    label_quantiles : Optional[int]
        Number of quantiles Q if `label_mode == QUANTILES`; `None` otherwise.
    metadata : Dict[str, Any]
        Additional metadata provided by the training routine.
    """
    model: object
    feature_columns: List[str]
    label_mode: str
    label_quantiles: Optional[int]
    metadata: Dict[str, Any]


@dataclass
class RankResult:
    """
    Result of applying a trained model to generate rankings.

    Attributes
    ----------
    df_ranked : pd.DataFrame
        Input DataFrame with added 'score' and 'rank_in_front' columns, sorted by (front, rank).
    feature_columns_used : List[str]
        Feature columns effectively used at predict time (including order).
    """
    df_ranked: pd.DataFrame
    feature_columns_used: List[str]


@dataclass
class ExprFeaturesResult:
    """
    Result of gene expression feature extraction.

    Attributes
    ----------
    metrics : Dict[str, Any]
        Flat dictionary with global/per-gene/per-condition stats, correlations, PCA, and/or time-series metrics.
    """
    metrics: Dict[str, Any]


@dataclass
class GRNFeaturesResult:
    """
    Result of weighted GRN feature extraction.

    Attributes
    ----------
    metrics : Dict[str, Any]
        Flat dictionary with global metrics, strength, assortativity, paths, clustering,
        communities, reciprocity, and/or advanced metrics (depending on flags).
    """
    metrics: Dict[str, Any]


# ---------------------------------------------------------------------
# Core logic (no disk I/O)
# ---------------------------------------------------------------------

def train_ltr_cv(
    df: pd.DataFrame,
    model_type: ModelType = ModelType.LGBMRanker,
    front_col: str = "front_id",
    target_col: str = "AUPR",
    id_col: str = "item_id",
    drop_cols: Optional[List[str]] = None,
    label_mode: LabelMode = LabelMode.RANK_DENSE,
    label_quantiles: int = 20,
    n_splits: int = 5,
    random_state: int = 42,
) -> TrainResult:
    """
    Train with GroupKFold by fronts (experimental workflow).

    Parameters
    ----------
    df : pd.DataFrame
        Table containing control columns and numeric feature columns.
    model_type : ModelType
        Ranker type to fit (e.g., LightGBM/XGBoost/CatBoost rankers).
    front_col : str
        Column that identifies the group/front (used as groups in GroupKFold).
    target_col : str
        Column with the target quality metric to transform into labels.
    id_col : str
        Identifier column for the individual (not used as a feature).
    drop_cols : list[str] | None
        Additional columns to exclude from features.
    label_mode : LabelMode
        Strategy for building intra-front labels.
    label_quantiles : int
        Number of bins if `label_mode == QUANTILES`.
    n_splits : int
        Number of folds for GroupKFold (splits by `front_col`).
    random_state : int
        Base seed for reproducibility (offset by fold index).

    Returns
    -------
    TrainResult
        Trained models, feature order, per-fold reports, and validation folds with scores.
    """
    drop_cols = drop_cols or []

    # Validate required columns exist
    for col in (front_col, target_col, id_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Build intra-front labels according to the selected mode
    targets = build_labels(df, front_col, target_col, mode=label_mode, n_quantiles=label_quantiles)

    # Select feature columns
    drops = set([front_col, target_col, id_col, *drop_cols])
    feature_columns = [c for c in df.columns if c not in drops]
    if not feature_columns:
        raise ValueError("No feature columns remain after excluding control columns.")

    features = df[feature_columns].astype(float)
    group_ids = df[front_col].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    fold_reports: List[TrainFoldReport] = []
    models: List[object] = []
    valid_folds: List[pd.DataFrame] = []

    # Cross-validation loop
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(features, targets, group_ids), start=1):
        X_tr, y_tr, gid_tr = features.iloc[tr_idx], targets[tr_idx], group_ids[tr_idx]
        X_va, y_va, gid_va = features.iloc[va_idx], targets[va_idx], group_ids[va_idx]

        # Fit model and get validation scores
        model, scores_va = fit_ranker(
            model_type,
            X_tr, y_tr, gid_tr,
            random_state=random_state + fold_idx,
            X_valid=X_va, y_valid=y_va, gid_valid=gid_va,
        )
        models.append(model)

        # Compose validation frame with scores
        df_va = df.iloc[va_idx].copy()
        df_va["score"] = scores_va
        valid_folds.append(df_va)

        # Compute ranking metrics on validation fold
        agg, per_group = compute_ranking_metrics(df_va, front_col, target_col, "score")
        fold_reports.append(
            TrainFoldReport(
                fold=fold_idx,
                agg=agg,
                groups=per_group,
                label_mode=label_mode.value,
                label_quantiles=label_quantiles if label_mode == LabelMode.QUANTILES else None,
            )
        )

        # Console-friendly summary (kept here so that CLI can capture stdout if desired)
        print(
            f"[Fold {fold_idx}] NDCG@1={agg.get('NDCG@1', 0.0):.4f}  "
            f"P@1={agg.get('P@1', 0.0):.4f}  "
            f"Regret@1={agg.get('Regret@1', 0.0):.4f}",
        )

    return TrainResult(
        models=models,
        feature_columns=feature_columns,
        fold_reports=fold_reports,
        valid_folds=valid_folds,
    )


def train_ltr_full(
    df: pd.DataFrame,
    model_type: ModelType = ModelType.LGBMRanker,
    front_col: str = "front_id",
    target_col: str = "AUPR",
    id_col: str = "item_id",
    drop_cols: Optional[List[str]] = None,
    label_mode: LabelMode = LabelMode.RANK_DENSE,
    label_quantiles: int = 20,
    random_state: int = 42,
) -> FitResult:
    """
    Train a single model on the entire CSV (production workflow).

    Parameters
    ----------
    df : pd.DataFrame
        Table containing control columns and numeric feature columns.
    model_type : ModelType
        Ranker type to fit.
    front_col, target_col, id_col : str
        Control columns (id is not used as a feature).
    drop_cols : list[str] | None
        Additional columns to exclude from features.
    label_mode : LabelMode
        Strategy for building intra-front labels.
    label_quantiles : int
        Number of bins if `label_mode == QUANTILES`.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    FitResult
        Trained model, feature order, label config, and training metadata.
    """
    drop_cols = drop_cols or []
    for col in (front_col, target_col, id_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    y = build_labels(df, front_col, target_col, mode=label_mode, n_quantiles=label_quantiles)

    drops = set([front_col, target_col, id_col, *drop_cols])
    feature_columns = [c for c in df.columns if c not in drops]
    if not feature_columns:
        raise ValueError("No feature columns remain after excluding control columns.")

    X = df[feature_columns].astype(float)
    groups = df[front_col].to_numpy()

    # Full training without validation set
    model, _ = fit_ranker(
        model_type,
        X, y, groups,
        random_state=random_state,
        X_valid=None, y_valid=None, gid_valid=None,
    )

    meta = {
        "model_type": getattr(model_type, "value", str(model_type)),
        "front_col": front_col,
        "target_col": target_col,
        "id_col": id_col,
        "drop_cols": drop_cols,
        "random_state": random_state,
        "label_mode": getattr(label_mode, "value", str(label_mode)),
        "label_quantiles": label_quantiles if label_mode == LabelMode.QUANTILES else None,
    }

    return FitResult(
        model=model,
        feature_columns=feature_columns,
        label_mode=meta["label_mode"],
        label_quantiles=meta["label_quantiles"],
        metadata=meta,
    )


def rank_with_model(
    df: pd.DataFrame,
    model: object,
    front_col: str = "front_id",
    id_col: str = "item_id",
    drop_cols: Optional[List[str]] = None,
    feature_columns_hint: Optional[List[str]] = None,
) -> RankResult:
    """
    Apply a trained LTR model to score and rank individuals within each front.

    Parameters
    ----------
    df : pd.DataFrame
        Table with control columns and numeric feature columns.
    model : object
        Trained model exposing `.predict(X) -> np.ndarray`.
    front_col : str, optional
        Group/front column; defines the ranking scope.
    id_col : str, optional
        Identifier column for the individual (not used as a feature).
    drop_cols : list[str] | None, optional
        Extra columns to exclude from features.
    feature_columns_hint : list[str] | None, optional
        Expected feature order (e.g., from training). If provided, presence is validated.

    Returns
    -------
    RankResult
        DataFrame with `score` and `rank_in_front`, and the list of feature columns used.

    Raises
    ------
    ValueError
        If required hinted columns are missing, or no valid features remain.
    """
    drop_cols = drop_cols or []
    if feature_columns_hint:
        missing = [c for c in feature_columns_hint if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns in inference CSV: {missing}")
        feature_columns = list(feature_columns_hint)
    else:
        drops = set([front_col, id_col, *drop_cols])
        feature_columns = [c for c in df.columns if c not in drops]

    if not feature_columns:
        raise ValueError("No feature columns available for prediction.")

    features = df[feature_columns].astype(float)
    scores = np.asarray(model.predict(features))

    df_out = df.copy()
    df_out["score"] = scores
    df_out["rank_in_front"] = (
        df_out.groupby(front_col)["score"].rank(ascending=False, method="first").astype(int)
    )
    df_out = df_out.sort_values([front_col, "rank_in_front"], ascending=[True, True])

    return RankResult(df_ranked=df_out, feature_columns_used=feature_columns)


def extract_expression_features(
    df_expr: pd.DataFrame,
    include_global: bool = True,
    include_gene_stats: bool = True,
    include_condition_stats: bool = True,
    include_correlations: bool = True,
    include_pca: bool = True,
    include_timeseries: bool = False,
    corr_max_pairs: Optional[int] = None,
    pca_top_k: int = 5,
    pca_center_by: str = "condition",
) -> ExprFeaturesResult:
    """
    Extract features from a gene expression matrix.

    Parameters
    ----------
    df_expr : pd.DataFrame
        Matrix with genes in rows and conditions/timepoints in columns.
    include_global : bool
        Include global size/statistics.
    include_gene_stats : bool
        Include aggregated per-gene statistics.
    include_condition_stats : bool
        Include aggregated per-condition statistics.
    include_correlations : bool
        Include gene–gene and condition–condition correlations (may sample pairs to speed up).
    include_pca : bool
        Include PCA/SVD and explained-variance ratios (top_k).
    include_timeseries : bool
        Include per-gene temporal metrics (assumes columns are time-ordered).
    corr_max_pairs : int | None
        Maximum sampled pairs per correlation family (None = no limit).
    pca_top_k : int
        Number of principal components to report.
    pca_center_by : str
        Centering prior to PCA: 'condition' (center by columns) or 'gene' (center by rows).

    Returns
    -------
    ExprFeaturesResult
        Flat dictionary with the selected metrics.
    """
    results: Dict[str, Any] = {}
    if include_global:            results.update(features_exp_global(df_expr))
    if include_gene_stats:        results.update(features_gene_stats(df_expr))
    if include_condition_stats:   results.update(features_condition_stats(df_expr))
    if include_correlations:      results.update(features_correlations(df_expr, max_pairs=corr_max_pairs))
    if include_pca:               results.update(features_pca(df_expr, top_k=pca_top_k, center_by=pca_center_by))
    if include_timeseries:        results.update(features_timeseries(df_expr, max_lag=1))
    return ExprFeaturesResult(metrics=results)


def extract_grnet_features(
    df_edges: pd.DataFrame,   # columns: Source, Target, Confidence
    include_global: bool = True,
    include_strength: bool = True,
    include_assortativity: bool = True,
    include_paths: bool = True,
    include_clustering: bool = True,
    include_communities: bool = True,
    include_reciprocity: bool = True,
    include_advanced: bool = False,
    top_frac: float = 0.10,
) -> GRNFeaturesResult:
    """
    Extract WEIGHTED features from a directed Gene Regulatory Network (GRN).

    Parameters
    ----------
    df_edges : pd.DataFrame
        Edge list with columns ['Source', 'Target', 'Confidence'] (numeric weights in 'Confidence').
    include_global : bool
        Include weighted global metrics (density, weight distribution, top-X concentration, etc.).
    include_strength : bool
        Include per-node in/out strength metrics (weighted sums).
    include_assortativity : bool
        Include weighted out→in assortativity.
    include_paths : bool
        Include path-based metrics: average distance and p95 diameter (weighted).
    include_clustering : bool
        Include weighted average clustering (treated as undirected for the coefficient).
    include_communities : bool
        Include community detection (Louvain) with weights.
    include_reciprocity : bool
        Include weighted reciprocity.
    include_advanced : bool
        Include advanced metrics (e.g., entropies).
    top_frac : float
        Fraction for top-X-by-weight indicators (e.g., 0.10 = top 10%).

    Returns
    -------
    GRNFeaturesResult
        Flat dictionary with the selected metrics.
    """
    # Build weighted digraph from edge list
    G = build_digraph(df_edges, source="Source", target="Target", weight="Confidence")
    weights = np.array([d["Confidence"] for *_, d in G.edges(data=True)], dtype=float)

    # Accumulate requested metric families
    results: Dict[str, Any] = {}
    if include_global:        results.update(features_grn_global(G, weights, top_frac=top_frac))
    if include_strength:      results.update(features_strength(G))
    if include_assortativity: results.update(features_assortativity(G))
    if include_paths:         results.update(features_paths(G))
    if include_clustering:    results.update(features_clustering(G))
    if include_communities:   results.update(features_communities(G))
    if include_reciprocity:   results.update(features_reciprocity(G))
    if include_advanced:      results.update(features_advanced(G, weights))
    return GRNFeaturesResult(metrics=results)


def build_data(
    df_front: pd.DataFrame,
    df_expr: pd.DataFrame,
    grn_dir: Path,
    front_id: int,
    drop_front_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build a 'training'-like dataset by combining:
      - ALL columns from the evaluated front (minus 'drop_front_cols'),
      - expr_features (shared by all rows),
      - grnet_features (specific per individual after consensus).

    Requirements
    ------------
    - `df_front` must contain 'AUPR' and 'GRN_*.csv' columns (per-technique weights).

    Parameters
    ----------
    df_front : pd.DataFrame
        Evaluated front with AUPR, objective levels and GRN weight columns.
    df_expr : pd.DataFrame
        Expression matrix (rows=genes, columns=conditions/timepoints).
    grn_dir : Path
        Directory containing GRN_*.csv files.
    front_id : int
        Numeric front identifier to assign to all rows.
    drop_front_cols : list[str] | None
        Front columns to exclude from the output (defaults to ['Accuracy Mean', 'AUROC']).

    Returns
    -------
    pd.DataFrame
        Consolidated dataset with control columns, preserved front columns, and feature blocks.
    """
    drop_front_cols = drop_front_cols or ["Accuracy Mean", "AUROC"]

    if "AUPR" not in df_front.columns:
        raise ValueError("The evaluated front CSV must contain the 'AUPR' column.")

    # 1) Expression feature block (shared across individuals)
    expr_res = extract_expression_features(df_expr)
    expr_feats = expr_res.metrics  # flat dict

    # 2) Output rows accumulator
    out_rows: List[Dict[str, Any]] = []

    # 3) Validate GRN directory
    grn_dir = Path(grn_dir)
    if not grn_dir.exists():
        raise FileNotFoundError(f"GRN directory not found: {grn_dir}")

    # 4) Iterate individuals in the front
    for idx, row in df_front.reset_index(drop=True).iterrows():
        # Preserve original front columns except those dropped
        base_front = row.drop(labels=[c for c in drop_front_cols if c in row.index], errors="ignore").to_dict()

        # Build consensus instruction from per-technique weights in the row
        weights = row_weights_from_front(row)
        summands: List[str] = []
        for fname, w in weights.items():
            fpath = grn_dir / fname
            if not fpath.exists():
                raise FileNotFoundError(f"GRN file not found: {fpath}")
            # weighted_confidence expects summands like "w*path"
            summands.append(f"{w}*{str(fpath)}")

        # Run consensus, then compute GRN features on the result
        df_cons = weighted_confidence(weight_file_summand=summands)
        grn_res = extract_grnet_features(df_edges=df_cons)
        grn_feats = grn_res.metrics

        # Compose output row
        row_out: Dict[str, Any] = {
            "front_id": int(front_id),
            "item_id": int(idx + 1),
        }

        # 2.1 Original front columns (avoid overwriting control columns)
        for k, v in base_front.items():
            if k not in ("front_id", "item_id"):
                row_out[k] = v

        # 2.2 Expression features (ensure 'expr_' prefix)
        for k, v in expr_feats.items():
            key = k if k.startswith(("expr_", "exp_")) else f"expr_{k}"
            # If the value is a list, expand into indexed keys
            if isinstance(v, list):
                for i, val in enumerate(v):
                    row_out[f"{key}_{i}"] = val
            else:
                row_out[key] = v

        # 2.3 GRN features (ensure 'grn_' prefix; handle collisions defensively)
        for k, v in grn_feats.items():
            key = k if k.startswith(("grn_", "net_")) else f"grn_{k}"
            if key in row_out:
                key = f"grn_{k}"
            row_out[key] = v

        out_rows.append(row_out)

    df_out = pd.DataFrame(out_rows)

    # Final column ordering: controls + preserved front columns + sorted feature columns
    control_cols = ["front_id", "item_id"]
    front_cols_kept = [c for c in df_front.columns if c not in drop_front_cols and c not in ("front_id", "item_id")]
    feature_cols = [c for c in df_out.columns if c not in control_cols + front_cols_kept]

    df_out = df_out[control_cols + front_cols_kept + sorted(feature_cols)]
    return df_out
