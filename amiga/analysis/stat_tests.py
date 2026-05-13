from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, norm, rankdata
from statsmodels.stats.multitest import multipletests

def is_lower_better(metric_name: str) -> bool:
    return metric_name.startswith("Regret@")


def build_front_metric_matrix_from_frames(
    metrics_by_model: Mapping[str, pd.DataFrame],
    metric_name: str,
    *,
    front_col: str = "front_id",
) -> pd.DataFrame:
    series = []
    for model_name, metrics_df in metrics_by_model.items():
        if front_col not in metrics_df.columns or metric_name not in metrics_df.columns:
            continue
        metric_series = (
            metrics_df[[front_col, metric_name]]
            .rename(columns={metric_name: model_name})
            .set_index(front_col)
        )
        series.append(metric_series)
    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1, join="inner").sort_index()


def rank_matrix(values_df: pd.DataFrame, *, lower_is_better: bool) -> pd.DataFrame:
    if values_df.empty:
        return values_df.copy()
    ranked = np.zeros_like(values_df.to_numpy(dtype=float), dtype=float)
    for idx, row in enumerate(values_df.to_numpy(dtype=float)):
        ranked[idx, :] = rankdata(row if lower_is_better else -row, method="average")
    return pd.DataFrame(ranked, index=values_df.index, columns=values_df.columns)


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return np.asarray([], dtype=float)
    return multipletests(p_values, method="holm")[1]


def friedman_with_holm(rank_matrix_df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, float]]:
    matrix_df = rank_matrix_df.dropna(axis=0, how="any").copy()
    if matrix_df.empty or matrix_df.shape[1] < 2:
        return pd.DataFrame(), {"friedman_stat": np.nan, "friedman_p": np.nan, "n_fronts": 0}

    n_fronts = int(matrix_df.shape[0])
    n_methods = int(matrix_df.shape[1])
    avg_ranks = matrix_df.mean(axis=0).sort_values(ascending=True)
    winner = avg_ranks.index[0]

    if n_methods >= 3 and n_fronts >= 2:
        friedman_stat, friedman_p = friedmanchisquare(*(matrix_df[col].to_numpy() for col in matrix_df.columns))
        friedman_stat = float(friedman_stat)
        friedman_p = float(friedman_p)
    else:
        friedman_stat = np.nan
        friedman_p = np.nan

    scale = float(np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_fronts)))
    raw_rows = []
    for model_name, avg_rank in avg_ranks.items():
        if model_name == winner:
            raw_rows.append(
                {
                    "model": model_name,
                    "avg_rank": float(avg_rank),
                    "holm_p_adj": np.nan,
                    "is_winner": True,
                    "significantly_worse": False,
                }
            )
            continue
        z_value = abs(float(avg_rank - avg_ranks[winner])) / scale
        raw_rows.append(
            {
                "model": model_name,
                "avg_rank": float(avg_rank),
                "_raw_p": float(2.0 * norm.sf(z_value)),
                "is_winner": False,
            }
        )

    loser_rows = [row for row in raw_rows if not row["is_winner"]]
    adjusted = _holm_adjust([row["_raw_p"] for row in loser_rows])
    for row, p_adj in zip(loser_rows, adjusted):
        row["holm_p_adj"] = float(p_adj)
        row["significantly_worse"] = bool(p_adj < 0.05)
        row.pop("_raw_p", None)

    stats_rows = []
    for row in raw_rows:
        row["friedman_stat"] = friedman_stat
        row["friedman_p"] = friedman_p
        row["n_fronts"] = n_fronts
        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows).sort_values(["avg_rank", "model"]).reset_index(drop=True)
    return stats_df, {
        "friedman_stat": friedman_stat,
        "friedman_p": friedman_p,
        "n_fronts": n_fronts,
    }


def compute_metric_rank_stats_from_frames(
    metrics_by_model: Mapping[str, pd.DataFrame],
    metric_name: str,
    *,
    front_col: str = "front_id",
) -> pd.DataFrame:
    metric_matrix = build_front_metric_matrix_from_frames(
        metrics_by_model,
        metric_name,
        front_col=front_col,
    )
    if metric_matrix.empty:
        return pd.DataFrame()
    rank_df = rank_matrix(metric_matrix, lower_is_better=is_lower_better(metric_name))
    stats_df, _ = friedman_with_holm(rank_df)
    if stats_df.empty:
        return pd.DataFrame()
    stats_df["metric"] = metric_name
    return stats_df


def compute_metric_stats_from_frames(
    metrics_by_model: Mapping[str, pd.DataFrame],
    metric_names: Sequence[str],
    *,
    front_col: str = "front_id",
) -> pd.DataFrame:
    rows = []
    for metric_name in metric_names:
        stats_df = compute_metric_rank_stats_from_frames(
            metrics_by_model,
            metric_name,
            front_col=front_col,
        )
        if not stats_df.empty:
            rows.append(stats_df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compute_global_metric_stats(
    metrics_by_model: Mapping[str, pd.DataFrame],
    metric_names: Sequence[str],
    *,
    front_col: str = "front_id",
) -> pd.DataFrame:
    return compute_metric_stats_from_frames(
        metrics_by_model,
        metric_names,
        front_col=front_col,
    )
