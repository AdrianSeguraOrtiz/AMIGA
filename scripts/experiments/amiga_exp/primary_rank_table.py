"""Primary Regret@5 reporting table for AMIGA experimental phases."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from scripts.experiments.amiga_exp.config_selection import (
    PRIMARY_METRIC,
    selection_rank_stats_from_reports,
)


PRIMARY_RANK_TABLE_COLUMNS = (
    "config",
    "avg_rank",
    "p_value",
    "mean_regret5",
    "std_regret5",
    "n_fronts",
)
GROUPED_PRIMARY_RANK_TABLE_COLUMNS = (
    "comparison_group",
    *PRIMARY_RANK_TABLE_COLUMNS,
)


class PrimaryRankTableError(ValueError):
    """Raised when the primary rank reporting table cannot be built safely."""


def build_primary_rank_table(
    candidate_configs: Sequence[Mapping[str, Any]],
    cv_report_paths_by_run: Mapping[str, Path],
    metrics_summary: pd.DataFrame | Path,
    *,
    metric: str = PRIMARY_METRIC,
) -> pd.DataFrame:
    """Build the canonical phase-level primary rank table.

    The resulting table is intentionally compact and centered on the pre-defined
    primary metric. Generic summaries still keep all secondary metrics for audit
    and supplementary analyses.
    """
    if metric != PRIMARY_METRIC:
        raise PrimaryRankTableError(f"primary rank table is fixed to '{PRIMARY_METRIC}'")
    rank_stats = selection_rank_stats_from_reports(
        candidate_configs,
        cv_report_paths_by_run,
        metric=metric,
    )
    summary = _coerce_metrics_summary(metrics_summary)
    metric_summary = _primary_metric_summary(summary, metric=metric)

    table = (
        rank_stats[["model", "avg_rank", "holm_p_adj", "n_fronts"]]
        .rename(
            columns={
                "model": "config",
                "holm_p_adj": "p_value",
            }
        )
        .merge(metric_summary, on="config", how="left", validate="one_to_one")
    )
    if table[["mean_regret5", "std_regret5"]].isna().all(axis=1).any():
        missing = sorted(table.loc[table["mean_regret5"].isna(), "config"].astype(str).tolist())
        raise PrimaryRankTableError(
            f"metrics summary is missing '{metric}' mean/std for config(s): {missing}"
        )

    table = table.sort_values(["avg_rank", "config"], kind="mergesort").reset_index(drop=True)
    table = table.loc[:, PRIMARY_RANK_TABLE_COLUMNS]
    return table


def write_primary_rank_table(
    candidate_configs: Sequence[Mapping[str, Any]],
    cv_report_paths_by_run: Mapping[str, Path],
    metrics_summary: pd.DataFrame | Path,
    output_path: Path,
    *,
    metric: str = PRIMARY_METRIC,
) -> Path:
    """Write ``primary_rank_table.csv`` and return its path."""
    table = build_primary_rank_table(
        candidate_configs,
        cv_report_paths_by_run,
        metrics_summary,
        metric=metric,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return output_path


def build_grouped_primary_rank_table(
    candidate_configs: Sequence[Mapping[str, Any]],
    cv_report_paths_by_run: Mapping[str, Path],
    metrics_summary: pd.DataFrame | Path,
    *,
    group_field: str,
    output_group_column: str = "comparison_group",
    metric: str = PRIMARY_METRIC,
) -> pd.DataFrame:
    """Build a primary rank table with independent comparisons per group."""
    if not group_field:
        raise PrimaryRankTableError("group_field must be a non-empty string")
    if not candidate_configs:
        raise PrimaryRankTableError("at least one candidate configuration is required")

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for candidate in candidate_configs:
        if group_field not in candidate:
            run_id = str(candidate.get("run_id") or candidate.get("model") or "<unknown>")
            raise PrimaryRankTableError(
                f"candidate '{run_id}' is missing grouping field '{group_field}'"
            )
        group_value = str(candidate[group_field])
        grouped.setdefault(group_value, []).append(candidate)

    tables: list[pd.DataFrame] = []
    for group_value, group_candidates in grouped.items():
        group_table = build_primary_rank_table(
            group_candidates,
            cv_report_paths_by_run,
            metrics_summary,
            metric=metric,
        )
        group_table.insert(0, output_group_column, group_value)
        tables.append(group_table)

    if not tables:
        raise PrimaryRankTableError("no grouped primary rank table rows were produced")

    table = pd.concat(tables, ignore_index=True)
    return table


def write_grouped_primary_rank_table(
    candidate_configs: Sequence[Mapping[str, Any]],
    cv_report_paths_by_run: Mapping[str, Path],
    metrics_summary: pd.DataFrame | Path,
    output_path: Path,
    *,
    group_field: str,
    output_group_column: str = "comparison_group",
    metric: str = PRIMARY_METRIC,
) -> Path:
    """Write a grouped ``primary_rank_table.csv`` and return its path."""
    table = build_grouped_primary_rank_table(
        candidate_configs,
        cv_report_paths_by_run,
        metrics_summary,
        group_field=group_field,
        output_group_column=output_group_column,
        metric=metric,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return output_path


def _coerce_metrics_summary(metrics_summary: pd.DataFrame | Path) -> pd.DataFrame:
    summary = pd.read_csv(metrics_summary) if isinstance(metrics_summary, Path) else metrics_summary.copy()
    required = {"model", "metric", "mean", "std"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise PrimaryRankTableError(f"metrics summary is missing required column(s): {missing}")
    if summary.empty:
        raise PrimaryRankTableError("metrics summary is empty")
    return summary


def _primary_metric_summary(summary: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    metric_df = summary[summary["metric"].astype(str) == metric].copy()
    if metric_df.empty:
        raise PrimaryRankTableError(f"metrics summary is missing primary metric '{metric}'")
    metric_df = metric_df.rename(
        columns={
            "model": "config",
            "mean": "mean_regret5",
            "std": "std_regret5",
        }
    )
    return metric_df[["config", "mean_regret5", "std_regret5"]]
