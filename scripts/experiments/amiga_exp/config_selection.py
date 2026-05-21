"""Shared configuration selector for AMIGA experimental phases."""

from __future__ import annotations

from collections.abc import Mapping as AbcMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from amiga.analysis.cv_reports import load_report_group_metrics
from amiga.analysis.stat_tests import compute_metric_rank_stats_from_frames


PRIMARY_METRIC = "Regret@5"
PRIMARY_RANK_FIELD = f"{PRIMARY_METRIC}_avg_rank"
PRIMARY_HOLM_FIELD = f"{PRIMARY_METRIC}_holm_p_adj"
PRIMARY_WINNER_FIELD = f"{PRIMARY_METRIC}_is_statistical_winner"
PRIMARY_SIGNIFICANT_FIELD = f"{PRIMARY_METRIC}_significantly_worse"
PRIMARY_FRIEDMAN_P_FIELD = f"{PRIMARY_METRIC}_friedman_p"
PRIMARY_N_FRONTS_FIELD = f"{PRIMARY_METRIC}_rank_n_fronts"
DEFAULT_TIE_BREAKERS = ("Hit@5", "Regret@1", "BestAUPR@5")
MODEL_STABILITY_PRIORITY = {
    "LGBMRanker": 0,
    "XGBRanker": 1,
    "CatBoostRanker": 2,
}

CriterionSource = Literal["metric", "rank_stat", "model_priority"]


class ConfigSelectionError(ValueError):
    """Raised when experiment configurations cannot be selected safely."""


@dataclass(frozen=True)
class SelectionCriterion:
    """One ordered criterion in the official selection rule."""

    field: str
    source: CriterionSource
    ascending: bool
    description: str

    @property
    def direction(self) -> str:
        return "minimize" if self.ascending else "maximize"


@dataclass(frozen=True)
class ConfigurationSelection:
    """Result of applying the official selection rule."""

    selected_configs: tuple[dict[str, Any], ...]
    excluded_configs: tuple[dict[str, Any], ...]
    candidate_table: pd.DataFrame
    selection_rule: dict[str, Any]


DEFAULT_SELECTION_CRITERIA = (
    SelectionCriterion(
        PRIMARY_RANK_FIELD,
        "rank_stat",
        True,
        "lower paired average rank for Regret@5",
    ),
    SelectionCriterion(PRIMARY_METRIC, "metric", True, "lower mean Regret@5"),
    SelectionCriterion("Hit@5", "metric", False, "higher Hit@5"),
    SelectionCriterion("Regret@1", "metric", True, "lower Regret@1"),
    SelectionCriterion("BestAUPR@5", "metric", False, "higher BestAUPR@5"),
    SelectionCriterion("model_priority", "model_priority", True, "simpler/stable model family"),
)


def select_configurations(
    candidate_configs: Sequence[Mapping[str, Any]],
    metrics_summary: pd.DataFrame | Path,
    metric_rank_stats: pd.DataFrame | Path | None = None,
    *,
    selection_size: int = 1,
    criteria: Sequence[SelectionCriterion] = DEFAULT_SELECTION_CRITERIA,
    model_stability_priority: Mapping[str, int] = MODEL_STABILITY_PRIORITY,
) -> ConfigurationSelection:
    """Select configurations using the experiment-wide primary-plus-tie-breaker rule."""
    if selection_size < 1:
        raise ConfigSelectionError("selection_size must be >= 1")
    if not candidate_configs:
        raise ConfigSelectionError("at least one candidate configuration is required")
    if not criteria:
        raise ConfigSelectionError("at least one selection criterion is required")
    if criteria[0].field != PRIMARY_RANK_FIELD:
        raise ConfigSelectionError(f"first selection criterion must be '{PRIMARY_RANK_FIELD}'")

    summary = _coerce_metrics_summary(metrics_summary)
    metrics_by_run = _metrics_by_run(summary)
    rank_stats_source = (
        metric_rank_stats
        if metric_rank_stats is not None
        else _infer_metric_rank_stats_path(metrics_summary)
    )
    if rank_stats_source is None:
        raise ConfigSelectionError(
            "metric_rank_stats with paired average ranks is required for configuration selection"
        )
    rank_stats = _coerce_metric_rank_stats(rank_stats_source)
    rank_stats_by_run = _rank_stats_by_run(rank_stats, metric=PRIMARY_METRIC)
    rows = [
        _candidate_row(
            candidate,
            metrics_by_run=metrics_by_run,
            rank_stats_by_run=rank_stats_by_run,
            model_stability_priority=model_stability_priority,
        )
        for candidate in candidate_configs
    ]
    candidate_table = pd.DataFrame(rows)
    selectable = candidate_table[~candidate_table["selection_excluded"]].copy()
    if selectable.empty:
        raise ConfigSelectionError("no selectable configurations remain after exclusions")

    sort_columns: list[str] = []
    ascending: list[bool] = []
    for criterion in criteria:
        if criterion.field not in selectable.columns:
            selectable[criterion.field] = _missing_sort_value(criterion)
        selectable[criterion.field] = selectable[criterion.field].fillna(_missing_sort_value(criterion))
        sort_columns.append(criterion.field)
        ascending.append(criterion.ascending)

    selectable = selectable.sort_values(
        [*sort_columns, "run_id"],
        ascending=[*ascending, True],
        kind="mergesort",
    ).reset_index(drop=True)

    selected_rows = selectable.head(selection_size).copy()
    selected_configs = tuple(
        _selected_record(row, rank=rank, n_available=len(selectable), criteria=criteria)
        for rank, (_, row) in enumerate(selected_rows.iterrows(), start=1)
    )
    excluded_configs = tuple(
        _excluded_record(row)
        for _, row in candidate_table[candidate_table["selection_excluded"]].iterrows()
    )
    return ConfigurationSelection(
        selected_configs=selected_configs,
        excluded_configs=excluded_configs,
        candidate_table=candidate_table,
        selection_rule=selection_rule_payload(
            criteria=criteria,
            model_stability_priority=model_stability_priority,
        ),
    )


def selection_rule_payload(
    *,
    criteria: Sequence[SelectionCriterion] = DEFAULT_SELECTION_CRITERIA,
    model_stability_priority: Mapping[str, int] = MODEL_STABILITY_PRIORITY,
) -> dict[str, Any]:
    """Return the explicit rule recorded in selection artifacts."""
    return {
        "primary_metric": PRIMARY_METRIC,
        "primary_selection_stat": PRIMARY_RANK_FIELD,
        "selection_basis": "paired_front_avg_rank",
        "tie_breakers": list(DEFAULT_TIE_BREAKERS),
        "p_value_policy": "Holm-adjusted p-values are reported as evidence and are not used as a selection gate.",
        "statistical_tests": {
            "global": "Friedman test over per-front ranks when at least three methods and two fronts are available",
            "post_hoc": "Holm-adjusted comparison against the best average rank",
            "alpha": 0.05,
            "selection_uses_p_value_gate": False,
        },
        "criteria": [
            {
                "field": criterion.field,
                "source": criterion.source,
                "direction": criterion.direction,
                "description": criterion.description,
            }
            for criterion in criteria
        ],
        "model_stability_priority": dict(model_stability_priority),
        "deterministic_fallback": "run_id",
    }


def selection_rank_stats_from_reports(
    candidate_configs: Sequence[Mapping[str, Any]],
    cv_report_paths_by_run: Mapping[str, Path],
    *,
    metric: str = PRIMARY_METRIC,
) -> pd.DataFrame:
    """Compute the paired rank statistics used for selecting among candidates."""
    report_paths = {str(run_id): Path(path) for run_id, path in cv_report_paths_by_run.items()}
    metrics_by_run: dict[str, pd.DataFrame] = {}
    missing_reports: list[str] = []

    for candidate in candidate_configs:
        record = dict(candidate)
        run_id = str(record.get("run_id") or record.get("model") or "")
        if not run_id:
            raise ConfigSelectionError("candidate configuration is missing 'run_id'")
        if _candidate_exclusion_reasons(record):
            continue
        report_path = report_paths.get(run_id)
        if report_path is None:
            missing_reports.append(run_id)
            continue
        metrics_by_run[run_id] = load_report_group_metrics(report_path)

    if missing_reports:
        raise ConfigSelectionError(
            "missing cv_report path(s) for selectable candidate(s): "
            + ", ".join(sorted(missing_reports))
        )
    if not metrics_by_run:
        raise ConfigSelectionError("at least one selectable configuration is required for rank selection")
    if len(metrics_by_run) == 1:
        run_id, metrics_df = next(iter(metrics_by_run.items()))
        if metric not in metrics_df.columns:
            raise ConfigSelectionError(f"selectable candidate '{run_id}' is missing '{metric}' in cv_report groups")
        return pd.DataFrame(
            [
                {
                    "model": run_id,
                    "metric": metric,
                    "avg_rank": 1.0,
                    "holm_p_adj": np.nan,
                    "is_winner": True,
                    "significantly_worse": False,
                    "friedman_stat": np.nan,
                    "friedman_p": np.nan,
                    "n_fronts": int(metrics_df["front_id"].nunique())
                    if "front_id" in metrics_df.columns
                    else int(len(metrics_df)),
                }
            ]
        )

    stats = compute_metric_rank_stats_from_frames(metrics_by_run, metric)
    if stats.empty:
        raise ConfigSelectionError(f"could not compute paired rank statistics for '{metric}'")
    return stats


def _coerce_metrics_summary(metrics_summary: pd.DataFrame | Path) -> pd.DataFrame:
    summary = pd.read_csv(metrics_summary) if isinstance(metrics_summary, Path) else metrics_summary.copy()
    required = {"model", "metric", "mean"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ConfigSelectionError(f"metrics summary is missing required column(s): {missing}")
    if summary.empty:
        raise ConfigSelectionError("metrics summary is empty")
    return summary


def _infer_metric_rank_stats_path(metrics_summary: pd.DataFrame | Path) -> Path | None:
    if not isinstance(metrics_summary, Path):
        return None
    inferred = metrics_summary.with_name("metric_rank_stats.csv")
    return inferred if inferred.exists() else None


def _coerce_metric_rank_stats(metric_rank_stats: pd.DataFrame | Path) -> pd.DataFrame:
    stats = pd.read_csv(metric_rank_stats) if isinstance(metric_rank_stats, Path) else metric_rank_stats.copy()
    required = {"model", "metric", "avg_rank"}
    missing = sorted(required - set(stats.columns))
    if missing:
        raise ConfigSelectionError(f"metric_rank_stats is missing required column(s): {missing}")
    stats = stats[stats["metric"].astype(str) == PRIMARY_METRIC].copy()
    if stats.empty:
        raise ConfigSelectionError(f"metric_rank_stats is missing primary metric '{PRIMARY_METRIC}'")
    return stats


def _metrics_by_run(summary: pd.DataFrame) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for row in summary.itertuples(index=False):
        run_metrics = metrics.setdefault(str(row.model), {})
        run_metrics[str(row.metric)] = float(row.mean)
    return metrics


def _rank_stats_by_run(stats: pd.DataFrame, *, metric: str) -> dict[str, dict[str, Any]]:
    stats_for_metric = stats[stats["metric"].astype(str) == metric]
    rank_stats: dict[str, dict[str, Any]] = {}
    for row in stats_for_metric.to_dict(orient="records"):
        run_id = str(row["model"])
        rank_stats[run_id] = {
            f"{metric}_avg_rank": float(row["avg_rank"]),
            f"{metric}_holm_p_adj": _optional_float(row.get("holm_p_adj")),
            f"{metric}_is_statistical_winner": _optional_bool(row.get("is_winner")),
            f"{metric}_significantly_worse": _optional_bool(row.get("significantly_worse")),
            f"{metric}_friedman_p": _optional_float(row.get("friedman_p")),
            f"{metric}_rank_n_fronts": _optional_int(row.get("n_fronts")),
        }
    return rank_stats


def _candidate_exclusion_reasons(
    record: Mapping[str, Any],
) -> list[str]:
    selectable = bool(record.get("selectable", True))
    exclusion_reasons: list[str] = []
    if not selectable:
        exclusion_reasons.append("selectable=false")
    return exclusion_reasons


def _candidate_row(
    candidate: Mapping[str, Any],
    *,
    metrics_by_run: Mapping[str, Mapping[str, float]],
    rank_stats_by_run: Mapping[str, Mapping[str, Any]],
    model_stability_priority: Mapping[str, int],
) -> dict[str, Any]:
    record = dict(candidate)
    run_id = str(record.get("run_id") or record.get("model") or "")
    if not run_id:
        raise ConfigSelectionError("candidate configuration is missing 'run_id'")
    metrics = dict(metrics_by_run.get(run_id, {}))
    if PRIMARY_METRIC not in metrics:
        raise ConfigSelectionError(f"candidate '{run_id}' is missing '{PRIMARY_METRIC}' in metrics summary")
    rank_stats = dict(rank_stats_by_run.get(run_id, {}))

    label_mode = str(record.get("label_mode", ""))
    selectable = bool(record.get("selectable", True))
    exclusion_reasons = _candidate_exclusion_reasons(record)
    if not exclusion_reasons and PRIMARY_RANK_FIELD not in rank_stats:
        raise ConfigSelectionError(
            f"candidate '{run_id}' is missing '{PRIMARY_RANK_FIELD}' in metric_rank_stats"
        )

    model_type = str(record.get("model_type", ""))
    return {
        **record,
        "run_id": run_id,
        "label_mode": label_mode,
        "selectable": selectable,
        **metrics,
        **rank_stats,
        "model_priority": int(model_stability_priority.get(model_type, 999)),
        "selection_excluded": bool(exclusion_reasons),
        "exclusion_reason": "; ".join(exclusion_reasons) if exclusion_reasons else None,
    }


def _selected_record(
    row: pd.Series,
    *,
    rank: int,
    n_available: int,
    criteria: Sequence[SelectionCriterion],
) -> dict[str, Any]:
    record = _public_record(row, drop={"selection_excluded", "exclusion_reason", "model_priority"})
    record["selection_rank"] = int(rank)
    record["selection_sort_values"] = {
        criterion.field: _native(row.get(criterion.field))
        for criterion in criteria
    }
    record["selection_statistical_evidence"] = _selection_statistical_evidence(row)
    record["selection_reason"] = _selection_reason(row, rank=rank, n_available=n_available, criteria=criteria)
    return record


def _excluded_record(row: pd.Series) -> dict[str, Any]:
    return _public_record(row, drop={"selection_excluded", "model_priority"})


def _public_record(row: pd.Series, *, drop: set[str]) -> dict[str, Any]:
    return {
        str(key): _native(value)
        for key, value in row.to_dict().items()
        if key not in drop
    }


def _selection_reason(
    row: pd.Series,
    *,
    rank: int,
    n_available: int,
    criteria: Sequence[SelectionCriterion],
) -> str:
    parts = []
    for criterion in criteria:
        value = _native(row.get(criterion.field))
        if criterion.source == "model_priority":
            parts.append(
                f"{criterion.description} ({row.get('model_type', 'unknown')} priority={value})"
            )
        else:
            parts.append(f"{criterion.description} ({criterion.field}={value})")
    p_value = _native(row.get(PRIMARY_HOLM_FIELD))
    p_value_text = "winner" if p_value is None else p_value
    return (
        f"Ranked {rank} of {n_available} selectable configurations: "
        + "; ".join(parts)
        + f". Holm p-value versus best average-rank configuration: {p_value_text}; "
        "p-values are reported, not used as a gate."
    )


def _selection_statistical_evidence(row: pd.Series) -> dict[str, Any]:
    return {
        "primary_metric": PRIMARY_METRIC,
        "selection_stat": PRIMARY_RANK_FIELD,
        "avg_rank": _native(row.get(PRIMARY_RANK_FIELD)),
        "holm_p_adj_vs_best_avg_rank": _native(row.get(PRIMARY_HOLM_FIELD)),
        "is_best_avg_rank": _native(row.get(PRIMARY_WINNER_FIELD)),
        "significantly_worse_than_best": _native(row.get(PRIMARY_SIGNIFICANT_FIELD)),
        "friedman_p": _native(row.get(PRIMARY_FRIEDMAN_P_FIELD)),
        "n_fronts": _native(row.get(PRIMARY_N_FRONTS_FIELD)),
        "alpha": 0.05,
        "p_value_used_as_gate": False,
    }


def _missing_sort_value(criterion: SelectionCriterion) -> float:
    return float("inf") if criterion.ascending else float("-inf")


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _native(value: Any) -> Any:
    if isinstance(value, AbcMapping):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value):
        return None
    return value
