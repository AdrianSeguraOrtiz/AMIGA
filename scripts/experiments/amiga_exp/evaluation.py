"""Ranking evaluation helpers for AMIGA experimental phases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from amiga.selection.learn2rank import compute_ranking_metrics, compute_ranking_metrics_by_front


DEFAULT_EVALUATION_KS = (1, 3, 5, 10)
DEFAULT_REQUIRED_METRIC = "Regret@5"


class RankingEvaluationError(ValueError):
    """Raised when a ranked frame cannot be evaluated safely."""


@dataclass(frozen=True)
class RankingEvaluation:
    """Evaluation result for one ranked split."""

    report: list[dict[str, Any]]
    agg: dict[str, float]
    groups: list[dict[str, Any]]
    per_front_metrics: pd.DataFrame


def validate_ranked_frame(
    ranked_df: pd.DataFrame,
    *,
    front_col: str,
    target_col: str,
    score_col: str,
) -> None:
    """Validate the minimum columns and values needed for ranking metrics."""
    if ranked_df.empty:
        raise RankingEvaluationError("ranked dataframe is empty")

    required_columns = [front_col, target_col, score_col]
    missing = [column for column in required_columns if column not in ranked_df.columns]
    if missing:
        raise RankingEvaluationError(f"ranked dataframe is missing required column(s): {missing}")

    null_counts = {
        column: int(ranked_df[column].isna().sum())
        for column in required_columns
        if int(ranked_df[column].isna().sum()) > 0
    }
    if null_counts:
        raise RankingEvaluationError(f"ranked dataframe contains null values: {null_counts}")

    for column in (target_col, score_col):
        numeric = pd.to_numeric(ranked_df[column], errors="coerce")
        if numeric.isna().any():
            raise RankingEvaluationError(f"ranked dataframe column '{column}' must be numeric")


def _normalize_group_keys(groups: list[dict[str, Any]], *, front_col: str) -> list[dict[str, Any]]:
    if front_col == "front_id":
        return groups

    normalized = []
    for group in groups:
        group_copy = dict(group)
        group_copy["front_id"] = group_copy.pop(front_col)
        normalized.append(group_copy)
    return normalized


def _normalize_per_front_columns(per_front_metrics: pd.DataFrame, *, front_col: str) -> pd.DataFrame:
    if front_col == "front_id":
        return per_front_metrics
    return per_front_metrics.rename(columns={front_col: "front_id"})


def evaluate_ranked_frame(
    ranked_df: pd.DataFrame,
    *,
    front_col: str = "front_id",
    target_col: str = "AUPR",
    score_col: str = "score",
    model_name: str,
    evaluation_split: str = "test",
    fold: int = 1,
    ks: Sequence[int] = DEFAULT_EVALUATION_KS,
    meta: dict[str, Any] | None = None,
    required_metric: str = DEFAULT_REQUIRED_METRIC,
) -> RankingEvaluation:
    """Evaluate one ranked dataframe and build a cv_report-compatible payload."""
    validate_ranked_frame(
        ranked_df,
        front_col=front_col,
        target_col=target_col,
        score_col=score_col,
    )

    working = ranked_df.copy()
    working[target_col] = pd.to_numeric(working[target_col], errors="raise")
    working[score_col] = pd.to_numeric(working[score_col], errors="raise")

    per_front_metrics = compute_ranking_metrics_by_front(
        working,
        front_col=front_col,
        target_col=target_col,
        score_col=score_col,
        ks=ks,
    )
    agg, groups = compute_ranking_metrics(
        working,
        front_col=front_col,
        target_col=target_col,
        score_col=score_col,
        ks=ks,
    )

    if required_metric not in agg:
        raise RankingEvaluationError(
            f"required metric '{required_metric}' was not produced; "
            "check front sizes and requested cutoffs"
        )

    normalized_groups = _normalize_group_keys(groups, front_col=front_col)
    normalized_per_front = _normalize_per_front_columns(per_front_metrics, front_col=front_col)
    report_meta = {
        "model": model_name,
        "evaluation_split": evaluation_split,
        "front_col": front_col,
        "target_col": target_col,
        "score_col": score_col,
        "ks": [int(k) for k in ks],
        **(meta or {}),
    }
    report = [
        {
            "fold": int(fold),
            "agg": agg,
            "groups": normalized_groups,
            "label_mode": None,
            "label_quantiles": None,
            "meta": report_meta,
        }
    ]
    return RankingEvaluation(
        report=report,
        agg=agg,
        groups=normalized_groups,
        per_front_metrics=normalized_per_front,
    )


def write_cv_report(path: Path, report: list[dict[str, Any]]) -> Path:
    """Write a cv_report.json payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def evaluate_ranked_csv(
    ranked_csv: Path,
    out_report: Path,
    *,
    front_col: str = "front_id",
    target_col: str = "AUPR",
    score_col: str = "score",
    model_name: str,
    evaluation_split: str = "test",
    fold: int = 1,
    ks: Sequence[int] = DEFAULT_EVALUATION_KS,
    meta: dict[str, Any] | None = None,
) -> RankingEvaluation:
    """Load a ranked CSV, evaluate it, and write a cv_report.json."""
    ranked_df = pd.read_csv(ranked_csv)
    evaluation = evaluate_ranked_frame(
        ranked_df,
        front_col=front_col,
        target_col=target_col,
        score_col=score_col,
        model_name=model_name,
        evaluation_split=evaluation_split,
        fold=fold,
        ks=ks,
        meta=meta,
    )
    write_cv_report(out_report, evaluation.report)
    return evaluation


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    return value
