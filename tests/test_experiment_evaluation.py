from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.experiments.amiga_exp.evaluation import (
    RankingEvaluationError,
    evaluate_ranked_csv,
    evaluate_ranked_frame,
    write_cv_report,
)
from scripts.experiments.amiga_exp.reports import summarize_phase_cv_reports


def _ranked_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "front_id": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            "item_id": [1, 2, 3, 4, 5, 6] * 2,
            "AUPR": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            "score": [0.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0],
        }
    )


def test_evaluate_ranked_frame_builds_cv_report_with_official_metrics():
    evaluation = evaluate_ranked_frame(
        _ranked_df(),
        model_name="FinalAMIGA",
        evaluation_split="test",
        ks=(1, 5),
    )

    assert len(evaluation.report) == 1
    assert evaluation.report[0]["fold"] == 1
    assert evaluation.report[0]["meta"]["model"] == "FinalAMIGA"
    assert evaluation.report[0]["meta"]["evaluation_split"] == "test"
    assert evaluation.agg["BestAUPR@5"] == pytest.approx(0.7)
    assert evaluation.agg["Regret@5"] == pytest.approx(0.05)
    assert evaluation.agg["Hit@5"] == pytest.approx(0.5)
    assert len(evaluation.groups) == 2
    assert set(evaluation.per_front_metrics["front_id"]) == {1, 2}


def test_write_cv_report_is_compatible_with_phase_summary(tmp_path):
    evaluation = evaluate_ranked_frame(
        _ranked_df(),
        model_name="FinalAMIGA",
        evaluation_split="test",
        ks=(1, 5),
    )
    report_path = write_cv_report(tmp_path / "final" / "cv_report.json", evaluation.report)

    outputs = summarize_phase_cv_reports([report_path], tmp_path / "summary")

    assert set(outputs) == {"metrics_long", "metrics_summary", "metric_ranks", "metric_rank_stats"}
    summary = pd.read_csv(outputs["metrics_summary"])
    assert "Regret@5" in set(summary["metric"])


def test_evaluate_ranked_csv_writes_report(tmp_path):
    ranked_csv = tmp_path / "ranked.csv"
    _ranked_df().to_csv(ranked_csv, index=False)
    report_path = tmp_path / "cv_report.json"

    evaluation = evaluate_ranked_csv(
        ranked_csv,
        report_path,
        model_name="Baseline",
        evaluation_split="test",
        ks=(1, 5),
    )

    assert report_path.exists()
    payload = json.loads(report_path.read_text())
    assert payload[0]["meta"]["model"] == "Baseline"
    assert payload[0]["agg"]["Regret@5"] == pytest.approx(evaluation.agg["Regret@5"])


def test_evaluate_ranked_frame_normalizes_custom_group_column_to_front_id():
    df = _ranked_df().rename(columns={"front_id": "front"})

    evaluation = evaluate_ranked_frame(
        df,
        front_col="front",
        model_name="CustomFrontColumn",
        ks=(1, 5),
    )

    assert "front_id" in evaluation.groups[0]
    assert "front" not in evaluation.groups[0]
    assert "front_id" in evaluation.per_front_metrics.columns


def test_evaluate_ranked_frame_requires_nonempty_input():
    with pytest.raises(RankingEvaluationError, match="empty"):
        evaluate_ranked_frame(pd.DataFrame(), model_name="Broken")


def test_evaluate_ranked_frame_requires_score_column():
    df = _ranked_df().drop(columns=["score"])

    with pytest.raises(RankingEvaluationError, match="missing"):
        evaluate_ranked_frame(df, model_name="Broken")


def test_evaluate_ranked_frame_requires_regret5_by_default():
    small_front_df = pd.DataFrame(
        {
            "front_id": [1, 1, 1],
            "AUPR": [1.0, 0.8, 0.2],
            "score": [0.0, 0.8, 0.2],
        }
    )

    with pytest.raises(RankingEvaluationError, match="Regret@5"):
        evaluate_ranked_frame(small_front_df, model_name="TooSmallForK5")
