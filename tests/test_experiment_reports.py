from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.experiments.amiga_exp.reports import ReportSummaryError, summarize_phase_cv_reports


def _write_report(path: Path, model: str, regret5: float, hit5: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "fold": 1,
            "agg": {
                "Regret@1": regret5 + 0.1,
                "Regret@5": regret5,
                "Hit@5": hit5,
                "BestAUPR@5": 0.9 - regret5,
                "n_items": 3,
            },
            "groups": [
                {
                    "front_id": 1,
                    "Regret@1": regret5 + 0.1,
                    "Regret@5": regret5,
                    "Hit@5": hit5,
                    "BestAUPR@5": 0.9 - regret5,
                    "n_items": 3,
                },
                {
                    "front_id": 2,
                    "Regret@1": regret5 + 0.2,
                    "Regret@5": regret5 + 0.05,
                    "Hit@5": hit5,
                    "BestAUPR@5": 0.85 - regret5,
                    "n_items": 3,
                },
            ],
            "label_mode": "continuous",
            "label_quantiles": None,
            "meta": {"model": model, "label_mode": "continuous", "label_quantiles": None},
        }
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_summarize_phase_cv_reports_writes_expected_outputs(tmp_path):
    report_a = _write_report(tmp_path / "run_a" / "cv_report.json", "ModelA", regret5=0.1, hit5=1.0)
    report_b = _write_report(tmp_path / "run_b" / "cv_report.json", "ModelB", regret5=0.2, hit5=0.0)
    summary_dir = tmp_path / "summary"

    outputs = summarize_phase_cv_reports([report_a, report_b], summary_dir)

    assert set(outputs) == {"metrics_long", "metrics_summary", "metric_ranks", "metric_rank_stats"}
    for path in outputs.values():
        assert path.exists()
    summary = pd.read_csv(outputs["metrics_summary"])
    assert "Regret@5" in set(summary["metric"])
    rank_stats = pd.read_csv(outputs["metric_rank_stats"])
    assert "Regret@5" in set(rank_stats["metric"])


def test_summarize_phase_cv_reports_fails_without_reports(tmp_path):
    with pytest.raises(ReportSummaryError, match="at least one"):
        summarize_phase_cv_reports([], tmp_path / "summary")


def test_summarize_phase_cv_reports_fails_for_missing_report_path(tmp_path):
    with pytest.raises(ReportSummaryError, match="missing cv_report"):
        summarize_phase_cv_reports([tmp_path / "missing" / "cv_report.json"], tmp_path / "summary")


def test_summarize_phase_cv_reports_requires_regret5(tmp_path):
    path = tmp_path / "run" / "cv_report.json"
    path.parent.mkdir(parents=True)
    payload = [
        {
            "fold": 1,
            "agg": {"Regret@1": 0.1, "Hit@1": 1.0, "n_items": 3},
            "groups": [{"front_id": 1, "Regret@1": 0.1, "Hit@1": 1.0, "n_items": 3}],
            "label_mode": "continuous",
            "label_quantiles": None,
            "meta": {"model": "NoRegret5"},
        }
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ReportSummaryError, match="Regret@5"):
        summarize_phase_cv_reports([path], tmp_path / "summary")
