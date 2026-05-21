"""Cross-validation report summarization helpers for experimental phases."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from amiga.analysis.cv_reports import summarize_cv_reports


DEFAULT_SUMMARY_STATS = ("metric_rank_stats",)
EXPECTED_SUMMARY_OUTPUTS = (
    "metrics_long",
    "metrics_summary",
    "metric_ranks",
    "metric_rank_stats",
)
DEFAULT_REQUIRED_METRIC = "Regret@5"


class ReportSummaryError(ValueError):
    """Raised when CV report summarization cannot produce valid phase outputs."""


def summarize_phase_cv_reports(
    report_paths: Sequence[Path],
    summary_dir: Path,
    *,
    metrics: Sequence[str] | None = None,
    stats: Sequence[str] = DEFAULT_SUMMARY_STATS,
    required_metric: str = DEFAULT_REQUIRED_METRIC,
) -> dict[str, Path]:
    """Summarize phase CV reports with the experiment-wide defaults.

    This is the single wrapper experimental phases should use instead of
    calling ``summarize_cv_reports`` directly.
    """
    paths = [Path(path) for path in report_paths]
    if not paths:
        raise ReportSummaryError("at least one cv_report.json path is required")

    missing_paths = [path for path in paths if not path.exists()]
    if missing_paths:
        raise ReportSummaryError(f"missing cv_report.json path(s): {missing_paths}")

    outputs = summarize_cv_reports(
        report_paths=paths,
        out_dir=summary_dir,
        metrics=metrics,
        stats=stats,
    )

    missing_outputs = [name for name in EXPECTED_SUMMARY_OUTPUTS if name not in outputs]
    if missing_outputs:
        raise ReportSummaryError(f"summarize-cv did not produce expected output(s): {missing_outputs}")

    summary_path = outputs["metrics_summary"]
    summary_df = pd.read_csv(summary_path)
    if required_metric not in set(summary_df.get("metric", [])):
        raise ReportSummaryError(
            f"required metric '{required_metric}' is missing from {summary_path}"
        )

    return outputs
