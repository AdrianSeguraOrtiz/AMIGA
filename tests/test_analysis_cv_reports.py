from __future__ import annotations

import pandas as pd
import pytest

from amiga.analysis.cv_reports import plot_cv_summary, summarize_cv_reports


def test_summarize_cv_reports_uses_only_json_groups(sample_cv_reports, tmp_path):
    out_dir = tmp_path / "summary"
    outputs = summarize_cv_reports(list(sample_cv_reports.values()), out_dir, stats=["metric_rank_stats"])

    assert set(outputs) == {"metrics_long", "metrics_summary", "metric_ranks", "metric_rank_stats"}
    rank_stats = pd.read_csv(outputs["metric_rank_stats"])
    assert not rank_stats.empty
    assert set(rank_stats["metric"]) >= {"Regret@1", "Hit@1", "NDCG@1", "Spearman"}


def test_summarize_cv_reports_requires_groups_for_metric_rank_stats(sample_cv_report_without_groups, tmp_path):
    out_dir = tmp_path / "summary"
    with pytest.raises(ValueError, match="requires per-front 'groups' data"):
        summarize_cv_reports([sample_cv_report_without_groups], out_dir, stats=["metric_rank_stats"])


def test_plot_cv_summary_generates_all_supported_plots(sample_cv_reports, tmp_path):
    out_dir = tmp_path / "summary"
    summarize_cv_reports(list(sample_cv_reports.values()), out_dir, stats=["metric_rank_stats"])

    dotplot = plot_cv_summary(out_dir, plot="dotplot_overview")
    topk = plot_cv_summary(out_dir, plot="topk_curves", metric_prefix="Regret")
    heatmap = plot_cv_summary(out_dir, plot="metric_rank_heatmap", metrics=["Regret@1", "Hit@1"])
    scatter = plot_cv_summary(out_dir, plot="metric_scatter", x_metric="Regret@1", y_metric="Hit@1")

    for path in (dotplot, topk, heatmap, scatter):
        assert path.exists()
        assert path.stat().st_size > 0


def test_plot_cv_summary_validates_inputs(sample_cv_reports, tmp_path):
    out_dir = tmp_path / "summary"
    summarize_cv_reports(list(sample_cv_reports.values()), out_dir, stats=["metric_rank_stats"])

    with pytest.raises(ValueError, match="requires --metric-prefix"):
        plot_cv_summary(out_dir, plot="topk_curves")

    with pytest.raises(ValueError, match="Unsupported metric prefix"):
        plot_cv_summary(out_dir, plot="topk_curves", metric_prefix="BadMetric")

    with pytest.raises(ValueError, match="requires at least one metric"):
        plot_cv_summary(out_dir, plot="metric_rank_heatmap")

    with pytest.raises(ValueError, match="requires both --x-metric and --y-metric"):
        plot_cv_summary(out_dir, plot="metric_scatter", x_metric="Regret@1")

    with pytest.raises(ValueError, match="must be different"):
        plot_cv_summary(out_dir, plot="metric_scatter", x_metric="Regret@1", y_metric="Regret@1")
