from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.experiments.amiga_exp.config_selection import (
    ConfigSelectionError,
    select_configurations,
    selection_rank_stats_from_reports,
)


def _summary(metric_rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for run_id, metrics in metric_rows.items():
        for metric, mean in metrics.items():
            rows.append(
                {
                    "model": run_id,
                    "metric": metric,
                    "tier": "primary",
                    "priority": 1,
                    "mean": mean,
                    "std": 0.0,
                    "n": 2,
                }
            )
    return pd.DataFrame(rows)


def _rank_stats(avg_ranks: dict[str, float]) -> pd.DataFrame:
    winner = min(avg_ranks, key=avg_ranks.get)
    rows = []
    for run_id, avg_rank in avg_ranks.items():
        rows.append(
            {
                "model": run_id,
                "metric": "Regret@5",
                "avg_rank": avg_rank,
                "holm_p_adj": None if run_id == winner else 0.2,
                "is_winner": run_id == winner,
                "significantly_worse": False,
                "friedman_stat": 1.0,
                "friedman_p": 0.3,
                "n_fronts": 4,
            }
        )
    return pd.DataFrame(rows)


def _write_cv_report(tmp_path, run_id: str, regret_values: list[float]):
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    report_path = run_dir / "cv_report.json"
    report = [
        {
            "fold": 0,
            "agg": {"Regret@5": sum(regret_values) / len(regret_values)},
            "groups": [
                {"front_id": f"front_{idx}", "Regret@5": value}
                for idx, value in enumerate(regret_values)
            ],
        }
    ]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path


def test_selection_rank_stats_from_reports_includes_control_label_modes(tmp_path):
    candidates = [
        {"run_id": "run_candidate", "model_type": "LGBMRanker", "label_mode": "continuous"},
        {"run_id": "run_reversed", "model_type": "LGBMRanker", "label_mode": "reversed"},
    ]
    stats = selection_rank_stats_from_reports(
        candidates,
        {
            "run_candidate": _write_cv_report(tmp_path, "run_candidate", [0.3, 0.2]),
            "run_reversed": _write_cv_report(tmp_path, "run_reversed", [0.0, 0.0]),
        },
    )

    assert stats["model"].tolist() == ["run_reversed", "run_candidate"]
    assert stats.loc[stats["model"] == "run_reversed", "avg_rank"].iloc[0] == 1.0


def test_select_configurations_uses_paired_avg_rank_before_mean_regret5():
    candidates = [
        {"run_id": "run_a", "model_type": "LGBMRanker", "label_mode": "continuous"},
        {"run_id": "run_b", "model_type": "XGBRanker", "label_mode": "continuous"},
    ]
    metrics = _summary(
        {
            "run_a": {"Regret@5": 0.05, "Hit@5": 1.0, "Regret@1": 0.1, "BestAUPR@5": 0.8},
            "run_b": {"Regret@5": 0.20, "Hit@5": 0.0, "Regret@1": 0.4, "BestAUPR@5": 0.4},
        }
    )

    selection = select_configurations(
        candidates,
        metrics,
        _rank_stats({"run_a": 2.0, "run_b": 1.0}),
        selection_size=1,
    )

    selected = selection.selected_configs[0]
    assert selected["run_id"] == "run_b"
    assert selected["selection_sort_values"]["Regret@5_avg_rank"] == 1.0
    assert selected["selection_statistical_evidence"]["p_value_used_as_gate"] is False
    assert selection.selection_rule["selection_basis"] == "paired_front_avg_rank"


def test_select_configurations_uses_hit5_as_first_tie_breaker():
    candidates = [
        {"run_id": "run_a", "model_type": "LGBMRanker", "label_mode": "continuous"},
        {"run_id": "run_b", "model_type": "XGBRanker", "label_mode": "continuous"},
    ]
    metrics = _summary(
        {
            "run_a": {"Regret@5": 0.1, "Hit@5": 0.5, "Regret@1": 0.2, "BestAUPR@5": 0.8},
            "run_b": {"Regret@5": 0.1, "Hit@5": 1.0, "Regret@1": 0.3, "BestAUPR@5": 0.7},
        }
    )

    selection = select_configurations(
        candidates,
        metrics,
        _rank_stats({"run_a": 1.5, "run_b": 1.5}),
        selection_size=1,
    )

    selected = selection.selected_configs[0]
    assert selected["run_id"] == "run_b"
    assert selected["selection_rank"] == 1
    assert "Hit@5=1.0" in selected["selection_reason"]
    assert selection.selection_rule["primary_metric"] == "Regret@5"


def test_select_configurations_uses_model_stability_after_metric_ties():
    candidates = [
        {"run_id": "run_xgb", "model_type": "XGBRanker", "label_mode": "continuous"},
        {"run_id": "run_lgbm", "model_type": "LGBMRanker", "label_mode": "continuous"},
    ]
    tied = {"Regret@5": 0.1, "Hit@5": 1.0, "Regret@1": 0.2, "BestAUPR@5": 0.8}
    metrics = _summary({"run_xgb": tied, "run_lgbm": tied})

    selection = select_configurations(
        candidates,
        metrics,
        _rank_stats({"run_xgb": 1.5, "run_lgbm": 1.5}),
        selection_size=1,
    )

    selected = selection.selected_configs[0]
    assert selected["run_id"] == "run_lgbm"
    assert "simpler/stable model family" in selected["selection_reason"]
    assert selected["selection_sort_values"]["model_priority"] == 0


def test_select_configurations_allows_control_label_modes_to_compete():
    candidates = [
        {"run_id": "run_reversed", "model_type": "LGBMRanker", "label_mode": "reversed"},
        {"run_id": "run_continuous", "model_type": "LGBMRanker", "label_mode": "continuous"},
    ]
    metrics = _summary(
        {
            "run_reversed": {"Regret@5": 0.0, "Hit@5": 1.0, "Regret@1": 0.0, "BestAUPR@5": 0.9},
            "run_continuous": {"Regret@5": 0.2, "Hit@5": 0.0, "Regret@1": 0.4, "BestAUPR@5": 0.6},
        }
    )

    selection = select_configurations(
        candidates,
        metrics,
        _rank_stats({"run_reversed": 1.0, "run_continuous": 2.0}),
        selection_size=1,
    )

    assert selection.selected_configs[0]["run_id"] == "run_reversed"
    assert selection.excluded_configs == ()


def test_select_configurations_excludes_explicitly_non_selectable_candidates():
    candidates = [
        {
            "run_id": "run_control",
            "model_type": "LGBMRanker",
            "label_mode": "continuous",
            "selectable": False,
        },
        {"run_id": "run_candidate", "model_type": "LGBMRanker", "label_mode": "continuous"},
    ]
    metrics = _summary(
        {
            "run_control": {"Regret@5": 0.0, "Hit@5": 1.0, "Regret@1": 0.0, "BestAUPR@5": 0.9},
            "run_candidate": {"Regret@5": 0.2, "Hit@5": 0.0, "Regret@1": 0.4, "BestAUPR@5": 0.6},
        }
    )

    selection = select_configurations(
        candidates,
        metrics,
        _rank_stats({"run_candidate": 1.0}),
        selection_size=1,
    )

    assert selection.selected_configs[0]["run_id"] == "run_candidate"
    assert selection.excluded_configs[0]["exclusion_reason"] == "selectable=false"


def test_select_configurations_requires_primary_metric():
    candidates = [{"run_id": "run_a", "model_type": "LGBMRanker", "label_mode": "continuous"}]
    metrics = _summary({"run_a": {"Hit@5": 1.0}})

    with pytest.raises(ConfigSelectionError, match="Regret@5"):
        select_configurations(candidates, metrics, _rank_stats({"run_a": 1.0}))
