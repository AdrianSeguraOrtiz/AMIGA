from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.experiments.amiga_exp.primary_rank_table import (
    PRIMARY_RANK_TABLE_COLUMNS,
    PrimaryRankTableError,
    build_grouped_primary_rank_table,
    build_primary_rank_table,
    write_grouped_primary_rank_table,
    write_primary_rank_table,
)


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


def _metrics_summary(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": run_id,
                "metric": metric,
                "tier": "primary",
                "priority": 1,
                "mean": values["mean"],
                "std": values["std"],
                "n": 3,
            }
            for run_id, metrics in rows.items()
            for metric, values in metrics.items()
        ]
    )


def test_write_primary_rank_table_includes_control_label_modes_and_adds_regret_summary(tmp_path):
    candidates = [
        {"run_id": "run_a", "label_mode": "continuous"},
        {"run_id": "run_b", "label_mode": "continuous"},
        {"run_id": "run_reversed", "label_mode": "reversed"},
    ]
    reports = {
        "run_a": _write_cv_report(tmp_path, "run_a", [0.1, 0.2, 0.3]),
        "run_b": _write_cv_report(tmp_path, "run_b", [0.3, 0.1, 0.4]),
        "run_reversed": _write_cv_report(tmp_path, "run_reversed", [0.0, 0.0, 0.0]),
    }
    summary = _metrics_summary(
        {
            "run_a": {"Regret@5": {"mean": 0.2, "std": 0.1}},
            "run_b": {"Regret@5": {"mean": 0.2667, "std": 0.1528}},
            "run_reversed": {"Regret@5": {"mean": 0.0, "std": 0.0}},
        }
    )

    output_path = write_primary_rank_table(
        candidates,
        reports,
        summary,
        tmp_path / "summary" / "primary_rank_table.csv",
    )

    table = pd.read_csv(output_path)
    assert list(table.columns) == list(PRIMARY_RANK_TABLE_COLUMNS)
    assert table["config"].tolist() == ["run_reversed", "run_a", "run_b"]
    assert table.loc[0, "avg_rank"] == pytest.approx(1.0)
    assert pd.isna(table.loc[0, "p_value"])
    assert table.loc[0, "mean_regret5"] == pytest.approx(0.0)
    assert table.loc[0, "std_regret5"] == pytest.approx(0.0)
    assert table.loc[0, "n_fronts"] == 3


def test_build_primary_rank_table_handles_one_config_smoke_case(tmp_path):
    candidates = [{"run_id": "run_single", "label_mode": "continuous"}]
    reports = {"run_single": _write_cv_report(tmp_path, "run_single", [0.4, 0.2])}
    summary = _metrics_summary(
        {
            "run_single": {"Regret@5": {"mean": 0.3, "std": 0.1414}},
        }
    )

    table = build_primary_rank_table(candidates, reports, summary)

    row = table.iloc[0]
    assert row["config"] == "run_single"
    assert row["avg_rank"] == 1.0
    assert pd.isna(row["p_value"])
    assert row["mean_regret5"] == 0.3
    assert row["std_regret5"] == 0.1414
    assert row["n_fronts"] == 2


def test_write_grouped_primary_rank_table_compares_candidates_within_each_group(tmp_path):
    candidates = [
        {"run_id": "lgbm_a", "model_type": "LGBMRanker", "label_mode": "continuous"},
        {"run_id": "lgbm_b", "model_type": "LGBMRanker", "label_mode": "rank_avg"},
        {"run_id": "xgb_a", "model_type": "XGBRanker", "label_mode": "continuous"},
        {"run_id": "xgb_b", "model_type": "XGBRanker", "label_mode": "rank_avg"},
    ]
    reports = {
        "lgbm_a": _write_cv_report(tmp_path, "lgbm_a", [0.1, 0.2, 0.1]),
        "lgbm_b": _write_cv_report(tmp_path, "lgbm_b", [0.3, 0.4, 0.3]),
        "xgb_a": _write_cv_report(tmp_path, "xgb_a", [0.5, 0.6, 0.5]),
        "xgb_b": _write_cv_report(tmp_path, "xgb_b", [0.2, 0.1, 0.2]),
    }
    summary = _metrics_summary(
        {
            "lgbm_a": {"Regret@5": {"mean": 0.1333, "std": 0.0577}},
            "lgbm_b": {"Regret@5": {"mean": 0.3333, "std": 0.0577}},
            "xgb_a": {"Regret@5": {"mean": 0.5333, "std": 0.0577}},
            "xgb_b": {"Regret@5": {"mean": 0.1667, "std": 0.0577}},
        }
    )

    output_path = write_grouped_primary_rank_table(
        candidates,
        reports,
        summary,
        tmp_path / "summary" / "primary_rank_table.csv",
        group_field="model_type",
        output_group_column="model_type",
    )

    table = pd.read_csv(output_path)
    assert list(table.columns) == ["model_type", *PRIMARY_RANK_TABLE_COLUMNS]
    assert table["config"].tolist() == ["lgbm_a", "lgbm_b", "xgb_b", "xgb_a"]
    winners = table[table["p_value"].isna()]
    assert winners[["model_type", "config"]].to_dict(orient="records") == [
        {"model_type": "LGBMRanker", "config": "lgbm_a"},
        {"model_type": "XGBRanker", "config": "xgb_b"},
    ]
    assert table.groupby("model_type")["avg_rank"].min().to_dict() == {
        "LGBMRanker": 1.0,
        "XGBRanker": 1.0,
    }


def test_build_grouped_primary_rank_table_requires_group_field(tmp_path):
    candidates = [{"run_id": "run_a", "label_mode": "continuous"}]
    reports = {"run_a": _write_cv_report(tmp_path, "run_a", [0.1, 0.2])}
    summary = _metrics_summary({"run_a": {"Regret@5": {"mean": 0.15, "std": 0.07}}})

    with pytest.raises(PrimaryRankTableError, match="grouping field"):
        build_grouped_primary_rank_table(
            candidates,
            reports,
            summary,
            group_field="model_type",
        )


def test_build_primary_rank_table_requires_primary_metric(tmp_path):
    candidates = [{"run_id": "run_a", "label_mode": "continuous"}]
    reports = {"run_a": _write_cv_report(tmp_path, "run_a", [0.1, 0.2])}
    summary = _metrics_summary({"run_a": {"Hit@5": {"mean": 1.0, "std": 0.0}}})

    with pytest.raises(PrimaryRankTableError, match="Regret@5"):
        build_primary_rank_table(candidates, reports, summary)


def test_build_primary_rank_table_is_fixed_to_regret5(tmp_path):
    candidates = [{"run_id": "run_a", "label_mode": "continuous"}]
    reports = {"run_a": _write_cv_report(tmp_path, "run_a", [0.1, 0.2])}
    summary = _metrics_summary({"run_a": {"Regret@5": {"mean": 0.15, "std": 0.07}}})

    with pytest.raises(PrimaryRankTableError, match="Regret@5"):
        build_primary_rank_table(candidates, reports, summary, metric="Hit@5")
