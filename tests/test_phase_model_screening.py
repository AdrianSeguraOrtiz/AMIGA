from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from amiga.selection.learn2rank import LabelMode, ModelType
from scripts.experiments.amiga_exp.context import load_case_context
from scripts.experiments.amiga_exp.manifests import WriteOptions
from scripts.experiments.amiga_exp.phases.phase_01_model_screening import (
    PHASE_NAME,
    REFERENCE_MODEL_PARAMS,
    ScreeningRunConfig,
    ScreeningRunResult,
    _model_params_for_run,
    _select_best_labeling_per_model_type,
    build_screening_run_configs,
    run_model_screening,
)


SMALL_LGBM_PARAMS = {
    "n_estimators": 8,
    "num_leaves": 7,
    "learning_rate": 0.1,
    "min_child_samples": 1,
    "verbose": -1,
}


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_screening_case(tmp_path: Path) -> tuple[Path, Path]:
    case_dir = tmp_path / "CASE"
    data_dir = case_dir / "data"
    data_dir.mkdir(parents=True)
    rows = []
    for front_id in range(1, 6):
        for item_id in range(1, 7):
            quality = item_id / 10.0 + front_id / 1000.0
            rows.append(
                {
                    "front_id": front_id,
                    "item_id": item_id,
                    "AUPR": quality,
                    "objective": 1.0 - quality,
                    "f_signal": quality,
                    "f_inverse": 1.0 - quality,
                }
            )
    pd.DataFrame(rows).to_csv(data_dir / "data_1.csv", index=False)

    split_manifest = _write_json(
        tmp_path / "CASE_split_manifest.json",
        {
            "case": "CASE",
            "assignments": [
                {"front_id": front_id, "front_name": f"front_{front_id}", "split": "development"}
                for front_id in range(1, 5)
            ]
            + [{"front_id": 5, "front_name": "front_5", "split": "test"}],
        },
    )
    feature_contract = _write_json(
        tmp_path / "CASE_feature_columns.json",
        {
            "case": "CASE",
            "target_column": "AUPR",
            "control_columns": ["front_id", "item_id"],
            "objective_columns": ["objective"],
            "objective_directions": {"objective": "minimize"},
            "feature_sets": {"full": ["objective", "f_signal", "f_inverse"]},
        },
    )
    config = _write_json(
        tmp_path / "CASE.json",
        {
            "case": "CASE",
            "data_csv": str(data_dir / "data_1.csv"),
            "split_manifest": str(split_manifest),
            "feature_contract": str(feature_contract),
            "group_column": "front_id",
            "item_column": "item_id",
            "target_column": "AUPR",
        },
    )
    return case_dir, config


def test_build_screening_run_configs_treats_control_label_modes_as_selectable():
    configs = build_screening_run_configs(
        model_types=[ModelType.LGBMRanker],
        label_modes=[LabelMode.CONTINUOUS, LabelMode.REVERSED, LabelMode.QUANTILES],
        label_quantiles=[5, 10],
    )

    assert [config.run_id for config in configs] == [
        "LGBMRanker__continuous",
        "LGBMRanker__reversed",
        "LGBMRanker__quantiles_q5",
        "LGBMRanker__quantiles_q10",
    ]
    selectable = {config.run_id: config.selectable for config in configs}
    assert selectable["LGBMRanker__continuous"] is True
    assert selectable["LGBMRanker__reversed"] is True
    assert selectable["LGBMRanker__quantiles_q5"] is True


def test_phase01_reference_model_params_are_explicit_and_overrideable():
    assert _model_params_for_run(ModelType.LGBMRanker, None) == REFERENCE_MODEL_PARAMS[ModelType.LGBMRanker]
    assert _model_params_for_run(ModelType.XGBRanker, None)["min_child_weight"] == 5
    catboost_params = _model_params_for_run(ModelType.CatBoostRanker, None)
    assert catboost_params["l2_leaf_reg"] == 5
    assert catboost_params["allow_writing_files"] is False

    lgbm_params = _model_params_for_run(
        ModelType.LGBMRanker,
        {"LGBMRanker": {"n_estimators": 8, "num_leaves": 7}},
    )
    assert lgbm_params["n_estimators"] == 8
    assert lgbm_params["num_leaves"] == 7
    assert lgbm_params["min_child_samples"] == 50

    forced_catboost_params = _model_params_for_run(
        ModelType.CatBoostRanker,
        {"CatBoostRanker": {"allow_writing_files": True, "iterations": 8}},
    )
    assert forced_catboost_params["iterations"] == 8
    assert forced_catboost_params["allow_writing_files"] is False


def test_run_model_screening_writes_run_summary_and_shortlist_without_test_fronts(tmp_path):
    case_dir, config = _write_screening_case(tmp_path)
    context = load_case_context(case_dir, config_path=config)

    result = run_model_screening(
        context,
        seed=7,
        options=WriteOptions(),
        model_types=[ModelType.LGBMRanker],
        label_modes=[LabelMode.CONTINUOUS],
        n_splits=2,
        model_params=SMALL_LGBM_PARAMS,
    )

    assert result.status == "written"
    assert result.phase_dir == context.results_root / PHASE_NAME
    assert len(result.run_results) == 1
    run = result.run_results[0]
    assert run.cv_report.exists()
    assert run.feature_columns.exists()
    assert run.run_manifest.exists()
    assert len(run.valid_fold_ranked) == 2
    assert all(path.exists() for path in run.valid_fold_ranked)

    report = json.loads(run.cv_report.read_text(encoding="utf-8"))
    assert len(report) == 2
    assert "Regret@5" in report[0]["agg"]
    features = json.loads(run.feature_columns.read_text(encoding="utf-8"))
    assert features["feature_columns"] == ["objective", "f_signal", "f_inverse"]

    valid_fronts = set()
    for ranked_path in run.valid_fold_ranked:
        ranked = pd.read_csv(ranked_path)
        valid_fronts.update(ranked["front_id"].unique().tolist())
    assert valid_fronts <= {1, 2, 3, 4}
    assert 5 not in valid_fronts

    assert (result.phase_dir / "summary" / "metrics_summary.csv").exists()
    assert (result.phase_dir / "summary" / "metrics_long.csv").exists()
    assert result.table_outputs["primary_rank_table"].exists()
    primary_table = pd.read_csv(result.table_outputs["primary_rank_table"])
    assert list(primary_table.columns) == [
        "model_type",
        "config",
        "avg_rank",
        "p_value",
        "mean_regret5",
        "std_regret5",
        "n_fronts",
    ]
    assert primary_table["model_type"].tolist() == ["LGBMRanker"]
    assert primary_table["config"].tolist() == ["LGBMRanker__continuous"]
    screening_manifest = json.loads(result.screening_manifest.read_text(encoding="utf-8"))
    assert screening_manifest["run_count"] == 1
    assert screening_manifest["tables"]["primary_rank_table"].endswith("primary_rank_table.csv")
    assert "plot_cv" not in screening_manifest
    assert screening_manifest["plots"]["plot_manifest"].endswith("plots/plot_manifest.json")
    assert screening_manifest["plots"]["primary"]["png"].endswith(
        "plots/model_screening_heatmap.png"
    )
    assert screening_manifest["plots"]["primary"]["pdf"].endswith(
        "plots/model_screening_heatmap.pdf"
    )
    phase_manifest = json.loads((result.phase_dir / "phase_manifest.json").read_text(encoding="utf-8"))
    assert phase_manifest["status"] == "completed"
    assert phase_manifest["outputs"]["screening_manifest"].endswith("screening_manifest.json")
    assert phase_manifest["outputs"]["run_count"] == 1
    assert phase_manifest["outputs"]["table_outputs"]["primary_rank_table"].endswith("primary_rank_table.csv")
    assert phase_manifest["outputs"]["plots"]["plot_manifest"].endswith("plots/plot_manifest.json")
    shortlist = json.loads(result.shortlisted_configs.read_text(encoding="utf-8"))
    assert shortlist["primary_metric"] == "Regret@5"
    assert shortlist["selection_rule"]["tie_breakers"] == ["Hit@5", "Regret@1", "BestAUPR@5"]
    assert shortlist["selection_rule"]["primary_selection_stat"] == "Regret@5_avg_rank"
    assert shortlist["selection_scope"] == "best_label_per_model_type"
    assert shortlist["selection_group_field"] == "model_type"
    assert shortlist["selected_per_group"] == 1
    assert shortlist["shortlist_count"] == 1
    assert shortlist["selection_metric_rank_stats"][0]["model"] == "LGBMRanker__continuous"
    assert len(shortlist["configs"]) == 1
    assert shortlist["configs"][0]["run_id"] == "LGBMRanker__continuous"
    assert shortlist["configs"][0]["selection_group"] == "LGBMRanker"
    assert "selection_reason" in shortlist["configs"][0]


def _screening_report(regret5_by_front: dict[int, float]) -> list[dict]:
    return [
        {
            "fold": 1,
            "agg": {
                "Regret@5": sum(regret5_by_front.values()) / len(regret5_by_front),
                "Hit@5": 1.0,
                "Regret@1": sum(regret5_by_front.values()) / len(regret5_by_front) + 0.1,
                "BestAUPR@5": 1.0 - sum(regret5_by_front.values()) / len(regret5_by_front),
            },
            "groups": [
                {
                    "front_id": front_id,
                    "Regret@5": regret5,
                    "Hit@5": 1.0,
                    "Regret@1": regret5 + 0.1,
                    "BestAUPR@5": 1.0 - regret5,
                    "n_items": 6,
                }
                for front_id, regret5 in regret5_by_front.items()
            ],
        }
    ]


def test_phase01_shortlist_selects_best_labeling_per_model_type(tmp_path):
    configs = [
        ScreeningRunConfig(
            "LGBMRanker__continuous",
            ModelType.LGBMRanker,
            LabelMode.CONTINUOUS,
            None,
            True,
        ),
        ScreeningRunConfig(
            "LGBMRanker__rank_avg",
            ModelType.LGBMRanker,
            LabelMode.RANK_AVG,
            None,
            True,
        ),
        ScreeningRunConfig(
            "LGBMRanker__shuffled",
            ModelType.LGBMRanker,
            LabelMode.SHUFFLED,
            None,
            True,
        ),
        ScreeningRunConfig(
            "XGBRanker__continuous",
            ModelType.XGBRanker,
            LabelMode.CONTINUOUS,
            None,
            True,
        ),
        ScreeningRunConfig(
            "XGBRanker__rank_avg",
            ModelType.XGBRanker,
            LabelMode.RANK_AVG,
            None,
            True,
        ),
    ]
    regrets = {
        "LGBMRanker__continuous": {1: 0.30, 2: 0.25},
        "LGBMRanker__rank_avg": {1: 0.10, 2: 0.12},
        "LGBMRanker__shuffled": {1: 0.01, 2: 0.01},
        "XGBRanker__continuous": {1: 0.15, 2: 0.11},
        "XGBRanker__rank_avg": {1: 0.24, 2: 0.20},
    }
    run_results = []
    summary_rows = []
    for config in configs:
        run_dir = tmp_path / config.run_id
        run_dir.mkdir()
        cv_report = _write_json(run_dir / "cv_report.json", _screening_report(regrets[config.run_id]))
        run_results.append(
            ScreeningRunResult(
                config=config,
                run_dir=run_dir,
                cv_report=cv_report,
                feature_columns=run_dir / "feature_columns.json",
                valid_fold_ranked=(),
                run_manifest=run_dir / "run_manifest.json",
                status="written",
            )
        )
        mean_regret = sum(regrets[config.run_id].values()) / len(regrets[config.run_id])
        for metric, value in {
            "Regret@5": mean_regret,
            "Hit@5": 1.0,
            "Regret@1": mean_regret + 0.1,
            "BestAUPR@5": 1.0 - mean_regret,
        }.items():
            summary_rows.append(
                {
                    "model": config.run_id,
                    "metric": metric,
                    "tier": "primary",
                    "priority": 1,
                    "mean": value,
                    "std": 0.0,
                    "n": 2,
                    "rank": 1,
                }
            )
    summary_path = tmp_path / "metrics_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    selection = _select_best_labeling_per_model_type(
        run_configs=configs,
        run_results=run_results,
        summary_path=summary_path,
    )

    assert [config["run_id"] for config in selection["configs"]] == [
        "LGBMRanker__shuffled",
        "XGBRanker__continuous",
    ]
    assert selection["shortlist_count"] == 2
    assert selection["selection_rule"]["selection_scope"] == "best_label_per_model_type"
    assert selection["excluded_configs"] == []
