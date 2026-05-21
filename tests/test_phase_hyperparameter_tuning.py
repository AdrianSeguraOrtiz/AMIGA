from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from amiga.selection.learn2rank import ModelType
from scripts.experiments.amiga_exp.context import load_case_context
from scripts.experiments.amiga_exp.manifests import WriteOptions
from scripts.experiments.amiga_exp.phases.phase_02_hyperparameter_tuning import (
    PHASE_NAME,
    build_tuning_run_configs,
    parameter_sets_for_model,
    run_hyperparameter_tuning,
)


SMALL_LGBM_PARAMS = {
    "n_estimators": 8,
    "num_leaves": 7,
    "learning_rate": 0.1,
    "min_child_samples": 1,
    "verbose": -1,
}


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_tuning_case(tmp_path: Path) -> tuple[Path, Path]:
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


def _write_shortlist(context) -> Path:
    return _write_json(
        context.results_root / "01_model_screening" / "shortlisted_configs.json",
        {
            "manifest_type": "shortlisted_configs",
            "phase": "01_model_screening",
            "configs": [
                {
                    "run_id": "LGBMRanker__continuous",
                    "model_type": "LGBMRanker",
                    "label_mode": "continuous",
                    "label_quantiles": None,
                    "feature_set": "full",
                    "selectable": True,
                }
            ],
        },
    )


def test_parameter_sets_for_model_default_counts_and_catboost_file_writes_disabled():
    lgbm_sets = parameter_sets_for_model(ModelType.LGBMRanker)
    assert len(lgbm_sets) == 27
    assert {param_set.model_params["num_leaves"] for param_set in lgbm_sets} == {31, 63, 127}
    assert {param_set.model_params["min_child_samples"] for param_set in lgbm_sets} == {30, 50, 100}
    assert {param_set.model_params["learning_rate"] for param_set in lgbm_sets} == {0.03, 0.05, 0.10}

    xgb_sets = parameter_sets_for_model(ModelType.XGBRanker)
    assert len(xgb_sets) == 36
    assert {param_set.model_params["max_depth"] for param_set in xgb_sets} == {4, 6, 8}
    assert {param_set.model_params["min_child_weight"] for param_set in xgb_sets} == {1, 5, 10}

    catboost_sets = parameter_sets_for_model(ModelType.CatBoostRanker)
    assert len(catboost_sets) == 36
    assert {param_set.model_params["depth"] for param_set in catboost_sets} == {4, 6, 8}
    assert {param_set.model_params["l2_leaf_reg"] for param_set in catboost_sets} == {3, 5, 7, 10}
    assert {param_set.model_params["learning_rate"] for param_set in catboost_sets} == {0.03, 0.05, 0.10}
    assert all(param_set.model_params["allow_writing_files"] is False for param_set in catboost_sets)


def test_build_tuning_run_configs_expands_shortlist_with_custom_grid():
    configs = build_tuning_run_configs(
        [
            {
                "run_id": "LGBMRanker__continuous",
                "model_type": "LGBMRanker",
                "label_mode": "continuous",
                "feature_set": "full",
            }
        ],
        tuning_grids={"LGBMRanker": [{"tag": "tiny", "model_params": SMALL_LGBM_PARAMS}]},
    )

    assert len(configs) == 1
    assert configs[0].run_id == "LGBMRanker__continuous__tiny"
    assert configs[0].base_run_id == "LGBMRanker__continuous"
    assert configs[0].model_params["n_estimators"] == 8


def test_run_hyperparameter_tuning_writes_summary_and_selected_config_without_test_fronts(tmp_path):
    case_dir, config = _write_tuning_case(tmp_path)
    context = load_case_context(case_dir, config_path=config)
    shortlist_path = _write_shortlist(context)

    result = run_hyperparameter_tuning(
        context,
        seed=7,
        options=WriteOptions(),
        shortlist_path=shortlist_path,
        n_splits=2,
        tuning_grids={"LGBMRanker": [{"tag": "tiny", "model_params": SMALL_LGBM_PARAMS}]},
    )

    assert result.status == "written"
    assert result.phase_dir == context.results_root / PHASE_NAME
    assert len(result.run_results) == 1
    run = result.run_results[0]
    assert run.cv_report.exists()
    assert run.feature_columns.exists()
    assert run.run_manifest.exists()
    assert len(run.valid_fold_ranked) == 2

    report = json.loads(run.cv_report.read_text(encoding="utf-8"))
    assert report[0]["meta"]["phase"] == "02_hyperparameter_tuning"
    assert report[0]["meta"]["model_params"]["n_estimators"] == 8
    assert "Regret@5" in report[0]["agg"]

    valid_fronts = set()
    for ranked_path in run.valid_fold_ranked:
        ranked = pd.read_csv(ranked_path)
        valid_fronts.update(ranked["front_id"].unique().tolist())
    assert valid_fronts <= {1, 2, 3, 4}
    assert 5 not in valid_fronts

    assert (result.phase_dir / "summary" / "metrics_summary.csv").exists()
    assert result.table_outputs["primary_rank_table"].exists()
    primary_table = pd.read_csv(result.table_outputs["primary_rank_table"])
    assert primary_table["config"].tolist() == ["LGBMRanker__continuous__tiny"]
    assert primary_table.loc[0, "avg_rank"] == 1.0
    tuning_manifest = json.loads(result.tuning_manifest.read_text(encoding="utf-8"))
    assert tuning_manifest["run_count"] == 1
    assert tuning_manifest["source_shortlist"].endswith("shortlisted_configs.json")
    assert tuning_manifest["tables"]["primary_rank_table"].endswith("primary_rank_table.csv")
    assert "plot_cv" not in tuning_manifest
    assert tuning_manifest["plots"]["plot_manifest"].endswith("plots/plot_manifest.json")
    assert tuning_manifest["plots"]["primary"]["png"].endswith(
        "plots/hyperparameter_regret_scatter.png"
    )
    assert tuning_manifest["plots"]["primary"]["pdf"].endswith(
        "plots/hyperparameter_regret_scatter.pdf"
    )
    assert tuning_manifest["plots"]["primary"]["csv"].endswith(
        "plots/hyperparameter_regret_scatter.csv"
    )
    phase_manifest = json.loads((result.phase_dir / "phase_manifest.json").read_text(encoding="utf-8"))
    assert phase_manifest["status"] == "completed"
    assert phase_manifest["outputs"]["tuning_manifest"].endswith("tuning_manifest.json")
    assert phase_manifest["outputs"]["selected_config"].endswith("selected_config.json")
    assert phase_manifest["outputs"]["table_outputs"]["primary_rank_table"].endswith("primary_rank_table.csv")
    assert phase_manifest["outputs"]["plots"]["plot_manifest"].endswith("plots/plot_manifest.json")
    selected = json.loads(result.selected_config.read_text(encoding="utf-8"))
    assert selected["primary_metric"] == "Regret@5"
    assert selected["tie_breakers"] == ["Hit@5", "Regret@1", "BestAUPR@5"]
    assert selected["selection_rule"]["primary_selection_stat"] == "Regret@5_avg_rank"
    assert selected["selection_metric_rank_stats"][0]["model"] == "LGBMRanker__continuous__tiny"
    selected_config = selected["selected_config"]
    assert selected_config["run_id"] == "LGBMRanker__continuous__tiny"
    assert selected_config["model_type"] == "LGBMRanker"
    assert selected_config["label_mode"] == "continuous"
    assert selected_config["feature_set"] == "full"
    assert selected_config["model_params"]["n_estimators"] == 8
    assert "selection_reason" in selected_config
