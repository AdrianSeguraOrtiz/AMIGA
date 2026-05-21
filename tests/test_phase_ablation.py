from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.experiments.amiga_exp.context import load_case_context
from scripts.experiments.amiga_exp.manifests import WriteOptions
from scripts.experiments.amiga_exp.phases.phase_03_ablation import (
    PHASE_NAME,
    resolve_ablation_feature_sets,
    run_ablation,
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


def _write_ablation_case(tmp_path: Path) -> tuple[Path, Path]:
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
                    "expr_signal": quality,
                    "network_signal": quality * 0.5,
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
            "feature_sets": {
                "full": ["objective", "expr_signal", "network_signal"],
                "expression_only": ["expr_signal"],
            },
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


def _write_selected_config(context) -> Path:
    return _write_json(
        context.results_root / "02_hyperparameter_tuning" / "selected_config.json",
        {
            "manifest_type": "selected_config",
            "phase": "02_hyperparameter_tuning",
            "selected_config": {
                "run_id": "LGBMRanker__continuous__tiny",
                "model_type": "LGBMRanker",
                "label_mode": "continuous",
                "label_quantiles": None,
                "feature_set": "full",
                "model_params": SMALL_LGBM_PARAMS,
            },
        },
    )


def test_resolve_ablation_feature_sets_uses_contract_order_and_request_subset(tmp_path):
    case_dir, config = _write_ablation_case(tmp_path)
    context = load_case_context(case_dir, config_path=config)

    assert resolve_ablation_feature_sets(context) == ("full", "expression_only")
    assert resolve_ablation_feature_sets(context, feature_sets=["expression_only"]) == ("expression_only",)


def test_run_ablation_writes_cv_final_test_summary_and_manifest(tmp_path):
    case_dir, config = _write_ablation_case(tmp_path)
    context = load_case_context(case_dir, config_path=config)
    selected_config = _write_selected_config(context)

    result = run_ablation(
        context,
        seed=7,
        options=WriteOptions(),
        selected_config_path=selected_config,
        feature_sets=["full", "expression_only"],
        n_splits=2,
    )

    assert result.status == "written"
    assert result.phase_dir == context.results_root / PHASE_NAME
    assert len(result.run_results) == 2
    assert len(result.final_test_results) == 2
    feature_columns_by_set = {
        run.config.feature_set: json.loads(run.feature_columns.read_text(encoding="utf-8"))["feature_columns"]
        for run in result.run_results
    }
    assert feature_columns_by_set["full"] == ["objective", "expr_signal", "network_signal"]
    assert feature_columns_by_set["expression_only"] == ["expr_signal"]

    for run in result.run_results:
        report = json.loads(run.cv_report.read_text(encoding="utf-8"))
        assert report[0]["meta"]["model_selection"] == "not_performed"
        valid_fronts = set()
        for ranked_path in run.valid_fold_ranked:
            ranked = pd.read_csv(ranked_path)
            valid_fronts.update(ranked["front_id"].unique().tolist())
        assert valid_fronts <= {1, 2, 3, 4}
        assert 5 not in valid_fronts

    for final_test in result.final_test_results:
        ranked = pd.read_csv(final_test.final_test_ranked)
        assert set(ranked["front_id"]) == {5}
        report = json.loads(final_test.cv_report.read_text(encoding="utf-8"))
        assert report[0]["meta"]["feature_set"] == final_test.config.feature_set
        assert [group["front_id"] for group in report[0]["groups"]] == [5]

    assert (result.phase_dir / "summary" / "metrics_summary.csv").exists()
    assert result.table_outputs["primary_rank_table"].exists()
    primary_table = pd.read_csv(result.table_outputs["primary_rank_table"])
    assert set(primary_table["config"]) == {"full", "expression_only"}
    assert set(primary_table.columns) == {
        "config",
        "avg_rank",
        "p_value",
        "mean_regret5",
        "std_regret5",
        "n_fronts",
    }
    manifest = json.loads(result.ablation_manifest.read_text(encoding="utf-8"))
    assert manifest["model_selection"] == "not_performed"
    assert manifest["feature_set_count"] == 2
    assert manifest["tables"]["primary_rank_table"].endswith("primary_rank_table.csv")
    assert "plot_cv" not in manifest
    assert manifest["plots"]["plot_manifest"].endswith("plots/plot_manifest.json")
    assert manifest["plots"]["primary"]["png"].endswith("plots/ablation_feature_matrix.png")
    assert manifest["plots"]["primary"]["pdf"].endswith("plots/ablation_feature_matrix.pdf")
    assert manifest["plots"]["primary"]["csv"].endswith("plots/ablation_feature_matrix.csv")
    phase_manifest = json.loads((result.phase_dir / "phase_manifest.json").read_text(encoding="utf-8"))
    assert phase_manifest["status"] == "completed"
    assert phase_manifest["outputs"]["ablation_manifest"].endswith("ablation_manifest.json")
    assert phase_manifest["outputs"]["final_test_count"] == 2
    assert phase_manifest["outputs"]["table_outputs"]["primary_rank_table"].endswith("primary_rank_table.csv")
    assert phase_manifest["outputs"]["plots"]["plot_manifest"].endswith("plots/plot_manifest.json")
    assert manifest["expression_only_audit"] == {
        "present": True,
        "n_features": 1,
        "feature_columns": ["expr_signal"],
    }
    assert manifest["feature_set_audit"]["expression_only"]["feature_columns"] == ["expr_signal"]
