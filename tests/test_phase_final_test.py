from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.experiments.amiga_exp.context import load_case_context
from scripts.experiments.amiga_exp.manifests import WriteOptions
from scripts.experiments.amiga_exp.phases.phase_02_final_test import (
    STEP_NAME,
    load_selected_config,
    run_final_test_evaluation,
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


def _write_final_test_case(tmp_path: Path) -> tuple[Path, Path]:
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


def test_load_selected_config_reads_frozen_parameters(tmp_path):
    case_dir, config = _write_final_test_case(tmp_path)
    context = load_case_context(case_dir, config_path=config)
    selected_path = _write_selected_config(context)

    selected = load_selected_config(selected_path)

    assert selected.run_id == "LGBMRanker__continuous__tiny"
    assert selected.model_type.value == "LGBMRanker"
    assert selected.label_mode.value == "continuous"
    assert selected.feature_set == "full"
    assert selected.model_params["n_estimators"] == 8


def test_run_final_test_evaluation_writes_outputs_and_uses_only_test_fronts(tmp_path):
    case_dir, config = _write_final_test_case(tmp_path)
    context = load_case_context(case_dir, config_path=config)
    selected_path = _write_selected_config(context)

    result = run_final_test_evaluation(
        context,
        seed=7,
        options=WriteOptions(),
        selected_config_path=selected_path,
    )

    assert result.status == "written"
    assert result.model.exists()
    assert result.feature_columns.exists()
    assert result.final_test_ranked.exists()
    assert result.final_test_report.exists()
    assert result.cv_report.exists()

    ranked = pd.read_csv(result.final_test_ranked)
    assert set(ranked["front_id"]) == {5}
    assert "rank_in_front" in ranked.columns

    cv_report = json.loads(result.cv_report.read_text(encoding="utf-8"))
    assert cv_report[0]["meta"]["step"] == STEP_NAME
    assert "Regret@5" in cv_report[0]["agg"]
    assert [group["front_id"] for group in cv_report[0]["groups"]] == [5]

    final_report = json.loads(result.final_test_report.read_text(encoding="utf-8"))
    assert final_report["step"] == STEP_NAME
    assert final_report["selected_config"]["run_id"] == "LGBMRanker__continuous__tiny"
    assert "Regret@5" in final_report["agg"]
    assert [group["front_id"] for group in final_report["groups"]] == [5]
