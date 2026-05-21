from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from scripts.experiments.amiga_exp import cli as cli_module
from scripts.experiments.amiga_exp.cli import app


runner = CliRunner()


def _case_dir(tmp_path: Path) -> Path:
    case_dir = tmp_path / "CASE"
    data_dir = case_dir / "data"
    audit_dir = data_dir / "audit"
    audit_dir.mkdir(parents=True)
    (data_dir / "data_1.csv").write_text(
        "front_id,item_id,AUPR,objective,feature\n"
        "1,1,0.8,0.1,1.0\n"
        "2,1,0.4,0.2,0.5\n",
        encoding="utf-8",
    )
    return case_dir


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _case_config(tmp_path: Path, case_dir: Path, *, case_name: str = "CASE") -> Path:
    split_manifest = _write_json(
        tmp_path / f"{case_name}_split_manifest.json",
        {
            "case": case_name,
            "assignments": [
                {"front_id": 1, "front_name": "front_1", "split": "development"},
                {"front_id": 2, "front_name": "front_2", "split": "test"},
            ],
        },
    )
    feature_contract = _write_json(
        tmp_path / f"{case_name}_feature_columns.json",
        {
            "case": case_name,
            "target_column": "AUPR",
            "control_columns": ["front_id", "item_id"],
            "objective_columns": ["objective"],
            "objective_directions": {"objective": "minimize"},
            "feature_sets": {
                "full": ["objective", "feature"],
                "objectives_only": ["objective"],
            },
        },
    )
    return _write_json(
        tmp_path / f"{case_name}.json",
        {
            "case": case_name,
            "data_csv": str(case_dir / "data" / "data_1.csv"),
            "split_manifest": str(split_manifest),
            "feature_contract": str(feature_contract),
            "group_column": "front_id",
            "item_column": "item_id",
            "target_column": "AUPR",
        },
    )


def _screening_case_config(tmp_path: Path) -> tuple[Path, Path]:
    case_dir = tmp_path / "SCREENING"
    data_dir = case_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "data_1.csv").write_text(
        "front_id,item_id,AUPR,objective,feature\n"
        "1,1,0.8,0.1,1.0\n"
        "2,1,0.4,0.2,0.5\n"
        "3,1,0.6,0.3,0.8\n",
        encoding="utf-8",
    )
    split_manifest = _write_json(
        tmp_path / "SCREENING_split_manifest.json",
        {
            "case": "SCREENING",
            "assignments": [
                {"front_id": 1, "front_name": "front_1", "split": "development"},
                {"front_id": 2, "front_name": "front_2", "split": "development"},
                {"front_id": 3, "front_name": "front_3", "split": "test"},
            ],
        },
    )
    feature_contract = _write_json(
        tmp_path / "SCREENING_feature_columns.json",
        {
            "case": "SCREENING",
            "target_column": "AUPR",
            "control_columns": ["front_id", "item_id"],
            "objective_columns": ["objective"],
            "objective_directions": {"objective": "minimize"},
            "feature_sets": {"full": ["objective", "feature"]},
        },
    )
    config = _write_json(
        tmp_path / "SCREENING.json",
        {
            "case": "SCREENING",
            "data_csv": str(data_dir / "data_1.csv"),
            "split_manifest": str(split_manifest),
            "feature_contract": str(feature_contract),
            "group_column": "front_id",
            "item_column": "item_id",
            "target_column": "AUPR",
        },
    )
    return case_dir, config


def test_amiga_exp_help_smoke():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "inspect" in result.output
    assert "validate" in result.output
    assert "init-results" in result.output
    assert "run-phase" in result.output
    assert "run-all" in result.output
    assert "plot-phase" in result.output
    assert "plot-all" in result.output
    assert "summarize-paper" in result.output


def test_amiga_exp_inspect_and_validate_case_dir(tmp_path):
    case_dir = _case_dir(tmp_path)
    config = _case_config(tmp_path, case_dir)

    inspect_result = runner.invoke(app, ["inspect", str(case_dir)])
    assert inspect_result.exit_code == 0
    assert "case_name: CASE" in inspect_result.output
    assert "data_csv_count: 1" in inspect_result.output
    assert "audit_dir_exists: True" in inspect_result.output

    validate_result = runner.invoke(app, ["validate", str(case_dir), "--config", str(config)])
    assert validate_result.exit_code == 0
    assert "Case context validation passed." in validate_result.output
    assert "development_rows: 1" in validate_result.output
    assert "test_rows: 1" in validate_result.output
    assert "feature_sets: ['full', 'objectives_only']" in validate_result.output


def test_amiga_exp_validate_reports_missing_case_dir(tmp_path):
    missing = tmp_path / "missing-case"

    result = runner.invoke(app, ["validate", str(missing)])

    assert result.exit_code == 1
    assert "case directory does not exist" in result.output


def test_amiga_exp_validate_requires_data_csv(tmp_path):
    case_dir = tmp_path / "CASE"
    (case_dir / "data").mkdir(parents=True)

    result = runner.invoke(app, ["validate", str(case_dir)])

    assert result.exit_code == 1
    assert "no data_*.csv file found" in result.output


def test_amiga_exp_run_all_dry_run_plans_full_pipeline(tmp_path):
    case_dir = _case_dir(tmp_path)
    config = _case_config(tmp_path, case_dir)

    result = runner.invoke(app, ["run-all", str(case_dir), "--config", str(config), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Full pipeline dry-run complete." in result.output
    assert "planned_step: 01_model_screening" in result.output
    assert "planned_step: 02_hyperparameter_tuning" in result.output
    assert "planned_step: final_test" in result.output
    assert "planned_step: 03_ablation" in result.output
    assert "planned_step: 04_decision_baselines" in result.output
    assert "planned_step: summarize-paper" in result.output
    assert not (case_dir / "results").exists()


def test_amiga_exp_plot_phase_prepares_plot_manifest(tmp_path):
    case_dir = _case_dir(tmp_path)
    config = _case_config(tmp_path, case_dir)
    table_path = case_dir / "results" / "amiga-exp" / "01_model_screening" / "summary" / "primary_rank_table.csv"
    table_path.parent.mkdir(parents=True)
    table_path.write_text(
        "config,avg_rank,p_value,mean_regret5,std_regret5,n_fronts\n"
        "LGBMRanker__continuous,1.0,,0.1,0.01,2\n",
        encoding="utf-8",
    )
    (table_path.parent / "metrics_summary.csv").write_text(
        "model,metric,tier,priority,mean,std,n,rank\n"
        "LGBMRanker__continuous,Regret@1,primary,10,0.2,0.01,2,1\n"
        "LGBMRanker__continuous,Regret@5,primary,12,0.1,0.01,2,1\n"
        "LGBMRanker__continuous,Hit@1,primary,30,1.0,0.0,2,1\n"
        "LGBMRanker__continuous,Hit@5,primary,32,1.0,0.0,2,1\n"
        "LGBMRanker__continuous,BestAUPR@1,primary,20,0.8,0.01,2,1\n"
        "LGBMRanker__continuous,BestAUPR@5,primary,22,0.9,0.01,2,1\n",
        encoding="utf-8",
    )
    _write_json(
        case_dir / "results" / "amiga-exp" / "01_model_screening" / "shortlisted_configs.json",
        {"configs": [{"run_id": "LGBMRanker__continuous"}]},
    )

    result = runner.invoke(
        app,
        [
            "plot-phase",
            "--case-dir",
            str(case_dir),
            "--config",
            str(config),
            "--phase",
            "01_model_screening",
            "--include-secondary",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Phase plot preparation complete." in result.output
    assert "plot_manifest:" in result.output
    plot_manifest = table_path.parents[1] / "plots" / "plot_manifest.json"
    assert plot_manifest.exists()
    assert (table_path.parents[1] / "plots" / "model_screening_heatmap.png").exists()
    assert (table_path.parents[1] / "plots" / "supplementary" / "topk_metric_curves.png").exists()


def test_amiga_exp_run_all_executes_standard_pipeline_in_order(tmp_path, monkeypatch):
    case_dir = _case_dir(tmp_path)
    config = _case_config(tmp_path, case_dir)
    calls: list[str] = []

    def fake_init(context, *, seed, options):
        calls.append("init-results")
        return {
            "directories": [],
            "environment_manifest": context.results_root / "manifests" / "environment_manifest.json",
            "environment_status": "written",
            "phase_statuses": {},
        }

    def fake_screening(context, **kwargs):
        calls.append("01_model_screening")
        assert kwargs["n_splits"] == 2
        return SimpleNamespace(shortlisted_configs=context.results_root / "01_model_screening" / "shortlisted_configs.json")

    def fake_tuning(context, **kwargs):
        calls.append("02_hyperparameter_tuning")
        assert kwargs["n_splits"] == 2
        assert kwargs["selection_size"] == 1
        return SimpleNamespace(selected_config=context.results_root / "02_hyperparameter_tuning" / "selected_config.json")

    def fake_final_test(context, **kwargs):
        calls.append("final_test")
        return SimpleNamespace(final_test_report=context.results_root / "02_hyperparameter_tuning" / "final_test" / "final_test_report.json")

    def fake_ablation(context, **kwargs):
        calls.append("03_ablation")
        assert kwargs["feature_sets"] == ["full"]
        assert kwargs["n_splits"] == 2
        return SimpleNamespace(ablation_manifest=context.results_root / "03_ablation" / "ablation_manifest.json")

    def fake_baselines(context, **kwargs):
        calls.append("04_decision_baselines")
        return SimpleNamespace(baseline_manifest=context.results_root / "04_decision_baselines" / "baseline_manifest.json")

    def fake_summary(context, **kwargs):
        calls.append("summarize-paper")
        assert kwargs["primary_metric"] == "Regret@5"
        return SimpleNamespace(
            outputs={
                "final_test_comparison": context.results_root / "summaries" / "final_test_comparison.csv",
                "ablation_comparison": context.results_root / "summaries" / "ablation_comparison.csv",
                "baseline_comparison": context.results_root / "summaries" / "baseline_comparison.csv",
                "statistical_tests_csv": context.results_root / "summaries" / "statistical_tests.csv",
                "statistical_tests_json": context.results_root / "summaries" / "statistical_tests.json",
            }
        )

    monkeypatch.setattr(cli_module, "initialize_results_layout", fake_init)
    monkeypatch.setattr(cli_module, "run_model_screening", fake_screening)
    monkeypatch.setattr(cli_module, "run_hyperparameter_tuning", fake_tuning)
    monkeypatch.setattr(cli_module, "run_final_test_evaluation", fake_final_test)
    monkeypatch.setattr(cli_module, "run_ablation", fake_ablation)
    monkeypatch.setattr(cli_module, "run_decision_baselines", fake_baselines)
    monkeypatch.setattr(cli_module, "summarize_paper", fake_summary)

    result = runner.invoke(
        app,
        [
            "run-all",
            str(case_dir),
            "--config",
            str(config),
            "--n-splits",
            "2",
            "--selection-size",
            "1",
            "--feature-set",
            "full",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        "init-results",
        "01_model_screening",
        "02_hyperparameter_tuning",
        "final_test",
        "03_ablation",
        "04_decision_baselines",
        "summarize-paper",
    ]
    assert "Full pipeline complete." in result.output


def test_amiga_exp_run_phase_model_screening_dry_run(tmp_path):
    case_dir, config = _screening_case_config(tmp_path)

    result = runner.invoke(
        app,
        [
            "run-phase",
            str(case_dir),
            "01_model_screening",
            "--config",
            str(config),
            "--dry-run",
            "--n-splits",
            "2",
            "--model",
            "LGBMRanker",
            "--label-mode",
            "continuous",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Phase 01 model screening dry-run complete." in result.output
    assert "planned_runs: 1" in result.output
    assert "LGBMRanker__continuous" in result.output
    assert not (case_dir / "results").exists()


def test_amiga_exp_run_phase_hyperparameter_tuning_dry_run(tmp_path):
    case_dir, config = _screening_case_config(tmp_path)
    _write_json(
        case_dir / "results" / "amiga-exp" / "01_model_screening" / "shortlisted_configs.json",
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
    grid = _write_json(
        tmp_path / "tuning_grid.json",
        {"LGBMRanker": [{"tag": "tiny", "model_params": {"n_estimators": 8}}]},
    )

    result = runner.invoke(
        app,
        [
            "run-phase",
            str(case_dir),
            "02_hyperparameter_tuning",
            "--config",
            str(config),
            "--dry-run",
            "--n-splits",
            "2",
            "--tuning-grid-json",
            str(grid),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Phase 02 hyperparameter tuning dry-run complete." in result.output
    assert "planned_runs: 1" in result.output
    assert "LGBMRanker__continuous__tiny" in result.output
    assert not (case_dir / "results" / "amiga-exp" / "02_hyperparameter_tuning").exists()


def test_amiga_exp_run_phase_final_test_dry_run(tmp_path):
    case_dir, config = _screening_case_config(tmp_path)
    _write_json(
        case_dir / "results" / "amiga-exp" / "02_hyperparameter_tuning" / "selected_config.json",
        {
            "manifest_type": "selected_config",
            "phase": "02_hyperparameter_tuning",
            "selected_config": {
                "run_id": "LGBMRanker__continuous__tiny",
                "model_type": "LGBMRanker",
                "label_mode": "continuous",
                "label_quantiles": None,
                "feature_set": "full",
                "model_params": {"n_estimators": 8},
            },
        },
    )

    result = runner.invoke(
        app,
        [
            "run-phase",
            str(case_dir),
            "final_test",
            "--config",
            str(config),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Held-out final test dry-run complete." in result.output
    assert "cv_report" in result.output
    assert not (case_dir / "results" / "amiga-exp" / "02_hyperparameter_tuning" / "final_test").exists()


def test_amiga_exp_run_phase_ablation_dry_run(tmp_path):
    case_dir, config = _screening_case_config(tmp_path)
    _write_json(
        case_dir / "results" / "amiga-exp" / "02_hyperparameter_tuning" / "selected_config.json",
        {
            "manifest_type": "selected_config",
            "phase": "02_hyperparameter_tuning",
            "selected_config": {
                "run_id": "LGBMRanker__continuous__tiny",
                "model_type": "LGBMRanker",
                "label_mode": "continuous",
                "label_quantiles": None,
                "feature_set": "full",
                "model_params": {"n_estimators": 8},
            },
        },
    )

    result = runner.invoke(
        app,
        [
            "run-phase",
            str(case_dir),
            "03_ablation",
            "--config",
            str(config),
            "--dry-run",
            "--n-splits",
            "2",
            "--feature-set",
            "full",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Phase 03 ablation dry-run complete." in result.output
    assert "planned_feature_sets: 1" in result.output
    assert "feature_set: full" in result.output
    assert not (case_dir / "results" / "amiga-exp" / "03_ablation").exists()


def test_amiga_exp_run_phase_decision_baselines_dry_run(tmp_path):
    case_dir, config = _screening_case_config(tmp_path)
    final_dir = case_dir / "results" / "amiga-exp" / "02_hyperparameter_tuning" / "final_test"
    final_dir.mkdir(parents=True)
    (final_dir / "final_test_ranked.csv").write_text("front_id,item_id,AUPR,score\n3,1,0.6,0.6\n", encoding="utf-8")
    (final_dir / "cv_report.json").write_text("[]", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "run-phase",
            str(case_dir),
            "04_decision_baselines",
            "--config",
            str(config),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Phase 04 decision baselines dry-run complete." in result.output
    assert "planned_runs: 9" in result.output
    assert "run: AMIGA_final" in result.output
    assert "family=learned_reference" in result.output
    assert "run: objective__objective" in result.output
    assert "family=single_objective" in result.output
    assert "run: objective_normalized_mean" in result.output
    assert "family=weighted_sum" in result.output
    assert "run: objective_ideal_l2" in result.output
    assert "family=ideal_distance" in result.output
    assert "run: objective_topsis" in result.output
    assert "family=ideal_antiideal_distance" in result.output
    assert "run: objective_vikor" in result.output
    assert "family=compromise_ranking" in result.output
    assert "run: objective_augmented_tchebycheff" in result.output
    assert "family=augmented_tchebycheff" in result.output
    assert not (case_dir / "results" / "amiga-exp" / "04_decision_baselines").exists()


def test_amiga_exp_init_results_writes_layout_and_manifests(tmp_path):
    case_dir = _case_dir(tmp_path)
    config = _case_config(tmp_path, case_dir)

    result = runner.invoke(app, ["init-results", str(case_dir), "--config", str(config), "--seed", "9"])

    assert result.exit_code == 0, result.output
    assert "Result layout initialization complete." in result.output
    root = case_dir / "results" / "amiga-exp"
    assert (root / "manifests" / "environment_manifest.json").exists()
    assert (root / "01_model_screening" / "phase_manifest.json").exists()


def test_amiga_exp_init_results_dry_run_does_not_write(tmp_path):
    case_dir = _case_dir(tmp_path)
    config = _case_config(tmp_path, case_dir)

    result = runner.invoke(app, ["init-results", str(case_dir), "--config", str(config), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Would create/check result directories" in result.output
    assert not (case_dir / "results").exists()


def test_amiga_exp_init_results_refuses_existing_manifest_without_force(tmp_path):
    case_dir = _case_dir(tmp_path)
    config = _case_config(tmp_path, case_dir)

    first = runner.invoke(app, ["init-results", str(case_dir), "--config", str(config)])
    assert first.exit_code == 0, first.output
    second = runner.invoke(app, ["init-results", str(case_dir), "--config", str(config)])

    assert second.exit_code == 1
    assert "refusing to overwrite" in second.output


def test_amiga_exp_unknown_phase_reports_known_phases(tmp_path):
    case_dir = _case_dir(tmp_path)
    config = _case_config(tmp_path, case_dir)

    result = runner.invoke(app, ["run-phase", str(case_dir), "bad_phase", "--config", str(config)])

    assert result.exit_code == 2
    assert "unknown phase" in result.output
    assert "01_model_screening" in result.output


def test_amiga_exp_wrapper_help_smoke():
    repo_root = Path(__file__).resolve().parents[1]
    wrapper = repo_root / "scripts" / "experiments" / "amiga-exp"
    env = {**os.environ, "PYTHON": sys.executable}

    result = subprocess.run(
        [str(wrapper), "--help"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "run-phase" in result.stdout
