from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from scripts.experiments.amiga_exp.cli import app
from scripts.experiments.amiga_exp.context import load_case_context
from scripts.experiments.amiga_exp.manifests import WriteOptions
from scripts.experiments.amiga_exp.paper_summary import (
    DEFAULT_PRIMARY_METRIC,
    build_comparison_table,
    load_report_groups,
    summarize_paper,
)


runner = CliRunner()


def _write_json(path: Path, payload: dict | list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_case(tmp_path: Path) -> tuple[Path, Path]:
    case_dir = tmp_path / "CASE"
    data_dir = case_dir / "data"
    data_dir.mkdir(parents=True)
    rows = []
    for front_id in (1, 5, 6, 7):
        for item_id in range(1, 7):
            rows.append(
                {
                    "front_id": front_id,
                    "item_id": item_id,
                    "AUPR": item_id / 10.0,
                    "objective": 7.0 - item_id,
                    "feature": item_id / 10.0,
                }
            )
    pd.DataFrame(rows).to_csv(data_dir / "data_1.csv", index=False)

    split_manifest = _write_json(
        tmp_path / "CASE_split_manifest.json",
        {
            "case": "CASE",
            "assignments": [
                {"front_id": 1, "front_name": "front_1", "split": "development"},
                {"front_id": 5, "front_name": "front_5", "split": "test"},
                {"front_id": 6, "front_name": "front_6", "split": "test"},
                {"front_id": 7, "front_name": "front_7", "split": "test"},
            ],
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
            "feature_sets": {"full": ["objective", "feature"]},
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


def _report_payload(method: str, regrets: dict[int, float]) -> list[dict]:
    groups = []
    for front_id, regret5 in regrets.items():
        groups.append(
            {
                "front_id": front_id,
                "Regret@1": regret5 + 0.1,
                "Regret@5": regret5,
                "Hit@5": 1.0 if regret5 <= 0.05 else 0.0,
                "BestAUPR@5": 1.0 - regret5,
                "n_items": 6,
            }
        )
    return [
        {
            "fold": 1,
            "agg": {
                "Regret@5": float(sum(regrets.values()) / len(regrets)),
                "Hit@5": 1.0,
                "BestAUPR@5": 0.95,
            },
            "groups": groups,
            "label_mode": None,
            "label_quantiles": None,
            "meta": {"model": method, "evaluation_split": "test"},
        }
    ]


def _write_completed_reports(context) -> None:
    root = context.results_root
    _write_json(
        root / "02_hyperparameter_tuning" / "final_test" / "cv_report.json",
        _report_payload("AMIGA_final", {5: 0.01, 6: 0.02, 7: 0.03}),
    )
    _write_json(
        root / "03_ablation" / "final_test" / "full" / "cv_report.json",
        _report_payload("full", {5: 0.02, 6: 0.03}),
    )
    _write_json(
        root / "03_ablation" / "final_test" / "expression_only" / "cv_report.json",
        _report_payload("expression_only", {5: 0.2, 6: 0.25, 7: 0.3}),
    )
    _write_json(
        root / "04_decision_baselines" / "runs" / "AMIGA_final" / "cv_report.json",
        _report_payload("AMIGA_final", {5: 0.01, 6: 0.02, 7: 0.03}),
    )
    _write_json(
        root / "04_decision_baselines" / "runs" / "objective__objective" / "cv_report.json",
        _report_payload("objective__objective", {5: 0.05, 6: 0.06, 7: 0.08}),
    )
    _write_json(
        root / "04_decision_baselines" / "baseline_manifest.json",
        {
            "manifest_type": "decision_baselines",
            "baselines": [
                {
                    "baseline_id": "AMIGA_final",
                    "outputs": {
                        "cv_report": str(
                            root / "04_decision_baselines" / "runs" / "AMIGA_final" / "cv_report.json"
                        )
                    },
                },
                {
                    "baseline_id": "objective__objective",
                    "outputs": {
                        "cv_report": str(
                            root
                            / "04_decision_baselines"
                            / "runs"
                            / "objective__objective"
                            / "cv_report.json"
                        )
                    },
                },
            ],
        },
    )


def test_build_comparison_table_uses_only_common_fronts(tmp_path):
    report_a = _write_json(tmp_path / "a" / "cv_report.json", _report_payload("a", {5: 0.1, 6: 0.2, 7: 0.9}))
    report_b = _write_json(tmp_path / "b" / "cv_report.json", _report_payload("b", {5: 0.2, 6: 0.3}))
    metrics_a = load_report_groups(report_a, method="a")
    metrics_b = load_report_groups(report_b, method="b")

    table = build_comparison_table(
        {"a": metrics_a, "b": metrics_b},
        comparison_type="synthetic",
        primary_metric=DEFAULT_PRIMARY_METRIC,
    )

    primary = table[table["metric"] == DEFAULT_PRIMARY_METRIC]
    assert set(primary["method"]) == {"a", "b"}
    assert set(primary["n_fronts_common"]) == {2}
    assert primary.loc[primary["method"] == "a", "mean"].item() == 0.15000000000000002


def test_summarize_paper_writes_comparisons_and_paired_tests(tmp_path):
    case_dir, config = _write_case(tmp_path)
    context = load_case_context(case_dir, config_path=config)
    _write_completed_reports(context)

    result = summarize_paper(context, options=WriteOptions())

    assert result.status == "written"
    for path in result.outputs.values():
        assert path.exists()

    final_test = pd.read_csv(result.final_test_comparison)
    assert set(final_test["method"]) == {"AMIGA_final"}
    assert DEFAULT_PRIMARY_METRIC in set(final_test["metric"])

    ablation = pd.read_csv(result.ablation_comparison)
    ablation_primary = ablation[ablation["metric"] == DEFAULT_PRIMARY_METRIC]
    assert set(ablation_primary["method"]) == {"AMIGA_final", "full", "expression_only"}
    assert set(ablation_primary["n_fronts_common"]) == {2}

    baseline = pd.read_csv(result.baseline_comparison)
    baseline_primary = baseline[baseline["metric"] == DEFAULT_PRIMARY_METRIC]
    assert set(baseline_primary["method"]) == {
        "AMIGA_final",
        "objective__objective",
    }
    assert set(baseline_primary["n_fronts_common"]) == {3}

    tests = pd.read_csv(result.statistical_tests_csv)
    assert set(tests["comparison_type"]) == {"ablation", "baseline"}
    assert set(tests["metric"]) == {DEFAULT_PRIMARY_METRIC}
    assert "AMIGA_final" in set(tests["method"])

    payload = json.loads(result.statistical_tests_json.read_text(encoding="utf-8"))
    assert payload["primary_metric"] == DEFAULT_PRIMARY_METRIC
    assert payload["paired_by"] == "front_id"
    assert payload["comparisons"]["ablation"]["n_fronts_common"] == 2


def test_amiga_exp_summarize_paper_dry_run(tmp_path):
    case_dir, config = _write_case(tmp_path)
    context = load_case_context(case_dir, config_path=config)
    _write_completed_reports(context)

    result = runner.invoke(
        app,
        [
            "summarize-paper",
            str(case_dir),
            "--config",
            str(config),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Paper summary dry-run complete." in result.output
    assert "final_test_comparison" in result.output
    assert not (context.results_root / "summaries").exists()
