from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pymcdm", reason="phase-4 package-backed baseline tests require pymcdm")

from amiga.selection.learn2rank import assign_rank_in_front
from scripts.experiments.amiga_exp.context import load_case_context
from scripts.experiments.amiga_exp.evaluation import evaluate_ranked_frame, write_cv_report
from scripts.experiments.amiga_exp.manifests import WriteOptions
from scripts.experiments.amiga_exp.phases.phase_04_decision_baselines import (
    AMIGA_REFERENCE_ID,
    BaselineSpec,
    DecisionBaselineError,
    build_baseline_specs,
    run_decision_baselines,
    score_baseline,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_decision_case(tmp_path: Path) -> tuple[Path, Path]:
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
                    "obj_quality": 7.0 - item_id,
                    "obj_size": 14.0 - (2 * item_id),
                    "f_signal": quality,
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
            "objective_columns": ["obj_quality", "obj_size"],
            "objective_directions": {"obj_quality": "minimize", "obj_size": "minimize"},
            "feature_sets": {"full": ["obj_quality", "obj_size", "f_signal"]},
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


def _write_amiga_reference(context) -> Path:
    final_dir = context.results_root / "02_hyperparameter_tuning" / "final_test"
    columns = [
        context.group_column,
        context.item_column,
        context.target_column,
        *context.objective_columns,
    ]
    ranked = context.test_df.loc[:, columns].copy()
    ranked["score"] = ranked[context.target_column]
    ranked = assign_rank_in_front(
        ranked,
        front_col=context.group_column,
        score_col="score",
        id_col=context.item_column,
    )
    final_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(final_dir / "final_test_ranked.csv", index=False)
    evaluation = evaluate_ranked_frame(
        ranked,
        front_col=context.group_column,
        target_col=context.target_column,
        score_col="score",
        model_name=AMIGA_REFERENCE_ID,
        evaluation_split="test",
        fold=1,
        meta={"phase": "02_hyperparameter_tuning", "step": "final_test"},
    )
    write_cv_report(final_dir / "cv_report.json", evaluation.report)
    return final_dir


def test_score_baseline_respects_objective_direction():
    df = pd.DataFrame(
        {
            "front_id": [1, 1, 1],
            "min_obj": [0.1, 0.5, 0.9],
            "max_obj": [0.1, 0.5, 0.9],
        }
    )

    min_score = score_baseline(
        df,
        BaselineSpec("single_min", "single_objective", objective="min_obj"),
        objective_columns=["min_obj"],
        objective_directions={"min_obj": "minimize"},
        front_col="front_id",
    )
    max_score = score_baseline(
        df,
        BaselineSpec("single_max", "single_objective", objective="max_obj"),
        objective_columns=["max_obj"],
        objective_directions={"max_obj": "maximize"},
        front_col="front_id",
    )
    normalized_min = score_baseline(
        df,
        BaselineSpec("normalized_min", "normalized_mean"),
        objective_columns=["min_obj"],
        objective_directions={"min_obj": "minimize"},
        front_col="front_id",
    )
    normalized_max = score_baseline(
        df,
        BaselineSpec("normalized_max", "normalized_mean"),
        objective_columns=["max_obj"],
        objective_directions={"max_obj": "maximize"},
        front_col="front_id",
    )

    assert min_score.iloc[0] > min_score.iloc[-1]
    assert max_score.iloc[-1] > max_score.iloc[0]
    assert normalized_min.iloc[0] > normalized_min.iloc[-1]
    assert normalized_max.iloc[-1] > normalized_max.iloc[0]


def test_score_baseline_applies_vikor_within_each_front():
    df = pd.DataFrame(
        {
            "front_id": [1, 1, 1, 2, 2, 2],
            "obj_a": [1.0, 0.0, 0.5, 0.0, 1.0, 1.0],
            "obj_b": [0.0, 1.0, 0.5, 0.0, 1.0, 0.0],
        }
    )

    scores = score_baseline(
        df,
        BaselineSpec("vikor", "vikor"),
        objective_columns=["obj_a", "obj_b"],
        objective_directions={"obj_a": "minimize", "obj_b": "minimize"},
        front_col="front_id",
    )

    assert scores.tolist() == pytest.approx([-0.5, -0.5, -0.0, -0.0, -1.0, -0.75])


def test_every_decision_baseline_produces_one_finite_score_per_row(tmp_path):
    case_dir, config = _write_decision_case(tmp_path)
    context = load_case_context(case_dir, config_path=config)

    for spec in build_baseline_specs(context):
        scores = score_baseline(
            context.test_df,
            spec,
            objective_columns=context.objective_columns,
            objective_directions=context.objective_directions,
            front_col=context.group_column,
        )

        assert len(scores) == len(context.test_df), spec.baseline_id
        assert scores.index.equals(context.test_df.index), spec.baseline_id
        assert np.isfinite(scores.to_numpy(dtype=float)).all(), spec.baseline_id


def test_constant_objective_fronts_do_not_produce_nan_scores(tmp_path):
    case_dir, config = _write_decision_case(tmp_path)
    context = load_case_context(case_dir, config_path=config)
    constant_df = pd.DataFrame(
        {
            "front_id": [1, 1, 1, 2, 2, 2],
            "item_id": [1, 2, 3, 1, 2, 3],
            "AUPR": [0.6, 0.5, 0.4, 0.7, 0.6, 0.5],
            "obj_quality": [3.0, 3.0, 3.0, 8.0, 8.0, 8.0],
            "obj_size": [10.0, 10.0, 10.0, 4.0, 4.0, 4.0],
        }
    )

    for spec in build_baseline_specs(context):
        scores = score_baseline(
            constant_df,
            spec,
            objective_columns=context.objective_columns,
            objective_directions=context.objective_directions,
            front_col=context.group_column,
        )

        assert len(scores) == len(constant_df), spec.baseline_id
        assert not scores.isna().any(), spec.baseline_id
        assert np.isfinite(scores.to_numpy(dtype=float)).all(), spec.baseline_id


def test_build_baseline_specs_includes_objective_baselines_by_default(tmp_path):
    case_dir, config = _write_decision_case(tmp_path)
    context = load_case_context(case_dir, config_path=config)

    specs = build_baseline_specs(context)

    assert [spec.baseline_id for spec in specs] == [
        "objective__obj_quality",
        "objective__obj_size",
        "objective_mean_rank",
        "objective_normalized_mean",
        "objective_ideal_l2",
        "objective_topsis",
        "objective_vikor",
        "objective_augmented_tchebycheff",
        "objective_tchebycheff",
    ]
    assert specs[0].method_family == "single_objective"
    assert specs[0].implementation == "local_numpy_pandas"
    assert specs[0].parameters["objective_direction"] == "minimize"
    normalized = next(spec for spec in specs if spec.baseline_id == "objective_normalized_mean")
    assert normalized.method_family == "weighted_sum"
    assert normalized.implementation == "pymcdm_1.4.0_adapter"
    assert normalized.parameters["package"] == "pymcdm"
    assert normalized.parameters["package_method"] == "WSM"
    assert normalized.parameters["normalization"] == "per_front_minmax_badness"
    assert normalized.parameters["weight_policy"] == "uniform"
    assert normalized.parameters["objective_weights"] == {
        "obj_quality": 0.5,
        "obj_size": 0.5,
    }
    ideal_l2 = next(spec for spec in specs if spec.baseline_id == "objective_ideal_l2")
    assert ideal_l2.method_family == "ideal_distance"
    assert ideal_l2.implementation == "local_numpy"
    assert ideal_l2.parameters["distance"] == "weighted_l2_to_ideal"
    assert ideal_l2.parameters["objective_weights"] == {
        "obj_quality": 0.5,
        "obj_size": 0.5,
    }
    topsis = next(spec for spec in specs if spec.baseline_id == "objective_topsis")
    assert topsis.method_family == "ideal_antiideal_distance"
    assert topsis.implementation == "pymcdm_1.4.0_adapter"
    assert topsis.parameters["package_method"] == "TOPSIS"
    assert topsis.parameters["score_formula"] == "D_minus / (D_plus + D_minus)"
    assert topsis.parameters["objective_weights"] == {
        "obj_quality": 0.5,
        "obj_size": 0.5,
    }
    vikor = next(spec for spec in specs if spec.baseline_id == "objective_vikor")
    assert vikor.method_family == "compromise_ranking"
    assert vikor.implementation == "pymcdm_1.4.0_adapter"
    assert vikor.parameters["package_method"] == "VIKOR"
    assert vikor.parameters["v"] == 0.5
    assert vikor.parameters["constant_criteria_policy"] == "drop_within_front"
    assert vikor.parameters["objective_weights"] == {
        "obj_quality": 0.5,
        "obj_size": 0.5,
    }
    augmented_tchebycheff = next(
        spec for spec in specs if spec.baseline_id == "objective_augmented_tchebycheff"
    )
    assert augmented_tchebycheff.method_family == "augmented_tchebycheff"
    assert augmented_tchebycheff.implementation == "local_numpy"
    assert augmented_tchebycheff.parameters["rho"] == 1e-6
    assert augmented_tchebycheff.parameters["objective_weights"] == {
        "obj_quality": 0.5,
        "obj_size": 0.5,
    }
    assert "max_i" in augmented_tchebycheff.parameters["score_formula"]


def test_run_decision_baselines_writes_runs_and_summary(tmp_path):
    case_dir, config = _write_decision_case(tmp_path)
    context = load_case_context(case_dir, config_path=config)
    _write_amiga_reference(context)

    result = run_decision_baselines(
        context,
        seed=100,
        options=WriteOptions(),
    )

    assert result.status == "written"
    assert len(result.run_results) == 10
    assert result.baseline_manifest.exists()
    assert set(result.summary_outputs) == {
        "metrics_long",
        "metrics_summary",
        "metric_ranks",
        "metric_rank_stats",
    }
    assert result.table_outputs["primary_rank_table"].exists()
    primary_table = pd.read_csv(result.table_outputs["primary_rank_table"])
    assert AMIGA_REFERENCE_ID in set(primary_table["config"])
    assert len(primary_table) == 10
    expected_run_ids = [run_result.baseline.baseline_id for run_result in result.run_results]
    assert set(primary_table["config"]) == set(expected_run_ids)
    for run_result in result.run_results:
        assert run_result.ranked.exists()
        assert run_result.cv_report.exists()
        assert run_result.run_manifest.exists()

    manifest = json.loads(result.baseline_manifest.read_text(encoding="utf-8"))
    assert manifest["baseline_count"] == 10
    assert [baseline["baseline_id"] for baseline in manifest["baselines"]] == expected_run_ids
    assert all(baseline["method_family"] for baseline in manifest["baselines"])
    assert all(baseline["implementation"] for baseline in manifest["baselines"])
    assert all("method_parameters" in baseline for baseline in manifest["baselines"])
    assert manifest["tables"]["primary_rank_table"].endswith("primary_rank_table.csv")
    assert "plot_cv" not in manifest
    assert manifest["plots"]["plot_manifest"].endswith("plots/plot_manifest.json")
    assert manifest["plots"]["primary"]["png"].endswith("plots/decision_baseline_rank.png")
    assert manifest["plots"]["primary"]["pdf"].endswith("plots/decision_baseline_rank.pdf")
    assert manifest["plots"]["primary"]["csv"].endswith("plots/decision_baseline_rank.csv")
    phase_manifest = json.loads((result.phase_dir / "phase_manifest.json").read_text(encoding="utf-8"))
    assert phase_manifest["status"] == "completed"
    assert phase_manifest["outputs"]["baseline_manifest"].endswith("baseline_manifest.json")
    assert phase_manifest["outputs"]["baseline_count"] == 10
    assert phase_manifest["outputs"]["table_outputs"]["primary_rank_table"].endswith("primary_rank_table.csv")
    assert phase_manifest["outputs"]["plots"]["plot_manifest"].endswith("plots/plot_manifest.json")
    assert [baseline["baseline_id"] for baseline in manifest["baselines"][:3]] == [
        AMIGA_REFERENCE_ID,
        "objective__obj_quality",
        "objective__obj_size",
    ]
    amiga_manifest_entry = manifest["baselines"][0]
    assert amiga_manifest_entry["method_family"] == "learned_reference"
    assert amiga_manifest_entry["implementation"] == "copied_phase_02_final_test"
    normalized_manifest_entry = next(
        baseline
        for baseline in manifest["baselines"]
        if baseline["baseline_id"] == "objective_normalized_mean"
    )
    assert normalized_manifest_entry["method_family"] == "weighted_sum"
    assert normalized_manifest_entry["implementation"] == "pymcdm_1.4.0_adapter"
    assert normalized_manifest_entry["method_parameters"]["package_method"] == "WSM"
    assert normalized_manifest_entry["method_parameters"]["normalization"] == "per_front_minmax_badness"
    assert normalized_manifest_entry["method_parameters"]["objective_weights"] == {
        "obj_quality": 0.5,
        "obj_size": 0.5,
    }
    ideal_l2_manifest_entry = next(
        baseline
        for baseline in manifest["baselines"]
        if baseline["baseline_id"] == "objective_ideal_l2"
    )
    assert ideal_l2_manifest_entry["method_family"] == "ideal_distance"
    assert ideal_l2_manifest_entry["implementation"] == "local_numpy"
    assert ideal_l2_manifest_entry["method_parameters"]["score_formula"] == "-sqrt(sum_i(weight_i * badness_i^2))"
    topsis_manifest_entry = next(
        baseline
        for baseline in manifest["baselines"]
        if baseline["baseline_id"] == "objective_topsis"
    )
    assert topsis_manifest_entry["method_family"] == "ideal_antiideal_distance"
    assert topsis_manifest_entry["implementation"] == "pymcdm_1.4.0_adapter"
    assert topsis_manifest_entry["method_parameters"]["package_method"] == "TOPSIS"
    vikor_manifest_entry = next(
        baseline
        for baseline in manifest["baselines"]
        if baseline["baseline_id"] == "objective_vikor"
    )
    assert vikor_manifest_entry["method_family"] == "compromise_ranking"
    assert vikor_manifest_entry["implementation"] == "pymcdm_1.4.0_adapter"
    assert vikor_manifest_entry["method_parameters"]["package_method"] == "VIKOR"
    assert vikor_manifest_entry["method_parameters"]["v"] == 0.5
    assert vikor_manifest_entry["method_parameters"]["constant_criteria_policy"] == "drop_within_front"
    augmented_manifest_entry = next(
        baseline
        for baseline in manifest["baselines"]
        if baseline["baseline_id"] == "objective_augmented_tchebycheff"
    )
    assert augmented_manifest_entry["method_family"] == "augmented_tchebycheff"
    assert augmented_manifest_entry["implementation"] == "local_numpy"
    assert augmented_manifest_entry["method_parameters"]["rho"] == 1e-6

    objective_ranked = pd.read_csv(
        context.results_root
        / "04_decision_baselines"
        / "runs"
        / "objective__obj_quality"
        / "ranked.csv"
    )
    assert set(objective_ranked["front_id"]) == {5}
    assert int(objective_ranked.iloc[0]["item_id"]) == 6

    cv_report = json.loads(
        (
            context.results_root
            / "04_decision_baselines"
            / "runs"
            / "objective__obj_quality"
            / "cv_report.json"
        ).read_text(encoding="utf-8")
    )
    assert cv_report[0]["meta"]["evaluation_split"] == "test"
    assert cv_report[0]["meta"]["method_family"] == "single_objective"
    assert cv_report[0]["meta"]["implementation"] == "local_numpy_pandas"
    assert cv_report[0]["meta"]["method_parameters"]["normalization"] == "none"
    assert cv_report[0]["groups"][0]["front_id"] == 5
    assert "Regret@5" in cv_report[0]["agg"]

    run_manifest = json.loads(
        (
            context.results_root
            / "04_decision_baselines"
            / "runs"
            / "objective_normalized_mean"
            / "run_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert run_manifest["parameters"]["method_family"] == "weighted_sum"
    assert run_manifest["parameters"]["implementation"] == "pymcdm_1.4.0_adapter"
    assert run_manifest["parameters"]["method_parameters"]["weight_policy"] == "uniform"


def test_run_decision_baselines_rejects_mismatched_amiga_reference_fronts(tmp_path):
    case_dir, config = _write_decision_case(tmp_path)
    context = load_case_context(case_dir, config_path=config)
    final_dir = _write_amiga_reference(context)
    ranked_path = final_dir / "final_test_ranked.csv"
    ranked = pd.read_csv(ranked_path)
    ranked["front_id"] = 99
    ranked.to_csv(ranked_path, index=False)

    with pytest.raises(DecisionBaselineError, match="does not match held-out test fronts"):
        run_decision_baselines(
            context,
            seed=100,
            options=WriteOptions(),
        )
