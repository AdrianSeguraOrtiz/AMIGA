from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.experiments.amiga_exp.manifests import WriteOptions
from scripts.experiments.amiga_exp.plots import (
    PLOT_PHASES,
    PlotPreparationError,
    prepare_all_phase_plots,
    prepare_phase_plots,
)


def _context(tmp_path: Path):
    case_dir = tmp_path / "CASE"
    case_dir.mkdir()
    return SimpleNamespace(
        case_name="CASE",
        case_dir=case_dir,
        data_csv=case_dir / "data" / "data_1.csv",
        config_path=tmp_path / "CASE.json",
        split_manifest_path=tmp_path / "CASE_split_manifest.json",
        feature_contract_path=tmp_path / "CASE_feature_columns.json",
        results_root=case_dir / "results" / "amiga-exp",
        n_development_fronts=4,
        n_development_rows=16,
        n_test_fronts=1,
        n_test_rows=4,
    )


def _write_primary_rank_table(context, phase: str) -> Path:
    table_path = context.results_root / phase / "summary" / "primary_rank_table.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "config,avg_rank,p_value,mean_regret5,std_regret5,n_fronts",
        "run_a,1.0,,0.10,0.01,4",
        "run_b,2.0,0.2,0.20,0.02,4",
    ]
    if phase == "01_model_screening":
        rows = [
            "config,avg_rank,p_value,mean_regret5,std_regret5,n_fronts",
            "LGBMRanker__continuous,1.0,,0.10,0.01,4",
            "XGBRanker__rank_dense,2.0,0.2,0.20,0.02,4",
            "XGBRanker__reversed,9.0,0.01,0.70,0.05,4",
            "XGBRanker__shuffled,10.0,0.01,0.80,0.05,4",
        ]
    if phase == "02_hyperparameter_tuning":
        rows = [
            "config,avg_rank,p_value,mean_regret5,std_regret5,n_fronts",
            "LGBMRanker__continuous__tiny,1.0,,0.10,0.01,4",
            "LGBMRanker__continuous__large,2.0,0.2,0.20,0.02,4",
            "XGBRanker__continuous__tiny,3.0,0.01,0.30,0.03,4",
        ]
    if phase == "03_ablation":
        rows = [
            "config,avg_rank,p_value,mean_regret5,std_regret5,n_fronts",
            "full,1.0,,0.10,0.01,4",
            "no_network,2.0,0.8,0.08,0.02,4",
            "expression_only,3.0,0.01,0.18,0.02,4",
        ]
    if phase == "04_decision_baselines":
        rows = [
            "config,avg_rank,p_value,mean_regret5,std_regret5,n_fronts",
            "AMIGA_final,1.0,,0.10,0.01,4",
            "objective__obj_quality,2.0,0.2,0.20,0.02,4",
            "objective_mean_rank,3.0,0.05,0.30,0.02,4",
            "objective_normalized_mean,4.0,0.01,0.40,0.03,4",
            "objective_ideal_l2,5.0,0.02,0.50,0.03,4",
            "objective_topsis,6.0,0.01,0.60,0.03,4",
            "objective_vikor,7.0,0.01,0.70,0.04,4",
            "objective_augmented_tchebycheff,8.0,0.01,0.80,0.04,4",
            "objective_tchebycheff,9.0,0.01,0.90,0.05,4",
        ]
    table_path.write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    return table_path


def _write_metrics_summary(context, phase: str) -> Path:
    summary_path = context.results_root / phase / "summary" / "metrics_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    configs_by_phase = {
        "01_model_screening": ["LGBMRanker__continuous", "XGBRanker__rank_dense"],
        "02_hyperparameter_tuning": [
            "LGBMRanker__continuous__tiny",
            "LGBMRanker__continuous__large",
            "XGBRanker__continuous__tiny",
        ],
        "03_ablation": ["full", "expression_only", "no_network"],
        "04_decision_baselines": [
            "AMIGA_final",
            "objective__obj_quality",
            "objective_mean_rank",
            "objective_normalized_mean",
            "objective_ideal_l2",
            "objective_topsis",
            "objective_vikor",
            "objective_augmented_tchebycheff",
            "objective_tchebycheff",
        ],
    }
    metrics = [
        ("Regret@1", "primary", 10),
        ("Regret@3", "primary", 11),
        ("Regret@5", "primary", 12),
        ("Hit@1", "primary", 30),
        ("Hit@3", "primary", 31),
        ("Hit@5", "primary", 32),
        ("BestAUPR@1", "primary", 20),
        ("BestAUPR@3", "primary", 21),
        ("BestAUPR@5", "primary", 22),
    ]
    rows = ["model,metric,tier,priority,mean,std,n,rank"]
    for config_idx, config in enumerate(configs_by_phase[phase], start=1):
        for metric, tier, priority in metrics:
            mean = 0.05 * config_idx
            if metric.startswith("Hit@") or metric.startswith("BestAUPR@"):
                mean = 1.0 - 0.05 * config_idx
            rows.append(f"{config},{metric},{tier},{priority},{mean:.3f},0.010,4,{config_idx}")
    summary_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return summary_path


def _write_statistical_tests(context) -> Path:
    path = context.results_root / "summaries" / "statistical_tests.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "comparison_type,metric,method,avg_rank,holm_p_adj,is_winner,significantly_worse,friedman_stat,friedman_p,n_fronts,n_methods,n_fronts_common",
                "ablation,Regret@5,AMIGA_final,1.5,,True,False,5.0,0.08,1,4,1",
                "ablation,Regret@5,full,1.5,1.0,False,False,5.0,0.08,1,4,1",
                "ablation,Regret@5,no_network,2.0,0.4,False,False,5.0,0.08,1,4,1",
                "ablation,Regret@5,expression_only,3.0,0.01,False,True,5.0,0.08,1,4,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_shortlisted_configs(context) -> Path:
    path = context.results_root / "01_model_screening" / "shortlisted_configs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "configs": [
                    {
                        "run_id": "LGBMRanker__continuous",
                        "model_type": "LGBMRanker",
                        "label_mode": "continuous",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_selected_config(context, *, run_id: str = "LGBMRanker__continuous__tiny") -> Path:
    path = context.results_root / "02_hyperparameter_tuning" / "selected_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"selected_config": {"run_id": run_id}}),
        encoding="utf-8",
    )
    return path


def test_prepare_phase_plots_writes_manifest_from_primary_rank_table(tmp_path):
    context = _context(tmp_path)
    table_path = _write_primary_rank_table(context, "01_model_screening")
    _write_shortlisted_configs(context)

    result = prepare_phase_plots(context, "01_model_screening", options=WriteOptions())

    assert result.status == "written"
    assert result.primary_rank_table == table_path
    assert result.plots_dir.exists()
    assert result.plot_manifest.exists()
    manifest = json.loads(result.plot_manifest.read_text(encoding="utf-8"))
    assert manifest["manifest_type"] == "plot_phase"
    assert manifest["phase"] == "01_model_screening"
    assert manifest["status"] == "generated"
    assert manifest["primary_rank_table_rows"] == 4
    assert manifest["planned_outputs"]["primary"]["png"].endswith("model_screening_heatmap.png")
    assert manifest["planned_outputs"]["primary"]["pdf"].endswith("model_screening_heatmap.pdf")
    assert manifest["generated_outputs"]["primary"]["png"].endswith("model_screening_heatmap.png")
    assert manifest["generated_outputs"]["primary"]["pdf"].endswith("model_screening_heatmap.pdf")
    assert result.generated_outputs["primary"]["png"].stat().st_size > 1000
    assert result.generated_outputs["primary"]["pdf"].stat().st_size > 1000


def test_prepare_phase02_plots_writes_regret_scatter_and_plotted_data(tmp_path):
    context = _context(tmp_path)
    _write_primary_rank_table(context, "02_hyperparameter_tuning")
    _write_selected_config(context)

    result = prepare_phase_plots(context, "02_hyperparameter_tuning", options=WriteOptions())

    assert result.status == "written"
    outputs = result.generated_outputs["primary"]
    assert outputs["png"].name == "hyperparameter_regret_scatter.png"
    assert outputs["pdf"].name == "hyperparameter_regret_scatter.pdf"
    assert outputs["csv"].name == "hyperparameter_regret_scatter.csv"
    assert outputs["png"].stat().st_size > 1000
    plotted = outputs["csv"].read_text(encoding="utf-8").splitlines()
    assert plotted[0].startswith("config,avg_rank,p_value")
    assert "base_config" in plotted[0]
    assert "param_label" in plotted[0]
    assert "point_label" in plotted[0]
    assert plotted[1].startswith("LGBMRanker__continuous__tiny,1.0,")
    assert ",True," in plotted[1]
    manifest = json.loads(result.plot_manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "generated"
    assert manifest["generated_outputs"]["primary"]["csv"].endswith("hyperparameter_regret_scatter.csv")


def test_prepare_phase_plots_can_include_supplementary_topk_context(tmp_path):
    context = _context(tmp_path)
    _write_primary_rank_table(context, "02_hyperparameter_tuning")
    _write_metrics_summary(context, "02_hyperparameter_tuning")
    _write_selected_config(context)

    result = prepare_phase_plots(
        context,
        "02_hyperparameter_tuning",
        options=WriteOptions(),
        include_secondary=True,
    )

    topk_outputs = result.generated_outputs["supplementary_topk_curves"]
    appendix_outputs = result.generated_outputs["supplementary_metric_appendix"]
    assert topk_outputs["png"].name == "topk_metric_curves.png"
    assert topk_outputs["pdf"].name == "topk_metric_curves.pdf"
    assert topk_outputs["csv"].name == "topk_metric_curves.csv"
    assert topk_outputs["png"].parent.name == "supplementary"
    assert topk_outputs["png"].stat().st_size > 1000
    assert appendix_outputs["csv"].name == "secondary_metric_appendix.csv"
    assert appendix_outputs["caption"].name == "secondary_metric_caption.txt"
    assert "Complementary context only" in appendix_outputs["caption"].read_text(encoding="utf-8")

    topk_rows = topk_outputs["csv"].read_text(encoding="utf-8").splitlines()
    assert topk_rows[0].startswith("config,metric_family,k,metric,mean")
    assert "Regret@5" in "\n".join(topk_rows)
    appendix_header = appendix_outputs["csv"].read_text(encoding="utf-8").splitlines()[0]
    assert "Regret@5_mean" in appendix_header

    manifest = json.loads(result.plot_manifest.read_text(encoding="utf-8"))
    assert manifest["secondary_outputs_policy"]["included"] is True
    assert "Complementary context only" in manifest["secondary_outputs_policy"]["role"]
    assert manifest["planned_outputs"]["supplementary_topk_curves"]["png"].endswith(
        "plots/supplementary/topk_metric_curves.png"
    )
    assert manifest["generated_outputs"]["supplementary_metric_appendix"]["caption"].endswith(
        "plots/supplementary/secondary_metric_caption.txt"
    )


def test_prepare_phase03_plots_writes_ablation_feature_matrix_and_plotted_data(tmp_path):
    context = _context(tmp_path)
    _write_primary_rank_table(context, "03_ablation")
    _write_statistical_tests(context)

    result = prepare_phase_plots(context, "03_ablation", options=WriteOptions())

    assert result.status == "written"
    outputs = result.generated_outputs["primary"]
    assert outputs["png"].name == "ablation_feature_matrix.png"
    assert outputs["pdf"].name == "ablation_feature_matrix.pdf"
    assert outputs["csv"].name == "ablation_feature_matrix.csv"
    assert outputs["png"].stat().st_size > 1000
    plotted = outputs["csv"].read_text(encoding="utf-8").splitlines()
    assert plotted[0].startswith(
        "config,display_label,objectives,technique_weights,expression,network,development_avg_rank"
    )
    assert "test_avg_rank" in plotted[0]
    assert "test_p_value_label" in plotted[0]
    assert "full,Full AMIGA,True,True,True,True,1.0,,winner" in plotted[1]
    assert ",1.5,,winner" in plotted[1]
    joined = "\n".join(plotted)
    assert "no_network,No network,True,True,True,False,2.0,0.8,p=0.800" in joined
    assert "expression_only,Expression only,False,False,True,False,3.0,0.01,p=0.010" in joined
    assert ",3.0,0.01,p=0.010" in joined
    manifest = json.loads(result.plot_manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "generated"
    assert manifest["generated_outputs"]["primary"]["csv"].endswith("ablation_feature_matrix.csv")


def test_prepare_phase04_plots_writes_decision_baseline_rank_plot(tmp_path):
    context = _context(tmp_path)
    _write_primary_rank_table(context, "04_decision_baselines")

    result = prepare_phase_plots(context, "04_decision_baselines", options=WriteOptions())

    assert result.status == "written"
    outputs = result.generated_outputs["primary"]
    assert outputs["png"].name == "decision_baseline_rank.png"
    assert outputs["pdf"].name == "decision_baseline_rank.pdf"
    assert outputs["csv"].name == "decision_baseline_rank.csv"
    assert outputs["png"].stat().st_size > 1000
    plotted = outputs["csv"].read_text(encoding="utf-8").splitlines()
    assert plotted[0].startswith("config,display_label,avg_rank,p_value,p_value_label")
    assert "p_value_significance" in plotted[0]
    assert "p_value_color" in plotted[0]
    assert "AMIGA_final,AMIGA,1.0,,winner" in plotted[1]
    joined = "\n".join(plotted)
    assert "objective__obj_quality,Single objective: obj quality,2.0,0.2,p=0.200" in joined
    assert "objective_mean_rank,Objective mean rank,3.0,0.05,p=0.050" in joined
    assert "objective_normalized_mean,Normalized objective mean,4.0,0.01,p=0.010" in joined
    assert "objective_ideal_l2,Ideal L2 distance,5.0,0.02,p=0.020" in joined
    assert "objective_topsis,TOPSIS,6.0,0.01,p=0.010" in joined
    assert "objective_vikor,VIKOR,7.0,0.01,p=0.010" in joined
    assert "objective_augmented_tchebycheff,Augmented Tchebycheff,8.0,0.01,p=0.010" in joined
    assert "objective_tchebycheff,Objective Tchebycheff,9.0,0.01,p=0.010" in joined
    manifest = json.loads(result.plot_manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "generated"
    assert manifest["primary_rank_table_rows"] == 9
    assert manifest["generated_outputs"]["primary"]["csv"].endswith("decision_baseline_rank.csv")


def test_prepare_all_phase_plots_writes_one_manifest_per_phase(tmp_path):
    context = _context(tmp_path)
    for phase in PLOT_PHASES:
        _write_primary_rank_table(context, phase)
    _write_shortlisted_configs(context)
    _write_selected_config(context)
    _write_statistical_tests(context)

    result = prepare_all_phase_plots(context, options=WriteOptions())

    assert [phase_result.phase for phase_result in result.results] == list(PLOT_PHASES)
    assert all(phase_result.plot_manifest.exists() for phase_result in result.results)


def test_prepare_phase_plots_requires_primary_rank_table(tmp_path):
    context = _context(tmp_path)

    with pytest.raises(PlotPreparationError, match="primary rank table"):
        prepare_phase_plots(context, "01_model_screening", options=WriteOptions())


def test_prepare_phase_plots_rejects_unknown_phase(tmp_path):
    context = _context(tmp_path)

    with pytest.raises(PlotPreparationError, match="unsupported plot phase"):
        prepare_phase_plots(context, "unknown_phase", options=WriteOptions())


def test_prepare_phase01_plots_requires_shortlisted_config_to_mark_selection(tmp_path):
    context = _context(tmp_path)
    _write_primary_rank_table(context, "01_model_screening")

    with pytest.raises(PlotPreparationError, match="shortlisted configs"):
        prepare_phase_plots(context, "01_model_screening", options=WriteOptions())


def test_prepare_phase02_plots_requires_selected_config_to_mark_selection(tmp_path):
    context = _context(tmp_path)
    _write_primary_rank_table(context, "02_hyperparameter_tuning")

    with pytest.raises(PlotPreparationError, match="selected config"):
        prepare_phase_plots(context, "02_hyperparameter_tuning", options=WriteOptions())


def test_prepare_phase03_plots_requires_full_reference(tmp_path):
    context = _context(tmp_path)
    table_path = context.results_root / "03_ablation" / "summary" / "primary_rank_table.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(
        "\n".join(
            [
                "config,avg_rank,p_value,mean_regret5,std_regret5,n_fronts",
                "expression_only,2.5,0.2,0.18,0.02,4",
                "no_network,1.0,0.8,0.08,0.02,4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_statistical_tests(context)

    with pytest.raises(PlotPreparationError, match="'full' reference"):
        prepare_phase_plots(context, "03_ablation", options=WriteOptions())


def test_prepare_phase03_plots_requires_held_out_statistical_tests(tmp_path):
    context = _context(tmp_path)
    _write_primary_rank_table(context, "03_ablation")

    with pytest.raises(PlotPreparationError, match="summarize-paper"):
        prepare_phase_plots(context, "03_ablation", options=WriteOptions())


def test_prepare_phase04_plots_requires_amiga_reference(tmp_path):
    context = _context(tmp_path)
    table_path = context.results_root / "04_decision_baselines" / "summary" / "primary_rank_table.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(
        "\n".join(
            [
                "config,avg_rank,p_value,mean_regret5,std_regret5,n_fronts",
                "objective_mean_rank,3.0,0.05,0.30,0.02,4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PlotPreparationError, match="'AMIGA_final' reference"):
        prepare_phase_plots(context, "04_decision_baselines", options=WriteOptions())
