"""Selection diagnostics, panel-wise inference and figure provenance."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from scripts.experiments.amiga_exp.cli import app
from scripts.experiments.amiga_exp.reporting.figure_data import (
    DISPLAY_METRICS, FAMILIES, FRACTIONS, LABELS, audit_selection,
    comparison_sets, feature_curve_data, load_importance, normalize_importance,
    rank_comparison, screening_data, tuning_data, validate_feature_summary, validate_selection,
)
from scripts.experiments.amiga_exp.reporting.supervised import METHODS, METRICS, build_tables, sha256


@pytest.fixture
def candidates():
    rows = []
    for arm in METHODS:
        for outer in range(5):
            for family in FAMILIES:
                label = LABELS[outer] if arm == "ranking" else "continuous"
                base = dict(case="CASE", arm=arm, family=family, outer_fold=outer,
                            label=label, parameters='{"learning_rate":0.03}',
                            fraction=1., n_features=4, std_regret5=.02)
                if arm == "ranking":
                    for i, relevance in enumerate(LABELS):
                        rows.append(dict(base, stage="phase1", label=relevance, config_id="screening",
                                         candidate=f"{arm}/{outer}/{family}/phase1/{relevance}",
                                         job=f"{arm}/{outer}/{family}/phase1/{relevance}",
                                         mean_regret5=.01+i*.001, selected_within_family=i == outer))
                for i in range(27 if family == "LightGBM" else 36):
                    rows.append(dict(base, stage="phase2", config_id=f"cfg-{i}",
                                     candidate=f"{arm}/{outer}/{family}/phase2/{i}",
                                     job=f"{arm}/{outer}/{family}/phase2/{i}",
                                     mean_regret5=.01+i*.001+outer*.0001,
                                     selected_within_family=i == outer))
                for i, fraction in enumerate(FRACTIONS):
                    rows.append(dict(base, stage="phase3", config_id="selected", fraction=fraction,
                                     n_features=i+1, candidate=f"{arm}/{outer}/{family}/phase3/{fraction}",
                                     job=f"{arm}/{outer}/{family}/phase3", mean_regret5=.02-fraction*.01,
                                     selected_within_family=i == outer % 4))
    return pd.DataFrame(rows)


def test_diagnostics_include_all_complements_and_original_selection_counts(candidates):
    validate_selection(candidates)
    screening = screening_data(candidates)
    assert len(screening) == 24
    assert set(screening.n_outer_folds) == {5}
    assert set(screening.groupby("family").selected_folds.sum()) == {5}
    tuning = tuning_data(candidates)
    assert len(tuning) == 99*5
    assert set(tuning.n_outer_folds) == {5}
    # The fixture selects five different configurations in each family. No
    # pooled "best" configuration may replace those five original selections.
    selected = tuning.query('arm == "ranking" and family == "LightGBM" and selected_folds > 0')
    assert len(selected) == 5 and set(selected.selected_folds) == {1}
    assert set(tuning.query('arm == "ranking"').label_modes) == {", ".join(sorted(LABELS[:5]))}
    assert tuning.query('arm == "ranking" and family == "LightGBM" and config_id == "cfg-0"').mean_regret5.item() == pytest.approx(.0102)
    curves = feature_curve_data(candidates)
    assert set(curves.n_outer_folds) == {5}
    assert set(curves.groupby(["arm", "family"]).selected_folds.sum()) == {5}


@pytest.mark.parametrize("mutation", ["missing_fold", "missing_family", "missing_label", "missing_budget", "missing_grid", "duplicate_grid", "no_selection", "extra_fold"])
def test_selection_coverage_cannot_be_silently_reduced(candidates, mutation):
    if mutation == "missing_fold":
        candidates = candidates[candidates.outer_fold != 2]
    elif mutation == "missing_family":
        candidates = candidates[candidates.family != "XGBoost"]
    elif mutation == "missing_label":
        candidates = candidates[~((candidates.stage == "phase1") & (candidates.label == "shuffled"))]
    elif mutation == "missing_budget":
        candidates = candidates[~((candidates.stage == "phase3") & (candidates.fraction == .25))]
    elif mutation == "missing_grid":
        candidates = candidates[~((candidates.stage == "phase2") & (candidates.config_id == "cfg-0"))]
    elif mutation == "duplicate_grid":
        idx = candidates.index[candidates.stage == "phase2"][1]
        candidates.loc[idx, "config_id"] = "cfg-0"
    elif mutation == "no_selection":
        candidates["selected_within_family"] = False
    else:
        candidates.loc[candidates.outer_fold == 4, "outer_fold"] = 5
    with pytest.raises(ValueError):
        validate_selection(candidates)


def importance_rows():
    return pd.DataFrame([dict(inner_fold=inner, fraction=1., feature=feature,
                              centered_mean_abs_shap=value*scale)
                         for inner, scale in enumerate((1, 100, .01))
                         for feature, value in zip(("a", "b", "c", "d"), (1, 3, 0, 0), strict=True)])


def test_shap_is_normalized_within_fit_before_averaging():
    rows = normalize_importance(importance_rows())
    expected = np.tile([.25, .75, 0, 0], 3)
    np.testing.assert_allclose(rows.relative_centered_shap, expected)
    rows.loc[rows.inner_fold == 0, "centered_mean_abs_shap"] = 0
    changed = normalize_importance(rows)
    np.testing.assert_array_equal(changed.query("inner_fold == 0").relative_centered_shap, 0)
    np.testing.assert_allclose(changed.query("inner_fold > 0").groupby("inner_fold").relative_centered_shap.sum(), 1)


@pytest.mark.parametrize("mutation", ["negative", "infinite", "duplicate", "missing_full"])
def test_invalid_shap_is_rejected(mutation):
    rows = importance_rows()
    if mutation == "negative":
        rows.loc[0, "centered_mean_abs_shap"] = -1
    elif mutation == "infinite":
        rows.loc[0, "centered_mean_abs_shap"] = np.inf
    elif mutation == "duplicate":
        rows = pd.concat([rows, rows.iloc[[0]]])
    else:
        rows["fraction"] = .75
    with pytest.raises(ValueError):
        normalize_importance(rows)


@pytest.mark.parametrize("mutation", [None, "missing_column", "duplicate", "wrong_fit_count", "invalid_rate"])
def test_importance_and_inclusion_require_matching_column_coverage(mutation):
    keys = dict(case=["CASE", "CASE"], arm=["ranking", "ranking"],
                family=["LightGBM", "LightGBM"], feature=["a", "b"])
    importance = pd.DataFrame(keys)
    stability = pd.DataFrame(dict(keys, selected_frequency=[1., .5], number_of_fits=[15, 15]))
    if mutation == "missing_column":
        stability = stability.iloc[:1]
    elif mutation == "duplicate":
        stability = pd.concat([stability, stability.iloc[[0]]])
    elif mutation == "wrong_fit_count":
        stability.loc[0, "number_of_fits"] = 14
    elif mutation == "invalid_rate":
        stability.loc[0, "selected_frequency"] = 1.1
    if mutation is None:
        validate_feature_summary(importance, stability)
    else:
        with pytest.raises(ValueError):
            validate_feature_summary(importance, stability)


def test_shap_is_bound_to_selected_job_hash_and_all_fifteen_fits(candidates, tmp_path):
    table = candidates.query('stage == "phase3" and family == "LightGBM" and arm == "ranking"')
    manifest = dict(dependency_result_hashes={})
    for job in table.job.unique():
        attempt = tmp_path/"jobs"/job/"attempt-001"
        attempt.mkdir(parents=True)
        importance_rows().to_csv(attempt/"feature_importance.csv", index=False)
        result = dict(status="complete", job=dict(id=job),
                      artifacts={"feature_importance.csv":sha256(attempt/"feature_importance.csv")})
        (attempt/"result.json").write_text(json.dumps(result))
        manifest["dependency_result_hashes"][job] = sha256(attempt/"result.json")
    output, identities = load_importance(table, {"ranking":(tmp_path, manifest)})
    assert set(output.n_inner_fits) == {15} and len(identities) == 10
    np.testing.assert_allclose(output.relative_centered_shap, [.25, .75, 0, 0])
    csv = next(tmp_path.rglob("feature_importance.csv"))
    csv.write_text(csv.read_text()+"tampering\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_importance(table, {"ranking":(tmp_path, manifest)})


@pytest.fixture
def metrics():
    values = [[.1, .1, .3, .4, .5], [.5, .4, .3, .2, .1],
              [.1, .2, .3, .4, .5], [.2, .1, .3, .5, .4]]
    return pd.DataFrame([dict(case="CASE", topology_id=f"topology-{i}", method=method,
                             **{metric:value if metric.startswith("Regret") else 1-value for metric in METRICS})
                         for i, row in enumerate(values)
                         for method, value in zip(METHODS, row, strict=True)])


def test_panel_statistics_reproduce_previous_five_method_report(metrics):
    rows, omnibus = rank_comparison(metrics, METHODS, expected_topologies=4)
    existing = build_tables(metrics, expected_topologies=4)
    for metric in DISPLAY_METRICS:
        ranks = existing["mean_ranks_all_metrics.csv"].query("metric == @metric").set_index("method")
        raw = existing["raw_metric_means.csv"].query("metric == @metric").set_index("method")
        np.testing.assert_allclose(rows[metric+"_mean_rank"], ranks.loc[list(METHODS), "mean_rank"])
        np.testing.assert_allclose(rows[metric+"_mean"], raw.loc[list(METHODS), "metric_mean"])
    np.testing.assert_allclose(rows.p_holm.iloc[1:], existing["regret5_posthoc_holm.csv"].p_holm_within_case_four)
    assert omnibus.p_value.item() == pytest.approx(existing["friedman_omnibus.csv"].p_value.item())
    assert set(omnibus.metric) == {"Regret@5"}
    assert rows.iloc[0]["Regret@5_mean_rank"] > rows.iloc[1]["Regret@5_mean_rank"]
    assert np.isnan(rows.iloc[0].p_holm)  # AMIGA remains control even if second.


def test_panel_sets_and_corrections_are_case_specific(metrics):
    extras = []
    for method in ("objective_knee", "objective_topsis", "random_uniform", "oracle", "reg_normalized"):
        extra = metrics.query('method == "ranking"').copy()
        extra["method"] = method
        extra.loc[:, list(METRICS)] = .25
        extras.append(extra)
    frame = pd.concat([metrics, *extras], ignore_index=True)
    groups = comparison_sets(frame, "separate")
    assert groups["supervised"] == METHODS
    assert set(groups["objectives"]) == {"ranking", "objective_knee", "objective_topsis", "random_uniform"}
    rows, tests = rank_comparison(frame, groups["objectives"], expected_topologies=4)
    assert tests.n_methods.item() == 4 and tests.n_holm_comparisons.item() == 3
    assert set(rows.standard_error) == {np.sqrt(4*5/(6*4))}
    assert len(comparison_sets(frame.query('method != "objective_topsis"'), "separate")["objectives"]) == 3
    # Removing an objective-only comparator cannot change the supervised panel.
    original = rank_comparison(frame, METHODS, expected_topologies=4)[0]
    changed = rank_comparison(frame.query('method != "objective_topsis"'), METHODS, expected_topologies=4)[0]
    pd.testing.assert_frame_equal(original, changed)


def test_all_tied_panel_and_metric_direction(metrics):
    metrics.loc[:, list(METRICS)] = .25
    rows, omnibus = rank_comparison(metrics, METHODS, expected_topologies=4)
    assert set(rows["Hit@5_mean_rank"]) == {3.}
    assert set(rows.p_holm.dropna()) == {1.} and not rows.significant.any()
    assert omnibus.statistic.item() == 0 and omnibus.p_value.item() == 1
    metrics.loc[metrics.method == "clf_top10", "Hit@1"] = 1
    rows, _ = rank_comparison(metrics, METHODS, expected_topologies=4)
    assert rows.query('method == "clf_top10"')["Hit@1_mean_rank"].item() == 1
    assert set(rows["Regret@5_mean_rank"]) == {3.}


@pytest.mark.parametrize("mutation", ["duplicate", "missing_pair", "missing_metric", "nan_id", "negative", "infinite", "above_one"])
def test_invalid_paired_evidence_stops_figure_statistics(metrics, mutation):
    if mutation == "duplicate":
        metrics = pd.concat([metrics, metrics.iloc[[0]]])
    elif mutation == "missing_pair":
        metrics = metrics.iloc[1:]
    elif mutation == "missing_metric":
        metrics = metrics.drop(columns="Hit@5")
    elif mutation == "nan_id":
        metrics.loc[0, "topology_id"] = np.nan
    else:
        metrics.loc[0, "Regret@5"] = dict(negative=-.01, infinite=np.inf, above_one=1.01)[mutation]
    with pytest.raises(ValueError):
        rank_comparison(metrics, METHODS, expected_topologies=4)


def test_selection_artifact_audit_rejects_tampering(candidates, tmp_path):
    candidates.to_csv(tmp_path/"selection_candidates.csv", index=False)
    pd.DataFrame(dict(feature=["a"])).to_csv(tmp_path/"feature_stability.csv", index=False)
    artifacts = {name:sha256(tmp_path/name) for name in ("selection_candidates.csv", "feature_stability.csv")}
    (tmp_path/"summary_manifest.json").write_text(json.dumps(dict(status="complete", artifacts=artifacts)))
    rows, _, _, identities = audit_selection(tmp_path)
    assert len(rows) == len(candidates) and len(identities) == 3
    path = tmp_path/"selection_candidates.csv"
    path.write_text(path.read_text()+"changed\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit_selection(tmp_path)


def test_new_figure_cli_is_discoverable_and_existing_output_is_protected(tmp_path):
    runner = CliRunner()
    help_result = runner.invoke(app, ["report", "figures", "--help"])
    assert help_result.exit_code == 0 and "--feature-count" in help_result.stdout
    args = ["report", "figures", "--output", str(tmp_path)]
    for option in ("selection-summary", "selection-run", "top5-summary", "top5-run", "top10-summary", "top10-run", "outer-summary"):
        args.extend(["--"+option, str(tmp_path)])
    result = runner.invoke(app, args)
    assert result.exit_code == 1 and "new figure directory" in result.output


@pytest.mark.parametrize("objective_count", [13, 16])
def test_horizontal_primary_comparison_keeps_labels_and_titles_clear(metrics, monkeypatch, objective_count):
    from scripts.experiments.amiga_exp.reporting import publication_plots as plots
    supervised, _ = rank_comparison(metrics, METHODS, expected_topologies=4)
    names = ["ranking", "objective__reducenonessentialsinteractions",
             *[f"objective_{i}" for i in range(objective_count-2)]]
    objectives = pd.DataFrame([dict(supervised.iloc[0], method=method, n_methods=objective_count,
                                    **{metric+"_mean_rank":1+i*.5 for metric in DISPLAY_METRICS})
                               for i, method in enumerate(names)])
    captured = []
    monkeypatch.setattr(plots, "save", lambda fig, prefix:captured.append(fig))
    plots.decision_comparison([("supervised", supervised), ("objectives", objectives)], Path("unused"), "CASE")
    fig = captured[0]
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        titles = {text.get_gid():text.get_window_extent(renderer) for text in fig.texts if text.get_gid()}
        assert fig.get_figwidth() > fig.get_figheight()
        assert len(fig.axes) == 6
        assert fig.axes[1].get_position().y0 == fig.axes[4].get_position().y0
        assert fig.axes[1].get_position().x1 < fig.axes[4].get_position().x0
        for i in range(2):
            assert titles[f"panel-title-{i}"].y1 < fig.legends[0].get_window_extent(renderer).y0
            assert titles[f"panel-title-{i}"].y0 > fig.axes[3*i+1].title.get_window_extent(renderer).y1
        footer = next(text for text in fig.texts if text.get_text().startswith("Regret@5 only"))
        assert footer.get_window_extent(renderer).y1 < fig.axes[1].xaxis.label.get_window_extent(renderer).y0
        assert not any("Hit@" in text.get_text() for text in fig.findobj(plots.matplotlib.text.Text))
        # Both lines of each numeric label need room within a method row.
        for axis in (fig.axes[1], fig.axes[4]):
            for rank, raw in zip(axis.texts[::2], axis.texts[1::2], strict=True):
                assert not rank.get_window_extent(renderer).overlaps(raw.get_window_extent(renderer))
        for axis in (fig.axes[0], fig.axes[3]):
            for first, second in zip(axis.texts, axis.texts[1:]):
                assert not first.get_window_extent(renderer).overlaps(second.get_window_extent(renderer))
    finally:
        plots.plt.close(fig)


@pytest.mark.parametrize("count, expected", [(4, ["a", "b", "e", "f"]),
                                            (3, ["a", "b", "f"]),
                                            (20, ["a", "b", "c", "d", "e", "f"])])
def test_feature_extremes_are_disjoint_and_keep_full_model_normalization(count, expected):
    from scripts.experiments.amiga_exp.reporting.publication_plots import feature_matrix
    importance = pd.DataFrame([dict(feature=feature, family=family, relative_centered_shap=value/21)
                               for family in FAMILIES
                               for feature, value in zip("abcdef", [6,5,4,3,2,1], strict=True)])
    stability = importance.assign(selected_frequency=.5)
    values, inclusion = feature_matrix(importance, stability, count=count, extremes=True)
    assert list(values.index) == expected and values.index.is_unique
    assert values.loc["f", "CatBoost"] == pytest.approx(100/21)
    assert (inclusion == 50).all().all()
    if count < 6:
        assert (values.sum() < 100).all()  # The displayed subset is never renormalized.
