"""Paired ranking semantics, numerical inference and immutable reporting inputs."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2, norm
from typer.testing import CliRunner

from scripts.experiments.amiga_exp.cli import app
from scripts.experiments.amiga_exp.reporting.supervised import (
    METRICS, METHODS, build_tables, sha256, verify_manifest, write_report,
)
from scripts.experiments.amiga_exp.version import __version__


@pytest.fixture
def metrics():
    values = [[.1, .1, .3, .4, .5], [.5, .4, .3, .2, .1],
              [.1, .2, .3, .4, .5], [.2, .1, .3, .5, .4]]
    rows = []
    for topology, row in enumerate(values):
        for method, value in zip(METHODS, row, strict=True):
            rows.append(dict(case="CASE", topology_id=f"topology-{topology}", method=method,
                             **{m: value if m.startswith("Regret") else 1-value for m in METRICS}))
    return pd.DataFrame(rows)


def save_summary(path, frame):
    path.mkdir()
    source = path / "topology_metrics.csv"
    frame.to_csv(source, index=False)
    (path / "manifest.json").write_text(json.dumps(dict(status="complete", artifacts={source.name: sha256(source)})))
    return path


def test_direction_ties_and_friedman_match_hand_computed_ranks(metrics):
    tables = build_tables(metrics, expected_topologies=4)
    expected = np.array([[1.5, 1.5, 3, 4, 5], [5, 4, 3, 2, 1],
                         [1, 2, 3, 4, 5], [2, 1, 3, 5, 4]])
    np.testing.assert_array_equal(tables["CASE-regret5-ranks.csv"].to_numpy(), expected)
    ranks = tables["mean_ranks_all_metrics.csv"]
    for metric in METRICS:
        np.testing.assert_array_equal(ranks.loc[ranks.metric == metric, "mean_rank"], expected.mean(axis=0))
    uncorrected = 12 * 4 / (5 * 6) * np.square(expected.mean(axis=0)).sum() - 3 * 4 * 6
    statistic = uncorrected / (1 - 6 / (4 * 5 * (5**2 - 1)))
    row = tables["friedman_omnibus.csv"].iloc[0]
    assert row.statistic == pytest.approx(statistic)
    assert row.p_value == pytest.approx(chi2.sf(statistic, 4))
    assert row.statistic != pytest.approx(uncorrected)
    raw = tables["raw_metric_means.csv"]
    assert raw.loc[(raw.method == "ranking") & (raw.metric == "Regret@5"), "metric_mean"].item() == pytest.approx(.225)


def test_control_is_fixed_even_when_another_method_has_better_mean_rank(metrics):
    tables = build_tables(metrics, expected_topologies=4)
    contrasts = tables["regret5_posthoc_holm.csv"]
    assert set(contrasts.control) == {"ranking"}
    reg = contrasts.set_index("comparator").loc["reg_aupr"]
    assert reg.mean_rank_delta == pytest.approx(-.25)
    assert reg.standard_error == pytest.approx(np.sqrt(5*6/(6*4)))
    assert reg.p_raw == pytest.approx(2 * norm.sf(abs(-.25 / np.sqrt(1.25))))
    assert (contrasts.p_holm_within_case_four <= 1).all()
    assert (contrasts.p_holm_within_case_four >= contrasts.p_raw).all()
    assert not contrasts.reject_005_after_omnibus_and_holm.any()


def test_global_multiplicity_keeps_case_comparisons_separate(metrics):
    second = metrics.copy()
    second["case"] = "OTHER"
    result = build_tables(pd.concat([metrics, second]), expected_topologies=4)
    contrasts = result["regret5_posthoc_holm.csv"]
    assert len(result["friedman_omnibus.csv"]) == 2 and len(contrasts) == 8
    assert (contrasts.p_holm_joint_eight >= contrasts.p_holm_within_case_four).all()


def test_numerical_tie_rounding_does_not_round_raw_means(metrics):
    original = build_tables(metrics, expected_topologies=4)
    metrics.loc[0, "Regret@5"] += 4e-13
    changed = build_tables(metrics, expected_topologies=4)
    pd.testing.assert_frame_equal(original["CASE-regret5-ranks.csv"], changed["CASE-regret5-ranks.csv"])
    assert original["raw_metric_means.csv"].equals(changed["raw_metric_means.csv"]) is False


def test_all_tied_methods_are_uninformative(metrics):
    metrics.loc[:, list(METRICS)] = .25
    tables = build_tables(metrics, expected_topologies=4)
    assert tables["friedman_omnibus.csv"].iloc[0].p_value == 1
    assert tables["friedman_omnibus.csv"].iloc[0].statistic == 0
    assert set(tables["mean_ranks_all_metrics.csv"].mean_rank) == {3}
    assert set(tables["regret5_posthoc_holm.csv"].p_holm_within_case_four) == {1}


@pytest.mark.parametrize("mutation", ["duplicate", "missing_pair", "missing_topology", "nan", "infinite", "negative", "above_one", "unsafe_case", "missing_metric"])
def test_incomplete_or_invalid_evidence_is_rejected(metrics, mutation):
    if mutation == "duplicate":
        metrics = pd.concat([metrics, metrics.iloc[[0]]])
    elif mutation == "missing_pair":
        metrics = metrics.iloc[1:]
    elif mutation == "missing_topology":
        metrics = metrics.loc[metrics.topology_id != "topology-0"]
    elif mutation == "unsafe_case":
        metrics["case"] = "../escape"
    elif mutation == "missing_metric":
        metrics = metrics.drop(columns="Hit@5")
    else:
        metrics.loc[0, "Regret@5"] = dict(nan=np.nan, infinite=np.inf, negative=-.01, above_one=1.01)[mutation]
    with pytest.raises(ValueError):
        build_tables(metrics, expected_topologies=4)


def test_other_methods_are_preserved_in_source_and_omitted_from_display(metrics, tmp_path):
    extra = metrics.loc[metrics.method == "ranking"].copy()
    extra["method"] = "reg_normalized"
    source = save_summary(tmp_path / "summary", pd.concat([metrics, extra]))
    identities = {p: sha256(p) for p in source.iterdir()}
    report = write_report(source, tmp_path / "report", expected_topologies=4)
    assert report["methods"] == list(METHODS)
    assert report["workflow_version"] == __version__
    assert (tmp_path / "report/CASE-publication-table.tex").is_file()
    assert (tmp_path / "report/mean_rank_tables.pdf").is_file()
    assert identities == {p: sha256(p) for p in source.iterdir()}
    verify_manifest(tmp_path / "report/manifest.json")
    with pytest.raises(ValueError, match="new output"):
        write_report(source, tmp_path / "report", expected_topologies=4)


def test_checksum_failure_does_not_leave_a_report(metrics, tmp_path):
    source = save_summary(tmp_path / "summary", metrics)
    with (source / "topology_metrics.csv").open("a") as handle:
        handle.write("tampering\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        write_report(source, tmp_path / "report", expected_topologies=4)
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("name", ["../outside.csv", "/absolute.csv", "nested/../../outside.csv", "..\\outside.csv"])
def test_manifest_rejects_paths_outside_summary(tmp_path, name):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(dict(status="complete", artifacts={name: "0" * 64})))
    with pytest.raises(ValueError, match="Invalid artifact"):
        verify_manifest(path)


def test_cli_reports_version_and_creates_audited_tables(metrics, tmp_path):
    runner = CliRunner()
    version = runner.invoke(app, ["--version"])
    assert version.exit_code == 0 and version.stdout.strip() == f"amiga-exp {__version__}"
    source = save_summary(tmp_path / "summary", metrics)
    result = runner.invoke(app, ["report", "supervised", "--summary", str(source),
                                 "--output", str(tmp_path / "report"), "--expected-topologies", "4", "--no-figures"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "complete"
    assert not (tmp_path / "report/mean_rank_tables.png").exists()
    failed = runner.invoke(app, ["report", "supervised", "--summary", str(source),
                                 "--output", str(tmp_path / "report")])
    assert failed.exit_code == 1 and "Report error" in failed.output
