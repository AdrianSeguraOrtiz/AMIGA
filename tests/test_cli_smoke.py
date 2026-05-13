from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

from amiga.cli import app
from amiga.utils import save_json, save_pickle


runner = CliRunner()


SMALL_LGBM_PARAMS = {
    "n_estimators": 8,
    "num_leaves": 7,
    "learning_rate": 0.1,
    "min_child_samples": 1,
    "verbose": -1,
}


class SumModel:
    def predict(self, X: pd.DataFrame):
        return X.sum(axis=1).to_numpy(dtype=float)


def _write_training_csv(path):
    rows = []
    for front_id in range(4):
        for item_id, quality in enumerate((0.1, 0.5, 0.9), start=1):
            rows.append(
                {
                    "front_id": front_id,
                    "item_id": item_id,
                    "AUPR": quality + front_id * 0.01,
                    "f_signal": quality,
                    "f_inverse": 1.0 - quality,
                    "ignored": front_id,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_cli_summarize_and_plot_smoke(sample_cv_reports, tmp_path):
    out_dir = tmp_path / "summary"
    summarize_result = runner.invoke(
        app,
        [
            "summarize-cv",
            str(sample_cv_reports["ModelA"]),
            str(sample_cv_reports["ModelB"]),
            "--out",
            str(out_dir),
            "--stats",
            "metric_rank_stats",
        ],
    )
    assert summarize_result.exit_code == 0, summarize_result.stdout

    for args in (
        ["plot-cv", "--input-dir", str(out_dir), "--plot", "dotplot_overview"],
        ["plot-cv", "--input-dir", str(out_dir), "--plot", "topk_curves", "--metric-prefix", "Regret"],
        ["plot-cv", "--input-dir", str(out_dir), "--plot", "metric_rank_heatmap", "--metrics", "Regret@1", "--metrics", "Hit@1"],
        ["plot-cv", "--input-dir", str(out_dir), "--plot", "metric_scatter", "--x-metric", "Regret@1", "--y-metric", "Hit@1"],
    ):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.stdout


def test_cli_build_data_smoke(tmp_path):
    front_csv = tmp_path / "front.csv"
    expr_csv = tmp_path / "expr.csv"
    grn_dir = tmp_path / "grns"
    out_csv = tmp_path / "data.csv"
    grn_dir.mkdir()

    pd.DataFrame(
        {
            "AUPR": [0.8],
            "obj": [1.0],
            "GRN_A.csv": [0.7],
            "GRN_B.csv": [0.3],
        }
    ).to_csv(front_csv, index=False)
    pd.DataFrame(
        {
            "gene": ["g1", "g2", "g3"],
            "c1": [1.0, 2.0, 3.0],
            "c2": [2.0, 1.0, 4.0],
            "c3": [3.0, 2.0, 5.0],
        }
    ).to_csv(expr_csv, index=False)
    (grn_dir / "GRN_A.csv").write_text("A,B,1.0\nB,C,0.5\n", encoding="utf-8")
    (grn_dir / "GRN_B.csv").write_text("A,B,0.2\nC,A,0.8\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "build-data",
            str(front_csv),
            str(expr_csv),
            str(grn_dir),
            "--front-id",
            "9",
            "--out",
            str(out_csv),
        ],
    )
    assert result.exit_code == 0, result.stdout
    built = pd.read_csv(out_csv)
    assert not built.empty
    assert built.loc[0, "front_id"] == 9


def test_cli_extract_feature_commands_smoke(tmp_path):
    expr_csv = tmp_path / "expr.csv"
    grn_csv = tmp_path / "grn.csv"
    expr_out = tmp_path / "expr.json"
    grn_out = tmp_path / "grn.json"

    pd.DataFrame(
        {
            "gene": ["g1", "g2", "g3"],
            "c1": [1.0, 2.0, 3.0],
            "c2": [2.0, 1.0, 4.0],
            "c3": [3.0, 2.0, 5.0],
        }
    ).to_csv(expr_csv, index=False)
    grn_csv.write_text("A,B,1.0\nB,C,0.5\nC,A,0.3\n", encoding="utf-8")

    expr_result = runner.invoke(
        app,
        ["extract-expr-features", str(expr_csv), "--out", str(expr_out), "--include-timeseries"],
    )
    assert expr_result.exit_code == 0, expr_result.stdout
    assert expr_out.exists()

    grn_result = runner.invoke(
        app,
        ["extract-grn-features", str(grn_csv), "--out", str(grn_out), "--include-advanced"],
    )
    assert grn_result.exit_code == 0, grn_result.stdout
    assert grn_out.exists()


def test_cli_train_and_rank_commands_smoke(tmp_path):
    train_csv = tmp_path / "train.csv"
    params_json = tmp_path / "params.json"
    full_out = tmp_path / "full"
    cv_out = tmp_path / "cv"
    ranked_csv = tmp_path / "ranked.csv"

    _write_training_csv(train_csv)
    save_json(SMALL_LGBM_PARAMS, params_json)

    full_result = runner.invoke(
        app,
        [
            "train-full",
            str(train_csv),
            "--model",
            "LGBMRanker",
            "--drop-cols",
            "ignored",
            "--model-params-json",
            str(params_json),
            "--out-dir",
            str(full_out),
        ],
    )
    assert full_result.exit_code == 0, full_result.stdout
    assert (full_out / "model.pkl").exists()
    assert (full_out / "feature_columns.json").exists()

    rank_result = runner.invoke(
        app,
        [
            "rank-csv",
            str(train_csv),
            str(full_out / "model.pkl"),
            "--drop-cols",
            "ignored",
            "--out-csv",
            str(ranked_csv),
        ],
    )
    assert rank_result.exit_code == 0, rank_result.stdout
    ranked = pd.read_csv(ranked_csv)
    assert {"score", "rank_in_front"}.issubset(ranked.columns)

    cv_result = runner.invoke(
        app,
        [
            "train-cv",
            str(train_csv),
            "--model",
            "LGBMRanker",
            "--drop-cols",
            "ignored",
            "--n-splits",
            "2",
            "--model-params-json",
            str(params_json),
            "--out-dir",
            str(cv_out),
        ],
    )
    assert cv_result.exit_code == 0, cv_result.stdout
    assert (cv_out / "cv_report.json").exists()
    assert (cv_out / "valid_fold1_ranked.csv").exists()


def test_cli_rank_csv_uses_feature_metadata(tmp_path):
    model_path = tmp_path / "model.pkl"
    meta_path = tmp_path / "feature_columns.json"
    input_csv = tmp_path / "input.csv"
    out_csv = tmp_path / "ranked.csv"

    save_pickle(SumModel(), model_path)
    save_json({"feature_columns": ["f2", "f1"]}, meta_path)
    pd.DataFrame(
        {
            "front_id": [1, 1],
            "item_id": [1, 2],
            "f1": [0.1, 0.2],
            "f2": [0.0, 1.0],
        }
    ).to_csv(input_csv, index=False)

    result = runner.invoke(
        app,
        [
            "rank-csv",
            str(input_csv),
            str(model_path),
            "--out-csv",
            str(out_csv),
        ],
    )
    assert result.exit_code == 0, result.stdout
    ranked = pd.read_csv(out_csv)
    assert ranked.loc[0, "item_id"] == 2
