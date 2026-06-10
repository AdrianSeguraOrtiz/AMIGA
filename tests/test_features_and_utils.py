from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from amiga.core.main import build_data
from amiga.features.expression import (
    features_condition_stats,
    features_correlations,
    features_exp_global,
    features_gene_stats,
    features_pca,
    features_timeseries,
)
from amiga.features.grn import (
    build_digraph,
    features_advanced,
    features_assortativity,
    features_clustering,
    features_communities,
    features_grn_global,
    features_paths,
    features_reciprocity,
    features_strength,
)
from amiga.utils import (
    clean,
    load_expression_matrix,
    load_json,
    load_pickle,
    row_weights_from_front,
    save_json,
    save_pickle,
    weighted_confidence,
)


def test_expression_feature_functions_handle_small_inputs(tiny_expression_df):
    expr_with_nan = tiny_expression_df.copy()
    expr_with_nan.iloc[0, 0] = np.nan

    assert features_exp_global(expr_with_nan)["n_genes"] == 3
    assert "gene_cv_p90" in features_gene_stats(expr_with_nan)
    assert "cond_mean_max_abs_z" in features_condition_stats(expr_with_nan)
    assert "gene_gene_abs_corr_mean" in features_correlations(expr_with_nan, max_pairs=2)
    assert "pca_eff_rank" in features_pca(expr_with_nan, top_k=2)
    assert "ts_slope_mean" in features_timeseries(expr_with_nan)
    assert "ts_lag2_autocorr_mean" in features_timeseries(expr_with_nan, max_lag=2)
    with pytest.raises(ValueError, match="max_lag"):
        features_timeseries(expr_with_nan, max_lag=0)


def test_grn_feature_functions_handle_empty_and_small_graphs():
    empty_edges = pd.DataFrame(columns=["Source", "Target", "Confidence"])
    empty_graph = build_digraph(empty_edges)
    assert np.isnan(features_paths(empty_graph)["w_avg_spl"])
    assert np.isnan(features_clustering(empty_graph)["avg_clustering_w_undirected"])

    edges = pd.DataFrame(
        {
            "Source": ["A", "B", "B", "C"],
            "Target": ["B", "A", "C", "A"],
            "Confidence": [1.0, 0.5, 0.2, 0.4],
        }
    )
    graph = build_digraph(edges)
    weights = np.array([1.0, 0.5, 0.2, 0.4], dtype=float)

    assert features_grn_global(graph, weights, top_frac=0.5)["num_edges"] == 4
    assert "strength_hubs_n" in features_strength(graph)
    assert "assortativity_out_in_w" in features_assortativity(graph)
    assert "reciprocity_weighted" in features_reciprocity(graph)
    assert "comm_n" in features_communities(graph)
    assert "entropy_weights" in features_advanced(graph, weights)


def test_utils_and_build_data_smoke(tmp_path, tiny_expression_df):
    expr_csv = tmp_path / "expr.csv"
    pd.DataFrame(
        {
            "gene": ["g1", "g2", "g3"],
            "c1": [1.0, 2.0, 3.0],
            "c2": [2.0, 1.0, 4.0],
            "c3": [3.0, 2.0, 5.0],
        }
    ).to_csv(expr_csv, index=False)
    loaded_expr = load_expression_matrix(expr_csv)
    assert loaded_expr.shape == (3, 3)

    grn_dir = tmp_path / "grn"
    grn_dir.mkdir()
    (grn_dir / "GRN_A.csv").write_text("A,B,1.0\nB,C,0.4\n", encoding="utf-8")
    (grn_dir / "GRN_B.csv").write_text("A,B,0.5\nC,A,0.9\n", encoding="utf-8")

    front = pd.DataFrame(
        {
            "AUPR": [0.8],
            "objective_x": [1.2],
            "GRN_A.csv": [0.6],
            "GRN_B.csv": [0.4],
        }
    )
    weights = row_weights_from_front(front.iloc[0])
    assert weights == {"GRN_A.csv": 0.6, "GRN_B.csv": 0.4}

    merged = weighted_confidence([f"0.6*{grn_dir / 'GRN_A.csv'}", f"0.4*{grn_dir / 'GRN_B.csv'}"])
    assert list(merged.columns) == ["Source", "Target", "Confidence"]
    assert not merged.empty

    built = build_data(front, loaded_expr, grn_dir, front_id=7, threads=1)
    assert built.loc[0, "front_id"] == 7
    assert built.loc[0, "item_id"] == 1
    assert any(col.startswith("expr_") for col in built.columns)
    assert any(col.startswith("grn_") for col in built.columns)


def test_build_data_supports_unlabelled_fronts(tmp_path, tiny_expression_df):
    grn_dir = tmp_path / "grn"
    grn_dir.mkdir()
    (grn_dir / "GRN_A.csv").write_text("A,B,1.0\nB,C,0.4\n", encoding="utf-8")
    (grn_dir / "GRN_B.csv").write_text("A,B,0.5\nC,A,0.9\n", encoding="utf-8")

    front = pd.DataFrame(
        {
            "objective_x": [1.2],
            "GRN_A.csv": [0.6],
            "GRN_B.csv": [0.4],
        }
    )

    built = build_data(front, tiny_expression_df, grn_dir, front_id=7, threads=1, require_target=False)
    assert built.loc[0, "front_id"] == 7
    assert built.loc[0, "item_id"] == 1
    assert "AUPR" not in built.columns
    assert any(col.startswith("expr_") for col in built.columns)
    assert any(col.startswith("grn_") for col in built.columns)


def test_weighted_confidence_validates_inputs():
    with pytest.raises(ValueError, match="Use '<weight>\\*<file_path>'"):
        weighted_confidence(["bad_spec"])


def test_utils_serialization_cleaning_and_validation_branches(tmp_path):
    payload = {"alpha": 1, "beta": [1, 2, 3]}
    json_path = tmp_path / "payload.json"
    pickle_path = tmp_path / "payload.pkl"

    save_json(payload, json_path)
    save_pickle(payload, pickle_path)

    assert load_json(json_path) == payload
    assert load_pickle(pickle_path) == payload

    cleaned = clean(
        {
            "float": np.float64(1.5),
            "int": np.int64(2),
            "bool": np.bool_(True),
            "array": np.array([1, 2]),
            "nan": np.float64(np.nan),
            "inf": float("inf"),
        }
    )
    assert cleaned == {
        "float": 1.5,
        "int": 2,
        "bool": True,
        "array": [1, 2],
        "nan": None,
        "inf": None,
    }

    with pytest.raises(ValueError, match="does not contain"):
        row_weights_from_front(pd.Series({"AUPR": 0.8}))

    with pytest.raises(ValueError, match="zero or NaN"):
        row_weights_from_front(pd.Series({"GRN_A.csv": 0.0, "GRN_B.csv": np.nan}))

    with pytest.warns(UserWarning, match="normalized"):
        weights = row_weights_from_front(pd.Series({"GRN_A.csv": 2.0, "GRN_B.csv": 1.0}))
    assert weights == {"GRN_A.csv": pytest.approx(2 / 3), "GRN_B.csv": pytest.approx(1 / 3)}

    with pytest.raises(ValueError, match="Invalid weight"):
        weighted_confidence(["not_a_number*missing.csv"])

    with pytest.raises(ValueError, match="File not found"):
        weighted_confidence(["1.0*missing.csv"])

    edge_file = tmp_path / "edges.csv"
    out_file = tmp_path / "merged.csv"
    edge_file.write_text("A,B,1.0\nA,B,0.5\n", encoding="utf-8")
    merged = weighted_confidence([f"1.0*{edge_file}"], output_file=out_file)
    assert out_file.exists()
    assert merged.loc[0, "Confidence"] == pytest.approx(0.75)
