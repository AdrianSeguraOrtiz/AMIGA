from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from amiga.selection.learn2rank import (
    LabelMode,
    ModelType,
    assign_rank_in_front,
    build_labels,
    compute_ranking_metrics,
    compute_ranking_metrics_by_front,
    fit_ranker,
)


@pytest.fixture
def ranking_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "front_id": [1, 1, 1, 2, 2, 2],
            "AUPR": [0.2, 0.5, 0.5, 0.1, 0.4, 0.3],
            "score": [0.1, 0.9, 0.8, 0.2, 0.7, 0.6],
        }
    )


@pytest.mark.parametrize(
    ("mode", "expected_dtype"),
    [
        (LabelMode.RANK_DENSE, np.integer),
        (LabelMode.RANK_AVG, np.integer),
        (LabelMode.QUANTILES, np.integer),
        (LabelMode.CONTINUOUS, np.floating),
        (LabelMode.REVERSED, np.floating),
        (LabelMode.SHUFFLED, np.floating),
    ],
)
def test_build_labels_all_modes(ranking_df, mode, expected_dtype):
    labels = build_labels(ranking_df, front_col="front_id", target_col="AUPR", mode=mode, random_state=7)
    assert labels.shape == (len(ranking_df),)
    assert np.issubdtype(labels.dtype, expected_dtype)
    if mode in {LabelMode.CONTINUOUS, LabelMode.REVERSED, LabelMode.SHUFFLED}:
        assert np.all(labels >= 0.0)
        assert np.all(labels <= 1.0)


def test_compute_ranking_metrics_handles_small_fronts_and_ties(ranking_df):
    per_front = compute_ranking_metrics_by_front(
        ranking_df,
        front_col="front_id",
        target_col="AUPR",
        score_col="score",
        ks=(1, 3, 5),
    )
    agg, groups = compute_ranking_metrics(
        ranking_df,
        front_col="front_id",
        target_col="AUPR",
        score_col="score",
        ks=(1, 3, 5),
    )

    assert len(per_front) == 2
    assert len(groups) == 2
    assert "Regret@1" in agg
    assert "NDCG@1" in agg
    assert "Hit@5" not in per_front.columns


def test_topk_metrics_match_decision_first_definitions():
    df = pd.DataFrame(
        {
            "front_id": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            "AUPR": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            "score": [0.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0],
        }
    )

    per_front = compute_ranking_metrics_by_front(
        df,
        front_col="front_id",
        target_col="AUPR",
        score_col="score",
        ks=(1, 5),
    ).set_index("front_id")
    agg, _ = compute_ranking_metrics(
        df,
        front_col="front_id",
        target_col="AUPR",
        score_col="score",
        ks=(1, 5),
    )

    assert per_front.loc[1, "BestAUPR@5"] == pytest.approx(0.9)
    assert per_front.loc[1, "Regret@5"] == pytest.approx(0.1)
    assert per_front.loc[1, "Hit@5"] == 0.0
    assert per_front.loc[2, "BestAUPR@5"] == pytest.approx(0.5)
    assert per_front.loc[2, "Regret@5"] == pytest.approx(0.0)
    assert per_front.loc[2, "Hit@5"] == 1.0

    assert agg["BestAUPR@5"] == pytest.approx(0.7)
    assert agg["Regret@5"] == pytest.approx(0.05)
    assert agg["Hit@5"] == pytest.approx(0.5)


def test_ranking_metric_aggregation_is_front_balanced_not_row_weighted():
    df = pd.DataFrame(
        {
            "front_id": [1] * 10 + [2, 2],
            "AUPR": [1.0] + [0.0] * 9 + [1.0, 0.0],
            "score": [0.0] + list(range(9, 0, -1)) + [1.0, 0.0],
        }
    )

    agg, groups = compute_ranking_metrics(
        df,
        front_col="front_id",
        target_col="AUPR",
        score_col="score",
        ks=(1,),
    )

    assert len(groups) == 2
    assert agg["Regret@1"] == pytest.approx(0.5)
    assert agg["Hit@1"] == pytest.approx(0.5)


def test_predicted_score_ties_are_evaluated_fairly_not_by_input_order():
    df = pd.DataFrame(
        {
            "front_id": [1, 1, 1],
            "AUPR": [0.2, 1.0, 0.8],
            "score": [0.5, 0.5, 0.1],
        }
    )

    per_front = compute_ranking_metrics_by_front(
        df,
        front_col="front_id",
        target_col="AUPR",
        score_col="score",
        ks=(1, 2),
    ).iloc[0]

    assert per_front["BestAUPR@1"] == pytest.approx(0.6)
    assert per_front["Regret@1"] == pytest.approx(0.4)
    assert per_front["Hit@1"] == pytest.approx(0.5)
    assert per_front["BestAUPR@2"] == pytest.approx(1.0)
    assert per_front["Regret@2"] == pytest.approx(0.0)
    assert per_front["Hit@2"] == 1.0


def test_constant_scores_are_invariant_to_input_order_for_topk_metrics():
    df = pd.DataFrame(
        {
            "front_id": [1, 1, 1],
            "AUPR": [1.0, 0.8, 0.2],
            "score": [0.0, 0.0, 0.0],
        }
    )
    shuffled = df.iloc[[2, 0, 1]].reset_index(drop=True)

    original = compute_ranking_metrics_by_front(
        df,
        front_col="front_id",
        target_col="AUPR",
        score_col="score",
        ks=(1, 2),
    ).iloc[0]
    reordered = compute_ranking_metrics_by_front(
        shuffled,
        front_col="front_id",
        target_col="AUPR",
        score_col="score",
        ks=(1, 2),
    ).iloc[0]

    assert original["BestAUPR@1"] == pytest.approx(2.0 / 3.0)
    assert original["Hit@1"] == pytest.approx(1.0 / 3.0)
    assert original["BestAUPR@2"] == pytest.approx((1.0 + 1.0 + 0.8) / 3.0)
    assert original["Hit@2"] == pytest.approx(2.0 / 3.0)
    assert original[["BestAUPR@1", "Regret@1", "Hit@1", "BestAUPR@2", "Regret@2", "Hit@2"]].to_dict() == pytest.approx(
        reordered[["BestAUPR@1", "Regret@1", "Hit@1", "BestAUPR@2", "Regret@2", "Hit@2"]].to_dict()
    )


def test_assign_rank_in_front_breaks_score_ties_without_row_order():
    df = pd.DataFrame(
        {
            "front_id": [1, 1, 1, 2, 2],
            "item_id": [3, 1, 2, 2, 1],
            "score": [0.5, 0.5, 0.5, 0.9, 0.9],
        }
    )
    shuffled = df.iloc[[2, 0, 4, 1, 3]].reset_index(drop=True)

    ranked = assign_rank_in_front(df, front_col="front_id", id_col="item_id", tie_seed=11)
    ranked_shuffled = assign_rank_in_front(
        shuffled,
        front_col="front_id",
        id_col="item_id",
        tie_seed=11,
    )

    assert ranked.groupby("front_id")["rank_in_front"].apply(list).to_dict() == {1: [1, 2, 3], 2: [1, 2]}
    assert ranked.groupby("front_id")["item_id"].apply(list).to_dict() == (
        ranked_shuffled.groupby("front_id")["item_id"].apply(list).to_dict()
    )


def _tiny_ranker_data():
    X_train = pd.DataFrame(
        {
            "f1": [0.1, 0.2, 0.3, 0.9, 1.0, 1.1, 0.4, 0.5, 0.6, 1.2, 1.3, 1.4],
            "f2": [1.0, 0.8, 0.6, 0.2, 0.1, 0.0, 0.9, 0.7, 0.5, 0.4, 0.3, 0.2],
        }
    )
    y_train = np.array([0.1, 0.5, 0.9, 0.2, 0.4, 0.8, 0.0, 0.3, 0.7, 0.15, 0.45, 0.95], dtype=float)
    gid_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=int)

    X_valid = pd.DataFrame({"f1": [0.15, 0.25, 0.95, 1.05], "f2": [0.95, 0.75, 0.15, 0.05]})
    y_valid = np.array([0.2, 0.8, 0.3, 0.9], dtype=float)
    gid_valid = np.array([10, 10, 11, 11], dtype=int)
    return X_train, y_train, gid_train, X_valid, y_valid, gid_valid


@pytest.mark.parametrize(
    ("model_type", "params"),
    [
        (ModelType.LGBMRanker, {"n_estimators": 20, "num_leaves": 7, "learning_rate": 0.1, "min_child_samples": 1}),
        (ModelType.XGBRanker, {"n_estimators": 20, "max_depth": 2, "learning_rate": 0.1}),
        (ModelType.CatBoostRanker, {"iterations": 20, "depth": 2, "learning_rate": 0.1, "verbose": False}),
    ],
)
def test_fit_ranker_with_and_without_validation(model_type, params, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    X_train, y_train, gid_train, X_valid, y_valid, gid_valid = _tiny_ranker_data()

    model, scores_valid = fit_ranker(
        model_type,
        X_train,
        y_train,
        gid_train,
        random_state=42,
        model_params=params,
        X_valid=X_valid,
        y_valid=y_valid,
        gid_valid=gid_valid,
    )
    assert model is not None
    assert scores_valid is not None
    assert scores_valid.shape == (len(X_valid),)

    model_no_valid, scores_none = fit_ranker(
        model_type,
        X_train,
        y_train,
        gid_train,
        random_state=42,
        model_params=params,
    )
    assert model_no_valid is not None
    assert scores_none is None
    assert not (tmp_path / "catboost_info").exists()
