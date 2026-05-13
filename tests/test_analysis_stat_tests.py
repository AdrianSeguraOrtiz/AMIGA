from __future__ import annotations

import numpy as np
import pandas as pd

from amiga.analysis.stat_tests import (
    build_front_metric_matrix_from_frames,
    compute_global_metric_stats,
    friedman_with_holm,
    rank_matrix,
)


def test_rank_matrix_respects_direction():
    values = pd.DataFrame(
        {
            "A": [0.1, 0.9],
            "B": [0.2, 0.3],
            "C": [0.3, 0.2],
        },
        index=[101, 102],
    )
    lower = rank_matrix(values, lower_is_better=True)
    higher = rank_matrix(values, lower_is_better=False)

    assert lower.loc[101, "A"] == 1.0
    assert higher.loc[101, "C"] == 1.0


def test_friedman_with_holm_returns_winner_stats():
    ranks = pd.DataFrame(
        {
            "ModelA": [1.0, 1.0, 1.0, 1.0],
            "ModelB": [2.0, 2.0, 2.0, 2.0],
            "ModelC": [3.0, 3.0, 3.0, 3.0],
        },
        index=[1, 2, 3, 4],
    )
    stats_df, summary = friedman_with_holm(ranks)

    assert summary["n_fronts"] == 4
    assert stats_df.iloc[0]["model"] == "ModelA"
    assert bool(stats_df.iloc[0]["is_winner"]) is True
    assert np.isfinite(summary["friedman_p"])


def test_compute_global_metric_stats_from_per_front_frames():
    metrics_by_model = {
        "ModelA": pd.DataFrame(
            {
                "front_id": [1, 2, 3],
                "Regret@1": [0.0, 0.1, 0.0],
                "Hit@1": [1.0, 0.0, 1.0],
            }
        ),
        "ModelB": pd.DataFrame(
            {
                "front_id": [1, 2, 3],
                "Regret@1": [0.3, 0.2, 0.1],
                "Hit@1": [0.0, 0.0, 0.0],
            }
        ),
    }
    stats_df = compute_global_metric_stats(metrics_by_model, ["Regret@1", "Hit@1"])

    assert not stats_df.empty
    assert set(stats_df["metric"]) == {"Regret@1", "Hit@1"}
    winner = stats_df[stats_df["metric"] == "Regret@1"].sort_values("avg_rank").iloc[0]
    assert winner["model"] == "ModelA"


def test_build_front_metric_matrix_from_frames_aligns_common_fronts_only():
    metrics_by_model = {
        "ModelA": pd.DataFrame({"front_id": [1, 2], "Hit@1": [1.0, 0.0]}),
        "ModelB": pd.DataFrame({"front_id": [2, 3], "Hit@1": [0.5, 1.0]}),
    }

    matrix = build_front_metric_matrix_from_frames(metrics_by_model, "Hit@1")

    assert matrix.index.tolist() == [2]
    assert matrix.loc[2].to_dict() == {"ModelA": 0.0, "ModelB": 0.5}
    assert build_front_metric_matrix_from_frames(metrics_by_model, "Missing").empty
