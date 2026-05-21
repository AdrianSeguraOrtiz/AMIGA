from __future__ import annotations

import pandas as pd
import pytest

import scripts.experiments.amiga_exp.decision_baselines as decision_baselines_module
from scripts.experiments.amiga_exp.decision_baselines import (
    DecisionBaselineAdapterError,
    ObjectiveMatrixError,
    augmented_tchebycheff_scores_from_badness,
    ideal_l2_scores_from_badness,
    normalized_objective_badness_matrix,
    topsis_scores_from_badness,
    vikor_scores_from_badness,
    weighted_sum_scores_from_badness,
)


def _requires_pymcdm() -> None:
    pytest.importorskip("pymcdm", reason="package-backed decision baseline tests require pymcdm")


def test_normalized_objective_badness_matrix_handles_minimize_and_maximize():
    df = pd.DataFrame(
        {
            "front_id": [1, 1, 1],
            "loss": [10.0, 20.0, 30.0],
            "gain": [10.0, 20.0, 30.0],
        },
        index=[10, 20, 30],
    )

    badness = normalized_objective_badness_matrix(
        df,
        objective_columns=["loss", "gain"],
        objective_directions={"loss": "minimize", "gain": "maximize"},
        front_col="front_id",
    )

    assert list(badness.index) == [10, 20, 30]
    assert badness["loss"].tolist() == [0.0, 0.5, 1.0]
    assert badness["gain"].tolist() == [1.0, 0.5, 0.0]


def test_normalized_objective_badness_matrix_normalizes_each_front_independently():
    df = pd.DataFrame(
        {
            "front_id": [1, 1, 2, 2],
            "loss": [1.0, 3.0, 100.0, 200.0],
        }
    )

    badness = normalized_objective_badness_matrix(
        df,
        objective_columns=["loss"],
        objective_directions={"loss": "minimize"},
        front_col="front_id",
    )

    assert badness["loss"].tolist() == [0.0, 1.0, 0.0, 1.0]


def test_normalized_objective_badness_matrix_maps_constant_fronts_to_zero():
    df = pd.DataFrame(
        {
            "front_id": [1, 1, 1, 2, 2],
            "constant": [5.0, 5.0, 5.0, 7.0, 9.0],
        }
    )

    badness = normalized_objective_badness_matrix(
        df,
        objective_columns=["constant"],
        objective_directions={"constant": "minimize"},
        front_col="front_id",
    )

    assert badness["constant"].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_normalized_objective_badness_matrix_rejects_invalid_inputs():
    df = pd.DataFrame(
        {
            "front_id": [1, 1],
            "loss": [1.0, 2.0],
        }
    )

    with pytest.raises(ObjectiveMatrixError, match="missing required column"):
        normalized_objective_badness_matrix(
            df,
            objective_columns=["loss", "missing"],
            objective_directions={"loss": "minimize", "missing": "minimize"},
            front_col="front_id",
        )

    with pytest.raises(ObjectiveMatrixError, match="missing directions"):
        normalized_objective_badness_matrix(
            df,
            objective_columns=["loss"],
            objective_directions={},
            front_col="front_id",
        )

    with pytest.raises(ObjectiveMatrixError, match="invalid objective directions"):
        normalized_objective_badness_matrix(
            df,
            objective_columns=["loss"],
            objective_directions={"loss": "smaller"},
            front_col="front_id",
        )


def test_normalized_objective_badness_matrix_rejects_non_numeric_objectives():
    df = pd.DataFrame(
        {
            "front_id": [1, 1],
            "loss": ["low", "high"],
        }
    )

    with pytest.raises(ObjectiveMatrixError, match="must be numeric"):
        normalized_objective_badness_matrix(
            df,
            objective_columns=["loss"],
            objective_directions={"loss": "minimize"},
            front_col="front_id",
        )


def test_weighted_sum_scores_from_badness_uses_package_adapter():
    _requires_pymcdm()
    badness = pd.DataFrame(
        {
            "obj_a": [0.0, 0.5, 1.0],
            "obj_b": [0.0, 0.5, 1.0],
        },
        index=["best", "middle", "worst"],
    )

    scores = weighted_sum_scores_from_badness(badness)

    assert scores.index.tolist() == ["best", "middle", "worst"]
    assert scores.tolist() == pytest.approx([1.0, 0.5, 0.0])


def test_ideal_l2_scores_from_badness_uses_weighted_distance_to_ideal():
    badness = pd.DataFrame(
        {
            "obj_a": [0.0, 0.6, 1.0],
            "obj_b": [0.0, 0.8, 1.0],
        },
        index=["best", "tradeoff", "worst"],
    )

    scores = ideal_l2_scores_from_badness(badness)

    assert scores.index.tolist() == ["best", "tradeoff", "worst"]
    assert scores.tolist() == pytest.approx([0.0, -(0.5 ** 0.5), -1.0])
    assert scores.loc["best"] > scores.loc["tradeoff"] > scores.loc["worst"]


def test_ideal_l2_scores_from_badness_accepts_custom_weights():
    badness = pd.DataFrame(
        {
            "obj_a": [1.0],
            "obj_b": [0.0],
        }
    )

    scores = ideal_l2_scores_from_badness(badness, weights={"obj_a": 0.25, "obj_b": 0.75})

    assert scores.iloc[0] == pytest.approx(-0.5)


def test_augmented_tchebycheff_scores_from_badness_breaks_simple_tchebycheff_ties():
    badness = pd.DataFrame(
        {
            "obj_a": [0.6, 0.6],
            "obj_b": [0.0, 0.6],
            "obj_c": [0.0, 0.0],
        },
        index=["sparse_compromise", "dense_compromise"],
    )

    simple_scores = -badness.max(axis=1)
    augmented_scores = augmented_tchebycheff_scores_from_badness(badness, rho=0.1)

    assert simple_scores.tolist() == pytest.approx([-0.6, -0.6])
    assert augmented_scores.loc["sparse_compromise"] > augmented_scores.loc["dense_compromise"]
    assert augmented_scores.tolist() == pytest.approx([-0.22, -0.24])


def test_augmented_tchebycheff_scores_from_badness_rejects_invalid_rho():
    with pytest.raises(DecisionBaselineAdapterError, match="rho must be"):
        augmented_tchebycheff_scores_from_badness(pd.DataFrame({"obj": [0.0, 1.0]}), rho=-1.0)


def test_topsis_scores_from_badness_uses_package_adapter():
    _requires_pymcdm()
    badness = pd.DataFrame(
        {
            "obj_a": [0.0, 0.5, 1.0],
            "obj_b": [0.0, 0.5, 1.0],
        }
    )

    scores = topsis_scores_from_badness(badness)

    assert scores.tolist() == pytest.approx([1.0, 0.5, 0.0])


def test_topsis_scores_from_badness_matches_documented_formula():
    _requires_pymcdm()
    badness = pd.DataFrame(
        {
            "obj_a": [0.2, 0.8],
            "obj_b": [0.4, 0.1],
        }
    )
    weights = {"obj_a": 0.75, "obj_b": 0.25}

    scores = topsis_scores_from_badness(badness, weights=weights)

    expected = []
    for _, row in badness.iterrows():
        d_plus = ((0.75 * row["obj_a"] ** 2) + (0.25 * row["obj_b"] ** 2)) ** 0.5
        d_minus = ((0.75 * (1 - row["obj_a"]) ** 2) + (0.25 * (1 - row["obj_b"]) ** 2)) ** 0.5
        expected.append(d_minus / (d_plus + d_minus))
    assert scores.tolist() == pytest.approx(expected)


def test_topsis_scores_from_badness_handles_all_ideal_tie_matrix():
    _requires_pymcdm()
    badness = pd.DataFrame(
        {
            "obj_a": [0.0, 0.0, 0.0],
            "obj_b": [0.0, 0.0, 0.0],
        }
    )

    scores = topsis_scores_from_badness(badness)

    assert scores.tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_vikor_scores_from_badness_uses_package_adapter_and_drops_constant_columns():
    _requires_pymcdm()
    badness = pd.DataFrame(
        {
            "constant": [0.0, 0.0, 0.0],
            "obj_a": [0.0, 0.5, 1.0],
        }
    )

    scores = vikor_scores_from_badness(badness, v=0.5)

    assert scores.tolist() == pytest.approx([-0.0, -0.5, -1.0])


def test_vikor_scores_from_badness_handles_all_tie_matrix():
    badness = pd.DataFrame(
        {
            "obj_a": [0.0, 0.0],
            "obj_b": [0.0, 0.0],
        }
    )

    scores = vikor_scores_from_badness(badness)

    assert scores.tolist() == pytest.approx([0.0, 0.0])


def test_vikor_scores_from_badness_handles_equal_s_values():
    badness = pd.DataFrame(
        {
            "obj_a": [1.0, 0.0, 0.5],
            "obj_b": [0.0, 1.0, 0.5],
        }
    )

    scores = vikor_scores_from_badness(badness, v=0.5)

    assert scores.tolist() == pytest.approx([-0.5, -0.5, -0.0])


def test_vikor_scores_from_badness_handles_equal_r_values():
    badness = pd.DataFrame(
        {
            "obj_a": [1.0, 0.0, 1.0],
            "obj_b": [0.0, 1.0, 1.0],
        }
    )

    scores = vikor_scores_from_badness(badness, v=0.5)

    assert scores.tolist() == pytest.approx([-0.0, -0.0, -0.5])


def test_vikor_scores_from_badness_rejects_invalid_v():
    with pytest.raises(DecisionBaselineAdapterError, match="v must be in"):
        vikor_scores_from_badness(pd.DataFrame({"obj": [0.0, 1.0]}), v=1.5)


def test_package_adapter_reports_missing_pymcdm(monkeypatch):
    def fail_import(name: str):
        if name == "pymcdm.methods":
            raise ImportError("missing")
        raise AssertionError(name)

    monkeypatch.setattr(decision_baselines_module, "import_module", fail_import)

    with pytest.raises(DecisionBaselineAdapterError, match="requires optional dependency 'pymcdm==1.4.0'"):
        weighted_sum_scores_from_badness(pd.DataFrame({"obj": [0.0, 1.0]}))


def test_package_adapter_rejects_invalid_badness_or_weights():
    with pytest.raises(DecisionBaselineAdapterError, match="values must be in"):
        weighted_sum_scores_from_badness(pd.DataFrame({"obj": [-0.1, 1.0]}))

    with pytest.raises(DecisionBaselineAdapterError, match="missing objective"):
        weighted_sum_scores_from_badness(
            pd.DataFrame({"obj": [0.0, 1.0]}),
            weights={"other": 1.0},
        )

    with pytest.raises(DecisionBaselineAdapterError, match="non-negative"):
        weighted_sum_scores_from_badness(
            pd.DataFrame({"obj": [0.0, 1.0]}),
            weights={"obj": -1.0},
        )
