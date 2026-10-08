"""Training-data contract checks; no estimators are fitted or evaluated."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.experiments.amiga_exp.grouped_validation.training_data import (
    prepare_frame,
    prepare_training,
)


def example_frame() -> pd.DataFrame:
    quality = {
        1: [0.1, 0.2, 0.2, 0.2, 0.3],
        2: [0.1] * 7 + [0.9] * 3,
        3: [0.0] * 9 + [1.0],
        4: [0.7] * 4,
    }
    return pd.DataFrame([
        {"front_id": f, "item_id": i, "AUPR": q, "feature": float(f + i)}
        for f, values in quality.items() for i, q in enumerate(values, start=1)
    ])


TOPOLOGY = {1: "A", 2: "A", 3: "B", 4: "A"}


def test_ties_flat_exclusion_and_topology_weights():
    result = prepare_frame(example_frame(), "BIO-INSIGHT", TOPOLOGY, ["feature"])
    groups = result["group_id"]
    assert result["report"]["flat_front_ids_excluded"] == [4]
    assert result["report"]["usable_fronts_by_topology"] == {"A": 2, "B": 1}
    assert set(groups) == {1, 2, 3}
    assert result["X"].dtype == np.float64
    assert np.issubdtype(groups.dtype, np.integer)
    assert np.all(groups[:-1] <= groups[1:])
    assert np.array_equal(result["labels"]["ltr_catboost"][groups == 1], [2, 3, 3, 3, 4])
    assert np.allclose(result["labels"]["reg_normalized"][groups == 1], [0, .5, .5, .5, 1])
    assert np.array_equal(result["labels"]["clf_top20"][groups == 2], [0] * 7 + [1] * 3)
    assert np.array_equal(result["labels"]["clf_top20"][groups == 3], [0] * 9 + [1])
    assert result["report"]["class_fractions"][2]["minimum_threshold_fallback"]
    assert np.array_equal(result["group_weight"][groups != 3], np.full(15, .5))
    assert np.array_equal(result["group_weight"][groups == 3], np.ones(10))
    assert result["point_weight"].mean() == pytest.approx(1)
    assert result["point_weight"][groups != 3].sum() == pytest.approx(result["point_weight"][groups == 3].sum())
    assert result["point_weight"][groups == 1].sum() == pytest.approx(result["point_weight"][groups == 2].sum())
    json.dumps(result["report"], allow_nan=False)


def test_rank_avg_and_order_invariance():
    frame = example_frame()
    expected = prepare_frame(frame, "MO-GENECI", TOPOLOGY, ["feature"])
    shuffled = prepare_frame(frame.sample(frac=1, random_state=4), "MO-GENECI", TOPOLOGY, ["feature"])
    assert np.array_equal(expected["labels"]["ltr_catboost"][expected["group_id"] == 1], [0, 2, 2, 2, 4])
    for key in ("X", "group_id", "item_id", "point_weight", "group_weight"):
        assert np.array_equal(expected[key], shuffled[key])
    for label in expected["labels"]:
        assert np.array_equal(expected["labels"][label], shuffled["labels"][label])


def test_only_supplied_training_fronts_affect_labels_and_weights():
    selected = example_frame().query("front_id in [1, 3]")
    result = prepare_frame(selected, "BIO-INSIGHT", TOPOLOGY, ["feature"])
    assert result["report"]["selected_front_ids"] == [1, 3]
    assert result["report"]["usable_fronts_by_topology"] == {"A": 1, "B": 1}
    assert np.array_equal(result["group_weight"], np.ones(len(selected)))
    assert result["report"]["flat_front_ids_excluded"] == []


def test_nan_predictor_preserved_but_inf_and_nonfinite_targets_rejected():
    frame = example_frame()
    frame.loc[0, "feature"] = np.nan
    result = prepare_frame(frame, "BIO-INSIGHT", TOPOLOGY, ["feature"])
    assert np.isnan(result["X"][0, 0])
    assert result["report"]["finite_checks"]["predictor_nan_count"] == 1
    frame.loc[0, "feature"] = np.inf
    with pytest.raises(ValueError, match="predictors contain infinity"):
        prepare_frame(frame, "BIO-INSIGHT", TOPOLOGY, ["feature"])
    frame.loc[0, "feature"] = 1
    frame.loc[0, "AUPR"] = np.nan
    with pytest.raises(ValueError, match="targets contain NaN"):
        prepare_frame(frame, "BIO-INSIGHT", TOPOLOGY, ["feature"])


def test_rejects_controls_duplicates_missing_topologies_and_all_flat():
    frame = example_frame()
    with pytest.raises(ValueError, match="cannot be predictors"):
        prepare_frame(frame, "BIO-INSIGHT", TOPOLOGY, ["AUPR"])
    with pytest.raises(ValueError, match="duplicate item_id"):
        prepare_frame(pd.concat([frame, frame.iloc[[0]]]), "BIO-INSIGHT", TOPOLOGY, ["feature"])
    with pytest.raises(ValueError, match="mapping is missing"):
        prepare_frame(frame, "BIO-INSIGHT", {1: "A"}, ["feature"])
    with pytest.raises(ValueError, match="no informative training fronts"):
        prepare_frame(frame.query("front_id == 4"), "BIO-INSIGHT", TOPOLOGY, ["feature"])


def test_csv_loader_filters_before_labels_validation_and_topology_weights(tmp_path):
    case = "BIO-INSIGHT"
    data_path = tmp_path / "experiments" / case / "data" / "data_104.csv"
    contract_path = tmp_path / "docs" / "experiments" / "contracts" / f"{case}_feature_columns.json"
    data_path.parent.mkdir(parents=True)
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text(json.dumps({"feature_sets": {"full": ["feature"]}}))
    frame = example_frame()
    # Excluded rows deliberately violate training requirements. They must not
    # enter finite-value validation, label generation, or topology weighting.
    excluded = frame["front_id"] == 2
    frame.loc[excluded, "AUPR"] = np.nan
    frame.loc[excluded, "feature"] = np.inf
    frame.to_csv(data_path, index=False)

    result = prepare_training(tmp_path, case, [1, 3], {1: "A", 3: "B"}, ["feature"])

    assert result["report"]["input_fronts"] == 2
    assert result["report"]["input_rows"] == 15
    assert result["report"]["selected_front_ids"] == [1, 3]
    assert result["report"]["flat_front_ids_excluded"] == []
    assert result["report"]["usable_fronts_by_topology"] == {"A": 1, "B": 1}
    assert np.isfinite(result["X"]).all()
    assert np.isfinite(result["labels"]["reg_aupr"]).all()
    assert np.allclose(result["labels"]["reg_normalized"][:5], [0, .5, .5, .5, 1])
    assert np.array_equal(result["group_weight"], np.ones(15))
    groups = result["group_id"]
    assert result["point_weight"][groups == 1].sum() == pytest.approx(
        result["point_weight"][groups == 3].sum()
    )


@pytest.mark.parametrize("case", ["BIO-INSIGHT", "MO-GENECI"])
def test_real_selected_training_partition_without_models(case):
    repo = Path(__file__).resolve().parents[2]
    dataset_path = repo / "experiments" / case / "data" / "data_104.csv"
    if not dataset_path.is_file():
        pytest.skip(f"optional local benchmark dataset is unavailable: {case}")
    contract_path = repo / "docs" / "experiments" / "contracts" / f"{case}_feature_columns.json"
    features = json.loads(contract_path.read_text())["feature_sets"]["full"]
    # Fixed IDs only: this smoke check neither selects nor evaluates models.
    selected = [1, 2, 3]
    result = prepare_training(repo, case, selected, {f: f"smoke-front-{f}" for f in selected}, features)
    assert result["report"]["selected_front_ids"] == selected
    assert set(result["group_id"]) <= set(selected)
    assert result["X"].shape[1] == (104 if case == "BIO-INSIGHT" else 101)
    assert result["report"]["evaluation_performed"] is False
    assert set(result["labels"]) == {"ltr_catboost", "reg_aupr", "reg_normalized", "clf_top20"}
    assert np.isfinite(result["labels"]["reg_normalized"]).all()
    assert result["point_weight"].mean() == pytest.approx(1)
    json.dumps(result["report"], allow_nan=False)
