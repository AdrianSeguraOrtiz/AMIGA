from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from amiga.core.main import rank_with_model, train_ltr_cv, train_ltr_full
from amiga.selection.learn2rank import LabelMode, ModelType


SMALL_LGBM_PARAMS = {
    "n_estimators": 8,
    "num_leaves": 7,
    "learning_rate": 0.1,
    "min_child_samples": 1,
    "verbose": -1,
}


class LinearDummyModel:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X.sum(axis=1).to_numpy(dtype=float)


class ConstantDummyModel:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=float)


def tiny_training_df() -> pd.DataFrame:
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
    return pd.DataFrame(rows)


def test_rank_with_model_aligns_hinted_features_and_validates_missing_columns():
    df = pd.DataFrame(
        {
            "front_id": [1, 1, 2, 2],
            "item_id": [1, 2, 1, 2],
            "f1": [0.1, 0.9, 0.4, 0.2],
            "f2": [0.0, 0.0, 0.0, 0.0],
        }
    )

    result = rank_with_model(
        df,
        model=LinearDummyModel(),
        feature_columns_hint=["f1", "f2"],
    )

    assert result.feature_columns_used == ["f1", "f2"]
    assert result.df_ranked.groupby("front_id").first()["item_id"].to_dict() == {1: 2, 2: 1}

    with pytest.raises(ValueError, match="Missing feature columns"):
        rank_with_model(df, model=LinearDummyModel(), feature_columns_hint=["missing"])

    with pytest.raises(ValueError, match="No feature columns"):
        rank_with_model(
            df[["front_id", "item_id"]],
            model=LinearDummyModel(),
            drop_cols=[],
        )


def test_rank_with_model_ties_do_not_depend_on_input_order():
    df = pd.DataFrame(
        {
            "front_id": [1, 1, 1, 2, 2],
            "item_id": [1, 2, 3, 1, 2],
            "f1": [0.3, 0.2, 0.1, 0.5, 0.4],
        }
    )
    shuffled = df.iloc[[2, 0, 4, 1, 3]].reset_index(drop=True)

    ranked = rank_with_model(df, model=ConstantDummyModel(), feature_columns_hint=["f1"]).df_ranked
    ranked_shuffled = rank_with_model(
        shuffled,
        model=ConstantDummyModel(),
        feature_columns_hint=["f1"],
    ).df_ranked

    assert ranked.groupby("front_id")["item_id"].apply(list).to_dict() == (
        ranked_shuffled.groupby("front_id")["item_id"].apply(list).to_dict()
    )


def test_train_ltr_core_wrappers_return_expected_artifacts():
    df = tiny_training_df()

    cv_result = train_ltr_cv(
        df,
        model_type=ModelType.LGBMRanker,
        drop_cols=["ignored"],
        label_mode=LabelMode.CONTINUOUS,
        n_splits=2,
        model_params=SMALL_LGBM_PARAMS,
    )
    assert len(cv_result.models) == 2
    assert cv_result.feature_columns == ["f_signal", "f_inverse"]
    assert len(cv_result.valid_folds) == 2
    assert all("score" in fold.columns for fold in cv_result.valid_folds)

    fit = train_ltr_full(
        df,
        model_type=ModelType.LGBMRanker,
        drop_cols=["ignored"],
        label_mode=LabelMode.CONTINUOUS,
        model_params=SMALL_LGBM_PARAMS,
    )
    assert fit.model is not None
    assert fit.feature_columns == ["f_signal", "f_inverse"]
    assert fit.metadata["model_type"] == ModelType.LGBMRanker.value
