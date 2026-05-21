"""Shared helpers for post-Pareto decision baseline scoring."""

from __future__ import annotations

from importlib import import_module
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


VALID_OBJECTIVE_DIRECTIONS = {"minimize", "maximize"}
PYMCDM_PACKAGE = "pymcdm"
PYMCDM_VERSION = "1.4.0"
PYMCDM_ADAPTER = f"{PYMCDM_PACKAGE}_{PYMCDM_VERSION}_adapter"
AUGMENTED_TCHEBYCHEFF_RHO = 1e-6


class ObjectiveMatrixError(ValueError):
    """Raised when an objective matrix cannot be built safely."""


class DecisionBaselineAdapterError(ValueError):
    """Raised when an external decision-baseline adapter cannot score data."""


def normalized_objective_badness_matrix(
    df: pd.DataFrame,
    *,
    objective_columns: Sequence[str],
    objective_directions: Mapping[str, str],
    front_col: str,
) -> pd.DataFrame:
    """Build per-front objective badness in [0, 1].

    The returned dataframe has the same index as ``df`` and one column per
    objective. A value of 0 means best within the front for that objective, and
    1 means worst within the front. Objectives that are constant within a front
    contribute 0 for that front because they cannot distinguish alternatives.
    """
    objective_columns = list(objective_columns)
    if not objective_columns:
        raise ObjectiveMatrixError("at least one objective column is required")

    missing_columns = [column for column in (front_col, *objective_columns) if column not in df.columns]
    if missing_columns:
        raise ObjectiveMatrixError(f"decision dataframe is missing required column(s): {missing_columns}")

    missing_directions = [column for column in objective_columns if column not in objective_directions]
    if missing_directions:
        raise ObjectiveMatrixError(f"objective columns missing directions: {missing_directions}")

    invalid_directions = {
        column: objective_directions[column]
        for column in objective_columns
        if objective_directions[column] not in VALID_OBJECTIVE_DIRECTIONS
    }
    if invalid_directions:
        raise ObjectiveMatrixError(f"invalid objective directions: {invalid_directions}")

    front_ids = df[front_col]
    if front_ids.isna().any():
        raise ObjectiveMatrixError(f"front column '{front_col}' contains missing values")

    badness = {
        objective: _normalized_objective_badness(
            df[objective],
            front_ids,
            direction=objective_directions[objective],
        )
        for objective in objective_columns
    }
    return pd.DataFrame(badness, index=df.index, columns=objective_columns).astype(float)


def weighted_sum_scores_from_badness(
    badness_df: pd.DataFrame,
    *,
    weights: Mapping[str, float] | Sequence[float] | None = None,
) -> pd.Series:
    """Score alternatives with package-backed WSM over normalized goodness."""
    matrix = _coerce_badness_matrix(badness_df)
    weight_array = _weights_for_badness(badness_df, weights)
    goodness = 1.0 - matrix
    wsm = _pymcdm_method("WSM", baseline_id="objective_normalized_mean")
    scores = wsm(normalization_function=_identity_normalization)(
        goodness,
        weight_array,
        _profit_types(badness_df.shape[1]),
        validation=False,
    )
    return pd.Series(scores, index=badness_df.index, dtype=float)


def ideal_l2_scores_from_badness(
    badness_df: pd.DataFrame,
    *,
    weights: Mapping[str, float] | Sequence[float] | None = None,
) -> pd.Series:
    """Score alternatives by negative weighted L2 distance to the ideal."""
    matrix = _coerce_badness_matrix(badness_df)
    weight_array = _weights_for_badness(badness_df, weights)
    distances = np.sqrt(np.sum(weight_array * np.square(matrix), axis=1))
    return pd.Series(-distances, index=badness_df.index, dtype=float)


def augmented_tchebycheff_scores_from_badness(
    badness_df: pd.DataFrame,
    *,
    weights: Mapping[str, float] | Sequence[float] | None = None,
    rho: float = AUGMENTED_TCHEBYCHEFF_RHO,
) -> pd.Series:
    """Score alternatives by negative augmented weighted Tchebycheff."""
    matrix = _coerce_badness_matrix(badness_df)
    if not np.isfinite(rho) or rho < 0.0:
        raise DecisionBaselineAdapterError("augmented Tchebycheff rho must be finite and non-negative")
    weight_array = _weights_for_badness(badness_df, weights)
    weighted_badness = matrix * weight_array
    scores = -(np.max(weighted_badness, axis=1) + (rho * np.sum(weighted_badness, axis=1)))
    return pd.Series(scores, index=badness_df.index, dtype=float)


def topsis_scores_from_badness(
    badness_df: pd.DataFrame,
    *,
    weights: Mapping[str, float] | Sequence[float] | None = None,
) -> pd.Series:
    """Score alternatives with package-backed TOPSIS over normalized goodness."""
    matrix = _coerce_badness_matrix(badness_df)
    weight_array = _weights_for_badness(badness_df, weights)
    goodness = 1.0 - matrix
    topsis = _pymcdm_method("TOPSIS", baseline_id="objective_topsis")
    # pymcdm computes distances after multiplying normalized columns by the
    # supplied weights. Use sqrt(w) to obtain sqrt(sum(w_i * deviation_i^2)).
    # Append explicit ideal/anti-ideal rows because pymcdm derives them from
    # the provided matrix, while this baseline fixes them at goodness 1 and 0.
    distance_weights = np.sqrt(weight_array)
    reference_rows = np.vstack(
        [
            np.ones((1, goodness.shape[1]), dtype=float),
            np.zeros((1, goodness.shape[1]), dtype=float),
        ]
    )
    augmented_goodness = np.vstack([goodness, reference_rows])
    scores = topsis(normalization_function=_identity_normalization)(
        augmented_goodness,
        distance_weights,
        _profit_types(badness_df.shape[1]),
        validation=False,
    )
    return _finite_scores(scores[: matrix.shape[0]], index=badness_df.index, baseline_id="objective_topsis")


def vikor_scores_from_badness(
    badness_df: pd.DataFrame,
    *,
    weights: Mapping[str, float] | Sequence[float] | None = None,
    v: float = 0.5,
) -> pd.Series:
    """Score alternatives with package-backed VIKOR over normalized goodness."""
    matrix = _coerce_badness_matrix(badness_df)
    if not np.isfinite(v) or v < 0.0 or v > 1.0:
        raise DecisionBaselineAdapterError("VIKOR parameter v must be in [0, 1]")
    variable_columns = ~np.all(np.isclose(matrix, matrix[0, :]), axis=0)
    if matrix.shape[0] <= 1 or not np.any(variable_columns):
        return pd.Series(0.0, index=badness_df.index, dtype=float)

    retained_badness = badness_df.loc[:, list(np.asarray(badness_df.columns)[variable_columns])]
    retained_matrix = matrix[:, variable_columns]
    weight_array = _weights_for_badness(retained_badness, _retained_weights(badness_df, weights, variable_columns))
    weighted_badness = retained_matrix * weight_array
    s_values = np.sum(weighted_badness, axis=1)
    r_values = np.max(weighted_badness, axis=1)
    if _all_close(s_values) or _all_close(r_values):
        q_values = (v * _zero_if_tied_minmax(s_values)) + ((1.0 - v) * _zero_if_tied_minmax(r_values))
        return -_finite_scores(q_values, index=badness_df.index, baseline_id="objective_vikor")

    goodness = 1.0 - retained_matrix
    vikor = _pymcdm_method("VIKOR", baseline_id="objective_vikor")
    q_values = vikor(v=v)(
        goodness,
        weight_array,
        _profit_types(retained_badness.shape[1]),
        validation=False,
    )
    return -_finite_scores(q_values, index=badness_df.index, baseline_id="objective_vikor")


def _normalized_objective_badness(
    values: pd.Series,
    front_ids: pd.Series,
    *,
    direction: str,
) -> pd.Series:
    try:
        numeric = pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ObjectiveMatrixError(f"objective column '{values.name}' must be numeric") from exc

    grouped = numeric.groupby(front_ids)
    min_values = grouped.transform("min")
    max_values = grouped.transform("max")
    denom = (max_values - min_values).replace(0, np.nan)
    if direction == "minimize":
        badness = (numeric - min_values) / denom
    elif direction == "maximize":
        badness = (max_values - numeric) / denom
    else:
        raise ObjectiveMatrixError(f"unsupported objective direction: {direction}")
    return badness.fillna(0.0).rename(values.name)


def _coerce_badness_matrix(badness_df: pd.DataFrame) -> np.ndarray:
    if badness_df.empty:
        raise DecisionBaselineAdapterError("badness matrix must contain at least one row and one objective")
    if badness_df.shape[1] == 0:
        raise DecisionBaselineAdapterError("badness matrix must contain at least one objective")
    try:
        matrix = badness_df.astype(float).to_numpy()
    except (TypeError, ValueError) as exc:
        raise DecisionBaselineAdapterError("badness matrix must be numeric") from exc
    if not np.isfinite(matrix).all():
        raise DecisionBaselineAdapterError("badness matrix must contain only finite values")
    tolerance = 1e-12
    if matrix.min() < -tolerance or matrix.max() > 1.0 + tolerance:
        raise DecisionBaselineAdapterError("badness matrix values must be in [0, 1]")
    return np.clip(matrix, 0.0, 1.0)


def _weights_for_badness(
    badness_df: pd.DataFrame,
    weights: Mapping[str, float] | Sequence[float] | None,
) -> np.ndarray:
    if weights is None:
        weight_array = np.full(badness_df.shape[1], 1.0 / badness_df.shape[1], dtype=float)
    elif isinstance(weights, Mapping):
        missing = [str(column) for column in badness_df.columns if column not in weights]
        if missing:
            raise DecisionBaselineAdapterError(f"weights are missing objective(s): {missing}")
        weight_array = np.asarray([weights[column] for column in badness_df.columns], dtype=float)
    else:
        weight_array = np.asarray(list(weights), dtype=float)
        if weight_array.shape[0] != badness_df.shape[1]:
            raise DecisionBaselineAdapterError(
                f"expected {badness_df.shape[1]} objective weights, got {weight_array.shape[0]}"
            )

    if not np.isfinite(weight_array).all():
        raise DecisionBaselineAdapterError("objective weights must be finite")
    if np.any(weight_array < 0):
        raise DecisionBaselineAdapterError("objective weights must be non-negative")
    weight_sum = float(weight_array.sum())
    if weight_sum <= 0:
        raise DecisionBaselineAdapterError("at least one objective weight must be positive")
    return weight_array / weight_sum


def _retained_weights(
    badness_df: pd.DataFrame,
    weights: Mapping[str, float] | Sequence[float] | None,
    variable_columns: np.ndarray,
) -> Mapping[str, float] | Sequence[float] | None:
    if weights is None:
        return None
    if isinstance(weights, Mapping):
        return {
            column: float(weights[column])
            for column, keep in zip(badness_df.columns, variable_columns)
            if keep
        }
    weight_array = np.asarray(list(weights), dtype=float)
    if weight_array.shape[0] != badness_df.shape[1]:
        raise DecisionBaselineAdapterError(
            f"expected {badness_df.shape[1]} objective weights, got {weight_array.shape[0]}"
        )
    return weight_array[variable_columns].tolist()


def _pymcdm_method(method_name: str, *, baseline_id: str):
    try:
        methods = import_module("pymcdm.methods")
    except ImportError as exc:
        raise DecisionBaselineAdapterError(
            f"baseline '{baseline_id}' requires optional dependency '{PYMCDM_PACKAGE}=={PYMCDM_VERSION}'"
        ) from exc
    try:
        return getattr(methods, method_name)
    except AttributeError as exc:
        raise DecisionBaselineAdapterError(
            f"dependency '{PYMCDM_PACKAGE}' does not expose required method '{method_name}'"
        ) from exc


def _identity_normalization(values: np.ndarray, cost: bool = False) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _profit_types(n_objectives: int) -> np.ndarray:
    return np.ones(n_objectives, dtype=int)


def _all_rows_tied(matrix: np.ndarray) -> bool:
    return matrix.shape[0] <= 1 or np.all(np.isclose(matrix, matrix[0, :]))


def _all_close(values: np.ndarray) -> bool:
    return values.shape[0] <= 1 or np.all(np.isclose(values, values[0]))


def _zero_if_tied_minmax(values: np.ndarray) -> np.ndarray:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if np.isclose(minimum, maximum):
        return np.zeros_like(values, dtype=float)
    return (values - minimum) / (maximum - minimum)


def _finite_scores(scores: np.ndarray, *, index: pd.Index, baseline_id: str) -> pd.Series:
    score_array = np.asarray(scores, dtype=float)
    if not np.isfinite(score_array).all():
        raise DecisionBaselineAdapterError(f"baseline '{baseline_id}' produced non-finite scores")
    return pd.Series(score_array, index=index, dtype=float)
