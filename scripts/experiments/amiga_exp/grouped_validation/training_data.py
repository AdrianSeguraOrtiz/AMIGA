"""Training-only data preparation for grouped validation experiments.

No validation metrics, model fitting, or outcome-based selection occurs here.
Predictor NaNs are preserved for CatBoost; predictor infinities and nonfinite
targets are errors. Complete selected fronts are required by the caller.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from amiga.selection.learn2rank import LabelMode, build_labels


LABEL_MODES = {
    "BIO-INSIGHT": LabelMode.RANK_DENSE,
    "MO-GENECI": LabelMode.RANK_AVG,
}
RESERVED_FEATURES = {
    "front_id", "item_id", "topology_id", "AUPR", "label", "target",
    "score", "rank_in_front", "ltr_catboost", "reg_aupr",
    "reg_normalized", "clf_top20",
}


def _validate_features(feature_columns: list[str]) -> list[str]:
    columns = list(feature_columns)
    if not columns or any(not isinstance(c, str) or not c for c in columns):
        raise ValueError("feature_columns must contain nonempty column names")
    if len(set(columns)) != len(columns):
        raise ValueError("feature_columns contains duplicates")
    forbidden = sorted(set(columns) & RESERVED_FEATURES)
    if forbidden:
        raise ValueError(f"control or target columns cannot be predictors: {forbidden}")
    return columns


def _integer_ids(values: pd.Series, name: str) -> np.ndarray:
    try:
        numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite integer IDs") from exc
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} must contain finite integer IDs")
    if (np.abs(numeric) > 2**53 - 1).any():
        raise ValueError(f"{name} contains IDs outside the supported exact integer range")
    return numeric.astype(np.int64)


def prepare_frame(
    frame: pd.DataFrame,
    case: str,
    topology_by_front: dict[int, str],
    feature_columns: list[str],
) -> dict[str, Any]:
    """Prepare an already restricted training frame, preserving front boundaries.

    Flat-front exclusion and every supervised transformation apply only to the
    supplied frame. ``topology_by_front`` may contain additional, unused IDs.
    No transformation statistics are estimated from any other data.
    """
    if case not in LABEL_MODES:
        raise ValueError(f"unsupported case: {case!r}")
    features = _validate_features(feature_columns)
    required = ["front_id", "item_id", "AUPR", *features]
    if frame.columns.duplicated().any():
        raise ValueError("frame contains duplicate column names")
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"training frame is missing columns: {missing}")
    if frame.empty:
        raise ValueError("training frame is empty")

    data = frame.loc[:, required].copy()
    for column in ("front_id", "item_id"):
        data[column] = _integer_ids(data[column], column)
    if data.duplicated(["front_id", "item_id"]).any():
        raise ValueError("duplicate item_id within a training front")
    supplied_fronts = sorted(int(v) for v in data["front_id"].unique())
    missing_topologies = [f for f in supplied_fronts if f not in topology_by_front]
    if missing_topologies:
        raise ValueError(f"topology mapping is missing training fronts: {missing_topologies}")
    if any(not isinstance(topology_by_front[f], str) or not topology_by_front[f]
           for f in supplied_fronts):
        raise ValueError("topology IDs must be nonempty strings")
    try:
        data["AUPR"] = pd.to_numeric(data["AUPR"], errors="raise").astype(np.float64)
        data[features] = data[features].apply(pd.to_numeric, errors="raise").astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("targets and predictor values must be numeric") from exc
    if not np.isfinite(data["AUPR"].to_numpy()).all():
        raise ValueError("training targets contain NaN or infinity")
    if np.isinf(data[features].to_numpy()).any():
        raise ValueError("training predictors contain infinity; NaNs are allowed")

    quality_counts = data.groupby("front_id", sort=True)["AUPR"].nunique()
    flat_fronts = sorted(int(v) for v in quality_counts[quality_counts == 1].index)
    input_rows = len(data)
    data = data.loc[~data["front_id"].isin(flat_fronts)].copy()
    if data.empty:
        raise ValueError("no informative training fronts remain after flat-front exclusion")
    data = data.sort_values(["front_id", "item_id"], kind="mergesort").reset_index(drop=True)
    data["topology_id"] = data["front_id"].map(topology_by_front)
    front_sizes = data.groupby("front_id", sort=True).size()
    topology_front_counts = (
        data[["front_id", "topology_id"]].drop_duplicates()
        .groupby("topology_id")["front_id"].size()
    )
    m_g = data["topology_id"].map(topology_front_counts).to_numpy(dtype=np.float64)
    n_f = data["front_id"].map(front_sizes).to_numpy(dtype=np.float64)
    group_weight = 1.0 / m_g
    point_weight = 1.0 / (m_g * n_f)
    point_weight /= point_weight.mean()

    classification = np.zeros(len(data), dtype=np.int64)
    class_report: list[dict[str, Any]] = []
    for front_id, group in data.groupby("front_id", sort=True):
        values = group["AUPR"].to_numpy(dtype=np.float64)
        k = max(1, (len(values) + 4) // 5)
        threshold = float(np.partition(values, len(values) - k)[len(values) - k])
        minimum = float(values.min())
        minimum_fallback = threshold == minimum
        positive = values > minimum if minimum_fallback else values >= threshold
        if not positive.any() or positive.all():
            raise ValueError(f"classification has only one class in nonflat front {front_id}")
        classification[group.index.to_numpy()] = positive.astype(np.int64)
        class_report.append({
            "front_id": int(front_id), "n_candidates": len(values),
            "requested_top_k": k, "threshold": threshold,
            "n_positive": int(positive.sum()), "positive_fraction": float(positive.mean()),
            "minimum_threshold_fallback": bool(minimum_fallback),
        })

    labels = {
        "ltr_catboost": build_labels(data, "front_id", "AUPR", LABEL_MODES[case]),
        "reg_aupr": data["AUPR"].to_numpy(dtype=np.float64, copy=True),
        "reg_normalized": build_labels(data, "front_id", "AUPR", LabelMode.CONTINUOUS),
        "clf_top20": classification,
    }
    X = data[features].to_numpy(dtype=np.float64, copy=True)
    group_id = data["front_id"].to_numpy(dtype=np.int64, copy=True)
    item_id = data["item_id"].to_numpy(dtype=np.int64, copy=True)
    report = {
        "case": case, "scope": "supplied_training_fronts_only",
        "input_rows": input_rows, "input_fronts": len(supplied_fronts),
        "input_topologies": len({topology_by_front[f] for f in supplied_fronts}),
        "usable_rows": len(data), "usable_fronts": len(front_sizes),
        "usable_topologies": len(topology_front_counts),
        "selected_front_ids": supplied_fronts,
        "usable_front_ids": [int(v) for v in front_sizes.index],
        "flat_front_ids_excluded": flat_fronts,
        "flat_rows_excluded": input_rows - len(data),
        "feature_count": len(features), "rank_label_mode": LABEL_MODES[case].value,
        "usable_fronts_by_topology": {str(k): int(v) for k, v in topology_front_counts.items()},
        "class_fractions": class_report,
        "finite_checks": {
            "targets_finite": True, "predictor_infinities": 0,
            "predictor_nan_count": int(np.isnan(X).sum()),
            "predictor_nan_policy": "preserved for CatBoost",
            "all_labels_finite": bool(all(np.isfinite(y).all() for y in labels.values())),
        },
        "point_weight_mean": float(point_weight.mean()),
        "point_weight_total_by_topology": {
            str(topology): float(point_weight[data["topology_id"].to_numpy() == topology].sum())
            for topology in topology_front_counts.index
        },
        "group_weight_rule": "exactly 1 / usable_fronts_in_topology; repeated per candidate",
        "rows_ordered_by": ["front_id", "item_id"],
        "evaluation_performed": False,
    }
    return {
        "X": X, "labels": labels, "group_id": group_id, "item_id": item_id,
        "point_weight": point_weight, "group_weight": group_weight,
        "feature_names": features, "report": report,
    }


def prepare_training(
    repo_root: Path,
    case: str,
    front_ids: list[int],
    topology_by_front: dict[int, str],
    feature_columns: list[str],
) -> dict[str, Any]:
    """Load complete requested fronts and prepare training without evaluating them.

    CSV chunks are restricted to requested IDs before they are retained or any
    labels, flat-front decisions, or weights are constructed. Feature columns
    must be members of the existing case-specific predictor contract.
    """
    if case not in LABEL_MODES:
        raise ValueError(f"unsupported case: {case!r}")
    features = _validate_features(feature_columns)
    if not front_ids:
        raise ValueError("front_ids must not be empty")
    requested = _integer_ids(pd.Series(front_ids), "front_ids")
    if len(set(requested.tolist())) != len(requested):
        raise ValueError("front_ids contains duplicates")
    requested_set = set(requested.tolist())
    root = Path(repo_root).expanduser().resolve()
    source = root / "experiments" / case / "data" / "data_104.csv"
    contract_path = root / "docs" / "experiments" / "contracts" / f"{case}_feature_columns.json"
    if not source.is_file():
        raise FileNotFoundError(f"case dataset does not exist: {source}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    allowed = set(contract["feature_sets"]["full"])
    unknown = sorted(set(features) - allowed)
    if unknown:
        raise ValueError(f"predictors absent from the case feature contract: {unknown}")
    chunks = []
    for chunk in pd.read_csv(source, usecols=["front_id", "item_id", "AUPR", *features], chunksize=10000):
        selected = chunk.loc[chunk["front_id"].isin(requested_set)].copy()
        if not selected.empty:
            chunks.append(selected)
    if not chunks:
        raise ValueError("none of the requested training front IDs exists in the dataset")
    selected_frame = pd.concat(chunks, ignore_index=True)
    found = set(selected_frame["front_id"].astype(int).unique().tolist())
    missing = sorted(requested_set - found)
    if missing:
        raise ValueError(f"requested training fronts are missing from dataset: {missing}")
    prepared = prepare_frame(selected_frame, case, topology_by_front, features)
    prepared["report"]["source_csv"] = str(source)
    prepared["report"]["feature_contract"] = str(contract_path)
    return prepared
