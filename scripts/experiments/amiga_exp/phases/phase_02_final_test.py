"""Held-out final test evaluation for the selected AMIGA configuration."""

from __future__ import annotations

from collections.abc import Mapping as AbcMapping
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from amiga.core.main import rank_with_model, train_ltr_full
from amiga.selection.learn2rank import LabelMode, ModelType
from amiga.utils import save_pickle
from scripts.experiments.amiga_exp.context import CaseContext
from scripts.experiments.amiga_exp.evaluation import (
    DEFAULT_EVALUATION_KS,
    RankingEvaluationError,
    evaluate_ranked_frame,
    write_cv_report,
)
from scripts.experiments.amiga_exp.manifests import (
    ManifestError,
    WriteOptions,
    common_manifest_fields,
    repo_relative,
    results_layout,
)
from scripts.experiments.amiga_exp.phases.phase_01_model_screening import (
    DEFAULT_LABEL_QUANTILES,
    FEATURE_SET_NAME,
    parse_label_mode,
    parse_model_type,
)


STEP_NAME = "final_test"
PARENT_PHASE_NAME = "02_hyperparameter_tuning"


class FinalTestEvaluationError(ValueError):
    """Raised when the held-out final test evaluation cannot proceed safely."""


@dataclass(frozen=True)
class SelectedConfig:
    """Frozen selected AMIGA configuration."""

    run_id: str
    model_type: ModelType
    label_mode: LabelMode
    label_quantiles: int | None
    feature_set: str
    model_params: dict[str, Any]


@dataclass(frozen=True)
class FinalTestResult:
    """Artifacts produced by held-out final test evaluation."""

    final_test_dir: Path
    model: Path
    feature_columns: Path
    final_test_ranked: Path
    final_test_report: Path
    cv_report: Path
    status: str


def run_final_test_evaluation(
    context: CaseContext,
    *,
    seed: int = 42,
    options: WriteOptions | None = None,
    selected_config_path: Path | None = None,
) -> FinalTestResult:
    """Train the selected config on development and evaluate once on held-out test."""
    options = options or WriteOptions()
    options.validate()
    layout = results_layout(context)
    phase_dir = layout.phases[PARENT_PHASE_NAME]
    final_test_dir = phase_dir / "final_test"
    resolved_selected_config_path = selected_config_path or (phase_dir / "selected_config.json")
    outputs = _final_test_outputs(final_test_dir)
    selected_config = load_selected_config(resolved_selected_config_path)

    if options.dry_run:
        return FinalTestResult(final_test_dir=final_test_dir, status="dry_run", **outputs)
    if options.skip_existing and all(path.exists() for path in outputs.values()):
        return FinalTestResult(final_test_dir=final_test_dir, status="skipped", **outputs)
    if options.skip_existing and final_test_dir.exists() and any(final_test_dir.iterdir()):
        missing = [path for path in outputs.values() if not path.exists()]
        raise FinalTestEvaluationError(
            f"cannot skip incomplete final test outputs; missing: {missing}"
        )

    _prepare_output_dir(final_test_dir, outputs, options)
    feature_columns = _feature_columns(context, selected_config.feature_set)
    training_df = _split_training_frame(context.development_df, context, feature_columns)
    test_df = _split_training_frame(context.test_df, context, feature_columns)

    label_quantiles = selected_config.label_quantiles or DEFAULT_LABEL_QUANTILES[0]
    fit = train_ltr_full(
        training_df,
        model_type=selected_config.model_type,
        front_col=context.group_column,
        target_col=context.target_column,
        id_col=context.item_column,
        drop_cols=[],
        label_mode=selected_config.label_mode,
        label_quantiles=label_quantiles,
        random_state=seed,
        model_params=_model_params_for_training(selected_config),
    )
    if list(fit.feature_columns) != list(feature_columns):
        raise FinalTestEvaluationError(
            f"selected configuration used unexpected feature columns; "
            f"expected {len(feature_columns)}, got {len(fit.feature_columns)}"
        )

    ranked = rank_with_model(
        test_df,
        fit.model,
        front_col=context.group_column,
        id_col=context.item_column,
        feature_columns_hint=feature_columns,
    ).df_ranked
    ranked.to_csv(outputs["final_test_ranked"], index=False)

    meta = {
        "phase": PARENT_PHASE_NAME,
        "step": STEP_NAME,
        "selected_config_path": repo_relative(resolved_selected_config_path),
        "selected_run_id": selected_config.run_id,
        "model_type": selected_config.model_type.value,
        "label_mode": selected_config.label_mode.value,
        "label_quantiles": selected_config.label_quantiles,
        "feature_set": selected_config.feature_set,
        "model_params": selected_config.model_params,
        "random_state": seed,
        "development_fronts": context.n_development_fronts,
        "development_rows": context.n_development_rows,
        "test_fronts": context.n_test_fronts,
        "test_rows": context.n_test_rows,
    }
    try:
        evaluation = evaluate_ranked_frame(
            ranked,
            front_col=context.group_column,
            target_col=context.target_column,
            score_col="score",
            model_name=selected_config.run_id,
            evaluation_split="test",
            fold=1,
            ks=DEFAULT_EVALUATION_KS,
            meta=meta,
        )
    except RankingEvaluationError as exc:
        raise FinalTestEvaluationError(str(exc)) from exc

    save_pickle(fit.model, outputs["model"])
    _write_json(
        outputs["feature_columns"],
        {
            "feature_set": selected_config.feature_set,
            "n_features": len(feature_columns),
            "feature_columns": feature_columns,
        },
    )
    write_cv_report(outputs["cv_report"], evaluation.report)
    _write_json(
        outputs["final_test_report"],
        _final_test_report_payload(
            context,
            selected_config=selected_config,
            selected_config_path=resolved_selected_config_path,
            outputs=outputs,
            seed=seed,
            evaluation=evaluation,
        ),
    )

    return FinalTestResult(final_test_dir=final_test_dir, status="written", **outputs)


def load_selected_config(path: Path) -> SelectedConfig:
    """Load the frozen selected config written by phase 02."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FinalTestEvaluationError(f"selected config file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FinalTestEvaluationError(f"invalid selected config JSON: {path}") from exc

    selected = payload.get("selected_config") if isinstance(payload, dict) else None
    if not isinstance(selected, dict):
        raise FinalTestEvaluationError(f"selected config JSON is missing object 'selected_config': {path}")

    run_id = str(selected.get("run_id") or "")
    if not run_id:
        raise FinalTestEvaluationError("selected config is missing 'run_id'")
    feature_set = str(selected.get("feature_set") or FEATURE_SET_NAME)
    model_params = selected.get("model_params") or {}
    if not isinstance(model_params, AbcMapping):
        raise FinalTestEvaluationError("selected config 'model_params' must be an object")

    label_quantiles = selected.get("label_quantiles")
    return SelectedConfig(
        run_id=run_id,
        model_type=parse_model_type(str(selected.get("model_type", ""))),
        label_mode=parse_label_mode(str(selected.get("label_mode", ""))),
        label_quantiles=None if label_quantiles is None else int(label_quantiles),
        feature_set=feature_set,
        model_params=dict(model_params),
    )


def _prepare_output_dir(
    final_test_dir: Path,
    outputs: Mapping[str, Path],
    options: WriteOptions,
) -> None:
    if final_test_dir.exists() and any(final_test_dir.iterdir()) and not options.force:
        raise ManifestError(f"refusing to overwrite existing final test outputs without --force: {final_test_dir}")

    final_test_dir.mkdir(parents=True, exist_ok=True)


def _final_test_outputs(final_test_dir: Path) -> dict[str, Path]:
    return {
        "model": final_test_dir / "model.pkl",
        "feature_columns": final_test_dir / "feature_columns.json",
        "final_test_ranked": final_test_dir / "final_test_ranked.csv",
        "final_test_report": final_test_dir / "final_test_report.json",
        "cv_report": final_test_dir / "cv_report.json",
    }


def _feature_columns(context: CaseContext, feature_set: str) -> list[str]:
    columns = list(context.feature_sets.get(feature_set, []))
    if not columns:
        raise FinalTestEvaluationError(f"feature set '{feature_set}' is not defined in the feature contract")
    missing = [column for column in columns if column not in context.df.columns]
    if missing:
        raise FinalTestEvaluationError(f"data is missing feature columns: {missing}")
    return columns


def _split_training_frame(
    df: pd.DataFrame,
    context: CaseContext,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    control_columns = [context.group_column, context.item_column, context.target_column]
    columns = list(dict.fromkeys([*control_columns, *feature_columns]))
    frame = df.loc[:, columns].copy().reset_index(drop=True)
    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame[context.target_column] = pd.to_numeric(frame[context.target_column], errors="raise")
    return frame


def _model_params_for_training(selected_config: SelectedConfig) -> dict[str, Any]:
    params = dict(selected_config.model_params)
    if selected_config.model_type == ModelType.CatBoostRanker:
        params["allow_writing_files"] = False
    return params


def _final_test_report_payload(
    context: CaseContext,
    *,
    selected_config: SelectedConfig,
    selected_config_path: Path,
    outputs: Mapping[str, Path],
    seed: int,
    evaluation,
) -> dict[str, Any]:
    return {
        **common_manifest_fields(context, seed=seed),
        "manifest_type": "final_test_report",
        "phase": PARENT_PHASE_NAME,
        "step": STEP_NAME,
        "status": "completed",
        "selected_config_path": repo_relative(selected_config_path),
        "selected_config": {
            "run_id": selected_config.run_id,
            "model_type": selected_config.model_type.value,
            "label_mode": selected_config.label_mode.value,
            "label_quantiles": selected_config.label_quantiles,
            "feature_set": selected_config.feature_set,
            "model_params": selected_config.model_params,
        },
        "agg": evaluation.agg,
        "groups": evaluation.groups,
        "per_front_metrics": evaluation.per_front_metrics.to_dict(orient="records"),
        "outputs": {name: repo_relative(path) for name, path in outputs.items()},
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return repo_relative(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
