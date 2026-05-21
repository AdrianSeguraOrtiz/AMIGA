"""Phase 01: model and label-mode screening on development fronts."""

from __future__ import annotations

from collections.abc import Mapping as AbcMapping
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from amiga.core.main import TrainFoldReport, train_ltr_cv
from amiga.selection.learn2rank import LabelMode, ModelType, assign_rank_in_front
from scripts.experiments.amiga_exp.config_selection import (
    DEFAULT_SELECTION_CRITERIA,
    PRIMARY_METRIC,
    ConfigSelectionError,
    select_configurations,
    selection_rank_stats_from_reports,
)
from scripts.experiments.amiga_exp.context import CaseContext
from scripts.experiments.amiga_exp.evaluation import DEFAULT_EVALUATION_KS
from scripts.experiments.amiga_exp.manifests import (
    ManifestError,
    WriteOptions,
    common_manifest_fields,
    ensure_results_layout,
    repo_relative,
    results_layout,
    run_manifest,
    write_manifest,
    write_phase_status_manifest,
)
from scripts.experiments.amiga_exp.primary_rank_table import (
    PrimaryRankTableError,
    write_grouped_primary_rank_table,
)
from scripts.experiments.amiga_exp.plots import planned_phase_plot_outputs
from scripts.experiments.amiga_exp.reports import (
    ReportSummaryError,
    summarize_phase_cv_reports,
)


PHASE_NAME = "01_model_screening"
FEATURE_SET_NAME = "full"
DEFAULT_LABEL_QUANTILES = (5, 10, 15)
SHORTLIST_SELECTION_SCOPE = "best_label_per_model_type"
SHORTLIST_GROUP_FIELD = "model_type"
SHORTLIST_SELECTED_PER_GROUP = 1
DEFAULT_LABEL_MODES = (
    LabelMode.RANK_DENSE,
    LabelMode.RANK_AVG,
    LabelMode.QUANTILES,
    LabelMode.CONTINUOUS,
    LabelMode.REVERSED,
    LabelMode.SHUFFLED,
)
DEFAULT_MODEL_TYPES = (
    ModelType.LGBMRanker,
    ModelType.XGBRanker,
    ModelType.CatBoostRanker,
)
REFERENCE_MODEL_PARAMS: dict[ModelType, dict[str, Any]] = {
    ModelType.LGBMRanker: {
        "num_leaves": 63,
        "min_child_samples": 50,
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    ModelType.XGBRanker: {
        "max_depth": 6,
        "min_child_weight": 5,
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    ModelType.CatBoostRanker: {
        "depth": 6,
        "l2_leaf_reg": 5,
        "learning_rate": 0.05,
        "iterations": 2000,
        "allow_writing_files": False,
    },
}
LABEL_SELECTION_CRITERIA = tuple(
    criterion
    for criterion in DEFAULT_SELECTION_CRITERIA
    if criterion.source != "model_priority"
)


class ModelScreeningError(ValueError):
    """Raised when phase 01 cannot be planned or executed safely."""


@dataclass(frozen=True)
class ScreeningRunConfig:
    """One model-screening configuration."""

    run_id: str
    model_type: ModelType
    label_mode: LabelMode
    label_quantiles: int | None
    selectable: bool

    def to_manifest_parameters(self, *, model_params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "model_type": self.model_type.value,
            "label_mode": self.label_mode.value,
            "label_quantiles": self.label_quantiles,
            "feature_set": FEATURE_SET_NAME,
            "selectable": self.selectable,
            "model_params": dict(model_params or {}),
        }


@dataclass(frozen=True)
class ScreeningRunResult:
    """Artifacts produced or reused for one screening run."""

    config: ScreeningRunConfig
    run_dir: Path
    cv_report: Path
    feature_columns: Path
    valid_fold_ranked: tuple[Path, ...]
    run_manifest: Path
    status: str


@dataclass(frozen=True)
class ModelScreeningResult:
    """Phase-level result for model screening."""

    phase_dir: Path
    run_results: tuple[ScreeningRunResult, ...]
    summary_outputs: dict[str, Path]
    table_outputs: dict[str, Path]
    screening_manifest: Path
    shortlisted_configs: Path
    status: str


def parse_model_type(value: str | ModelType) -> ModelType:
    """Parse a CLI/config value into a supported model type."""
    if isinstance(value, ModelType):
        return value
    try:
        return ModelType(value)
    except ValueError as exc:
        allowed = ", ".join(model.value for model in ModelType)
        raise ModelScreeningError(f"unknown model type '{value}'. Allowed values: {allowed}") from exc


def parse_label_mode(value: str | LabelMode) -> LabelMode:
    """Parse a CLI/config value into a supported label mode."""
    if isinstance(value, LabelMode):
        return value
    try:
        return LabelMode(value)
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in LabelMode)
        raise ModelScreeningError(f"unknown label mode '{value}'. Allowed values: {allowed}") from exc


def build_screening_run_configs(
    *,
    model_types: Sequence[str | ModelType] | None = None,
    label_modes: Sequence[str | LabelMode] | None = None,
    label_quantiles: Sequence[int] = DEFAULT_LABEL_QUANTILES,
) -> tuple[ScreeningRunConfig, ...]:
    """Build the phase-01 run grid."""
    parsed_models = tuple(parse_model_type(model) for model in (model_types or DEFAULT_MODEL_TYPES))
    parsed_modes = tuple(parse_label_mode(mode) for mode in (label_modes or DEFAULT_LABEL_MODES))
    quantile_values = tuple(int(value) for value in label_quantiles)
    if not quantile_values:
        raise ModelScreeningError("at least one label-quantiles value is required")
    invalid_quantiles = [value for value in quantile_values if value < 2]
    if invalid_quantiles:
        raise ModelScreeningError(f"label-quantiles values must be >= 2: {invalid_quantiles}")

    run_configs: list[ScreeningRunConfig] = []
    seen_run_ids: set[str] = set()
    for model_type in parsed_models:
        for label_mode in parsed_modes:
            mode_quantiles: tuple[int | None, ...]
            if label_mode == LabelMode.QUANTILES:
                mode_quantiles = quantile_values
            else:
                mode_quantiles = (None,)
            for quantiles in mode_quantiles:
                run_id = _run_id(model_type=model_type, label_mode=label_mode, label_quantiles=quantiles)
                if run_id in seen_run_ids:
                    continue
                seen_run_ids.add(run_id)
                run_configs.append(
                    ScreeningRunConfig(
                        run_id=run_id,
                        model_type=model_type,
                        label_mode=label_mode,
                        label_quantiles=quantiles,
                        selectable=True,
                    )
                )

    if not run_configs:
        raise ModelScreeningError("screening grid is empty")
    return tuple(run_configs)


def run_model_screening(
    context: CaseContext,
    *,
    seed: int = 42,
    options: WriteOptions | None = None,
    model_types: Sequence[str | ModelType] | None = None,
    label_modes: Sequence[str | LabelMode] | None = None,
    label_quantiles: Sequence[int] = DEFAULT_LABEL_QUANTILES,
    n_splits: int = 5,
    model_params: Mapping[str, Any] | None = None,
) -> ModelScreeningResult:
    """Run phase 01 using only development fronts."""
    options = options or WriteOptions()
    options.validate()
    if n_splits < 2:
        raise ModelScreeningError("n_splits must be >= 2")
    if context.n_development_fronts < n_splits:
        raise ModelScreeningError(
            f"n_splits={n_splits} is larger than development fronts ({context.n_development_fronts})"
        )
    feature_columns = _full_feature_columns(context)
    control_columns = _training_control_columns(context)
    run_configs = build_screening_run_configs(
        model_types=model_types,
        label_modes=label_modes,
        label_quantiles=label_quantiles,
    )
    layout = results_layout(context)
    phase_dir = layout.phases[PHASE_NAME]
    runs_dir = phase_dir / "runs"
    summary_dir = phase_dir / "summary"

    if options.dry_run:
        result = ModelScreeningResult(
            phase_dir=phase_dir,
            run_results=tuple(
                ScreeningRunResult(
                    config=config,
                    run_dir=runs_dir / config.run_id,
                    cv_report=runs_dir / config.run_id / "cv_report.json",
                    feature_columns=runs_dir / config.run_id / "feature_columns.json",
                    valid_fold_ranked=tuple(
                        runs_dir / config.run_id / f"valid_fold{fold}_ranked.csv"
                        for fold in range(1, n_splits + 1)
                    ),
                    run_manifest=runs_dir / config.run_id / "run_manifest.json",
                    status="planned",
                )
                for config in run_configs
            ),
            summary_outputs={},
            table_outputs={},
            screening_manifest=phase_dir / "screening_manifest.json",
            shortlisted_configs=phase_dir / "shortlisted_configs.json",
            status="dry_run",
        )
        return result

    ensure_results_layout(layout, dry_run=False)
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    development_df = _development_training_frame(context, control_columns, feature_columns)
    run_results: list[ScreeningRunResult] = []
    report_paths: list[Path] = []
    for run_config in run_configs:
        run_result = _execute_or_reuse_run(
            context,
            run_config,
            development_df=development_df,
            expected_feature_columns=feature_columns,
            runs_dir=runs_dir,
            seed=seed,
            n_splits=n_splits,
            model_params=_model_params_for_run(run_config.model_type, model_params),
            options=options,
        )
        run_results.append(run_result)
        report_paths.append(run_result.cv_report)

    try:
        summary_outputs = summarize_phase_cv_reports(report_paths, summary_dir)
    except ReportSummaryError as exc:
        raise ModelScreeningError(str(exc)) from exc
    try:
        table_outputs = _write_primary_tables(
            run_configs=run_configs,
            run_results=run_results,
            summary_outputs=summary_outputs,
            summary_dir=summary_dir,
        )
    except PrimaryRankTableError as exc:
        raise ModelScreeningError(str(exc)) from exc

    try:
        shortlisted_payload = _build_shortlist_payload(
            context,
            run_configs=run_configs,
            run_results=run_results,
            summary_path=summary_outputs["metrics_summary"],
            seed=seed,
        )
    except ConfigSelectionError as exc:
        raise ModelScreeningError(str(exc)) from exc
    shortlist_status = write_manifest(phase_dir / "shortlisted_configs.json", shortlisted_payload, options)

    screening_payload = _screening_manifest_payload(
        context,
        phase_dir=phase_dir,
        run_results=run_results,
        summary_outputs=summary_outputs,
        table_outputs=table_outputs,
        shortlist_path=phase_dir / "shortlisted_configs.json",
        shortlist_status=shortlist_status,
        seed=seed,
        n_splits=n_splits,
        feature_columns=feature_columns,
    )
    manifest_status = write_manifest(phase_dir / "screening_manifest.json", screening_payload, options)
    write_phase_status_manifest(
        phase_dir / "phase_manifest.json",
        context,
        PHASE_NAME,
        seed=seed,
        status="completed",
        outputs={
            "phase_dir": repo_relative(phase_dir),
            "run_count": len(run_results),
            "summary_outputs": {
                name: repo_relative(path) for name, path in summary_outputs.items()
            },
            "table_outputs": {
                name: repo_relative(path) for name, path in table_outputs.items()
            },
            "screening_manifest": repo_relative(phase_dir / "screening_manifest.json"),
            "screening_manifest_status": manifest_status,
            "shortlisted_configs": repo_relative(phase_dir / "shortlisted_configs.json"),
            "shortlisted_configs_status": shortlist_status,
            "plots": planned_phase_plot_outputs(phase_dir, PHASE_NAME),
        },
        options=options,
    )
    return ModelScreeningResult(
        phase_dir=phase_dir,
        run_results=tuple(run_results),
        summary_outputs=summary_outputs,
        table_outputs=table_outputs,
        screening_manifest=phase_dir / "screening_manifest.json",
        shortlisted_configs=phase_dir / "shortlisted_configs.json",
        status="written" if manifest_status == "written" else manifest_status,
    )


def _execute_or_reuse_run(
    context: CaseContext,
    run_config: ScreeningRunConfig,
    *,
    development_df: pd.DataFrame,
    expected_feature_columns: Sequence[str],
    runs_dir: Path,
    seed: int,
    n_splits: int,
    model_params: Mapping[str, Any],
    options: WriteOptions,
) -> ScreeningRunResult:
    run_dir = runs_dir / run_config.run_id
    cv_report_path = run_dir / "cv_report.json"
    feature_columns_path = run_dir / "feature_columns.json"
    run_manifest_path = run_dir / "run_manifest.json"

    if cv_report_path.exists() and options.skip_existing:
        expected_ranked = tuple(run_dir / f"valid_fold{fold}_ranked.csv" for fold in range(1, n_splits + 1))
        missing = [
            path
            for path in (feature_columns_path, run_manifest_path, *expected_ranked)
            if not path.exists()
        ]
        if missing:
            raise ModelScreeningError(
                f"cannot skip incomplete existing run '{run_config.run_id}'; missing: {missing}"
            )
        return ScreeningRunResult(
            config=run_config,
            run_dir=run_dir,
            cv_report=cv_report_path,
            feature_columns=feature_columns_path,
            valid_fold_ranked=expected_ranked,
            run_manifest=run_manifest_path,
            status="skipped",
        )

    if run_dir.exists() and any(run_dir.iterdir()) and not options.force:
        raise ManifestError(f"refusing to overwrite existing run without --force: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    training_quantiles = run_config.label_quantiles or DEFAULT_LABEL_QUANTILES[0]
    result = train_ltr_cv(
        development_df,
        model_type=run_config.model_type,
        front_col=context.group_column,
        target_col=context.target_column,
        id_col=context.item_column,
        drop_cols=[],
        label_mode=run_config.label_mode,
        label_quantiles=training_quantiles,
        n_splits=n_splits,
        random_state=seed,
        model_params=dict(model_params),
    )
    if list(result.feature_columns) != list(expected_feature_columns):
        raise ModelScreeningError(
            f"run '{run_config.run_id}' used unexpected feature columns; "
            f"expected {len(expected_feature_columns)}, got {len(result.feature_columns)}"
        )

    ranked_paths = _write_ranked_validation_folds(
        run_dir,
        result.valid_folds,
        front_col=context.group_column,
        id_col=context.item_column,
        n_splits=n_splits,
        tie_seed=seed,
    )
    _write_json(
        feature_columns_path,
        {
            "feature_set": FEATURE_SET_NAME,
            "n_features": len(result.feature_columns),
            "feature_columns": result.feature_columns,
        },
    )
    _write_json(
        cv_report_path,
        _fold_reports_payload(
            result.fold_reports,
            run_config=run_config,
            context=context,
            seed=seed,
            n_splits=n_splits,
            model_params=model_params,
        ),
    )
    manifest_payload = run_manifest(
        context,
        PHASE_NAME,
        run_config.run_id,
        seed=seed,
        status="completed",
        parameters={
            **run_config.to_manifest_parameters(model_params=model_params),
            "n_splits": n_splits,
            "ks": list(DEFAULT_EVALUATION_KS),
            "development_fronts": context.n_development_fronts,
            "development_rows": context.n_development_rows,
        },
        outputs={
            "cv_report": repo_relative(cv_report_path),
            "feature_columns": repo_relative(feature_columns_path),
            "valid_fold_ranked": [repo_relative(path) for path in ranked_paths],
        },
    )
    write_manifest(run_manifest_path, manifest_payload, options)
    return ScreeningRunResult(
        config=run_config,
        run_dir=run_dir,
        cv_report=cv_report_path,
        feature_columns=feature_columns_path,
        valid_fold_ranked=ranked_paths,
        run_manifest=run_manifest_path,
        status="written",
    )


def _full_feature_columns(context: CaseContext) -> list[str]:
    columns = list(context.feature_sets.get(FEATURE_SET_NAME, []))
    if not columns:
        raise ModelScreeningError(f"feature set '{FEATURE_SET_NAME}' is not defined in the feature contract")
    missing = [column for column in columns if column not in context.development_df.columns]
    if missing:
        raise ModelScreeningError(f"development data is missing full feature columns: {missing}")
    return columns


def _training_control_columns(context: CaseContext) -> list[str]:
    control_columns = [context.group_column, context.item_column, context.target_column]
    missing = [column for column in control_columns if column not in context.development_df.columns]
    if missing:
        raise ModelScreeningError(f"development data is missing required control columns: {missing}")
    return control_columns


def _development_training_frame(
    context: CaseContext,
    control_columns: Sequence[str],
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    columns = list(dict.fromkeys([*control_columns, *feature_columns]))
    frame = context.development_df.loc[:, columns].copy().reset_index(drop=True)
    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame[context.target_column] = pd.to_numeric(frame[context.target_column], errors="raise")
    return frame


def _model_params_for_run(
    model_type: ModelType,
    model_params: Mapping[str, Any] | None,
) -> dict[str, Any]:
    params = dict(REFERENCE_MODEL_PARAMS.get(model_type, {}))
    if model_params is None:
        overrides: dict[str, Any] = {}
    elif any(model.value in model_params for model in ModelType):
        nested = model_params.get(model_type.value, {})
        if nested is not None and not isinstance(nested, AbcMapping):
            raise ModelScreeningError(
                f"model params for '{model_type.value}' must be a JSON object"
            )
        overrides = dict(nested or {})
    else:
        overrides = dict(model_params)

    params.update(overrides)
    if model_type == ModelType.CatBoostRanker:
        params["allow_writing_files"] = False
    return params


def _write_ranked_validation_folds(
    run_dir: Path,
    valid_folds: Sequence[pd.DataFrame],
    *,
    front_col: str,
    id_col: str,
    n_splits: int,
    tie_seed: int,
) -> tuple[Path, ...]:
    if len(valid_folds) != n_splits:
        raise ModelScreeningError(f"expected {n_splits} validation folds, got {len(valid_folds)}")
    paths: list[Path] = []
    for fold_idx, fold_df in enumerate(valid_folds, start=1):
        ranked = assign_rank_in_front(
            fold_df,
            front_col=front_col,
            score_col="score",
            id_col=id_col,
            rank_col="rank_in_front",
            tie_seed=tie_seed + fold_idx,
        )
        path = run_dir / f"valid_fold{fold_idx}_ranked.csv"
        ranked.to_csv(path, index=False)
        paths.append(path)
    return tuple(paths)


def _fold_reports_payload(
    fold_reports: Sequence[TrainFoldReport],
    *,
    run_config: ScreeningRunConfig,
    context: CaseContext,
    seed: int,
    n_splits: int,
    model_params: Mapping[str, Any],
) -> list[dict[str, Any]]:
    meta = {
        "model": run_config.run_id,
        "model_type": run_config.model_type.value,
        "label_mode": run_config.label_mode.value,
        "label_quantiles": run_config.label_quantiles,
        "feature_set": FEATURE_SET_NAME,
        "selectable": run_config.selectable,
        "phase": PHASE_NAME,
        "evaluation_split": "development_cv",
        "front_col": context.group_column,
        "target_col": context.target_column,
        "id_col": context.item_column,
        "score_col": "score",
        "random_state": seed,
        "n_splits": n_splits,
        "ks": list(DEFAULT_EVALUATION_KS),
        "model_params": dict(model_params),
    }
    payload = []
    for report in fold_reports:
        payload.append(
            {
                "fold": int(report.fold),
                "agg": report.agg,
                "groups": _normalize_groups(report.groups, front_col=context.group_column),
                "label_mode": report.label_mode,
                "label_quantiles": report.label_quantiles,
                "meta": meta,
            }
        )
    return _json_safe(payload)


def _normalize_groups(groups: list[dict[str, Any]], *, front_col: str) -> list[dict[str, Any]]:
    if front_col == "front_id":
        return groups
    normalized = []
    for group in groups:
        group_copy = dict(group)
        group_copy["front_id"] = group_copy.pop(front_col)
        normalized.append(group_copy)
    return normalized


def _build_shortlist_payload(
    context: CaseContext,
    *,
    run_configs: Sequence[ScreeningRunConfig],
    run_results: Sequence[ScreeningRunResult],
    summary_path: Path,
    seed: int,
) -> dict[str, Any]:
    selection_payload = _select_best_labeling_per_model_type(
        run_configs=run_configs,
        run_results=run_results,
        summary_path=summary_path,
    )

    return {
        **common_manifest_fields(context, seed=seed),
        "manifest_type": "shortlisted_configs",
        "phase": PHASE_NAME,
        "primary_metric": selection_payload["selection_rule"]["primary_metric"],
        "tie_breakers": selection_payload["selection_rule"]["tie_breakers"],
        **selection_payload,
    }


def _write_primary_tables(
    *,
    run_configs: Sequence[ScreeningRunConfig],
    run_results: Sequence[ScreeningRunResult],
    summary_outputs: Mapping[str, Path],
    summary_dir: Path,
) -> dict[str, Path]:
    primary_rank_table = write_grouped_primary_rank_table(
        _screening_candidate_records(run_configs),
        {result.config.run_id: result.cv_report for result in run_results},
        summary_outputs["metrics_summary"],
        summary_dir / "primary_rank_table.csv",
        group_field="model_type",
        output_group_column="model_type",
    )
    return {"primary_rank_table": primary_rank_table}


def _select_best_labeling_per_model_type(
    *,
    run_configs: Sequence[ScreeningRunConfig],
    run_results: Sequence[ScreeningRunResult],
    summary_path: Path,
) -> dict[str, Any]:
    """Select the best label formulation independently within each model family."""
    candidate_records = _screening_candidate_records(run_configs)
    cv_reports = {result.config.run_id: result.cv_report for result in run_results}
    selected_configs: list[dict[str, Any]] = []
    excluded_configs: list[dict[str, Any]] = []
    rank_stats_frames: list[pd.DataFrame] = []
    selection_groups: list[dict[str, Any]] = []
    selection_rule: dict[str, Any] | None = None

    for model_type in _model_type_order(run_configs):
        group_candidates = [
            candidate
            for candidate in candidate_records
            if candidate["model_type"] == model_type
        ]
        if not group_candidates:
            continue
        rank_stats = selection_rank_stats_from_reports(group_candidates, cv_reports)
        rank_stats = rank_stats.copy()
        rank_stats["selection_group"] = model_type
        rank_stats_frames.append(rank_stats)
        selection = select_configurations(
            group_candidates,
            summary_path,
            metric_rank_stats=rank_stats,
            selection_size=SHORTLIST_SELECTED_PER_GROUP,
            criteria=LABEL_SELECTION_CRITERIA,
        )
        selection_rule = selection.selection_rule
        selected = [
            {
                **config,
                "selection_scope": SHORTLIST_SELECTION_SCOPE,
                "selection_group_field": SHORTLIST_GROUP_FIELD,
                "selection_group": model_type,
            }
            for config in selection.selected_configs
        ]
        excluded = [
            {
                **config,
                "selection_scope": SHORTLIST_SELECTION_SCOPE,
                "selection_group_field": SHORTLIST_GROUP_FIELD,
                "selection_group": model_type,
            }
            for config in selection.excluded_configs
        ]
        selected_configs.extend(selected)
        excluded_configs.extend(excluded)
        selection_groups.append(
            {
                "model_type": model_type,
                "selectable_config_count": _selectable_count(group_candidates),
                "selected_run_id": selected[0]["run_id"],
                "selected_label_mode": selected[0]["label_mode"],
                "selected_label_quantiles": selected[0].get("label_quantiles"),
            }
        )

    if not selected_configs or selection_rule is None:
        raise ConfigSelectionError("no selectable model-specific label configurations were selected")

    selection_rule = {
        **selection_rule,
        "selection_scope": SHORTLIST_SELECTION_SCOPE,
        "selection_group_field": SHORTLIST_GROUP_FIELD,
        "selected_per_group": SHORTLIST_SELECTED_PER_GROUP,
        "description": (
            "Phase 01 selects the best label formulation independently within "
            "each model family, so every model family advances to tuning."
        ),
    }
    rank_stats = pd.concat(rank_stats_frames, ignore_index=True) if rank_stats_frames else pd.DataFrame()
    return {
        "selection_scope": SHORTLIST_SELECTION_SCOPE,
        "selection_group_field": SHORTLIST_GROUP_FIELD,
        "selected_per_group": SHORTLIST_SELECTED_PER_GROUP,
        "shortlist_count": len(selected_configs),
        "selection_rule": selection_rule,
        "selection_metric_rank_stats": _json_safe(rank_stats.to_dict(orient="records")),
        "selection_groups": _json_safe(selection_groups),
        "configs": _json_safe(selected_configs),
        "excluded_configs": _json_safe(excluded_configs),
    }


def _model_type_order(run_configs: Sequence[ScreeningRunConfig]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for config in run_configs:
        model_type = config.model_type.value
        if model_type not in seen:
            seen.add(model_type)
            ordered.append(model_type)
    return tuple(ordered)


def _selectable_count(candidates: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        1
        for candidate in candidates
        if bool(candidate.get("selectable", True))
    )


def _screening_candidate_records(run_configs: Sequence[ScreeningRunConfig]) -> list[dict[str, Any]]:
    return [
        {
            "run_id": config.run_id,
            "model_type": config.model_type.value,
            "label_mode": config.label_mode.value,
            "label_quantiles": config.label_quantiles,
            "feature_set": FEATURE_SET_NAME,
            "selectable": config.selectable,
        }
        for config in run_configs
    ]


def _screening_manifest_payload(
    context: CaseContext,
    *,
    phase_dir: Path,
    run_results: Sequence[ScreeningRunResult],
    summary_outputs: Mapping[str, Path],
    table_outputs: Mapping[str, Path],
    shortlist_path: Path,
    shortlist_status: str,
    seed: int,
    n_splits: int,
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    return {
        **common_manifest_fields(context, seed=seed),
        "manifest_type": "screening",
        "phase": PHASE_NAME,
        "status": "completed",
        "n_splits": int(n_splits),
        "feature_set": FEATURE_SET_NAME,
        "n_features": len(feature_columns),
        "primary_metric": PRIMARY_METRIC,
        "shortlist_selection_scope": SHORTLIST_SELECTION_SCOPE,
        "shortlist_group_field": SHORTLIST_GROUP_FIELD,
        "selected_per_group": SHORTLIST_SELECTED_PER_GROUP,
        "run_count": len(run_results),
        "runs": [
            {
                "run_id": result.config.run_id,
                "model_type": result.config.model_type.value,
                "label_mode": result.config.label_mode.value,
                "label_quantiles": result.config.label_quantiles,
                "selectable": result.config.selectable,
                "status": result.status,
                "outputs": {
                    "cv_report": repo_relative(result.cv_report),
                    "feature_columns": repo_relative(result.feature_columns),
                    "valid_fold_ranked": [repo_relative(path) for path in result.valid_fold_ranked],
                    "run_manifest": repo_relative(result.run_manifest),
                },
            }
            for result in run_results
        ],
        "summary_outputs": {
            name: repo_relative(path) for name, path in summary_outputs.items()
        },
        "tables": {
            name: repo_relative(path) for name, path in table_outputs.items()
        },
        "shortlisted_configs": {
            "path": repo_relative(shortlist_path),
            "status": shortlist_status,
        },
        "plots": planned_phase_plot_outputs(phase_dir, PHASE_NAME),
    }


def _run_id(
    *,
    model_type: ModelType,
    label_mode: LabelMode,
    label_quantiles: int | None,
) -> str:
    base = f"{model_type.value}__{label_mode.value}"
    if label_mode == LabelMode.QUANTILES:
        return f"{base}_q{label_quantiles}"
    return base


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
