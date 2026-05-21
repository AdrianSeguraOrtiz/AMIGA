"""Phase 02: hyperparameter tuning for shortlisted AMIGA configurations."""

from __future__ import annotations

from collections.abc import Mapping as AbcMapping
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from amiga.core.main import TrainFoldReport, train_ltr_cv
from amiga.selection.learn2rank import LabelMode, ModelType, assign_rank_in_front
from scripts.experiments.amiga_exp.config_selection import (
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
    write_primary_rank_table,
)
from scripts.experiments.amiga_exp.plots import planned_phase_plot_outputs
from scripts.experiments.amiga_exp.phases.phase_01_model_screening import (
    DEFAULT_LABEL_QUANTILES,
    FEATURE_SET_NAME,
    parse_label_mode,
    parse_model_type,
)
from scripts.experiments.amiga_exp.reports import (
    ReportSummaryError,
    summarize_phase_cv_reports,
)


PHASE_NAME = "02_hyperparameter_tuning"
SOURCE_PHASE_NAME = "01_model_screening"
DEFAULT_SELECTION_SIZE = 1
DEFAULT_LGBM_NUM_LEAVES = (31, 63, 127)
DEFAULT_LGBM_MIN_CHILD_SAMPLES = (30, 50, 100)
DEFAULT_LGBM_LEARNING_RATES = (0.03, 0.05, 0.10)
DEFAULT_XGB_MAX_DEPTH = (4, 6, 8)
DEFAULT_XGB_SUBSAMPLE = (0.8, 1.0)
DEFAULT_XGB_MIN_CHILD_WEIGHT = (1, 5, 10)
DEFAULT_XGB_LEARNING_RATES = (0.03, 0.05)
DEFAULT_CATBOOST_DEPTH = (4, 6, 8)
DEFAULT_CATBOOST_L2_LEAF_REG = (3, 5, 7, 10)
DEFAULT_CATBOOST_LEARNING_RATES = (0.03, 0.05, 0.10)


class HyperparameterTuningError(ValueError):
    """Raised when phase 02 cannot be planned or executed safely."""


@dataclass(frozen=True)
class TuningParameterSet:
    """One model-specific hyperparameter setting."""

    tag: str
    model_params: dict[str, Any]


@dataclass(frozen=True)
class TuningRunConfig:
    """One hyperparameter-tuning run."""

    run_id: str
    base_run_id: str
    model_type: ModelType
    label_mode: LabelMode
    label_quantiles: int | None
    feature_set: str
    model_params: dict[str, Any]
    param_tag: str

    def to_candidate_record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "base_run_id": self.base_run_id,
            "model_type": self.model_type.value,
            "label_mode": self.label_mode.value,
            "label_quantiles": self.label_quantiles,
            "feature_set": self.feature_set,
            "model_params": self.model_params,
            "param_tag": self.param_tag,
            "selectable": True,
        }


@dataclass(frozen=True)
class TuningRunResult:
    """Artifacts produced or reused for one tuning run."""

    config: TuningRunConfig
    run_dir: Path
    cv_report: Path
    feature_columns: Path
    valid_fold_ranked: tuple[Path, ...]
    run_manifest: Path
    status: str


@dataclass(frozen=True)
class HyperparameterTuningResult:
    """Phase-level result for hyperparameter tuning."""

    phase_dir: Path
    run_results: tuple[TuningRunResult, ...]
    summary_outputs: dict[str, Path]
    table_outputs: dict[str, Path]
    tuning_manifest: Path
    selected_config: Path
    status: str


def run_hyperparameter_tuning(
    context: CaseContext,
    *,
    seed: int = 42,
    options: WriteOptions | None = None,
    shortlist_path: Path | None = None,
    n_splits: int = 5,
    tuning_grids: Mapping[str | ModelType, Sequence[Mapping[str, Any] | TuningParameterSet]] | None = None,
    selection_size: int = DEFAULT_SELECTION_SIZE,
) -> HyperparameterTuningResult:
    """Run phase 02 using only development fronts and the phase-01 shortlist."""
    options = options or WriteOptions()
    options.validate()
    if n_splits < 2:
        raise HyperparameterTuningError("n_splits must be >= 2")
    if context.n_development_fronts < n_splits:
        raise HyperparameterTuningError(
            f"n_splits={n_splits} is larger than development fronts ({context.n_development_fronts})"
        )
    if selection_size < 1:
        raise HyperparameterTuningError("selection_size must be >= 1")

    layout = results_layout(context)
    phase_dir = layout.phases[PHASE_NAME]
    runs_dir = phase_dir / "runs"
    summary_dir = phase_dir / "summary"
    resolved_shortlist_path = shortlist_path or (
        layout.phases[SOURCE_PHASE_NAME] / "shortlisted_configs.json"
    )
    shortlisted_configs = load_shortlisted_configs(resolved_shortlist_path)
    run_configs = build_tuning_run_configs(shortlisted_configs, tuning_grids=tuning_grids)

    if options.dry_run:
        return HyperparameterTuningResult(
            phase_dir=phase_dir,
            run_results=tuple(
                TuningRunResult(
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
            tuning_manifest=phase_dir / "tuning_manifest.json",
            selected_config=phase_dir / "selected_config.json",
            status="dry_run",
        )

    ensure_results_layout(layout, dry_run=False)
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = _full_feature_columns(context)
    control_columns = _training_control_columns(context)
    development_df = _development_training_frame(context, control_columns, feature_columns)

    run_results: list[TuningRunResult] = []
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
            options=options,
        )
        run_results.append(run_result)
        report_paths.append(run_result.cv_report)

    try:
        summary_outputs = summarize_phase_cv_reports(report_paths, summary_dir)
    except ReportSummaryError as exc:
        raise HyperparameterTuningError(str(exc)) from exc
    try:
        table_outputs = _write_primary_tables(
            run_configs=run_configs,
            run_results=run_results,
            summary_outputs=summary_outputs,
            summary_dir=summary_dir,
        )
    except PrimaryRankTableError as exc:
        raise HyperparameterTuningError(str(exc)) from exc

    try:
        selected_payload = _build_selected_config_payload(
            context,
            run_configs=run_configs,
            run_results=run_results,
            summary_path=summary_outputs["metrics_summary"],
            shortlist_path=resolved_shortlist_path,
            selection_size=selection_size,
            seed=seed,
        )
    except ConfigSelectionError as exc:
        raise HyperparameterTuningError(str(exc)) from exc
    selected_status = write_manifest(phase_dir / "selected_config.json", selected_payload, options)

    tuning_payload = _tuning_manifest_payload(
        context,
        phase_dir=phase_dir,
        run_results=run_results,
        summary_outputs=summary_outputs,
        table_outputs=table_outputs,
        shortlist_path=resolved_shortlist_path,
        selected_config_path=phase_dir / "selected_config.json",
        selected_status=selected_status,
        seed=seed,
        n_splits=n_splits,
        feature_columns=feature_columns,
    )
    manifest_status = write_manifest(phase_dir / "tuning_manifest.json", tuning_payload, options)
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
            "tuning_manifest": repo_relative(phase_dir / "tuning_manifest.json"),
            "tuning_manifest_status": manifest_status,
            "selected_config": repo_relative(phase_dir / "selected_config.json"),
            "selected_config_status": selected_status,
            "plots": planned_phase_plot_outputs(phase_dir, PHASE_NAME),
        },
        options=options,
    )
    return HyperparameterTuningResult(
        phase_dir=phase_dir,
        run_results=tuple(run_results),
        summary_outputs=summary_outputs,
        table_outputs=table_outputs,
        tuning_manifest=phase_dir / "tuning_manifest.json",
        selected_config=phase_dir / "selected_config.json",
        status="written" if manifest_status == "written" else manifest_status,
    )


def load_shortlisted_configs(path: Path) -> list[dict[str, Any]]:
    """Load phase-01 shortlisted configs."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HyperparameterTuningError(f"shortlisted configs file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise HyperparameterTuningError(f"invalid shortlisted configs JSON: {path}") from exc

    configs = payload.get("configs") if isinstance(payload, dict) else None
    if not isinstance(configs, list) or not configs:
        raise HyperparameterTuningError(f"shortlisted configs JSON has no non-empty 'configs' list: {path}")
    return [dict(config) for config in configs]


def build_tuning_run_configs(
    shortlisted_configs: Sequence[Mapping[str, Any]],
    *,
    tuning_grids: Mapping[str | ModelType, Sequence[Mapping[str, Any] | TuningParameterSet]] | None = None,
) -> tuple[TuningRunConfig, ...]:
    """Expand shortlisted phase-01 configs into phase-02 tuning runs."""
    run_configs: list[TuningRunConfig] = []
    seen_run_ids: set[str] = set()
    for shortlisted in shortlisted_configs:
        base_run_id = str(shortlisted.get("run_id") or "")
        if not base_run_id:
            raise HyperparameterTuningError("shortlisted config is missing 'run_id'")
        model_type = parse_model_type(str(shortlisted.get("model_type", "")))
        label_mode = parse_label_mode(str(shortlisted.get("label_mode", "")))
        feature_set = str(shortlisted.get("feature_set") or FEATURE_SET_NAME)
        if feature_set != FEATURE_SET_NAME:
            raise HyperparameterTuningError(
                f"phase 02 currently supports feature_set='{FEATURE_SET_NAME}', got '{feature_set}'"
            )
        if not bool(shortlisted.get("selectable", True)):
            raise HyperparameterTuningError(f"shortlisted config '{base_run_id}' is not selectable")

        label_quantiles = shortlisted.get("label_quantiles")
        label_quantiles = None if label_quantiles is None else int(label_quantiles)
        for parameter_set in parameter_sets_for_model(model_type, tuning_grids=tuning_grids):
            run_id = f"{base_run_id}__{parameter_set.tag}"
            if run_id in seen_run_ids:
                raise HyperparameterTuningError(f"duplicated tuning run id: {run_id}")
            seen_run_ids.add(run_id)
            run_configs.append(
                TuningRunConfig(
                    run_id=run_id,
                    base_run_id=base_run_id,
                    model_type=model_type,
                    label_mode=label_mode,
                    label_quantiles=label_quantiles,
                    feature_set=feature_set,
                    model_params=parameter_set.model_params,
                    param_tag=parameter_set.tag,
                )
            )

    if not run_configs:
        raise HyperparameterTuningError("tuning grid is empty")
    return tuple(run_configs)


def parameter_sets_for_model(
    model_type: ModelType,
    *,
    tuning_grids: Mapping[str | ModelType, Sequence[Mapping[str, Any] | TuningParameterSet]] | None = None,
) -> tuple[TuningParameterSet, ...]:
    """Return custom or default parameter sets for one model type."""
    if tuning_grids is not None:
        custom = _custom_grid_for_model(model_type, tuning_grids)
        if custom is not None:
            return _normalize_custom_grid(custom)

    if model_type == ModelType.LGBMRanker:
        return tuple(
            TuningParameterSet(
                tag=f"nl{num_leaves}_mcs{min_child_samples}_lr{_float_tag(learning_rate)}",
                model_params={
                    "num_leaves": num_leaves,
                    "min_child_samples": min_child_samples,
                    "learning_rate": learning_rate,
                    "n_estimators": 3000,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                },
            )
            for num_leaves, min_child_samples, learning_rate in product(
                DEFAULT_LGBM_NUM_LEAVES,
                DEFAULT_LGBM_MIN_CHILD_SAMPLES,
                DEFAULT_LGBM_LEARNING_RATES,
            )
        )

    if model_type == ModelType.XGBRanker:
        return tuple(
            TuningParameterSet(
                tag=f"md{max_depth}_ss{_float_tag(subsample)}_mcw{min_child_weight}_lr{_float_tag(learning_rate)}",
                model_params={
                    "max_depth": max_depth,
                    "subsample": subsample,
                    "min_child_weight": min_child_weight,
                    "learning_rate": learning_rate,
                    "n_estimators": 3000,
                    "colsample_bytree": 0.8,
                },
            )
            for max_depth, subsample, min_child_weight, learning_rate in product(
                DEFAULT_XGB_MAX_DEPTH,
                DEFAULT_XGB_SUBSAMPLE,
                DEFAULT_XGB_MIN_CHILD_WEIGHT,
                DEFAULT_XGB_LEARNING_RATES,
            )
        )

    if model_type == ModelType.CatBoostRanker:
        return tuple(
            TuningParameterSet(
                tag=f"d{depth}_l2{l2_leaf_reg}_lr{_float_tag(learning_rate)}",
                model_params={
                    "depth": depth,
                    "l2_leaf_reg": l2_leaf_reg,
                    "learning_rate": learning_rate,
                    "iterations": 3000,
                    "allow_writing_files": False,
                },
            )
            for depth, l2_leaf_reg, learning_rate in product(
                DEFAULT_CATBOOST_DEPTH,
                DEFAULT_CATBOOST_L2_LEAF_REG,
                DEFAULT_CATBOOST_LEARNING_RATES,
            )
        )

    raise HyperparameterTuningError(f"unsupported model type for tuning: {model_type.value}")


def _custom_grid_for_model(
    model_type: ModelType,
    tuning_grids: Mapping[str | ModelType, Sequence[Mapping[str, Any] | TuningParameterSet]],
) -> Sequence[Mapping[str, Any] | TuningParameterSet] | None:
    for key in (model_type, model_type.value):
        if key in tuning_grids:
            return tuning_grids[key]
    return None


def _normalize_custom_grid(
    grid: Sequence[Mapping[str, Any] | TuningParameterSet],
) -> tuple[TuningParameterSet, ...]:
    if not grid:
        raise HyperparameterTuningError("custom tuning grid is empty")
    normalized: list[TuningParameterSet] = []
    for idx, item in enumerate(grid, start=1):
        if isinstance(item, TuningParameterSet):
            normalized.append(item)
            continue
        if not isinstance(item, AbcMapping):
            raise HyperparameterTuningError("custom tuning grid entries must be JSON objects")
        item_dict = dict(item)
        tag = str(item_dict.pop("tag", f"params{idx}"))
        model_params_value = item_dict.pop("model_params", item_dict)
        if not isinstance(model_params_value, AbcMapping):
            raise HyperparameterTuningError("custom tuning grid 'model_params' must be an object")
        normalized.append(TuningParameterSet(tag=tag, model_params=dict(model_params_value)))
    return tuple(normalized)


def _execute_or_reuse_run(
    context: CaseContext,
    run_config: TuningRunConfig,
    *,
    development_df: pd.DataFrame,
    expected_feature_columns: Sequence[str],
    runs_dir: Path,
    seed: int,
    n_splits: int,
    options: WriteOptions,
) -> TuningRunResult:
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
            raise HyperparameterTuningError(
                f"cannot skip incomplete existing run '{run_config.run_id}'; missing: {missing}"
            )
        return TuningRunResult(
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
    model_params = _model_params_for_training(run_config)
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
        model_params=model_params,
    )
    if list(result.feature_columns) != list(expected_feature_columns):
        raise HyperparameterTuningError(
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
    write_manifest(
        run_manifest_path,
        run_manifest(
            context,
            PHASE_NAME,
            run_config.run_id,
            seed=seed,
            status="completed",
            parameters={
                **run_config.to_candidate_record(),
                "model_params": model_params,
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
        ),
        options,
    )
    return TuningRunResult(
        config=run_config,
        run_dir=run_dir,
        cv_report=cv_report_path,
        feature_columns=feature_columns_path,
        valid_fold_ranked=ranked_paths,
        run_manifest=run_manifest_path,
        status="written",
    )


def _model_params_for_training(run_config: TuningRunConfig) -> dict[str, Any]:
    params = dict(run_config.model_params)
    if run_config.model_type == ModelType.CatBoostRanker:
        params["allow_writing_files"] = False
    return params


def _full_feature_columns(context: CaseContext) -> list[str]:
    columns = list(context.feature_sets.get(FEATURE_SET_NAME, []))
    if not columns:
        raise HyperparameterTuningError(f"feature set '{FEATURE_SET_NAME}' is not defined in the feature contract")
    missing = [column for column in columns if column not in context.development_df.columns]
    if missing:
        raise HyperparameterTuningError(f"development data is missing full feature columns: {missing}")
    return columns


def _training_control_columns(context: CaseContext) -> list[str]:
    control_columns = [context.group_column, context.item_column, context.target_column]
    missing = [column for column in control_columns if column not in context.development_df.columns]
    if missing:
        raise HyperparameterTuningError(f"development data is missing required control columns: {missing}")
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
        raise HyperparameterTuningError(f"expected {n_splits} validation folds, got {len(valid_folds)}")
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
    run_config: TuningRunConfig,
    context: CaseContext,
    seed: int,
    n_splits: int,
    model_params: Mapping[str, Any],
) -> list[dict[str, Any]]:
    meta = {
        "model": run_config.run_id,
        "base_run_id": run_config.base_run_id,
        "model_type": run_config.model_type.value,
        "label_mode": run_config.label_mode.value,
        "label_quantiles": run_config.label_quantiles,
        "feature_set": run_config.feature_set,
        "model_params": dict(model_params),
        "param_tag": run_config.param_tag,
        "phase": PHASE_NAME,
        "evaluation_split": "development_cv",
        "front_col": context.group_column,
        "target_col": context.target_column,
        "id_col": context.item_column,
        "score_col": "score",
        "random_state": seed,
        "n_splits": n_splits,
        "ks": list(DEFAULT_EVALUATION_KS),
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


def _build_selected_config_payload(
    context: CaseContext,
    *,
    run_configs: Sequence[TuningRunConfig],
    run_results: Sequence[TuningRunResult],
    summary_path: Path,
    shortlist_path: Path,
    selection_size: int,
    seed: int,
) -> dict[str, Any]:
    candidate_records = [config.to_candidate_record() for config in run_configs]
    rank_stats = selection_rank_stats_from_reports(
        candidate_records,
        {result.config.run_id: result.cv_report for result in run_results},
    )
    selection = select_configurations(
        candidate_records,
        summary_path,
        metric_rank_stats=rank_stats,
        selection_size=selection_size,
    )
    selected_configs = _json_safe(selection.selected_configs)
    selected_config = selected_configs[0]
    return {
        **common_manifest_fields(context, seed=seed),
        "manifest_type": "selected_config",
        "phase": PHASE_NAME,
        "source_shortlist": repo_relative(shortlist_path),
        "primary_metric": selection.selection_rule["primary_metric"],
        "tie_breakers": selection.selection_rule["tie_breakers"],
        "selection_rule": selection.selection_rule,
        "selection_metric_rank_stats": _json_safe(rank_stats.to_dict(orient="records")),
        "selected_config": selected_config,
        "selected_configs": selected_configs,
        "excluded_configs": _json_safe(selection.excluded_configs),
    }


def _write_primary_tables(
    *,
    run_configs: Sequence[TuningRunConfig],
    run_results: Sequence[TuningRunResult],
    summary_outputs: Mapping[str, Path],
    summary_dir: Path,
) -> dict[str, Path]:
    primary_rank_table = write_primary_rank_table(
        [config.to_candidate_record() for config in run_configs],
        {result.config.run_id: result.cv_report for result in run_results},
        summary_outputs["metrics_summary"],
        summary_dir / "primary_rank_table.csv",
    )
    return {"primary_rank_table": primary_rank_table}


def _tuning_manifest_payload(
    context: CaseContext,
    *,
    phase_dir: Path,
    run_results: Sequence[TuningRunResult],
    summary_outputs: Mapping[str, Path],
    table_outputs: Mapping[str, Path],
    shortlist_path: Path,
    selected_config_path: Path,
    selected_status: str,
    seed: int,
    n_splits: int,
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    return {
        **common_manifest_fields(context, seed=seed),
        "manifest_type": "tuning",
        "phase": PHASE_NAME,
        "status": "completed",
        "n_splits": int(n_splits),
        "feature_set": FEATURE_SET_NAME,
        "n_features": len(feature_columns),
        "primary_metric": PRIMARY_METRIC,
        "source_shortlist": repo_relative(shortlist_path),
        "run_count": len(run_results),
        "runs": [
            {
                "run_id": result.config.run_id,
                "base_run_id": result.config.base_run_id,
                "model_type": result.config.model_type.value,
                "label_mode": result.config.label_mode.value,
                "label_quantiles": result.config.label_quantiles,
                "feature_set": result.config.feature_set,
                "param_tag": result.config.param_tag,
                "model_params": result.config.model_params,
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
        "selected_config": {
            "path": repo_relative(selected_config_path),
            "status": selected_status,
        },
        "plots": planned_phase_plot_outputs(phase_dir, PHASE_NAME),
    }


def _float_tag(value: float) -> str:
    return str(value).replace(".", "")


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
