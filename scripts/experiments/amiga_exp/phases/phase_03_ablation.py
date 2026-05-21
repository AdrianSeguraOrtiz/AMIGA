"""Phase 03: feature-set ablation using the frozen AMIGA configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from amiga.core.main import TrainFoldReport, rank_with_model, train_ltr_cv, train_ltr_full
from amiga.selection.learn2rank import ModelType, assign_rank_in_front
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
)
from scripts.experiments.amiga_exp.phases.phase_02_final_test import (
    FinalTestEvaluationError,
    SelectedConfig,
    load_selected_config,
)
from scripts.experiments.amiga_exp.reports import (
    ReportSummaryError,
    summarize_phase_cv_reports,
)


PHASE_NAME = "03_ablation"
SELECTED_CONFIG_PHASE = "02_hyperparameter_tuning"
DEFAULT_FEATURE_SET_ORDER = (
    "full",
    "objectives_only",
    "technique_weights_only",
    "expression_only",
    "network_only",
    "no_objectives",
    "no_technique_weights",
    "no_expression",
    "no_network",
)


class AblationError(ValueError):
    """Raised when phase 03 cannot be planned or executed safely."""


@dataclass(frozen=True)
class AblationRunConfig:
    """One frozen-protocol ablation variant."""

    feature_set: str
    feature_columns: tuple[str, ...]
    selected_config: SelectedConfig

    @property
    def run_id(self) -> str:
        return self.feature_set


@dataclass(frozen=True)
class AblationRunResult:
    """Development-CV artifacts for one ablation variant."""

    config: AblationRunConfig
    run_dir: Path
    cv_report: Path
    feature_columns: Path
    valid_fold_ranked: tuple[Path, ...]
    run_manifest: Path
    status: str


@dataclass(frozen=True)
class AblationFinalTestResult:
    """Held-out-test artifacts for one ablation variant."""

    config: AblationRunConfig
    final_test_dir: Path
    final_test_ranked: Path
    final_test_report: Path
    cv_report: Path
    status: str


@dataclass(frozen=True)
class AblationResult:
    """Phase-level result for ablation."""

    phase_dir: Path
    run_results: tuple[AblationRunResult, ...]
    final_test_results: tuple[AblationFinalTestResult, ...]
    summary_outputs: dict[str, Path]
    table_outputs: dict[str, Path]
    ablation_manifest: Path
    status: str


def run_ablation(
    context: CaseContext,
    *,
    seed: int = 42,
    options: WriteOptions | None = None,
    selected_config_path: Path | None = None,
    feature_sets: Sequence[str] | None = None,
    n_splits: int = 5,
) -> AblationResult:
    """Run feature-set ablations with the frozen selected model protocol."""
    options = options or WriteOptions()
    options.validate()
    if n_splits < 2:
        raise AblationError("n_splits must be >= 2")
    if context.n_development_fronts < n_splits:
        raise AblationError(
            f"n_splits={n_splits} is larger than development fronts ({context.n_development_fronts})"
        )

    layout = results_layout(context)
    phase_dir = layout.phases[PHASE_NAME]
    runs_dir = phase_dir / "runs"
    final_test_root = phase_dir / "final_test"
    summary_dir = phase_dir / "summary"
    resolved_selected_config_path = selected_config_path or (
        layout.phases[SELECTED_CONFIG_PHASE] / "selected_config.json"
    )
    try:
        selected_config = load_selected_config(resolved_selected_config_path)
    except FinalTestEvaluationError as exc:
        raise AblationError(str(exc)) from exc

    selected_feature_sets = resolve_ablation_feature_sets(context, feature_sets=feature_sets)
    run_configs = tuple(
        AblationRunConfig(
            feature_set=feature_set,
            feature_columns=tuple(_feature_columns(context, feature_set)),
            selected_config=selected_config,
        )
        for feature_set in selected_feature_sets
    )

    if options.dry_run:
        return AblationResult(
            phase_dir=phase_dir,
            run_results=tuple(
                AblationRunResult(
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
            final_test_results=tuple(
                AblationFinalTestResult(
                    config=config,
                    final_test_dir=final_test_root / config.run_id,
                    final_test_ranked=final_test_root / config.run_id / "final_test_ranked.csv",
                    final_test_report=final_test_root / config.run_id / "final_test_report.json",
                    cv_report=final_test_root / config.run_id / "cv_report.json",
                    status="planned",
                )
                for config in run_configs
            ),
            summary_outputs={},
            table_outputs={},
            ablation_manifest=phase_dir / "ablation_manifest.json",
            status="dry_run",
        )

    ensure_results_layout(layout, dry_run=False)
    runs_dir.mkdir(parents=True, exist_ok=True)
    final_test_root.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    run_results: list[AblationRunResult] = []
    report_paths: list[Path] = []
    final_test_results: list[AblationFinalTestResult] = []
    for run_config in run_configs:
        cv_result = _execute_or_reuse_cv_run(
            context,
            run_config,
            runs_dir=runs_dir,
            seed=seed,
            n_splits=n_splits,
            options=options,
        )
        run_results.append(cv_result)
        report_paths.append(cv_result.cv_report)
        final_test_results.append(
            _execute_or_reuse_final_test(
                context,
                run_config,
                final_test_root=final_test_root,
                selected_config_path=resolved_selected_config_path,
                seed=seed,
                options=options,
            )
        )

    try:
        summary_outputs = summarize_phase_cv_reports(report_paths, summary_dir)
    except ReportSummaryError as exc:
        raise AblationError(str(exc)) from exc
    try:
        table_outputs = _write_primary_tables(
            run_configs=run_configs,
            run_results=run_results,
            summary_outputs=summary_outputs,
            summary_dir=summary_dir,
        )
    except PrimaryRankTableError as exc:
        raise AblationError(str(exc)) from exc

    manifest_status = write_manifest(
        phase_dir / "ablation_manifest.json",
        _ablation_manifest_payload(
            context,
            phase_dir=phase_dir,
            run_results=run_results,
            final_test_results=final_test_results,
            summary_outputs=summary_outputs,
            table_outputs=table_outputs,
            selected_config=selected_config,
            selected_config_path=resolved_selected_config_path,
            seed=seed,
            n_splits=n_splits,
        ),
        options,
    )
    write_phase_status_manifest(
        phase_dir / "phase_manifest.json",
        context,
        PHASE_NAME,
        seed=seed,
        status="completed",
        outputs={
            "phase_dir": repo_relative(phase_dir),
            "run_count": len(run_results),
            "final_test_count": len(final_test_results),
            "summary_outputs": {
                name: repo_relative(path) for name, path in summary_outputs.items()
            },
            "table_outputs": {
                name: repo_relative(path) for name, path in table_outputs.items()
            },
            "ablation_manifest": repo_relative(phase_dir / "ablation_manifest.json"),
            "ablation_manifest_status": manifest_status,
            "plots": planned_phase_plot_outputs(phase_dir, PHASE_NAME),
        },
        options=options,
    )
    return AblationResult(
        phase_dir=phase_dir,
        run_results=tuple(run_results),
        final_test_results=tuple(final_test_results),
        summary_outputs=summary_outputs,
        table_outputs=table_outputs,
        ablation_manifest=phase_dir / "ablation_manifest.json",
        status="written" if manifest_status == "written" else manifest_status,
    )


def resolve_ablation_feature_sets(
    context: CaseContext,
    *,
    feature_sets: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Resolve requested or default ablation feature sets from the feature contract."""
    available = set(context.feature_sets)
    if feature_sets:
        selected = tuple(dict.fromkeys(str(name) for name in feature_sets))
        missing = [name for name in selected if name not in available]
        if missing:
            raise AblationError(f"unknown feature set(s): {missing}")
        return selected

    ordered = [name for name in DEFAULT_FEATURE_SET_ORDER if name in available]
    extra = sorted(available - set(ordered))
    selected = tuple([*ordered, *extra])
    if not selected:
        raise AblationError("no feature sets are defined in the feature contract")
    return selected


def _execute_or_reuse_cv_run(
    context: CaseContext,
    run_config: AblationRunConfig,
    *,
    runs_dir: Path,
    seed: int,
    n_splits: int,
    options: WriteOptions,
) -> AblationRunResult:
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
            raise AblationError(
                f"cannot skip incomplete existing ablation run '{run_config.run_id}'; missing: {missing}"
            )
        return AblationRunResult(
            config=run_config,
            run_dir=run_dir,
            cv_report=cv_report_path,
            feature_columns=feature_columns_path,
            valid_fold_ranked=expected_ranked,
            run_manifest=run_manifest_path,
            status="skipped",
        )

    if run_dir.exists() and any(run_dir.iterdir()) and not options.force:
        raise ManifestError(f"refusing to overwrite existing ablation run without --force: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    development_df = _split_training_frame(context.development_df, context, run_config.feature_columns)
    label_quantiles = run_config.selected_config.label_quantiles or DEFAULT_LABEL_QUANTILES[0]
    model_params = _model_params_for_training(run_config.selected_config)
    result = train_ltr_cv(
        development_df,
        model_type=run_config.selected_config.model_type,
        front_col=context.group_column,
        target_col=context.target_column,
        id_col=context.item_column,
        drop_cols=[],
        label_mode=run_config.selected_config.label_mode,
        label_quantiles=label_quantiles,
        n_splits=n_splits,
        random_state=seed,
        model_params=model_params,
    )
    if list(result.feature_columns) != list(run_config.feature_columns):
        raise AblationError(
            f"ablation run '{run_config.run_id}' used unexpected feature columns; "
            f"expected {len(run_config.feature_columns)}, got {len(result.feature_columns)}"
        )

    ranked_paths = _write_ranked_validation_folds(
        run_dir,
        result.valid_folds,
        front_col=context.group_column,
        id_col=context.item_column,
        n_splits=n_splits,
        tie_seed=seed,
    )
    _write_feature_columns(feature_columns_path, run_config)
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
                **_protocol_payload(run_config.selected_config),
                "feature_set": run_config.feature_set,
                "n_features": len(run_config.feature_columns),
                "n_splits": n_splits,
                "ks": list(DEFAULT_EVALUATION_KS),
                "model_selection": "not_performed",
            },
            outputs={
                "cv_report": repo_relative(cv_report_path),
                "feature_columns": repo_relative(feature_columns_path),
                "valid_fold_ranked": [repo_relative(path) for path in ranked_paths],
            },
        ),
        options,
    )
    return AblationRunResult(
        config=run_config,
        run_dir=run_dir,
        cv_report=cv_report_path,
        feature_columns=feature_columns_path,
        valid_fold_ranked=ranked_paths,
        run_manifest=run_manifest_path,
        status="written",
    )


def _execute_or_reuse_final_test(
    context: CaseContext,
    run_config: AblationRunConfig,
    *,
    final_test_root: Path,
    selected_config_path: Path,
    seed: int,
    options: WriteOptions,
) -> AblationFinalTestResult:
    final_test_dir = final_test_root / run_config.run_id
    outputs = _final_test_outputs(final_test_dir)

    if outputs["cv_report"].exists() and options.skip_existing:
        missing = [path for path in outputs.values() if not path.exists()]
        if missing:
            raise AblationError(
                f"cannot skip incomplete ablation final-test '{run_config.run_id}'; missing: {missing}"
            )
        return AblationFinalTestResult(
            config=run_config,
            final_test_dir=final_test_dir,
            final_test_ranked=outputs["final_test_ranked"],
            final_test_report=outputs["final_test_report"],
            cv_report=outputs["cv_report"],
            status="skipped",
        )

    if final_test_dir.exists() and any(final_test_dir.iterdir()) and not options.force:
        raise ManifestError(
            f"refusing to overwrite existing ablation final-test without --force: {final_test_dir}"
        )

    final_test_dir.mkdir(parents=True, exist_ok=True)
    train_df = _split_training_frame(context.development_df, context, run_config.feature_columns)
    test_df = _split_training_frame(context.test_df, context, run_config.feature_columns)
    label_quantiles = run_config.selected_config.label_quantiles or DEFAULT_LABEL_QUANTILES[0]
    model_params = _model_params_for_training(run_config.selected_config)
    fit = train_ltr_full(
        train_df,
        model_type=run_config.selected_config.model_type,
        front_col=context.group_column,
        target_col=context.target_column,
        id_col=context.item_column,
        drop_cols=[],
        label_mode=run_config.selected_config.label_mode,
        label_quantiles=label_quantiles,
        random_state=seed,
        model_params=model_params,
    )
    if list(fit.feature_columns) != list(run_config.feature_columns):
        raise AblationError(
            f"ablation final-test '{run_config.run_id}' used unexpected feature columns; "
            f"expected {len(run_config.feature_columns)}, got {len(fit.feature_columns)}"
        )

    ranked = rank_with_model(
        test_df,
        fit.model,
        front_col=context.group_column,
        id_col=context.item_column,
        feature_columns_hint=list(run_config.feature_columns),
    ).df_ranked
    ranked.to_csv(outputs["final_test_ranked"], index=False)

    meta = {
        "phase": PHASE_NAME,
        "evaluation_split": "test",
        "selected_config_path": repo_relative(selected_config_path),
        "selected_run_id": run_config.selected_config.run_id,
        "feature_set": run_config.feature_set,
        "n_features": len(run_config.feature_columns),
        "model_selection": "not_performed",
        **_protocol_payload(run_config.selected_config),
        "random_state": seed,
    }
    try:
        evaluation = evaluate_ranked_frame(
            ranked,
            front_col=context.group_column,
            target_col=context.target_column,
            score_col="score",
            model_name=run_config.run_id,
            evaluation_split="test",
            fold=1,
            ks=DEFAULT_EVALUATION_KS,
            meta=meta,
        )
    except RankingEvaluationError as exc:
        raise AblationError(str(exc)) from exc

    write_cv_report(outputs["cv_report"], evaluation.report)
    _write_json(
        outputs["final_test_report"],
        {
            **common_manifest_fields(context, seed=seed),
            "manifest_type": "ablation_final_test_report",
            "phase": PHASE_NAME,
            "status": "completed",
            "feature_set": run_config.feature_set,
            "n_features": len(run_config.feature_columns),
            "selected_config_path": repo_relative(selected_config_path),
            "model_selection": "not_performed",
            "protocol": _protocol_payload(run_config.selected_config),
            "agg": evaluation.agg,
            "groups": evaluation.groups,
            "per_front_metrics": evaluation.per_front_metrics.to_dict(orient="records"),
            "outputs": {name: repo_relative(path) for name, path in outputs.items()},
        },
    )
    return AblationFinalTestResult(
        config=run_config,
        final_test_dir=final_test_dir,
        final_test_ranked=outputs["final_test_ranked"],
        final_test_report=outputs["final_test_report"],
        cv_report=outputs["cv_report"],
        status="written",
    )


def _feature_columns(context: CaseContext, feature_set: str) -> list[str]:
    columns = list(context.feature_sets.get(feature_set, []))
    if not columns:
        raise AblationError(f"feature set '{feature_set}' is empty or undefined")
    missing = [column for column in columns if column not in context.df.columns]
    if missing:
        raise AblationError(f"feature set '{feature_set}' has missing columns: {missing}")
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
        raise AblationError(f"expected {n_splits} validation folds, got {len(valid_folds)}")
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
    run_config: AblationRunConfig,
    context: CaseContext,
    seed: int,
    n_splits: int,
    model_params: Mapping[str, Any],
) -> list[dict[str, Any]]:
    meta = {
        "model": run_config.run_id,
        "phase": PHASE_NAME,
        "evaluation_split": "development_cv",
        "feature_set": run_config.feature_set,
        "n_features": len(run_config.feature_columns),
        "model_selection": "not_performed",
        "front_col": context.group_column,
        "target_col": context.target_column,
        "id_col": context.item_column,
        "score_col": "score",
        "random_state": seed,
        "n_splits": n_splits,
        "ks": list(DEFAULT_EVALUATION_KS),
        **_protocol_payload(run_config.selected_config),
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


def _write_feature_columns(path: Path, run_config: AblationRunConfig) -> None:
    _write_json(
        path,
        {
            "feature_set": run_config.feature_set,
            "n_features": len(run_config.feature_columns),
            "feature_columns": list(run_config.feature_columns),
        },
    )


def _final_test_outputs(final_test_dir: Path) -> dict[str, Path]:
    return {
        "final_test_ranked": final_test_dir / "final_test_ranked.csv",
        "final_test_report": final_test_dir / "final_test_report.json",
        "cv_report": final_test_dir / "cv_report.json",
    }


def _protocol_payload(selected_config: SelectedConfig) -> dict[str, Any]:
    return {
        "selected_run_id": selected_config.run_id,
        "model_type": selected_config.model_type.value,
        "label_mode": selected_config.label_mode.value,
        "label_quantiles": selected_config.label_quantiles,
        "selected_feature_set": selected_config.feature_set,
        "model_params": selected_config.model_params,
    }


def _write_primary_tables(
    *,
    run_configs: Sequence[AblationRunConfig],
    run_results: Sequence[AblationRunResult],
    summary_outputs: Mapping[str, Path],
    summary_dir: Path,
) -> dict[str, Path]:
    primary_rank_table = write_primary_rank_table(
        _ablation_candidate_records(run_configs),
        {result.config.run_id: result.cv_report for result in run_results},
        summary_outputs["metrics_summary"],
        summary_dir / "primary_rank_table.csv",
    )
    return {"primary_rank_table": primary_rank_table}


def _ablation_candidate_records(run_configs: Sequence[AblationRunConfig]) -> list[dict[str, Any]]:
    return [
        {
            "run_id": config.run_id,
            "feature_set": config.feature_set,
            "selectable": True,
        }
        for config in run_configs
    ]


def _ablation_manifest_payload(
    context: CaseContext,
    *,
    phase_dir: Path,
    run_results: Sequence[AblationRunResult],
    final_test_results: Sequence[AblationFinalTestResult],
    summary_outputs: Mapping[str, Path],
    table_outputs: Mapping[str, Path],
    selected_config: SelectedConfig,
    selected_config_path: Path,
    seed: int,
    n_splits: int,
) -> dict[str, Any]:
    feature_set_audit = {
        result.config.feature_set: {
            "n_features": len(result.config.feature_columns),
            "feature_columns": list(result.config.feature_columns),
        }
        for result in run_results
    }
    expression_columns = list(context.feature_sets.get("expression_only", []))
    return {
        **common_manifest_fields(context, seed=seed),
        "manifest_type": "ablation",
        "phase": PHASE_NAME,
        "status": "completed",
        "n_splits": int(n_splits),
        "selected_config_path": repo_relative(selected_config_path),
        "protocol": _protocol_payload(selected_config),
        "model_selection": "not_performed",
        "feature_set_count": len(run_results),
        "feature_set_audit": feature_set_audit,
        "expression_only_audit": {
            "present": "expression_only" in context.feature_sets,
            "n_features": len(expression_columns),
            "feature_columns": expression_columns,
        },
        "runs": [
            {
                "feature_set": result.config.feature_set,
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
        "final_test": [
            {
                "feature_set": result.config.feature_set,
                "status": result.status,
                "outputs": {
                    "final_test_ranked": repo_relative(result.final_test_ranked),
                    "final_test_report": repo_relative(result.final_test_report),
                    "cv_report": repo_relative(result.cv_report),
                },
            }
            for result in final_test_results
        ],
        "summary_outputs": {
            name: repo_relative(path) for name, path in summary_outputs.items()
        },
        "tables": {
            name: repo_relative(path) for name, path in table_outputs.items()
        },
        "plots": planned_phase_plot_outputs(phase_dir, PHASE_NAME),
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
