"""Phase 04: non-learned decision baselines on held-out test fronts."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from amiga.selection.learn2rank import assign_rank_in_front
from scripts.experiments.amiga_exp.context import CaseContext
from scripts.experiments.amiga_exp.decision_baselines import (
    AUGMENTED_TCHEBYCHEFF_RHO,
    DecisionBaselineAdapterError,
    ObjectiveMatrixError,
    PYMCDM_ADAPTER,
    PYMCDM_PACKAGE,
    PYMCDM_VERSION,
    VALID_OBJECTIVE_DIRECTIONS,
    augmented_tchebycheff_scores_from_badness,
    ideal_l2_scores_from_badness,
    normalized_objective_badness_matrix,
    topsis_scores_from_badness,
    vikor_scores_from_badness,
    weighted_sum_scores_from_badness,
)
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
from scripts.experiments.amiga_exp.reports import (
    ReportSummaryError,
    summarize_phase_cv_reports,
)


PHASE_NAME = "04_decision_baselines"
AMIGA_REFERENCE_ID = "AMIGA_final"
DEFAULT_SEED = 42


class DecisionBaselineError(ValueError):
    """Raised when phase 04 cannot be planned or executed safely."""


@dataclass(frozen=True)
class BaselineSpec:
    """One decision baseline run specification."""

    baseline_id: str
    baseline_type: str
    objective: str | None = None
    method_family: str = "custom"
    implementation: str = "local_numpy_pandas"
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BaselineRunResult:
    """Artifacts produced or reused for one decision baseline."""

    baseline: BaselineSpec
    run_dir: Path
    ranked: Path
    cv_report: Path
    run_manifest: Path
    status: str


@dataclass(frozen=True)
class DecisionBaselinesResult:
    """Phase-level result for decision baselines."""

    phase_dir: Path
    run_results: tuple[BaselineRunResult, ...]
    summary_outputs: dict[str, Path]
    table_outputs: dict[str, Path]
    baseline_manifest: Path
    status: str


def run_decision_baselines(
    context: CaseContext,
    *,
    seed: int = DEFAULT_SEED,
    options: WriteOptions | None = None,
    amiga_final_dir: Path | None = None,
) -> DecisionBaselinesResult:
    """Run non-learned decision baselines on held-out test fronts."""
    options = options or WriteOptions()
    options.validate()

    validate_objective_directions(context)
    layout = results_layout(context)
    phase_dir = layout.phases[PHASE_NAME]
    runs_dir = phase_dir / "runs"
    summary_dir = phase_dir / "summary"
    resolved_amiga_final_dir = amiga_final_dir or (
        layout.phases["02_hyperparameter_tuning"] / "final_test"
    )
    _validate_amiga_final_outputs(context, resolved_amiga_final_dir)
    baseline_specs = build_baseline_specs(context)

    if options.dry_run:
        return DecisionBaselinesResult(
            phase_dir=phase_dir,
            run_results=tuple(
                BaselineRunResult(
                    baseline=spec,
                    run_dir=runs_dir / spec.baseline_id,
                    ranked=runs_dir / spec.baseline_id / "ranked.csv",
                    cv_report=runs_dir / spec.baseline_id / "cv_report.json",
                    run_manifest=runs_dir / spec.baseline_id / "run_manifest.json",
                    status="planned",
                )
                for spec in (_amiga_reference_spec(), *baseline_specs)
            ),
            summary_outputs={},
            table_outputs={},
            baseline_manifest=phase_dir / "baseline_manifest.json",
            status="dry_run",
        )

    ensure_results_layout(layout, dry_run=False)
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    run_results: list[BaselineRunResult] = []
    amiga_result = _copy_or_reuse_amiga_reference(
        context,
        runs_dir=runs_dir,
        amiga_final_dir=resolved_amiga_final_dir,
        seed=seed,
        options=options,
    )
    run_results.append(amiga_result)
    for spec in baseline_specs:
        run_results.append(
            _execute_or_reuse_baseline(
                context,
                spec,
                runs_dir=runs_dir,
                seed=seed,
                options=options,
            )
        )

    report_paths = [result.cv_report for result in run_results]
    try:
        summary_outputs = summarize_phase_cv_reports(report_paths, summary_dir)
    except ReportSummaryError as exc:
        raise DecisionBaselineError(str(exc)) from exc
    try:
        table_outputs = _write_primary_tables(
            run_results=run_results,
            summary_outputs=summary_outputs,
            summary_dir=summary_dir,
        )
    except PrimaryRankTableError as exc:
        raise DecisionBaselineError(str(exc)) from exc

    manifest_status = write_manifest(
        phase_dir / "baseline_manifest.json",
        _baseline_manifest_payload(
            context,
            phase_dir=phase_dir,
            run_results=run_results,
            summary_outputs=summary_outputs,
            table_outputs=table_outputs,
            amiga_final_dir=resolved_amiga_final_dir,
            seed=seed,
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
            "baseline_count": len(run_results),
            "summary_outputs": {
                name: repo_relative(path) for name, path in summary_outputs.items()
            },
            "table_outputs": {
                name: repo_relative(path) for name, path in table_outputs.items()
            },
            "baseline_manifest": repo_relative(phase_dir / "baseline_manifest.json"),
            "baseline_manifest_status": manifest_status,
            "plots": planned_phase_plot_outputs(phase_dir, PHASE_NAME),
        },
        options=options,
    )
    return DecisionBaselinesResult(
        phase_dir=phase_dir,
        run_results=tuple(run_results),
        summary_outputs=summary_outputs,
        table_outputs=table_outputs,
        baseline_manifest=phase_dir / "baseline_manifest.json",
        status="written" if manifest_status == "written" else manifest_status,
    )


def build_baseline_specs(context: CaseContext) -> tuple[BaselineSpec, ...]:
    """Build decision-baseline specs for one case."""
    if not context.objective_columns:
        raise DecisionBaselineError("no objective columns are defined in the feature contract")
    specs: list[BaselineSpec] = []
    for objective in context.objective_columns:
        specs.append(
            BaselineSpec(
                baseline_id=f"objective__{_slug(objective)}",
                baseline_type="single_objective",
                objective=objective,
                method_family="single_objective",
                implementation="local_numpy_pandas",
                parameters={
                    "objective": objective,
                    "objective_direction": context.objective_directions[objective],
                    "normalization": "none",
                    "score_formula": "larger_is_better_objective_score",
                },
            )
        )
    specs.extend(
        [
            BaselineSpec(
                "objective_mean_rank",
                "mean_rank",
                method_family="rank_aggregation",
                implementation="local_numpy_pandas",
                parameters={
                    "objective_columns": list(context.objective_columns),
                    "objective_directions": _objective_directions_payload(context),
                    "normalization": "per_front_objective_ranks",
                    "rank_method": "average",
                    "aggregation": "uniform_mean_rank",
                    "score_formula": "-mean_objective_rank",
                },
            ),
            BaselineSpec(
                "objective_normalized_mean",
                "normalized_mean",
                method_family="weighted_sum",
                implementation=PYMCDM_ADAPTER,
                parameters={
                    "package": PYMCDM_PACKAGE,
                    "package_version": PYMCDM_VERSION,
                    "package_method": "WSM",
                    "objective_columns": list(context.objective_columns),
                    "objective_directions": _objective_directions_payload(context),
                    "normalization": "per_front_minmax_badness",
                    "input_transform": "goodness = 1 - badness",
                    "weight_policy": "uniform",
                    "objective_weights": _uniform_objective_weights(context.objective_columns),
                    "aggregation": "weighted_sum_goodness",
                    "score_formula": "WSM(goodness, uniform_weights)",
                },
            ),
            BaselineSpec(
                "objective_ideal_l2",
                "ideal_l2",
                method_family="ideal_distance",
                implementation="local_numpy",
                parameters={
                    "objective_columns": list(context.objective_columns),
                    "objective_directions": _objective_directions_payload(context),
                    "normalization": "per_front_minmax_badness",
                    "weight_policy": "uniform",
                    "objective_weights": _uniform_objective_weights(context.objective_columns),
                    "distance": "weighted_l2_to_ideal",
                    "score_formula": "-sqrt(sum_i(weight_i * badness_i^2))",
                },
            ),
            BaselineSpec(
                "objective_topsis",
                "topsis",
                method_family="ideal_antiideal_distance",
                implementation=PYMCDM_ADAPTER,
                parameters={
                    "package": PYMCDM_PACKAGE,
                    "package_version": PYMCDM_VERSION,
                    "package_method": "TOPSIS",
                    "objective_columns": list(context.objective_columns),
                    "objective_directions": _objective_directions_payload(context),
                    "normalization": "per_front_minmax_badness",
                    "input_transform": "goodness = 1 - badness",
                    "weight_policy": "uniform",
                    "objective_weights": _uniform_objective_weights(context.objective_columns),
                    "positive_ideal": "all_ones_goodness",
                    "negative_ideal": "all_zeros_goodness",
                    "score_formula": "D_minus / (D_plus + D_minus)",
                },
            ),
            BaselineSpec(
                "objective_vikor",
                "vikor",
                method_family="compromise_ranking",
                implementation=PYMCDM_ADAPTER,
                parameters={
                    "package": PYMCDM_PACKAGE,
                    "package_version": PYMCDM_VERSION,
                    "package_method": "VIKOR",
                    "objective_columns": list(context.objective_columns),
                    "objective_directions": _objective_directions_payload(context),
                    "normalization": "per_front_minmax_badness",
                    "input_transform": "goodness = 1 - badness",
                    "weight_policy": "uniform",
                    "objective_weights": _uniform_objective_weights(context.objective_columns),
                    "v": 0.5,
                    "constant_criteria_policy": "drop_within_front",
                    "score_formula": "-Q, where Q = v * normalized(S) + (1 - v) * normalized(R)",
                },
            ),
            BaselineSpec(
                "objective_augmented_tchebycheff",
                "augmented_tchebycheff",
                method_family="augmented_tchebycheff",
                implementation="local_numpy",
                parameters={
                    "objective_columns": list(context.objective_columns),
                    "objective_directions": _objective_directions_payload(context),
                    "normalization": "per_front_minmax_badness",
                    "weight_policy": "uniform",
                    "objective_weights": _uniform_objective_weights(context.objective_columns),
                    "rho": AUGMENTED_TCHEBYCHEFF_RHO,
                    "score_formula": "-(max_i(weight_i * badness_i) + rho * sum_i(weight_i * badness_i))",
                },
            ),
            BaselineSpec(
                "objective_tchebycheff",
                "tchebycheff",
                method_family="tchebycheff",
                implementation="local_numpy_pandas",
                parameters={
                    "objective_columns": list(context.objective_columns),
                    "objective_directions": _objective_directions_payload(context),
                    "normalization": "per_front_minmax_badness",
                    "weight_policy": "uniform",
                    "objective_weights": _uniform_objective_weights(context.objective_columns),
                    "aggregation": "max_badness",
                    "score_formula": "-max(badness)",
                },
            ),
        ]
    )
    baseline_ids = [spec.baseline_id for spec in specs]
    duplicates = sorted({baseline_id for baseline_id in baseline_ids if baseline_ids.count(baseline_id) > 1})
    if duplicates:
        raise DecisionBaselineError(f"duplicated baseline id(s): {duplicates}")
    return tuple(specs)


def validate_objective_directions(context: CaseContext) -> None:
    """Validate declared objective directions."""
    missing = [column for column in context.objective_columns if column not in context.objective_directions]
    if missing:
        raise DecisionBaselineError(f"objective columns missing directions: {missing}")
    invalid = {
        column: direction
        for column, direction in context.objective_directions.items()
        if column in context.objective_columns and direction not in VALID_OBJECTIVE_DIRECTIONS
    }
    if invalid:
        raise DecisionBaselineError(f"invalid objective directions: {invalid}")


def score_baseline(
    df: pd.DataFrame,
    spec: BaselineSpec,
    *,
    objective_columns: Sequence[str],
    objective_directions: Mapping[str, str],
    front_col: str,
) -> pd.Series:
    """Compute baseline score where larger means better rank."""
    if spec.baseline_type == "single_objective":
        if spec.objective is None:
            raise DecisionBaselineError(f"single-objective baseline '{spec.baseline_id}' has no objective")
        return _objective_score(df[spec.objective], objective_directions[spec.objective])

    if spec.baseline_type == "mean_rank":
        rank_columns = [
            df.groupby(front_col)[objective].rank(
                method="average",
                ascending=objective_directions[objective] == "minimize",
            )
            for objective in objective_columns
        ]
        return -pd.concat(rank_columns, axis=1).mean(axis=1)

    try:
        badness_df = normalized_objective_badness_matrix(
            df,
            objective_columns=objective_columns,
            objective_directions=objective_directions,
            front_col=front_col,
        )
    except ObjectiveMatrixError as exc:
        raise DecisionBaselineError(str(exc)) from exc
    if spec.baseline_type == "normalized_mean":
        try:
            return weighted_sum_scores_from_badness(badness_df)
        except DecisionBaselineAdapterError as exc:
            raise DecisionBaselineError(str(exc)) from exc
    if spec.baseline_type == "ideal_l2":
        try:
            return ideal_l2_scores_from_badness(badness_df)
        except DecisionBaselineAdapterError as exc:
            raise DecisionBaselineError(str(exc)) from exc
    if spec.baseline_type == "topsis":
        try:
            return topsis_scores_from_badness(badness_df)
        except DecisionBaselineAdapterError as exc:
            raise DecisionBaselineError(str(exc)) from exc
    if spec.baseline_type == "vikor":
        try:
            return _score_vikor_by_front(badness_df, df[front_col])
        except DecisionBaselineAdapterError as exc:
            raise DecisionBaselineError(str(exc)) from exc
    if spec.baseline_type == "augmented_tchebycheff":
        try:
            return augmented_tchebycheff_scores_from_badness(badness_df)
        except DecisionBaselineAdapterError as exc:
            raise DecisionBaselineError(str(exc)) from exc
    if spec.baseline_type == "tchebycheff":
        return -badness_df.max(axis=1)

    raise DecisionBaselineError(f"unsupported baseline type: {spec.baseline_type}")


def _score_vikor_by_front(badness_df: pd.DataFrame, front_ids: pd.Series) -> pd.Series:
    scores = pd.Series(index=badness_df.index, dtype=float)
    for _, front_index in front_ids.groupby(front_ids, sort=False).groups.items():
        front_badness = badness_df.loc[front_index]
        scores.loc[front_index] = vikor_scores_from_badness(front_badness)
    return scores


def _copy_or_reuse_amiga_reference(
    context: CaseContext,
    *,
    runs_dir: Path,
    amiga_final_dir: Path,
    seed: int,
    options: WriteOptions,
) -> BaselineRunResult:
    spec = _amiga_reference_spec()
    run_dir = runs_dir / spec.baseline_id
    outputs = _baseline_outputs(run_dir)
    if outputs["cv_report"].exists() and options.skip_existing:
        missing = [path for path in outputs.values() if not path.exists()]
        if missing:
            raise DecisionBaselineError(f"cannot skip incomplete AMIGA reference; missing: {missing}")
        return BaselineRunResult(spec, run_dir, outputs["ranked"], outputs["cv_report"], outputs["run_manifest"], "skipped")
    if run_dir.exists() and any(run_dir.iterdir()) and not options.force:
        raise ManifestError(f"refusing to overwrite existing baseline run without --force: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    ranked = pd.read_csv(amiga_final_dir / "final_test_ranked.csv")
    ranked.to_csv(outputs["ranked"], index=False)
    shutil.copy2(amiga_final_dir / "cv_report.json", outputs["cv_report"])
    write_manifest(
        outputs["run_manifest"],
        run_manifest(
            context,
            PHASE_NAME,
            spec.baseline_id,
            seed=seed,
            status="completed",
            parameters={
                **_baseline_parameters_payload(context, spec),
                "source": repo_relative(amiga_final_dir),
                "role": "reference",
            },
            outputs={
                "ranked": repo_relative(outputs["ranked"]),
                "cv_report": repo_relative(outputs["cv_report"]),
            },
        ),
        options,
    )
    return BaselineRunResult(spec, run_dir, outputs["ranked"], outputs["cv_report"], outputs["run_manifest"], "written")


def _execute_or_reuse_baseline(
    context: CaseContext,
    spec: BaselineSpec,
    *,
    runs_dir: Path,
    seed: int,
    options: WriteOptions,
) -> BaselineRunResult:
    run_dir = runs_dir / spec.baseline_id
    outputs = _baseline_outputs(run_dir)
    if outputs["cv_report"].exists() and options.skip_existing:
        missing = [path for path in outputs.values() if not path.exists()]
        if missing:
            raise DecisionBaselineError(f"cannot skip incomplete baseline '{spec.baseline_id}'; missing: {missing}")
        return BaselineRunResult(spec, run_dir, outputs["ranked"], outputs["cv_report"], outputs["run_manifest"], "skipped")
    if run_dir.exists() and any(run_dir.iterdir()) and not options.force:
        raise ManifestError(f"refusing to overwrite existing baseline run without --force: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    ranked = _ranked_baseline_frame(context, spec, seed=seed)
    ranked.to_csv(outputs["ranked"], index=False)
    try:
        evaluation = evaluate_ranked_frame(
            ranked,
            front_col=context.group_column,
            target_col=context.target_column,
            score_col="score",
            model_name=spec.baseline_id,
            evaluation_split="test",
            fold=1,
            ks=DEFAULT_EVALUATION_KS,
            meta={
                "phase": PHASE_NAME,
                "baseline_id": spec.baseline_id,
                "baseline_type": spec.baseline_type,
                "method_family": spec.method_family,
                "implementation": spec.implementation,
                "objective": spec.objective,
                "method_parameters": dict(spec.parameters),
                "objective_columns": list(context.objective_columns),
                "objective_directions": {
                    column: context.objective_directions[column]
                    for column in context.objective_columns
                },
            },
        )
    except RankingEvaluationError as exc:
        raise DecisionBaselineError(str(exc)) from exc
    write_cv_report(outputs["cv_report"], evaluation.report)
    write_manifest(
        outputs["run_manifest"],
        run_manifest(
            context,
            PHASE_NAME,
            spec.baseline_id,
            seed=seed,
            status="completed",
            parameters={
                **_baseline_parameters_payload(context, spec),
            },
            outputs={
                "ranked": repo_relative(outputs["ranked"]),
                "cv_report": repo_relative(outputs["cv_report"]),
            },
        ),
        options,
    )
    return BaselineRunResult(spec, run_dir, outputs["ranked"], outputs["cv_report"], outputs["run_manifest"], "written")


def _ranked_baseline_frame(context: CaseContext, spec: BaselineSpec, *, seed: int) -> pd.DataFrame:
    columns = list(
        dict.fromkeys(
            [
                context.group_column,
                context.item_column,
                context.target_column,
                *context.objective_columns,
            ]
        )
    )
    ranked = context.test_df.loc[:, columns].copy().reset_index(drop=True)
    for objective in context.objective_columns:
        ranked[objective] = pd.to_numeric(ranked[objective], errors="raise")
    ranked[context.target_column] = pd.to_numeric(ranked[context.target_column], errors="raise")
    ranked["score"] = score_baseline(
        ranked,
        spec,
        objective_columns=context.objective_columns,
        objective_directions=context.objective_directions,
        front_col=context.group_column,
    )
    ranked = assign_rank_in_front(
        ranked,
        front_col=context.group_column,
        score_col="score",
        id_col=context.item_column,
        rank_col="rank_in_front",
        tie_seed=seed,
    )
    return ranked


def _objective_score(values: pd.Series, direction: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    if direction == "minimize":
        return -numeric
    if direction == "maximize":
        return numeric
    raise DecisionBaselineError(f"unsupported objective direction: {direction}")


def _baseline_outputs(run_dir: Path) -> dict[str, Path]:
    return {
        "ranked": run_dir / "ranked.csv",
        "cv_report": run_dir / "cv_report.json",
        "run_manifest": run_dir / "run_manifest.json",
    }


def _validate_amiga_final_outputs(context: CaseContext, amiga_final_dir: Path) -> None:
    missing = [
        path
        for path in (
            amiga_final_dir / "final_test_ranked.csv",
            amiga_final_dir / "cv_report.json",
        )
        if not path.exists()
    ]
    if missing:
        raise DecisionBaselineError(f"missing final AMIGA output(s): {missing}")

    ranked = pd.read_csv(amiga_final_dir / "final_test_ranked.csv")
    required = [context.group_column, context.item_column, context.target_column, "score"]
    missing_columns = [column for column in required if column not in ranked.columns]
    if missing_columns:
        raise DecisionBaselineError(
            f"final AMIGA ranked output is missing required column(s): {missing_columns}"
        )
    expected_fronts = set(pd.to_numeric(context.test_df[context.group_column], errors="raise").astype(int))
    actual_fronts = set(pd.to_numeric(ranked[context.group_column], errors="raise").astype(int))
    if actual_fronts != expected_fronts:
        raise DecisionBaselineError(
            "final AMIGA ranked output does not match held-out test fronts; "
            f"expected={sorted(expected_fronts)}, actual={sorted(actual_fronts)}"
        )


def _write_primary_tables(
    *,
    run_results: Sequence[BaselineRunResult],
    summary_outputs: Mapping[str, Path],
    summary_dir: Path,
) -> dict[str, Path]:
    primary_rank_table = write_primary_rank_table(
        _baseline_candidate_records(run_results),
        {result.baseline.baseline_id: result.cv_report for result in run_results},
        summary_outputs["metrics_summary"],
        summary_dir / "primary_rank_table.csv",
    )
    return {"primary_rank_table": primary_rank_table}


def _baseline_candidate_records(run_results: Sequence[BaselineRunResult]) -> list[dict[str, Any]]:
    return [
        {
            "run_id": result.baseline.baseline_id,
            "baseline_type": result.baseline.baseline_type,
            "method_family": result.baseline.method_family,
            "implementation": result.baseline.implementation,
            "objective": result.baseline.objective,
            "method_parameters": dict(result.baseline.parameters),
            "selectable": True,
        }
        for result in run_results
    ]


def _baseline_manifest_payload(
    context: CaseContext,
    *,
    phase_dir: Path,
    run_results: Sequence[BaselineRunResult],
    summary_outputs: Mapping[str, Path],
    table_outputs: Mapping[str, Path],
    amiga_final_dir: Path,
    seed: int,
) -> dict[str, Any]:
    return {
        **common_manifest_fields(context, seed=seed),
        "manifest_type": "decision_baselines",
        "phase": PHASE_NAME,
        "status": "completed",
        "amiga_final_dir": repo_relative(amiga_final_dir),
        "objective_columns": list(context.objective_columns),
        "objective_directions": {
            column: context.objective_directions[column]
            for column in context.objective_columns
        },
        "baseline_count": len(run_results),
        "baselines": [
            {
                "baseline_id": result.baseline.baseline_id,
                "baseline_type": result.baseline.baseline_type,
                "method_family": result.baseline.method_family,
                "implementation": result.baseline.implementation,
                "objective": result.baseline.objective,
                "method_parameters": dict(result.baseline.parameters),
                "status": result.status,
                "outputs": {
                    "ranked": repo_relative(result.ranked),
                    "cv_report": repo_relative(result.cv_report),
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
        "plots": planned_phase_plot_outputs(phase_dir, PHASE_NAME),
    }


def _amiga_reference_spec() -> BaselineSpec:
    return BaselineSpec(
        AMIGA_REFERENCE_ID,
        "amiga_reference",
        method_family="learned_reference",
        implementation="copied_phase_02_final_test",
        parameters={
            "role": "reference",
            "source_phase": "02_hyperparameter_tuning/final_test",
        },
    )


def _baseline_parameters_payload(context: CaseContext, spec: BaselineSpec) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "baseline_type": spec.baseline_type,
        "method_family": spec.method_family,
        "implementation": spec.implementation,
        "objective": spec.objective,
        "method_parameters": dict(spec.parameters),
    }
    if spec.baseline_type != "amiga_reference":
        payload["objective_columns"] = list(context.objective_columns)
        payload["objective_directions"] = _objective_directions_payload(context)
    return payload


def _objective_directions_payload(context: CaseContext) -> dict[str, str]:
    return {
        column: context.objective_directions[column]
        for column in context.objective_columns
    }


def _uniform_objective_weights(objective_columns: Sequence[str]) -> dict[str, float]:
    if not objective_columns:
        return {}
    weight = 1.0 / len(objective_columns)
    return {column: weight for column in objective_columns}


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in value)
