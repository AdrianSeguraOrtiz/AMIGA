"""Command line skeleton for the AMIGA experimental pipeline."""

from __future__ import annotations

from pathlib import Path

import typer

from scripts.experiments.amiga_exp.context import (
    BasicCaseInfo,
    CaseContextError,
    context_summary,
    inspect_basic_case,
    load_case_context,
)
from scripts.experiments.amiga_exp.manifests import (
    ManifestError,
    WriteOptions,
    initialize_results_layout,
    repo_relative,
)
from scripts.experiments.amiga_exp.paper_summary import (
    DEFAULT_PRIMARY_METRIC,
    PaperSummaryError,
    summarize_paper,
)
from scripts.experiments.amiga_exp.phases.phase_01_model_screening import (
    DEFAULT_LABEL_QUANTILES,
    ModelScreeningError,
    run_model_screening,
)
from scripts.experiments.amiga_exp.phases.phase_02_hyperparameter_tuning import (
    DEFAULT_SELECTION_SIZE,
    HyperparameterTuningError,
    run_hyperparameter_tuning,
)
from scripts.experiments.amiga_exp.phases.phase_02_final_test import (
    FinalTestEvaluationError,
    run_final_test_evaluation,
)
from scripts.experiments.amiga_exp.phases.phase_03_ablation import (
    AblationError,
    run_ablation,
)
from scripts.experiments.amiga_exp.phases.phase_04_decision_baselines import (
    DecisionBaselineError,
    run_decision_baselines,
)
from scripts.experiments.amiga_exp.plots import (
    PlotPreparationError,
    prepare_all_phase_plots,
    prepare_phase_plots,
)

KNOWN_PHASES = (
    "01_model_screening",
    "02_hyperparameter_tuning",
    "final_test",
    "03_ablation",
    "04_decision_baselines",
)


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Experimental orchestration CLI for the AMIGA paper workflow. "
        "This command is intentionally separate from the public amiga package CLI."
    ),
)


def _fail(message: str, *, code: int = 1) -> None:
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=code)


def _print_case_info(case_info: BasicCaseInfo) -> None:
    typer.echo(f"case_name: {case_info.case_name}")
    typer.echo(f"case_dir: {case_info.case_dir}")
    typer.echo(f"data_dir: {case_info.data_dir}")
    typer.echo(f"data_csv_count: {len(case_info.data_csvs)}")
    for data_csv in case_info.data_csvs:
        typer.echo(f"data_csv: {data_csv}")
    typer.echo(f"audit_dir: {case_info.audit_dir}")
    typer.echo(f"audit_dir_exists: {case_info.audit_dir.exists()}")


def _print_context_summary(summary: dict[str, object]) -> None:
    for key, value in summary.items():
        typer.echo(f"{key}: {value}")


def _load_context_or_exit(case_dir: Path, *, case_name: str | None, config: Path | None):
    try:
        return load_case_context(case_dir, case_name=case_name, config_path=config)
    except CaseContextError as exc:
        _fail(str(exc))


def _write_options_or_exit(*, force: bool, skip_existing: bool, dry_run: bool) -> WriteOptions:
    options = WriteOptions(force=force, skip_existing=skip_existing, dry_run=dry_run)
    try:
        options.validate()
    except ManifestError as exc:
        _fail(str(exc))
    return options


def _load_model_params_or_exit(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        _fail(f"could not read model params JSON: {exc}")
    except json.JSONDecodeError as exc:
        _fail(f"invalid model params JSON: {path}: {exc}")
    if not isinstance(payload, dict):
        _fail(f"model params JSON must contain an object: {path}")
    return payload


def _print_layout_result(result: dict[str, object], *, dry_run: bool) -> None:
    action = "Would create/check" if dry_run else "Created/checked"
    typer.echo(f"{action} result directories:")
    for directory in result["directories"]:  # type: ignore[index]
        typer.echo(f"directory: {directory}")
    typer.echo(f"environment_manifest: {result['environment_manifest']}")
    typer.echo(f"environment_status: {result['environment_status']}")
    typer.echo(f"phase_statuses: {result['phase_statuses']}")


def _echo_primary_rank_table(result: object) -> None:
    table_outputs = getattr(result, "table_outputs", None)
    if not isinstance(table_outputs, dict):
        return
    primary_rank_table = table_outputs.get("primary_rank_table")
    if primary_rank_table is not None:
        typer.echo(f"primary_rank_table: {repo_relative(Path(primary_rank_table))}")


def _echo_plot_result(result: object) -> None:
    typer.echo(f"phase: {result.phase}")
    typer.echo(f"plots_dir: {repo_relative(result.plots_dir)}")
    typer.echo(f"primary_rank_table: {repo_relative(result.primary_rank_table)}")
    typer.echo(f"plot_manifest: {repo_relative(result.plot_manifest)}")
    typer.echo(f"plot_manifest_status: {result.status}")


@app.command(name="inspect")
def inspect_case(
    case_dir: Path = typer.Argument(..., help="Path to an experimental case directory."),
    case_name: str | None = typer.Option(
        None,
        "--case-name",
        help="Optional case name override when it cannot be inferred from the folder name.",
    ),
) -> None:
    """Inspect the minimal case-folder layout without running experimental phases."""
    try:
        case_info = inspect_basic_case(case_dir, case_name=case_name)
    except CaseContextError as exc:
        _fail(str(exc))
    _print_case_info(case_info)


@app.command(name="validate")
def validate_case(
    case_dir: Path = typer.Argument(..., help="Path to an experimental case directory."),
    case_name: str | None = typer.Option(
        None,
        "--case-name",
        help="Optional case name override when it cannot be inferred from the folder name.",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Optional path to the case JSON config. Defaults to docs/experiments/cases/<CASE>.json.",
    ),
) -> None:
    """Load and validate the shared case context required by experimental phases."""
    context = _load_context_or_exit(case_dir, case_name=case_name, config=config)
    typer.secho("Case context validation passed.", fg=typer.colors.GREEN)
    _print_context_summary(context_summary(context))


@app.command(name="init-results")
def init_results(
    case_dir: Path = typer.Argument(..., help="Path to an experimental case directory."),
    case_name: str | None = typer.Option(
        None,
        "--case-name",
        help="Optional case name override when it cannot be inferred from the folder name.",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Optional path to the case JSON config. Defaults to docs/experiments/cases/<CASE>.json.",
    ),
    seed: int = typer.Option(42, "--seed", help="Seed recorded in initialized manifests."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing manifests."),
    skip_existing: bool = typer.Option(
        False,
        "--skip-existing",
        help="Keep existing manifests instead of failing.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show the layout/manifests that would be initialized without writing files.",
    ),
) -> None:
    """Initialize the standard result layout and basic manifests for a case."""
    context = _load_context_or_exit(case_dir, case_name=case_name, config=config)
    options = _write_options_or_exit(force=force, skip_existing=skip_existing, dry_run=dry_run)
    try:
        result = initialize_results_layout(context, seed=seed, options=options)
    except ManifestError as exc:
        _fail(str(exc))
    typer.secho("Result layout initialization complete.", fg=typer.colors.GREEN)
    _print_layout_result(result, dry_run=dry_run)


@app.command(name="run-phase")
def run_phase(
    case_dir: Path = typer.Argument(..., help="Path to an experimental case directory."),
    phase: str = typer.Argument(..., help="Experimental phase name."),
    case_name: str | None = typer.Option(
        None,
        "--case-name",
        help="Optional case name override when it cannot be inferred from the folder name.",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Optional path to the case JSON config. Defaults to docs/experiments/cases/<CASE>.json.",
    ),
    seed: int = typer.Option(42, "--seed", help="Seed to use for phase execution."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing phase outputs."),
    skip_existing: bool = typer.Option(False, "--skip-existing", help="Skip phase outputs that already exist."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would run without writing outputs."),
    model: list[str] | None = typer.Option(
        None,
        "--model",
        help="Optional repeated model type for 01_model_screening.",
    ),
    label_mode: list[str] | None = typer.Option(
        None,
        "--label-mode",
        help="Optional repeated label mode for 01_model_screening.",
    ),
    label_quantiles: list[int] | None = typer.Option(
        None,
        "--label-quantiles",
        help="Optional repeated quantile count for the quantiles label mode.",
    ),
    n_splits: int = typer.Option(5, "--n-splits", help="GroupKFold split count for CV phases."),
    model_params_json: Path | None = typer.Option(
        None,
        "--model-params-json",
        help="Optional JSON object with model params, either generic or keyed by model type.",
    ),
    tuning_grid_json: Path | None = typer.Option(
        None,
        "--tuning-grid-json",
        help="Optional JSON object with phase-02 tuning grids keyed by model type.",
    ),
    selection_size: int = typer.Option(
        DEFAULT_SELECTION_SIZE,
        "--selection-size",
        help="Number of final configs to record for 02_hyperparameter_tuning.",
    ),
    selected_config: Path | None = typer.Option(
        None,
        "--selected-config",
        help="Optional selected_config.json path for the held-out final_test step.",
    ),
    feature_set: list[str] | None = typer.Option(
        None,
        "--feature-set",
        help="Optional repeated feature set for 03_ablation.",
    ),
    amiga_final_dir: Path | None = typer.Option(
        None,
        "--amiga-final-dir",
        help="Optional final_test directory to use as the AMIGA reference for 04_decision_baselines.",
    ),
) -> None:
    """Run one experimental phase."""
    context = _load_context_or_exit(case_dir, case_name=case_name, config=config)
    options = _write_options_or_exit(force=force, skip_existing=skip_existing, dry_run=dry_run)
    if phase not in KNOWN_PHASES:
        _fail(
            f"unknown phase '{phase}'. Known phases: {', '.join(KNOWN_PHASES)}",
            code=2,
        )
    if phase == "01_model_screening":
        try:
            result = run_model_screening(
                context,
                seed=seed,
                options=options,
                model_types=model,
                label_modes=label_mode,
                label_quantiles=label_quantiles or DEFAULT_LABEL_QUANTILES,
                n_splits=n_splits,
                model_params=_load_model_params_or_exit(model_params_json),
            )
        except (ManifestError, ModelScreeningError) as exc:
            _fail(str(exc))

        if dry_run:
            typer.secho("Phase 01 model screening dry-run complete.", fg=typer.colors.GREEN)
            typer.echo(f"planned_runs: {len(result.run_results)}")
            for run_result in result.run_results:
                typer.echo(f"run: {run_result.config.run_id}")
            return

        typer.secho("Phase 01 model screening complete.", fg=typer.colors.GREEN)
        typer.echo(f"phase_dir: {repo_relative(result.phase_dir)}")
        typer.echo(f"run_count: {len(result.run_results)}")
        typer.echo(f"summary_outputs: {result.summary_outputs}")
        typer.echo(f"table_outputs: {result.table_outputs}")
        typer.echo(f"screening_manifest: {repo_relative(result.screening_manifest)}")
        typer.echo(f"shortlisted_configs: {repo_relative(result.shortlisted_configs)}")
        return

    if phase == "02_hyperparameter_tuning":
        try:
            result = run_hyperparameter_tuning(
                context,
                seed=seed,
                options=options,
                n_splits=n_splits,
                tuning_grids=_load_model_params_or_exit(tuning_grid_json),
                selection_size=selection_size,
            )
        except (ManifestError, HyperparameterTuningError) as exc:
            _fail(str(exc))

        if dry_run:
            typer.secho("Phase 02 hyperparameter tuning dry-run complete.", fg=typer.colors.GREEN)
            typer.echo(f"planned_runs: {len(result.run_results)}")
            for run_result in result.run_results:
                typer.echo(f"run: {run_result.config.run_id}")
            return

        typer.secho("Phase 02 hyperparameter tuning complete.", fg=typer.colors.GREEN)
        typer.echo(f"phase_dir: {repo_relative(result.phase_dir)}")
        typer.echo(f"run_count: {len(result.run_results)}")
        typer.echo(f"summary_outputs: {result.summary_outputs}")
        typer.echo(f"table_outputs: {result.table_outputs}")
        typer.echo(f"tuning_manifest: {repo_relative(result.tuning_manifest)}")
        typer.echo(f"selected_config: {repo_relative(result.selected_config)}")
        return

    if phase == "final_test":
        try:
            result = run_final_test_evaluation(
                context,
                seed=seed,
                options=options,
                selected_config_path=selected_config,
            )
        except (ManifestError, FinalTestEvaluationError) as exc:
            _fail(str(exc))

        if dry_run:
            typer.secho("Held-out final test dry-run complete.", fg=typer.colors.GREEN)
            typer.echo(f"final_test_dir: {repo_relative(result.final_test_dir)}")
            typer.echo(f"cv_report: {repo_relative(result.cv_report)}")
            return

        typer.secho("Held-out final test complete.", fg=typer.colors.GREEN)
        typer.echo(f"final_test_dir: {repo_relative(result.final_test_dir)}")
        typer.echo(f"final_test_ranked: {repo_relative(result.final_test_ranked)}")
        typer.echo(f"final_test_report: {repo_relative(result.final_test_report)}")
        typer.echo(f"cv_report: {repo_relative(result.cv_report)}")
        return

    if phase == "03_ablation":
        try:
            result = run_ablation(
                context,
                seed=seed,
                options=options,
                selected_config_path=selected_config,
                feature_sets=feature_set,
                n_splits=n_splits,
            )
        except (ManifestError, AblationError) as exc:
            _fail(str(exc))

        if dry_run:
            typer.secho("Phase 03 ablation dry-run complete.", fg=typer.colors.GREEN)
            typer.echo(f"planned_feature_sets: {len(result.run_results)}")
            for run_result in result.run_results:
                typer.echo(f"feature_set: {run_result.config.feature_set}")
            return

        typer.secho("Phase 03 ablation complete.", fg=typer.colors.GREEN)
        typer.echo(f"phase_dir: {repo_relative(result.phase_dir)}")
        typer.echo(f"feature_set_count: {len(result.run_results)}")
        typer.echo(f"summary_outputs: {result.summary_outputs}")
        typer.echo(f"table_outputs: {result.table_outputs}")
        typer.echo(f"ablation_manifest: {repo_relative(result.ablation_manifest)}")
        return

    if phase == "04_decision_baselines":
        try:
            result = run_decision_baselines(
                context,
                seed=seed,
                options=options,
                amiga_final_dir=amiga_final_dir,
            )
        except (ManifestError, DecisionBaselineError) as exc:
            _fail(str(exc))

        if dry_run:
            typer.secho("Phase 04 decision baselines dry-run complete.", fg=typer.colors.GREEN)
            typer.echo(f"planned_runs: {len(result.run_results)}")
            for run_result in result.run_results:
                baseline = run_result.baseline
                details = (
                    f"type={baseline.baseline_type}, "
                    f"family={baseline.method_family}, "
                    f"implementation={baseline.implementation}"
                )
                if baseline.objective is not None:
                    details = f"{details}, objective={baseline.objective}"
                typer.echo(f"run: {baseline.baseline_id} ({details})")
            return

        typer.secho("Phase 04 decision baselines complete.", fg=typer.colors.GREEN)
        typer.echo(f"phase_dir: {repo_relative(result.phase_dir)}")
        typer.echo(f"baseline_count: {len(result.run_results)}")
        typer.echo(f"summary_outputs: {result.summary_outputs}")
        typer.echo(f"table_outputs: {result.table_outputs}")
        typer.echo(f"baseline_manifest: {repo_relative(result.baseline_manifest)}")
        return

    typer.secho(
        f"Phase '{phase}' for case '{context.case_name}' is not implemented yet. "
        f"Seed={seed}; results_root={repo_relative(context.results_root)}",
        fg=typer.colors.YELLOW,
        err=True,
    )
    raise typer.Exit(code=2)


@app.command(name="plot-phase")
def plot_phase_command(
    case_dir: Path = typer.Option(..., "--case-dir", help="Path to an experimental case directory."),
    phase: str = typer.Option(..., "--phase", help="Experimental phase to prepare plots for."),
    case_name: str | None = typer.Option(
        None,
        "--case-name",
        help="Optional case name override when it cannot be inferred from the folder name.",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Optional path to the case JSON config. Defaults to docs/experiments/cases/<CASE>.json.",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing plot manifests."),
    skip_existing: bool = typer.Option(False, "--skip-existing", help="Keep existing plot manifests instead of failing."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate inputs without writing plot manifests."),
    include_secondary: bool = typer.Option(
        False,
        "--include-secondary",
        help="Also write complementary Top-K context plots under plots/supplementary.",
    ),
) -> None:
    """Prepare plot outputs for one completed experimental phase."""
    context = _load_context_or_exit(case_dir, case_name=case_name, config=config)
    options = _write_options_or_exit(force=force, skip_existing=skip_existing, dry_run=dry_run)
    try:
        result = prepare_phase_plots(
            context,
            phase,
            options=options,
            include_secondary=include_secondary,
        )
    except (ManifestError, PlotPreparationError) as exc:
        _fail(str(exc))

    typer.secho("Phase plot preparation complete.", fg=typer.colors.GREEN)
    _echo_plot_result(result)


@app.command(name="plot-all")
def plot_all_command(
    case_dir: Path = typer.Option(..., "--case-dir", help="Path to an experimental case directory."),
    case_name: str | None = typer.Option(
        None,
        "--case-name",
        help="Optional case name override when it cannot be inferred from the folder name.",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Optional path to the case JSON config. Defaults to docs/experiments/cases/<CASE>.json.",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing plot manifests."),
    skip_existing: bool = typer.Option(False, "--skip-existing", help="Keep existing plot manifests instead of failing."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate inputs without writing plot manifests."),
    include_secondary: bool = typer.Option(
        False,
        "--include-secondary",
        help="Also write complementary Top-K context plots under plots/supplementary.",
    ),
) -> None:
    """Prepare plot outputs for every completed experimental phase."""
    context = _load_context_or_exit(case_dir, case_name=case_name, config=config)
    options = _write_options_or_exit(force=force, skip_existing=skip_existing, dry_run=dry_run)
    try:
        result = prepare_all_phase_plots(
            context,
            options=options,
            include_secondary=include_secondary,
        )
    except (ManifestError, PlotPreparationError) as exc:
        _fail(str(exc))

    typer.secho("All phase plot preparation complete.", fg=typer.colors.GREEN)
    for phase_result in result.results:
        _echo_plot_result(phase_result)


@app.command(name="run-all")
def run_all(
    case_dir: Path = typer.Argument(..., help="Path to an experimental case directory."),
    case_name: str | None = typer.Option(
        None,
        "--case-name",
        help="Optional case name override when it cannot be inferred from the folder name.",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Optional path to the case JSON config. Defaults to docs/experiments/cases/<CASE>.json.",
    ),
    seed: int = typer.Option(42, "--seed", help="Seed to use for phase execution."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing phase outputs."),
    skip_existing: bool = typer.Option(False, "--skip-existing", help="Skip phase outputs that already exist."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would run without writing outputs."),
    model: list[str] | None = typer.Option(
        None,
        "--model",
        help="Optional repeated model type for 01_model_screening.",
    ),
    label_mode: list[str] | None = typer.Option(
        None,
        "--label-mode",
        help="Optional repeated label mode for 01_model_screening.",
    ),
    label_quantiles: list[int] | None = typer.Option(
        None,
        "--label-quantiles",
        help="Optional repeated quantile count for the quantiles label mode.",
    ),
    n_splits: int = typer.Option(5, "--n-splits", help="GroupKFold split count for CV phases."),
    model_params_json: Path | None = typer.Option(
        None,
        "--model-params-json",
        help="Optional JSON object with model params for 01_model_screening.",
    ),
    tuning_grid_json: Path | None = typer.Option(
        None,
        "--tuning-grid-json",
        help="Optional JSON object with phase-02 tuning grids keyed by model type.",
    ),
    selection_size: int = typer.Option(
        DEFAULT_SELECTION_SIZE,
        "--selection-size",
        help="Number of final configs to record for 02_hyperparameter_tuning.",
    ),
    feature_set: list[str] | None = typer.Option(
        None,
        "--feature-set",
        help="Optional repeated feature set for 03_ablation.",
    ),
    primary_metric: str = typer.Option(
        DEFAULT_PRIMARY_METRIC,
        "--primary-metric",
        help="Primary metric for paper-level statistical tests.",
    ),
) -> None:
    """Run the full standard experimental pipeline for one case."""
    context = _load_context_or_exit(case_dir, case_name=case_name, config=config)
    options = _write_options_or_exit(force=force, skip_existing=skip_existing, dry_run=dry_run)
    model_params = _load_model_params_or_exit(model_params_json)
    tuning_grids = _load_model_params_or_exit(tuning_grid_json)

    if dry_run:
        try:
            layout_result = initialize_results_layout(context, seed=seed, options=options)
        except ManifestError as exc:
            _fail(str(exc))
        typer.secho("Full pipeline dry-run complete.", fg=typer.colors.GREEN)
        _print_layout_result(layout_result, dry_run=True)
        for step in (*KNOWN_PHASES, "summarize-paper"):
            typer.echo(f"planned_step: {step}")
        return

    try:
        layout_result = initialize_results_layout(context, seed=seed, options=options)
        typer.secho("Result layout initialization complete.", fg=typer.colors.GREEN)
        typer.echo(f"environment_manifest: {repo_relative(layout_result['environment_manifest'])}")

        screening = run_model_screening(
            context,
            seed=seed,
            options=options,
            model_types=model,
            label_modes=label_mode,
            label_quantiles=label_quantiles or DEFAULT_LABEL_QUANTILES,
            n_splits=n_splits,
            model_params=model_params,
        )
        typer.secho("Completed 01_model_screening.", fg=typer.colors.GREEN)
        typer.echo(f"shortlisted_configs: {repo_relative(screening.shortlisted_configs)}")
        _echo_primary_rank_table(screening)

        tuning = run_hyperparameter_tuning(
            context,
            seed=seed,
            options=options,
            n_splits=n_splits,
            tuning_grids=tuning_grids,
            selection_size=selection_size,
        )
        typer.secho("Completed 02_hyperparameter_tuning.", fg=typer.colors.GREEN)
        typer.echo(f"selected_config: {repo_relative(tuning.selected_config)}")
        _echo_primary_rank_table(tuning)

        final_test_result = run_final_test_evaluation(
            context,
            seed=seed,
            options=options,
        )
        typer.secho("Completed final_test.", fg=typer.colors.GREEN)
        typer.echo(f"final_test_report: {repo_relative(final_test_result.final_test_report)}")

        ablation = run_ablation(
            context,
            seed=seed,
            options=options,
            feature_sets=feature_set,
            n_splits=n_splits,
        )
        typer.secho("Completed 03_ablation.", fg=typer.colors.GREEN)
        typer.echo(f"ablation_manifest: {repo_relative(ablation.ablation_manifest)}")
        _echo_primary_rank_table(ablation)

        baselines = run_decision_baselines(
            context,
            seed=seed,
            options=options,
        )
        typer.secho("Completed 04_decision_baselines.", fg=typer.colors.GREEN)
        typer.echo(f"baseline_manifest: {repo_relative(baselines.baseline_manifest)}")
        _echo_primary_rank_table(baselines)

        paper_summary = summarize_paper(
            context,
            options=options,
            primary_metric=primary_metric,
        )
        typer.secho("Completed summarize-paper.", fg=typer.colors.GREEN)
        for name, path in paper_summary.outputs.items():
            typer.echo(f"{name}: {repo_relative(path)}")
    except (
        ManifestError,
        ModelScreeningError,
        HyperparameterTuningError,
        FinalTestEvaluationError,
        AblationError,
        DecisionBaselineError,
        PaperSummaryError,
    ) as exc:
        _fail(str(exc))

    typer.secho("Full pipeline complete.", fg=typer.colors.GREEN)


@app.command(name="summarize-paper")
def summarize_paper_command(
    case_dir: Path = typer.Argument(..., help="Path to an experimental case directory."),
    case_name: str | None = typer.Option(
        None,
        "--case-name",
        help="Optional case name override when it cannot be inferred from the folder name.",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Optional path to the case JSON config. Defaults to docs/experiments/cases/<CASE>.json.",
    ),
    primary_metric: str = typer.Option(
        DEFAULT_PRIMARY_METRIC,
        "--primary-metric",
        help="Primary metric for paired paper-level statistical tests.",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing summary outputs."),
    skip_existing: bool = typer.Option(False, "--skip-existing", help="Keep existing summary outputs instead of failing."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate inputs and show planned outputs without writing files."),
) -> None:
    """Build paper-level comparison tables from completed experimental phases."""
    context = _load_context_or_exit(case_dir, case_name=case_name, config=config)
    options = _write_options_or_exit(force=force, skip_existing=skip_existing, dry_run=dry_run)
    try:
        result = summarize_paper(
            context,
            options=options,
            primary_metric=primary_metric,
        )
    except PaperSummaryError as exc:
        _fail(str(exc))

    if dry_run:
        typer.secho("Paper summary dry-run complete.", fg=typer.colors.GREEN)
    else:
        typer.secho("Paper summary complete.", fg=typer.colors.GREEN)
    typer.echo(f"summary_dir: {repo_relative(result.summary_dir)}")
    for name, path in result.outputs.items():
        typer.echo(f"{name}: {repo_relative(path)}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
