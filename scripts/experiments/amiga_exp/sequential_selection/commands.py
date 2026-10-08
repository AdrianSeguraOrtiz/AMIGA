"""CLI for topology-grouped sequential model selection."""
from __future__ import annotations

import json
from pathlib import Path
import typer

app = typer.Typer(no_args_is_help=True, help='Training-only phases 0–3; no outer evaluation.')


@app.command('freeze')
def freeze_command(output: Path = typer.Option(...), mode: str = 'selection',
                   profile: str = 'original', budget_hours: float = 144.):
    from .spec import build_contract, freeze, counts
    budget = min(budget_hours, 2.) if mode == 'pilot' else budget_hours
    contract = build_contract(mode=mode, profile=profile, budget_seconds=budget*3600)
    freeze(output, contract)
    typer.echo(json.dumps(dict(contract=str(output), **counts(contract)), indent=2))


@app.command('run')
def run(contract: Path = typer.Option(..., exists=True, dir_okay=False),
        output: Path = typer.Option(...), jobs: int = 2, threads: int = 8,
        dry_run: bool = False, resume: bool = False, retry_failed: bool = False):
    from .runner import run_selection
    result = run_selection(contract, output, jobs=jobs, threads=threads,
                           dry_run=dry_run, resume=resume, retry_failed=retry_failed)
    typer.echo(json.dumps(result, indent=2))


@app.command('status')
def status(run: Path = typer.Option(..., exists=True, file_okay=False)):
    from .monitor import status as inspect
    typer.echo(json.dumps(inspect(run), indent=2))


@app.command('cost')
def cost(run: Path = typer.Option(..., exists=True, file_okay=False)):
    from .costing import project, choose_profile
    result = project(run)
    result['recommended_profile'] = choose_profile(result)
    typer.echo(json.dumps(result, indent=2))


@app.command('summarize')
def summarize(run: Path = typer.Option(..., exists=True, file_okay=False),
              output: Path = typer.Option(...), figures: bool = True):
    from .summary import summarize as make_summary
    report = make_summary(run, output, figures=figures)
    typer.echo(json.dumps(dict(status=report['status'], output=str(output),
                               selected_procedures=report['selected_procedures']), indent=2))
