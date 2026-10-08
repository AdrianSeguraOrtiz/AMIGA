"""CLI entry points for the grouped evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, help='Grouped nested evaluation and reproducible execution.')


@app.command('freeze')
def freeze(output: Path = typer.Option(..., help='New contract JSON; existing different contracts are refused.')):
    from .planning import build_evaluation_contract, freeze_contract, build_plan
    from .runner import REPO
    contract = build_evaluation_contract(REPO)
    plan = build_plan(contract)
    freeze_contract(output, contract)
    typer.echo(json.dumps(dict(contract=str(output), fits=len(plan) - 2, jobs=len(plan)), indent=2))


@app.command('run')
def run(contract: Path = typer.Option(..., exists=True, dir_okay=False),
        output: Path = typer.Option(...), jobs: int = 2, threads: int = 8,
        dry_run: bool = False, resume: bool = False, retry_failed: bool = False):
    from .runner import run_evaluation
    result = run_evaluation(contract, output, jobs=jobs, threads=threads,
                            dry_run=dry_run, resume=resume, retry_failed=retry_failed)
    typer.echo(json.dumps(result, indent=2))


@app.command('summarize')
def summarize(run: Path = typer.Option(..., exists=True, file_okay=False), output: Path = typer.Option(...)):
    from .summary import summarize as make_summary
    typer.echo(json.dumps(make_summary(run, output), indent=2))


@app.command('rank-deployment')
def rank_deployment(run: Path = typer.Option(..., exists=True, file_okay=False),
                    case: str = typer.Option(...),
                    data: Path = typer.Option(..., exists=True, dir_okay=False),
                    output: Path = typer.Option(...)):
    """Score unlabeled fronts with the four deployment models and five fixed rules."""
    from .deployment import score_deployment
    typer.echo(json.dumps(score_deployment(run, case, data, output), indent=2))
