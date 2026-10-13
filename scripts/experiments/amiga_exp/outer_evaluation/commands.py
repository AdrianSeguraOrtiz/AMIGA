"""Phase-4 contract, execution, monitoring and summary commands."""
import json
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, help='Phase 4: held-out topology evaluation after sequential selection.')


@app.command('freeze')
def freeze_command(selection_run: Path = typer.Option(..., exists=True, file_okay=False),
                   selection_summary: Path = typer.Option(..., exists=True, file_okay=False),
                   output: Path = typer.Option(...)):
    from .spec import build_contract, freeze, counts
    contract = build_contract(selection_run, selection_summary)
    freeze(output, contract)
    typer.echo(json.dumps(dict(contract=str(output), **counts(contract)), indent=2))


@app.command('run')
def run(contract: Path = typer.Option(..., exists=True, dir_okay=False),
        output: Path = typer.Option(...), jobs: int = 16, threads: int = 4,
        dry_run: bool = False, resume: bool = False, retry_failed: bool = False):
    from .runner import run_evaluation
    result = run_evaluation(contract, output, jobs=jobs, threads=threads, dry_run=dry_run,
                            resume=resume, retry_failed=retry_failed)
    typer.echo(json.dumps(result, indent=2))


def inspect_status(run):
    from scripts.experiments.amiga_exp.sequential_selection.monitor import status
    result = status(run)
    state = json.loads((Path(run) / 'state.json').read_text())
    if state['status'] == 'running_evaluation':
        result['evaluation'] = status(Path(state['run']))
        result['health'] = result['evaluation']['health']
    return result


@app.command('status')
def status_command(run: Path = typer.Option(..., exists=True, file_okay=False)):
    typer.echo(json.dumps(inspect_status(run), indent=2))


@app.command('summarize')
def summarize_command(run: Path = typer.Option(..., exists=True, file_okay=False),
                      output: Path = typer.Option(...), figures: bool = True):
    from .summary import summarize
    result = summarize(run, output, figures=figures)
    typer.echo(json.dumps(dict(status=result['status'], output=str(output), audit=result['audit']), indent=2))
