"""Freeze, execute and monitor top-ten-percent classification sensitivity."""
import json
from pathlib import Path

import typer

app=typer.Typer(no_args_is_help=True,help='Retuned top-10% classification on the existing grouped evaluation.')


@app.command('freeze')
def freeze_command(output:Path=typer.Option(...)):
    from .spec import build_contract,freeze,counts
    c=build_contract()
    freeze(output,c)
    typer.echo(json.dumps(dict(contract=str(output),**counts(c)),indent=2))


@app.command('run')
def run_command(contract:Path=typer.Option(...,exists=True,dir_okay=False),
                output:Path=typer.Option(...),work_output:Path=typer.Option(...),
                jobs:int=16,threads:int=4,resume:bool=False,retry_failed:bool=False):
    from .pipeline import run_pipeline
    typer.echo(json.dumps(run_pipeline(contract,output,work_output,jobs=jobs,threads=threads,
                                     resume=resume,retry_failed=retry_failed),indent=2))


def inspect_status(run):
    from scripts.experiments.amiga_exp.sequential_selection.monitor import status
    run=Path(run).resolve()
    result=status(run)
    state=json.loads((run/'state.json').read_text())
    if state['status'] in ('running_selection','running_outer'):
        child=Path(state['run'])
        if (child/'state.json').is_file():
            result['active_stage']=status(child)
            result['health']=result['active_stage']['health']
    return result


@app.command('status')
def status_command(run:Path=typer.Option(...,exists=True,file_okay=False)):
    typer.echo(json.dumps(inspect_status(run),indent=2))


@app.command('summarize')
def summarize_command(run:Path=typer.Option(...,exists=True,file_okay=False),output:Path=typer.Option(...)):
    from .summary import summarize
    result=summarize(run,output)
    typer.echo(json.dumps(dict(status=result['status'],output=str(output),audit=result['audit']),indent=2))
