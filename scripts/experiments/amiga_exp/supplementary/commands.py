"""CLI for fixed-AMIGA learning curves, real-case ranking and prediction timing."""
import json
from pathlib import Path
import typer

app=typer.Typer(no_args_is_help=True,help='Fixed-AMIGA learning curves, original real-case analysis and prediction costs.')


@app.command('freeze')
def freeze_command(output:Path=typer.Option(...)):
    from .spec import build_contract,build_plan,freeze
    c=build_contract()
    freeze(output,c)
    plan=build_plan(c)
    typer.echo(json.dumps(dict(contract=str(output),jobs=len(plan),
                              planned_fits=sum(j['planned_fits'] for j in plan)),indent=2))


@app.command('run')
def run_command(contract:Path=typer.Option(...,exists=True,dir_okay=False),output:Path=typer.Option(...),
                jobs:int=typer.Option(16,min=1),threads:int=typer.Option(4,min=1,max=8),
                resume:bool=False,retry_failed:bool=False):
    from .pipeline import run_all
    run_all(contract,output,jobs=jobs,threads=threads,resume=resume,retry_failed=retry_failed)


@app.command('status')
def status_command(run:Path=typer.Option(...,exists=True,file_okay=False)):
    result={}
    for name,path in [('pipeline',run/'state.json'),('execution',run/'run/state.json')]:
        if path.exists(): result[name]=json.loads(path.read_text())
    result['active_progress']={}
    for identifier in result.get('execution',{}).get('active_jobs',[]):
        paths=sorted((run/'run/jobs'/identifier).glob('attempt-*/progress.json'))
        if paths: result['active_progress'][identifier]=json.loads(paths[-1].read_text())
    typer.echo(json.dumps(result,indent=2))
