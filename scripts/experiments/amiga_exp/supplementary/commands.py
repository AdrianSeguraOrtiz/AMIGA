"""CLI for complete label-scarcity, deployment and computational-cost analyses."""
import json
from pathlib import Path
import typer

app=typer.Typer(no_args_is_help=True,help='Full-grid learning curves, deployment and empirical computational costs.')


@app.command('freeze')
def freeze_command(output:Path=typer.Option(...)):
    from .spec import build_contract,build_plan,freeze
    c=build_contract()
    freeze(output,c)
    plan=build_plan(c)
    typer.echo(json.dumps(dict(contract=str(output),jobs=len(plan),
                              planned_fits=sum(j['planned_fits'] for j in plan),
                              maximum_additional_mask_fits=sum(j.get('maximum_mask_parent_fits',0) for j in plan)),indent=2))


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


@app.command('archive')
def archive_command(run:Path=typer.Option(...,exists=True,file_okay=False),output:Path=typer.Option(...)):
    from .archive import build_archive
    result=build_archive(run,output)
    typer.echo(json.dumps(dict(output=str(output),archives=len(result['archives'])),indent=2))


@app.command('archive-comparison')
def comparison_archive_command(output:Path=typer.Option(...)):
    from .archive import build_comparison_archive
    result=build_comparison_archive(output)
    typer.echo(json.dumps(dict(output=str(output),archives=len(result['archives'])),indent=2))


@app.command('verify-archive')
def verify_archive_command(archive:Path=typer.Option(...,exists=True,file_okay=False)):
    from .archive import verify_archive
    result=verify_archive(archive)
    typer.echo(json.dumps(dict(status='verified',files=sum(len(x['files']) for x in result['archives'].values())),indent=2))


@app.command('restore-archive')
def restore_archive_command(archive:Path=typer.Option(...,exists=True,file_okay=False),
                            destination:Path=typer.Option(Path('.'))):
    from .archive import restore
    written=restore(archive,destination)
    typer.echo(json.dumps(dict(status='restored',new_files=written),indent=2))
