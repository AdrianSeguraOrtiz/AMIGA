"""Fixed-configuration AMIGA refits, learning curves and original real-case analysis."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import traceback
from tempfile import TemporaryDirectory

from scripts.experiments.amiga_exp.grouped_validation.pilot import utc,write_json
from .runner import run_evaluation
from .spec import REPO
from .summary import summarize
from .deployment import apply
from scripts.experiments.amiga_exp.reporting.supervised import verify_manifest


def atomic_stage(output,name,function):
    """Recover a published, checksummed stage after a missing checkpoint."""
    output=Path(output)
    target=output/name
    if target.exists():
        manifest,_=verify_manifest(target/'manifest.json')
        if manifest.get('status')!='complete':
            raise ValueError(f'Incomplete existing stage: {target}')
        return
    with TemporaryDirectory(prefix=f'.{name}-',dir=output) as temporary:
        staged=Path(temporary)/name
        function(staged)
        verify_manifest(staged/'manifest.json')
        staged.rename(target)


def run_all(contract,output,*,jobs=16,threads=4,resume=False,retry_failed=False):
    output=Path(output).resolve()
    if output.exists() and not resume: raise ValueError('Use a new pipeline destination or explicit resume')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.pipeline.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another supplementary pipeline holds this destination') from exc
        return run_stages(contract,output,jobs=jobs,threads=threads,retry_failed=retry_failed)


def run_stages(contract,output,*,jobs,threads,retry_failed):
    state_path=output/'state.json'
    previous=json.loads(state_path.read_text()) if state_path.exists() else {}
    completed=set(previous.get('completed_stages',[]))

    def checkpoint(status,**extra):
        write_json(state_path,dict(status=status,updated_at_utc=utc(),supervisor_pid=os.getpid(),
                                   completed_stages=sorted(completed),**extra))

    try:
        if 'fits' not in completed:
            checkpoint('running_fixed_amiga_fits')
            run=output/'run'
            result=run_evaluation(Path(contract),run,jobs=jobs,threads=threads,
                                  resume=run.exists(),retry_failed=retry_failed)
            if result['status']!='complete': raise RuntimeError('Supplementary fitting did not complete')
            completed.add('fits')
        if 'summary' not in completed:
            checkpoint('summarizing_learning_curves')
            atomic_stage(output,'summary',lambda path:summarize(output/'run',path))
            completed.add('summary')
        if 'application' not in completed:
            checkpoint('scoring_and_supporting_tcga')
            atomic_stage(output,'application',lambda path:apply(output/'run',REPO/'experiments/BIO-INSIGHT/real-world/tcga_brca',path))
            completed.add('application')
        checkpoint('complete',summary=str(output/'summary'),application=str(output/'application'))
    except BaseException:
        checkpoint('interrupted' if isinstance(__import__('sys').exc_info()[1],KeyboardInterrupt) else 'failed',error=traceback.format_exc())
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=16)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--retry-failed',action='store_true')
    a=p.parse_args()
    run_all(a.contract,a.output,jobs=a.jobs,threads=a.threads,resume=a.resume,retry_failed=a.retry_failed)
