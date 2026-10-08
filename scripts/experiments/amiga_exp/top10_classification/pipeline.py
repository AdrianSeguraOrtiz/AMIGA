"""Detached full inner selection, frozen outer fits and audited comparisons."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import fcntl
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

from scripts.experiments.amiga_exp.grouped_validation.pilot import sha256, utc, write_json
from .runner import environment,run_selection
from .spec import REPO,counts,freeze,outer_contract,verify_sources,verify_artifacts


def run_pipeline(contract_path,output,work_output,*,jobs=16,threads=4,resume=False,retry_failed=False):
    contract_path,output,work=[Path(p).resolve() for p in (contract_path,output,work_output)]
    if output==work or output in work.parents or work in output.parents:
        raise ValueError('Launch and work destinations must be separate')
    if retry_failed and not resume:
        raise ValueError('Explicit retry requires --resume')
    c=json.loads(contract_path.read_text())
    verify_sources(REPO,c)
    if c['mode']!='selection':
        raise ValueError('The full pipeline starts with the selection contract')
    definition=dict(contract=str(contract_path),contract_sha256=sha256(contract_path),work=str(work),
                    environment=environment(),workers=jobs,threads=threads,counts=counts(c))
    if not resume:
        if output.exists() or work.exists():
            raise ValueError('Use new launch and work destinations')
        output.mkdir(parents=True)
        work.mkdir(parents=True)
    elif not output.is_dir() or not work.is_dir():
        raise ValueError('Cannot resume nonexistent work')
    with (output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if resume:
            manifest=json.loads((output/'manifest.json').read_text())
            if any(manifest.get(k)!=v for k,v in definition.items()):
                raise ValueError('Launch definitions changed')
        else:
            write_json(output/'manifest.json',dict(created_at_utc=utc(),**definition))
        state,mutex,stop={},threading.Lock(),threading.Event()

        def persist():
            state.update(updated_at_utc=utc(),accounted_at_unix=time.time())
            write_json(output/'state.json',dict(state))

        def checkpoint(status,**extra):
            with mutex:
                state.clear()
                state.update(status=status,supervisor_pid=os.getpid(),work=str(work),
                             evidence_role='exploratory_sensitivity',**extra)
                persist()
                print(json.dumps(state),flush=True)

        def heartbeat():
            while not stop.wait(5):
                with mutex:
                    if state:
                        persist()

        thread=threading.Thread(target=heartbeat,daemon=True)
        thread.start()

        def execute(path,run,stage,contract):
            if not run.exists():
                run_selection(path,run,root=REPO,jobs=jobs,threads=threads,dry_run=True)
            checkpoint(stage,run=str(run),**counts(contract))
            with (run/'supervisor.log').open('a',buffering=1) as stream,redirect_stdout(stream):
                result=run_selection(path,run,root=REPO,jobs=jobs,threads=threads,resume=True,retry_failed=retry_failed)
            if result['status']!='complete':
                raise ValueError('Fitting stage did not complete')

        try:
            inner_run,inner_summary=work/'selection-run',work/'selection-summary'
            execute(contract_path,inner_run,'running_selection',c)
            checkpoint('summarizing_selection',run=str(inner_run),summary=str(inner_summary))
            from .selection_summary import summarize as inner_summarize
            if inner_summary.exists():
                verify_artifacts(inner_run,inner_summary,manifest_name='summary_manifest.json')
            else:
                inner_summarize(inner_run,inner_summary,figures=False)
            path=work/'outer-contract.json'
            checkpoint('freezing_outer',run=str(inner_run),summary=str(inner_summary))
            if path.exists():
                outer=json.loads(path.read_text())
                verify_sources(REPO,outer)
                if outer['selection_contract']!=c:
                    raise ValueError('Outer contract belongs to another selection')
            else:
                outer=outer_contract(inner_run,inner_summary,root=REPO)
                freeze(path,outer)
            outer_run,outer_summary=work/'outer-run',work/'outer-summary'
            execute(path,outer_run,'running_outer',outer)
            checkpoint('summarizing_outer',run=str(outer_run),summary=str(outer_summary))
            from .summary import summarize,verify_summary
            report=verify_summary(outer_run,outer_summary) if outer_summary.exists() else summarize(outer_run,outer_summary)
            if (report['status']!='complete' or report['audit']['completed_fits']!=counts(outer)['total_fits']
                    or not report['audit']['predictions_and_metrics_recomputed'] or not report['audit']['all_five_seeds_present']):
                raise ValueError('Final audit did not confirm complete evaluation')
            checkpoint('complete',run=str(outer_run),summary=str(outer_summary),audit=report['audit'],
                       selection_fits=counts(c)['total_fits'])
            return dict(state)
        except BaseException:
            checkpoint('interrupted' if isinstance(sys.exc_info()[1],KeyboardInterrupt) else 'failed',
                       error=traceback.format_exc())
            raise
        finally:
            stop.set()
            thread.join(timeout=2)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('contract','output','work-output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=16)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--retry-failed',action='store_true')
    args=parser.parse_args()
    run_pipeline(args.contract,args.output,args.work_output,jobs=args.jobs,threads=args.threads,
                 resume=args.resume,retry_failed=args.retry_failed)
