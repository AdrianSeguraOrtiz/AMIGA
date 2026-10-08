"""Run an explicitly frozen complete-grid selection and generate its figures."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import os
import sys
from pathlib import Path
import time
import traceback

from scripts.experiments.amiga_exp.grouped_validation.pilot import sha256, utc, write_json
from .runner import environment, run_selection, verify_sources
from .spec import counts, REPO
from .summary import summarize


def run_full(contract_path, output, run_output, summary_output, *, jobs, threads):
    contract_path, output, run_output, summary_output = [Path(p).resolve() for p in
                                                        (contract_path, output, run_output, summary_output)]
    if output.exists() or run_output.exists() or summary_output.exists():
        raise ValueError('Pipeline, execution and summary destinations must be new')
    contract = json.loads(contract_path.read_text())
    if contract['mode'] != 'selection' or contract['grid_profile'] != 'original':
        raise ValueError('This pipeline requires the complete original grids')
    verify_sources(REPO, contract)
    output.mkdir(parents=True)
    manifest = dict(created_at_utc=utc(), contract=str(contract_path), run=str(run_output),
                    summary=str(summary_output), environment=environment(),
                    contract_sha256=sha256(contract_path), workers=jobs, threads=threads,
                    grid_profile='original', automatic_grid_reduction=False)
    write_json(output / 'manifest.json', manifest)

    def checkpoint(state, **extra):
        value = dict(status=state, updated_at_utc=utc(), accounted_at_unix=time.time(),
                     supervisor_pid=os.getpid(), run=str(run_output),
                     summary=str(summary_output), **extra)
        write_json(output / 'state.json', value)
        print(json.dumps(value), flush=True)

    try:
        run_selection(contract_path, run_output, jobs=jobs, threads=threads, dry_run=True)
        checkpoint('running_selection', grid_profile=contract['grid_profile'], **counts(contract))
        # run_selection maintains a separate live heartbeat and per-job logs.
        with (run_output/'supervisor.log').open('w', buffering=1) as stream, redirect_stdout(stream):
            result = run_selection(contract_path, run_output, jobs=jobs, threads=threads, resume=True)
        if result['status'] != 'complete':
            raise RuntimeError('Selection did not complete')
        checkpoint('summarizing', grid_profile=contract['grid_profile'])
        report = summarize(run_output, summary_output)
        checkpoint('complete', grid_profile=contract['grid_profile'],
                   selected_procedures=report['selected_procedures'])
    except BaseException:
        checkpoint('interrupted' if isinstance(sys.exc_info()[1], KeyboardInterrupt) else 'failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('contract', 'output', 'run-output', 'summary-output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--jobs',type=int,required=True)
    parser.add_argument('--threads',type=int,required=True)
    args = parser.parse_args()
    run_full(args.contract, args.output, args.run_output, args.summary_output, jobs=args.jobs, threads=args.threads)
