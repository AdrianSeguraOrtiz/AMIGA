"""Execute phase 4 and automatically audit, summarize and plot all results."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
import time
import traceback

from scripts.experiments.amiga_exp.grouped_validation.pilot import sha256, utc, write_json
from .runner import environment, run_evaluation
from .spec import REPO, counts, verify_sources
from .summary import summarize


def run_full(contract_path, output, run_output, summary_output, *, jobs, threads):
    contract_path, output, run_output, summary_output = [Path(p).resolve() for p in
                                                        (contract_path, output, run_output, summary_output)]
    if any(p.exists() for p in (output, run_output, summary_output)):
        raise ValueError('Launch, run and summary destinations must be new')
    contract = json.loads(contract_path.read_text())
    verify_sources(REPO, contract)
    output.mkdir(parents=True)
    write_json(output / 'manifest.json', dict(created_at_utc=utc(), contract=str(contract_path),
               contract_sha256=sha256(contract_path), run=str(run_output), summary=str(summary_output),
               environment=environment(), workers=jobs, threads=threads, counts=counts(contract)))

    def checkpoint(status, **extra):
        value = dict(status=status, updated_at_utc=utc(), accounted_at_unix=time.time(),
                     supervisor_pid=os.getpid(), run=str(run_output), summary=str(summary_output), **extra)
        write_json(output / 'state.json', value)
        print(json.dumps(value), flush=True)

    try:
        run_evaluation(contract_path, run_output, jobs=jobs, threads=threads, dry_run=True)
        checkpoint('running_evaluation', **counts(contract))
        with (run_output / 'supervisor.log').open('w', buffering=1) as stream, redirect_stdout(stream):
            result = run_evaluation(contract_path, run_output, jobs=jobs, threads=threads, resume=True)
        if result['status'] != 'complete':
            raise RuntimeError('Outer evaluation did not complete')
        checkpoint('summarizing')
        report = summarize(run_output, summary_output)
        checkpoint('complete', audit=report['audit'])
    except BaseException:
        checkpoint('interrupted' if isinstance(sys.exc_info()[1], KeyboardInterrupt) else 'failed',
                   error=traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('contract', 'output', 'run-output', 'summary-output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--jobs', type=int, required=True)
    parser.add_argument('--threads', type=int, required=True)
    args = parser.parse_args()
    run_full(args.contract, args.output, args.run_output, args.summary_output, jobs=args.jobs, threads=args.threads)
