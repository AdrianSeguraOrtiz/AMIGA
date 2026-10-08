"""Bounded subprocess execution with dependency checks and verified resumption."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback

from .execution import completed_job, execute_job
from scripts.experiments.amiga_exp.grouped_validation.pilot import sha256, terminate_process_group, utc, write_json
from .spec import build_plan, validate_contract
from .resources import layout, worker_command, worker_environment

REPO = Path(__file__).resolve().parents[4]
MODULE = 'scripts.experiments.amiga_exp.sequential_selection.runner'


def environment():
    return dict(python=sys.version, platform=platform.platform(), machine=platform.machine(),
                packages={name: importlib.metadata.version(name) for name in
                          ('catboost', 'lightgbm', 'xgboost', 'numpy', 'pandas', 'scikit-learn', 'scipy', 'matplotlib', 'seaborn')})


def verify_sources(root: Path, contract: dict):
    validate_contract(contract)
    for relative, digest in contract['source_hashes'].items():
        if sha256(root / relative) != digest:
            raise ValueError(f'Source changed after the contract was frozen: {relative}')


def read_run(run: Path, *, verify_environment=True):
    manifest = json.loads((run / 'manifest.json').read_text())
    for name in ('contract.json', 'plan.json'):
        if sha256(run / name) != manifest['artifacts'][name]:
            raise ValueError(f'Run definition changed: {name}')
    contract = json.loads((run / 'contract.json').read_text())
    plan = json.loads((run / 'plan.json').read_text())
    if plan != build_plan(contract):
        raise ValueError('Stored job plan differs from the frozen specification')
    if verify_environment and manifest['environment'] != environment():
        raise ValueError('Runtime environment changed; resuming would mix environments')
    return manifest, contract, plan


def worker(run: Path, job_id: str, destination: Path):
    guard = (run / '.workers.lock').open('a')
    try:
        fcntl.flock(guard, fcntl.LOCK_SH)
        manifest, contract, jobs = read_run(run)
        root = Path(manifest['repo_root'])
        verify_sources(root, contract)
        plan = {j['id']: j for j in jobs}
        job = plan[job_id]
        if destination.parent != run / 'jobs' / job_id or not destination.name.startswith('attempt-'):
            raise ValueError('Worker destination differs from the planned job directory')
        execute_job(root, contract, job, plan, run, destination, manifest['threads'])
        return 0
    except Exception:
        write_json(destination / 'result.json', dict(status='failed', job_id=job_id,
                   ended_at_utc=utc(), error=traceback.format_exc()))
        return 1
    finally:
        guard.close()


def _interrupt(signum, frame):
    raise KeyboardInterrupt(f'Sequential supervisor received signal {signum}')


def run_selection(contract_path: Path, output: Path, *, root=REPO, jobs=2, threads=8,
                   dry_run=False, resume=False, retry_failed=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    resources = layout(jobs, threads)
    if retry_failed and not resume:
        raise ValueError('Explicit retry requires --resume')
    contract = json.loads(Path(contract_path).read_text())
    verify_sources(root, contract)
    plan = build_plan(contract)
    if not output.exists():
        if resume:
            raise ValueError('Cannot resume a run that does not exist')
        output.mkdir(parents=True)
    elif not resume:
        raise ValueError('Output already exists; use a new directory or explicit --resume')
    with (output / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another supervisor holds this run lock') from exc
        with (output / '.workers.lock').open('a') as workers_lock:
            try:
                fcntl.flock(workers_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError('Workers from an earlier supervisor are still active; wait for them to finish') from exc
        if resume:
            manifest, stored, _ = read_run(output)
            if stored != contract or manifest['repo_root'] != str(root):
                raise ValueError('Cannot resume with a different contract or repository')
            if (manifest['workers'], manifest['threads']) != (jobs, threads):
                raise ValueError('Cannot change worker/thread resources when resuming')
            if manifest.get('resources') != resources:
                raise ValueError('CPU allocation changed; cannot mix resource layouts')
            state = json.loads((output / 'state.json').read_text())
        else:
            write_json(output / 'contract.json', contract)
            write_json(output / 'plan.json', plan)
            manifest = dict(created_at_utc=utc(), repo_root=str(root), workers=jobs, threads=threads, resources=resources,
                            environment=environment(), artifacts={p: sha256(output / p) for p in
                                                                  ('contract.json', 'plan.json')})
            write_json(output / 'manifest.json', manifest)
            state = dict(status='planned', elapsed_seconds=0.0, completed_jobs=0)
            write_json(output / 'state.json', state)
        done, failed = set(), []
        for job in plan:
            if completed_job(output, job):
                done.add(job['id'])
            elif list((output / 'jobs' / job['id']).glob('attempt-*')):
                failed.append(job['id'])
        for job in plan:
            if job['id'] in done and not set(job['dependencies']) <= done:
                raise ValueError('Completed job has missing configuration-selection dependencies')
        if failed and not retry_failed:
            raise ValueError('Failed or interrupted attempts exist; inspect logs, then use --resume --retry-failed')
        if dry_run:
            return dict(status='planned', planned_jobs=len(plan), planned_fits=sum(j['planned_fits'] for j in plan),
                        completed_jobs=len(done), output=str(output))
        pending = [j for j in plan if j['id'] not in done]
        active = []
        start = time.monotonic()
        previous = float(state['elapsed_seconds'])
        # After a hard supervisor crash, charge the full unaccounted interval,
        # including downtime. This conservative rule cannot extend the budget.
        if state['status'] == 'running':
            previous += max(0.0, time.time() - state['accounted_at_unix'])
        handlers = {sig: signal.signal(sig, _interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}

        def checkpoint(status, **extra):
            if status in ('running', 'complete'):
                state.pop('error', None)
            state.update(status=status, elapsed_seconds=previous + time.monotonic() - start,
                         accounted_at_unix=time.time(), completed_jobs=len(done), total_jobs=len(plan), updated_at_utc=utc(),
                         supervisor_pid=os.getpid(), active_jobs=[j['id'] for _, j, _, _, _ in active],
                         workers=jobs, threads_per_worker=threads, cpu_budget=resources['cpu_budget'],
                         completed_by_stage={stage: sum(j['id'] in done for j in plan if j['stage']==stage)
                                             for stage in sorted({j['stage'] for j in plan})}, **extra)
            write_json(output / 'state.json', state)

        try:
            checkpoint('running')
            slots = {}
            while pending or active:
                if previous + time.monotonic() - start >= contract['failures']['total_budget_seconds']:
                    raise TimeoutError('Total active execution budget exhausted')
                for item in list(active):
                    process, job, destination, launched, log = item
                    if time.monotonic() - launched > contract['failures']['per_job_timeout_seconds']:
                        raise TimeoutError(f'Per-job time limit reached: {job["id"]}')
                    if process.poll() is None:
                        continue
                    log.close()
                    active.remove(item)
                    del slots[process.pid]
                    if process.returncode or not completed_job(output, job):
                        raise RuntimeError(f'Job failed: {job["id"]}; see {destination}')
                    done.add(job['id'])
                    print(f'Completed {len(done)}/{len(plan)}: {job["id"]}', flush=True)
                for job in list(pending):
                    if len(active) >= jobs:
                        break
                    if not set(job['dependencies']) <= done:
                        continue
                    parent = output / 'jobs' / job['id']
                    parent.mkdir(parents=True, exist_ok=True)
                    attempts = list(parent.glob('attempt-*'))
                    destination = parent / f'attempt-{len(attempts) + 1:03d}'
                    destination.mkdir()
                    log = (destination / 'worker.log').open('w')
                    slot = next(i for i in range(jobs) if i not in slots.values())
                    env = worker_environment(threads)
                    try:
                        command = worker_command(MODULE, ['--worker', '--run', str(output), '--job', job['id'],
                                                          '--attempt', str(destination)], resources['cpu_sets'][slot])
                        process = subprocess.Popen(command, cwd=root, env=env,
                                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    except BaseException:
                        log.close()
                        raise
                    active.append((process, job, destination, time.monotonic(), log))
                    slots[process.pid] = slot
                    pending.remove(job)
                if pending and not active:
                    raise RuntimeError('No runnable job remains; dependency graph is incomplete')
                checkpoint('running')
                if active:
                    time.sleep(0.2)
            verify_sources(root, contract)
            checkpoint('complete')
        except BaseException as exc:
            checkpoint('interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
            raise
        finally:
            for process, job, destination, _, log in active:
                terminate_process_group(process)
                log.close()
                if not (destination / 'result.json').exists():
                    write_json(destination / 'result.json', dict(status='interrupted', job=job, ended_at_utc=utc()))
            state.update(elapsed_seconds=previous + time.monotonic() - start, accounted_at_unix=time.time())
            write_json(output / 'state.json', state)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        return dict(state, output=str(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true', required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--attempt', type=Path, required=True)
    args = parser.parse_args()
    sys.exit(worker(args.run, args.job, args.attempt))
