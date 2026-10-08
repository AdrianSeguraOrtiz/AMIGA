"""Quality-blind comparison of parallel CPU layouts on real training fronts."""
from __future__ import annotations

import argparse
from itertools import product
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

from scripts.experiments.amiga_exp.grouped_validation.pilot import sha256, terminate_process_group, utc, write_json
from .execution import load_rows
from .models import prepare, fit, feature_importance, FAMILIES, ARMS
from .resources import cpu_limits, layout, worker_command, worker_environment
from .spec import build_contract, endpoint, REPO

MODULE = 'scripts.experiments.amiga_exp.sequential_selection.throughput'


def worker(directory, task, threads):
    configuration = json.loads((directory/'configuration.json').read_text())
    c = configuration['training_contract']
    case, family, arm = configuration['tasks'][task]
    ids = c['split_contract']['outer_folds'][0]['inner_folds'][0]['train_front_ids']
    mapping = {int(k):v for k,v in c['split_contract']['topology_by_front'].items()}
    started = time.monotonic()
    frame = load_rows(REPO, c, case, ids)
    data = prepare(frame, case, mapping, c['split_contract']['cases'][case]['feature_columns'])
    params = dict(endpoint(family), iterations=configuration['iterations'])
    fitted_at = time.monotonic()
    model, fitting = fit(data, family, arm, params, threads=threads)
    fit_seconds = time.monotonic()-fitted_at
    shap_at = time.monotonic()
    _, sampling = feature_importance(model, family, data, threads=threads,
                                     maximum=configuration['shap_rows_per_front'])
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return dict(task=task, case=case, family=family, arm=arm, threads=threads,
                seconds=time.monotonic()-started, fit_seconds=fit_seconds,
                shap_seconds=time.monotonic()-shap_at,
                process_cpu_seconds=usage.ru_utime+usage.ru_stime,
                peak_rss_mib=usage.ru_maxrss/1024, affinity=sorted(os.sched_getaffinity(0)),
                fitting=fitting, training_rows=len(frame), sampling=sampling,
                validation_metrics_computed=False, outer_predictions_computed=False)


def _cpu_stat():
    path = Path('/proc/stat')
    return list(map(int,path.read_text().splitlines()[0].split()[1:9])) if path.exists() else None


def benchmark(output, *, iterations=1000, repeats=4):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new throughput directory')
    limits = cpu_limits()
    profiles = [layout(limits['capacity']//threads, threads, limits) for threads in (8,4,2,1)
                if limits['capacity'] >= threads]
    contract = build_contract(mode='pilot')
    tasks = list(product(contract['split_contract']['cases'], FAMILIES, ARMS))*repeats
    output.mkdir(parents=True)
    config = dict(training_contract=contract, tasks=tasks, iterations=iterations,
                  shap_rows_per_front=8, profiles=profiles, no_quality_evaluation=True,
                  global_timeout_seconds=2400, per_task_timeout_seconds=600,
                  decision_rule='minimum batch wall time over identical training tasks')
    write_json(output/'configuration.json',config)
    started = time.monotonic()
    results, active = [], []
    prior = {}
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Throughput benchmark received signal {signum}')
    for sig in (signal.SIGINT,signal.SIGTERM):
        prior[sig] = signal.signal(sig,interrupt)
    def checkpoint(status, **extra):
        write_json(output/'state.json',dict(status=status, supervisor_pid=os.getpid(),
                   updated_at_utc=utc(), accounted_at_unix=time.time(),
                   elapsed_seconds=time.monotonic()-started, completed_profiles=len(results),
                   total_profiles=len(profiles), **extra))
    try:
        for profile in profiles:
            name = f'{profile["workers"]}x{profile["threads"]}'
            folder = output/name
            folder.mkdir()
            pending = list(range(len(tasks)))
            complete = []
            before_cpu = _cpu_stat()
            before = time.monotonic()
            while pending or active:
                if time.monotonic()-started > config['global_timeout_seconds']:
                    raise TimeoutError('Throughput benchmark budget exhausted')
                for item in list(active):
                    process,task,slot,launched,stream = item
                    if time.monotonic()-launched > config['per_task_timeout_seconds']:
                        raise TimeoutError(f'Throughput task timeout: {name}/{task}')
                    if process.poll() is None:
                        continue
                    stream.close()
                    active.remove(item)
                    if process.returncode:
                        raise RuntimeError(f'Throughput task failed: {name}/{task}')
                    record = json.loads((folder/f'{task:03d}.json').read_text())
                    assert not record['validation_metrics_computed'] and not record['outer_predictions_computed']
                    complete.append(record)
                while pending and len(active) < profile['workers']:
                    task = pending.pop(0)
                    slot = next(i for i in range(profile['workers']) if i not in {p[2] for p in active})
                    stream = (folder/f'{task:03d}.log').open('w')
                    args = ['--worker','--output',output,'--task',task,'--threads',profile['threads'],
                            '--result',folder/f'{task:03d}.json']
                    process = subprocess.Popen(worker_command(MODULE,args,profile['cpu_sets'][slot]),
                        cwd=REPO,env=worker_environment(profile['threads']),stdout=stream,
                        stderr=subprocess.STDOUT,start_new_session=True)
                    active.append((process,task,slot,time.monotonic(),stream))
                checkpoint('running',profile=name,completed_tasks=len(complete),total_tasks=len(tasks),
                           active_tasks=len(active))
                time.sleep(.2)
            elapsed = time.monotonic()-before
            after_cpu = _cpu_stat()
            cpu_seconds = sum(r['process_cpu_seconds'] for r in complete)
            row = dict(profile=profile, wall_seconds=elapsed, tasks=len(tasks),
                       tasks_per_hour=len(tasks)/elapsed*3600,
                       process_cpu_seconds=cpu_seconds, mean_busy_cpu_equivalents=cpu_seconds/elapsed,
                       maximum_worker_rss_mib=max(r['peak_rss_mib'] for r in complete))
            if before_cpu is not None:
                delta = [b-a for a,b in zip(before_cpu,after_cpu)]
                row.update(machine_busy_fraction=1-(delta[3]+delta[4])/sum(delta),
                           hypervisor_steal_fraction=delta[7]/sum(delta))
            results.append(row)
            write_json(output/'partial_results.json',results)
            print(json.dumps(row),flush=True)
        for name,digest in contract['source_hashes'].items():
            if sha256(REPO/name) != digest:
                raise ValueError(f'Source changed during throughput measurement: {name}')
        selected = min(results,key=lambda r:r['wall_seconds'])
        report = dict(status='complete',results=results,selected_profile=selected['profile'],
                      configuration_sha256=sha256(output/'configuration.json'),
                      quality_metrics_computed=False,
                      interpretation='Hardware calibration at 1000 iterations; full selection retains 3000.')
        write_json(output/'report.json',report)
        checkpoint('complete',selected_workers=selected['profile']['workers'],
                   selected_threads=selected['profile']['threads'])
        return report
    except BaseException as exc:
        checkpoint('failed',error=str(exc))
        raise
    finally:
        for process,task,slot,launched,stream in active:
            terminate_process_group(process)
            stream.close()
        for sig,handler in prior.items():
            signal.signal(sig,handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--task',type=int)
    parser.add_argument('--threads',type=int)
    parser.add_argument('--result',type=Path)
    args = parser.parse_args()
    if args.worker:
        write_json(args.result,worker(args.output,args.task,args.threads))
    else:
        benchmark(args.output)
