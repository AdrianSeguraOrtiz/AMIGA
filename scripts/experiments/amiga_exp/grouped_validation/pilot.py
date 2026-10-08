#!/usr/bin/env python3
"""Bounded, training-only CPU timing pilot. Runtime artifacts are written under the experiment results directory."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS_ROOT = REPO / "experiments" / "grouped-validation"
MODULE_NAME = "scripts.experiments.amiga_exp.grouped_validation.pilot"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def terminate_process_group(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def interrupt_supervisor(signum, frame):
    raise KeyboardInterrupt(f'Pilot supervisor received signal {signum}')


def worker(job_path):
    start = time.monotonic()
    job = json.loads(Path(job_path).read_text())
    destination = Path(job['result_path'])
    result = {'job': job, 'started_at_utc': utc(), 'status': 'running'}
    try:
        import numpy as np
        from catboost import CatBoostRanker, CatBoostRegressor, CatBoostClassifier, Pool
        from scripts.experiments.amiga_exp.grouped_validation.training_data import prepare_training

        if sha256(job['contract_path']) != job['contract_sha256']:
            raise ValueError('Contract changed after pilot was planned')
        contract = json.loads(Path(job['contract_path']).read_text())
        if job['train_front_ids'] != contract['pilot']['train_front_ids']:
            raise ValueError('Worker front IDs differ from the contracted pilot training scope')
        if job['arm'] not in contract['arms'] or not 1 <= job['threads'] <= 8:
            raise ValueError('Worker formulation or resources outside pilot contract')
        if job['params'] not in [dict(depth=d, learning_rate=0.03, l2_leaf_reg=3,
                                       iterations=3000) for d in (4, 8)]:
            raise ValueError('Worker configuration differs from the fixed timing endpoints')
        case = contract['cases'][job['case']]
        if sha256(REPO / case['data_path']) != case['sha256']:
            raise ValueError('Dataset hash differs from pilot contract')
        data_start = time.monotonic()
        prepared = prepare_training(
            REPO, job['case'], job['train_front_ids'],
            {int(k): v for k, v in contract['topology_by_front'].items()},
            case['feature_columns'],
        )
        result['data_preparation_seconds'] = time.monotonic() - data_start
        result['training_data'] = prepared['report']
        params = dict(job['params'])
        params.update(random_seed=contract['seeds']['tuning'], task_type='CPU',
                      thread_count=job['threads'], allow_writing_files=False,
                      use_best_model=False, verbose=False)
        arm = job['arm']
        common_pool = dict(data=prepared['X'], label=prepared['labels'][arm],
                           feature_names=prepared['feature_names'], thread_count=job['threads'])
        if arm == 'ltr_catboost':
            pool = Pool(**common_pool, group_id=prepared['group_id'],
                        group_weight=prepared['group_weight'])
            model = CatBoostRanker(loss_function='YetiRank', **params)
        else:
            pool = Pool(**common_pool, weight=prepared['point_weight'])
            if arm == 'clf_top20':
                model = CatBoostClassifier(loss_function='Logloss', **params)
            else:
                model = CatBoostRegressor(loss_function='RMSE', **params)
        fit_start = time.monotonic()
        model.fit(pool)  # No eval_set, stopping metric, or external-fold predictions.
        result['fit_seconds'] = time.monotonic() - fit_start
        if model.tree_count_ != params['iterations']:
            raise ValueError('Unexpected number of trained trees')
        effective = model.get_all_params()
        if effective.get('use_best_model') is not False:
            raise ValueError('use_best_model must remain disabled')
        if effective.get('od_type') not in (None, 'None'):
            raise ValueError('Unexpected overfitting detector')
        prediction_start = time.monotonic()
        if arm == 'clf_top20':
            score = model.predict_proba(prepared['X'], thread_count=job['threads'])[:, 1]
            if not np.all((score >= 0) & (score <= 1)):
                raise ValueError('Invalid predicted probabilities')
        else:
            score = model.predict(prepared['X'], thread_count=job['threads'])
        if score.shape != (len(prepared['X']),) or not np.isfinite(score).all():
            raise ValueError('Invalid training prediction shape or values')
        result['training_prediction_seconds'] = time.monotonic() - prediction_start
        result.update(status='complete', tree_count=int(model.tree_count_),
                      effective_parameters=effective, training_scores_finite=True,
                      external_predictions_computed=False,
                      quality_metrics_computed=False)
    except Exception:
        result.update(status='failed', error=traceback.format_exc())
    result.update(total_seconds=time.monotonic() - start, ended_at_utc=utc(),
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    write_json(destination, result)
    return 0 if result['status'] == 'complete' else 1


def timing_projection(results, jobs):
    """Bounded scenario, not a guarantee or a quality-based budget decision."""
    measured = {(r['job']['case'], r['job']['arm'], r['job']['params']['depth']): r
                for r in results if r['status'] == 'complete'}
    cases = ('BIO-INSIGHT', 'MO-GENECI')
    arms = ('ltr_catboost', 'reg_aupr', 'reg_normalized', 'clf_top20')
    if any((c, a, d) not in measured for c in cases for a in arms for d in (4, 8)):
        return {'status': 'incomplete_measurements',
                'reason': 'Need both depth endpoints for every case and formulation'}
    # Use depth-8 time for all 6 grid candidates and final fits. Scale every fit
    # up to all 31,200 rows; curves and leave-family-out are thus overestimated.
    blocks = dict(core=0.0, learning_curve=0.0, ablation=0.0,
                  family_exclusions=0.0, deployment=0.0)
    for case in cases:
        for arm in arms:
            r = measured[case, arm, 8]
            n = r['training_data']['usable_rows']
            unit = (r['fit_seconds'] + r['data_preparation_seconds'] +
                    r['training_prediction_seconds']) * max(1.0, 31200 / n)
            blocks['core'] += unit * (5 * 6 * 3 + 5 * 5)
            blocks['deployment'] += unit * (6 * 3 + 1)
            if arm in ('ltr_catboost', 'reg_normalized'):
                blocks['learning_curve'] += unit * 5 * 10
                blocks['family_exclusions'] += unit * 6
            if arm == 'ltr_catboost':
                blocks['ablation'] += unit * 5 * 8
    hours_serial = sum(blocks.values()) / 3600
    return {
        'status': 'scenario_from_measured_endpoints',
        'assumptions': [
            'Depth 8 runtime substituted for depths 4 and 6 and all selected models',
            'Each fit scaled linearly to 31,200 rows (upper count of either case)',
            'All 1,376 planned fits included, plus 25 percent processing/variation margin',
            'No accuracy results used; row scaling and parallel efficiency are approximations',
            'Feature regeneration, major code fixes and retries are not measured here',
        ],
        'scaled_serial_hours_by_block': {k: v / 3600 for k, v in blocks.items()},
        'scaled_serial_hours_with_25_percent_margin': hours_serial * 1.25,
        'scaled_parallel_hours_with_margin_and_80_percent_efficiency':
            hours_serial * 1.25 / max(1, jobs * 0.8),
        'jobs': jobs,
    }


def run(args):
    from scripts.experiments.amiga_exp.grouped_validation.contract import validate_contract
    if (not 1 <= args.jobs <= 2 or not 1 <= args.threads <= 8
            or not 1 <= args.budget_seconds <= 3600
            or not 1 <= args.job_timeout_seconds <= min(900, args.budget_seconds)):
        raise ValueError('Pilot limits: 1–2 jobs, 1–8 threads, global ≤3,600 s, per-job ≤900 s')
    path = args.contract.resolve()
    contract = json.loads(path.read_text())
    checks = validate_contract(contract)
    for relative, expected in contract['source_hashes'].items():
        if sha256(REPO / relative) != expected:
            raise ValueError(f'Source changed since contract creation: {relative}')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = {
        'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'versions': {name: importlib.metadata.version(name) for name in
                     ('catboost', 'numpy', 'pandas', 'scikit-learn', 'scipy')},
        'source_sha256': {p.name: sha256(p) for p in
                          (HERE / 'contract.py', HERE / 'training_data.py', Path(__file__))},
    }
    plan = []
    # Representative timing only: fixed learning rate; endpoints of tree depth.
    for depth in (4, 8):
        for case in contract['cases']:
            for arm in contract['arms']:
                identifier = f'{case}__{arm}__d{depth}'
                plan.append(dict(
                    id=identifier, case=case, arm=arm, threads=args.threads,
                    contract_path=str(path), contract_sha256=sha256(path),
                    train_front_ids=contract['pilot']['train_front_ids'],
                    params=dict(depth=depth, learning_rate=0.03, l2_leaf_reg=3,
                                iterations=3000),
                    result_path=str(output / (identifier + '.json')),
                ))
    for job in plan:
        write_json(output / (job['id'] + '.job.json'), job)
    manifest = dict(started_at_utc=utc(), status='planned', environment=environment,
                    contract_validation=checks, contract_sha256=sha256(path),
                    planned_jobs=len(plan), jobs=args.jobs, threads_per_job=args.threads,
                    budget_seconds=args.budget_seconds,
                    per_job_timeout_seconds=args.job_timeout_seconds,
                    external_evaluation=False, plan=plan)
    write_json(output / 'pilot-manifest.json', manifest)
    if args.dry_run:
        print(json.dumps({'status': 'dry_run', 'output': str(output), 'jobs': len(plan)}))
        return 0
    started = time.monotonic()
    running = []
    remaining = list(plan)
    results = []
    failed = False
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    manifest['status'] = 'running'
    write_json(output / 'pilot-manifest.json', manifest)
    previous_sigterm = signal.signal(signal.SIGTERM, interrupt_supervisor)
    try:
        while running or remaining:
            elapsed = time.monotonic() - started
            while remaining and len(running) < args.jobs and not failed and elapsed < args.budget_seconds:
                job = remaining.pop(0)
                log = (output / (job['id'] + '.log')).open('w')
                process = subprocess.Popen(
                    [sys.executable, '-m', MODULE_NAME, '--worker',
                     str(output / (job['id'] + '.job.json'))],
                    cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                running.append((process, job, time.monotonic(), log))
                print(json.dumps({'event': 'started', 'id': job['id'], 'elapsed_seconds': elapsed}), flush=True)
            for entry in list(running):
                process, job, job_start, log = entry
                timeout = (time.monotonic() - started >= args.budget_seconds or
                           time.monotonic() - job_start >= args.job_timeout_seconds)
                if process.poll() is None and timeout:
                    terminate_process_group(process)
                if process.poll() is not None:
                    log.close()
                    result_path = Path(job['result_path'])
                    if result_path.exists():
                        result = json.loads(result_path.read_text())
                    else:
                        result = dict(job=job, status='timeout' if timeout else 'failed',
                                      returncode=process.returncode,
                                      total_seconds=time.monotonic() - job_start)
                        write_json(result_path, result)
                    results.append(result)
                    running.remove(entry)
                    if result['status'] != 'complete':
                        failed = True  # Stop dispatch, preserve all failed evidence.
                    print(json.dumps({'event': 'finished', 'id': job['id'],
                                      'status': result['status'],
                                      'fit_seconds': result.get('fit_seconds')}), flush=True)
            if not running and (failed or time.monotonic() - started >= args.budget_seconds):
                break
            if running:
                time.sleep(0.5)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for process, job, _, log in running:
            terminate_process_group(process)
            log.close()
            result_path = Path(job['result_path'])
            if result_path.exists():
                result = json.loads(result_path.read_text())
            else:
                result = dict(job=job, status='cancelled', returncode=process.returncode)
                write_json(result_path, result)
            results.append(result)
        manifest.update(status='complete' if len(results) == len(plan) and not failed else 'incomplete',
                        ended_at_utc=utc(), elapsed_seconds=time.monotonic() - started,
                        completed_jobs=sum(r['status'] == 'complete' for r in results),
                        unstarted_job_ids=[j['id'] for j in remaining])
        write_json(output / 'pilot-manifest.json', manifest)
        summary = dict(status=manifest['status'], elapsed_seconds=manifest['elapsed_seconds'],
                       completed_jobs=manifest['completed_jobs'], planned_jobs=len(plan),
                       quality_metrics_computed=False, external_predictions_computed=False,
                       projection=timing_projection(results, args.jobs),
                       results=[{k: r.get(k) for k in
                                 ('status', 'fit_seconds', 'total_seconds', 'peak_rss_mib')} |
                                {'id': r['job']['id']} for r in results])
        write_json(output / 'pilot-summary.json', summary)
        print(json.dumps({'event': 'pilot_finished', 'summary': str(output / 'pilot-summary.json'),
                          'status': manifest['status']}), flush=True)
        signal.signal(signal.SIGTERM, previous_sigterm)
    return 0 if manifest['status'] == 'complete' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--contract', type=Path, default=DEFAULT_RESULTS_ROOT / 'protocol-contract.json')
    parser.add_argument('--output', type=Path,
                        default=DEFAULT_RESULTS_ROOT / 'runs' / dt.datetime.now().strftime('pilot-%Y%m%d-%H%M%S'))
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--budget-seconds', type=int, default=3600)
    parser.add_argument('--job-timeout-seconds', type=int, default=900)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    return worker(args.worker) if args.worker else run(args)


if __name__ == '__main__':
    raise SystemExit(main())
