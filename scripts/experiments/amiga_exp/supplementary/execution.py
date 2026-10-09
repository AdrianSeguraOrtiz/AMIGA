"""Fit only the saved AMIGA configuration and predictor columns."""
import json
from pathlib import Path
import resource
import time

import pandas as pd

from scripts.experiments.amiga_exp.grouped_validation.metrics import front_metrics
from scripts.experiments.amiga_exp.grouped_validation.pilot import sha256, utc, write_json
from scripts.experiments.amiga_exp.outer_evaluation.execution import load_rows
from scripts.experiments.amiga_exp.sequential_selection import models


def completed_job(run, job):
    successful = []
    for path in sorted((Path(run) / 'jobs' / job['id']).glob('attempt-*/result.json')):
        report = json.loads(path.read_text())
        if report.get('status') != 'complete':
            continue
        if report.get('job') != job or not report.get('artifacts'):
            raise ValueError('Completion identity or inventory differs')
        required = {'selected_procedures.json', 'fit_reports.json', 'progress.json'}
        seeds = [1201] if job['context_id'] == 'deployment' else range(1201, 1206)
        for seed in seeds:
            required.add(f'ranking-seed-{seed}.model.json')
            if job['context_id'] != 'deployment':
                required.add(f'ranking-seed-{seed}.predictions.csv.gz')
        if job['context_id'] != 'deployment':
            required.add('metrics.csv')
        if not required.issubset(report['artifacts']):
            raise ValueError('Completion inventory omits required scientific artifacts')
        for name, digest in report['artifacts'].items():
            p = Path(name)
            if p.is_absolute() or '..' in p.parts or sha256(path.parent / p) != digest:
                raise ValueError(f'Completed artifact changed: {path.parent / p}')
        if job['context_id'] == 'deployment':
            info = json.loads((path.parent / 'ranking-seed-1201.model.json').read_text())
            if report['artifacts'].get(info['model_file']) != info['model_sha256']:
                raise ValueError('Completion inventory omits the native deployment model')
        successful.append((report, path.parent))
    if len(successful) > 1:
        raise ValueError('Multiple successful attempts')
    return successful[0] if successful else None


def save_model(model, family, prefix):
    path = Path(str(prefix) + {'CatBoost': '.cbm', 'LightGBM': '.txt', 'XGBoost': '.json'}[family])
    if family == 'LightGBM':
        model.booster_.save_model(str(path))
    else:
        model.save_model(str(path))
    return path


def execute_job(root, c, job, plan, run, destination, threads):
    start = time.monotonic()
    scope = next(s for s in c['contexts'] if s['id'] == job['context_id'])
    procedure = job['recipe']['procedure']
    features = job['recipe']['feature_columns']
    mapping = {int(k): v for k, v in c['original']['split_contract']['topology_by_front'].items()}
    if {mapping[f] for f in scope['train_front_ids']} & {mapping[f] for f in scope['test_front_ids']}:
        raise ValueError('Training and evaluation topologies overlap')
    train = load_rows(root, c['original'], job['case'], scope['train_front_ids'], features)
    prepared = models.prepare(train, job['case'], mapping, features, procedure['label'], seed=1101)
    seeds = c['final_seeds'] if scope['kind'] == 'learning' else [c['deployment_seed']]
    metrics, reports = [], []
    write_json(destination / 'selected_procedures.json', [procedure])
    for index, seed in enumerate(seeds):
        write_json(destination / 'progress.json', dict(status='running', completed_fits=index,
            planned_fits=len(seeds), seed=seed, family=procedure['family'], updated_at_utc=utc()))
        before = time.monotonic()
        model, fitting = models.fit(prepared, procedure['family'], 'ranking', procedure['config'],
                                    seed=seed, threads=threads)
        info = dict(procedure=procedure, feature_columns=features, seed=seed,
                    recipe_source=job['recipe']['source_model_metadata'], selection_repeated=False,
                    train_front_ids=scope['train_front_ids'], fitting=fitting,
                    fit_seconds=time.monotonic() - before, training=prepared['report'])
        if scope['kind'] == 'learning':
            evaluation = load_rows(root, c['original'], job['case'], scope['test_front_ids'], features, labels=False)
            score = models.scores(model, procedure['family'], 'ranking', evaluation[features].to_numpy(), threads)
            prediction = evaluation[['front_id', 'item_id']].assign(method='ranking', seed=seed, score=score)
            prediction.to_csv(destination / f'ranking-seed-{seed}.predictions.csv.gz', index=False,
                              compression={'method': 'gzip', 'mtime': 0})
            labels = load_rows(root, c['original'], job['case'], scope['test_front_ids'], [], labels=True)
            if not labels[['front_id', 'item_id']].equals(evaluation[['front_id', 'item_id']]):
                raise ValueError('Prediction identifiers and quality labels differ')
            metrics.append(front_metrics(labels, score).assign(method='ranking', seed=seed))
        else:
            path = save_model(model, procedure['family'], destination / 'ranking')
            info.update(model_file=path.name, model_sha256=sha256(path))
        write_json(destination / f'ranking-seed-{seed}.model.json', info)
        reports.append(dict(method='ranking', seed=seed, family=procedure['family'],
                            fit_seconds=info['fit_seconds'], **fitting))
    if metrics:
        pd.concat(metrics, ignore_index=True).to_csv(destination / 'metrics.csv', index=False)
    write_json(destination / 'fit_reports.json', reports)
    write_json(destination / 'progress.json', dict(status='complete', completed_fits=len(seeds),
        planned_fits=len(seeds), family=procedure['family'], updated_at_utc=utc()))
    artifacts = {p.relative_to(destination).as_posix(): sha256(p) for p in sorted(destination.rglob('*'))
                 if p.is_file() and p.name != 'worker.log'}
    report = dict(status='complete', job=job, ended_at_utc=utc(), seconds=time.monotonic() - start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, artifacts=artifacts,
        scope=scope['kind'], training_size=scope['training_size'], selection_repeated=False)
    write_json(destination / 'result.json', report)
    return report
