"""Training-only mask refits, five-seed outer fits, and fixed baseline scores."""
from __future__ import annotations

import json
import math
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from scripts.experiments.amiga_exp.grouped_validation.baselines import score_front
from scripts.experiments.amiga_exp.grouped_validation.metrics import front_metrics
from scripts.experiments.amiga_exp.grouped_validation.pilot import sha256, utc, write_json
from . import models


def completed_job(run, job):
    expected = ({'mask.json', 'feature_importance.csv'} if job['stage'] == 'mask' else
                {'metrics.csv', 'predictions.csv', 'model.json'} if job['stage'] == 'outer' else
                {'metrics.csv', 'predictions.csv'})
    successful = []
    for path in sorted((Path(run) / 'jobs' / job['id']).glob('attempt-*/result.json')):
        report = json.loads(path.read_text())
        if report.get('status') != 'complete':
            continue
        if report.get('job') != job or set(report.get('artifacts', {})) != expected:
            raise ValueError(f'Completion identity or inventory differs: {path}')
        for name, digest in report['artifacts'].items():
            if not (path.parent / name).is_file() or sha256(path.parent / name) != digest:
                raise ValueError(f'Completed artifact changed: {path.parent / name}')
        successful.append((report, path.parent))
    if len(successful) > 1:
        raise ValueError('Multiple successful attempts for one job')
    return successful[0] if successful else None


def load_rows(root, contract, case, fronts, features, *, labels=True):
    """Predictions can load identifiers and predictors without reading quality."""
    columns = ['front_id', 'item_id', *(['AUPR'] if labels else []), *features]
    source = Path(root) / contract['split_contract']['cases'][case]['data_path']
    parts = [part.loc[part['front_id'].isin(fronts)] for part in
             pd.read_csv(source, usecols=list(dict.fromkeys(columns)), chunksize=10000)]
    frame = pd.concat(parts, ignore_index=True).sort_values(['front_id', 'item_id']).reset_index(drop=True)
    if set(frame['front_id']) != set(fronts) or frame.duplicated(['front_id', 'item_id']).any():
        raise ValueError('Front/candidate coverage differs')
    return frame


def learn_mask(prepared, procedure, *, threads):
    """Fit only the parents needed to reach the preselected feature fraction."""
    fraction = procedure['fraction']
    stop = models.FRACTIONS.index(fraction)
    original = list(prepared['feature_names'])
    current = list(range(len(original)))
    paths, importances = [], []
    for parent, child in zip(models.FRACTIONS[:stop], models.FRACTIONS[1:stop + 1]):
        data = dict(prepared, X=prepared['X'][:, current], feature_names=[original[i] for i in current])
        start = time.monotonic()
        model, fit_report = models.fit(data, procedure['family'], procedure['arm'],
                                       procedure['config'], seed=1101, threads=threads)
        fit_seconds = time.monotonic() - start
        start = time.monotonic()
        importance, sampling = models.feature_importance(model, procedure['family'], data, threads=threads)
        importances.append(importance.assign(fraction=parent))
        ordered = importance.sort_values(['centered_mean_abs_shap', 'feature'], ascending=[False, True])
        keep = set(ordered['feature'].iloc[:math.ceil(len(original) * child)])
        paths.append(dict(fraction=parent, features=data['feature_names'], n_features=len(current),
                          fit_seconds=fit_seconds, shap_seconds=time.monotonic() - start,
                          sampling=sampling, **fit_report))
        current = [i for i in current if original[i] in keep]
    features = [original[i] for i in current]
    if len(features) != procedure['n_features']:
        raise ValueError('Relearned feature count differs from selected budget')
    return features, paths, pd.concat(importances, ignore_index=True)


def selected_features(contract, job, plan, run):
    full = contract['split_contract']['cases'][job['case']]['feature_columns']
    if job['procedure']['fraction'] == 1.0:
        if job['dependencies']:
            raise ValueError('Full representation must not depend on a mask')
        return full, None
    if len(job['dependencies']) != 1:
        raise ValueError('Reduced representation needs exactly one mask dependency')
    parent = plan[job['dependencies'][0]]
    if (parent['stage'] != 'mask' or parent['procedure'] != job['procedure']
            or parent['train_front_ids'] != job['train_front_ids'] or parent['evaluation_front_ids']):
        raise ValueError('Feature-mask dependency uses a different training scope')
    complete = completed_job(run, parent)
    if complete is None:
        raise ValueError('Feature-mask dependency is incomplete')
    _, directory = complete
    mask = json.loads((directory / 'mask.json').read_text())
    features = mask['features']
    if (mask['train_front_ids'] != job['train_front_ids'] or mask['training_seed'] != 1101
            or len(features) != job['procedure']['n_features']
            or len(set(features)) != len(features) or not set(features) <= set(full)
            or features != [f for f in full if f in features]):
        raise ValueError('Invalid learned feature mask')
    return features, dict(job_id=parent['id'], result_sha256=sha256(directory / 'result.json'))


def execute_job(root, contract, job, plan, run, destination, threads):
    started = time.monotonic()
    destination = Path(destination)
    report = dict(job=job, status='running', started_at_utc=utc(), classification_policy=dict(models.TARGET))
    names = []

    def csv(name, frame):
        frame.to_csv(destination / name, index=False)
        names.append(name)

    base = contract['split_contract']
    mapping = {int(k): v for k, v in base['topology_by_front'].items()}
    if {mapping[f] for f in job['train_front_ids']} & {mapping[f] for f in job['evaluation_front_ids']}:
        raise ValueError('Training and evaluation topologies overlap')
    if job['stage'] == 'baselines':
        info = contract['cases'][job['case']]
        frame = load_rows(root, contract, job['case'], job['evaluation_front_ids'], info['objective_columns'])
        metrics, predictions, diagnostics = [], [], []
        for front, group in frame.groupby('front_id', sort=True):
            scores, diagnostic = score_front(group.drop(columns='AUPR'), info['objective_columns'],
                                              info['objective_directions'])
            scores['oracle'] = group['AUPR'].to_numpy()
            if set(scores) != set(info['baseline_ids']):
                raise ValueError('Baseline coverage differs from the contract')
            pred = group[['front_id', 'item_id']].copy()
            for method, values in scores.items():
                pred[method] = values
                metrics.append(front_metrics(group, values).assign(method=method))
            predictions.append(pred)
            diagnostics.append(dict(front_id=int(front), **diagnostic))
        csv('metrics.csv', pd.concat(metrics, ignore_index=True))
        csv('predictions.csv', pd.concat(predictions, ignore_index=True))
        report['knee_diagnostics'] = diagnostics
    else:
        procedure = job['procedure']
        if job['stage'] == 'mask':
            features = base['cases'][job['case']]['feature_columns']
            dependency = None
        else:
            features, dependency = selected_features(contract, job, plan, run)
        training = load_rows(root, contract, job['case'], job['train_front_ids'], features)
        prepared = models.prepare(training, job['case'], mapping, features, procedure['label'], seed=1101)
        report['training'] = prepared['report']
        if job['stage'] == 'mask':
            if job['evaluation_front_ids'] or job['seed'] != 1101:
                raise ValueError('A mask job must use training data and selection seed only')
            retained, paths, importance = learn_mask(prepared, procedure, threads=threads)
            write_json(destination / 'mask.json', dict(features=retained, paths=paths,
                       training_seed=1101, train_front_ids=job['train_front_ids'], procedure=procedure,
                       scope='outer_training_only', shared_across_final_seeds=True))
            names.append('mask.json')
            csv('feature_importance.csv', importance)
            report['fit_reports'] = paths
        else:
            start = time.monotonic()
            model, fitting = models.fit(prepared, procedure['family'], job['arm'],
                                        procedure['config'], seed=job['seed'], threads=threads)
            report.update(fit_seconds=time.monotonic() - start, fitting=fitting, mask_dependency=dependency)
            # Held-out AUPR is not loaded until after predictions have been saved.
            evaluation = load_rows(root, contract, job['case'], job['evaluation_front_ids'], features, labels=False)
            scores = models.scores(model, procedure['family'], job['arm'], evaluation[features].to_numpy(), threads)
            csv('predictions.csv', evaluation[['front_id', 'item_id']].assign(score=scores))
            labels = load_rows(root, contract, job['case'], job['evaluation_front_ids'], [], labels=True)
            if not evaluation[['front_id', 'item_id']].equals(labels[['front_id', 'item_id']]):
                raise ValueError('Quality labels and saved predictions are misaligned')
            csv('metrics.csv', front_metrics(labels, scores))
            params = model.get_all_params() if procedure['family'] == 'CatBoost' else model.get_params()
            write_json(destination / 'model.json', dict(procedure=procedure, feature_columns=features,
                       seed=job['seed'], relevance_label_seed=1101, train_front_ids=job['train_front_ids'],
                       evaluation_front_ids=job['evaluation_front_ids'], mask_dependency=dependency,
                       effective_parameters={k: v for k, v in params.items()
                                             if isinstance(v, (str, int, bool, type(None), list, dict))
                                             or isinstance(v, float) and np.isfinite(v)},
                       fitting=fitting, evaluation_scope='outer_held_out_topologies'))
            names.append('model.json')
    report.update(status='complete', ended_at_utc=utc(), seconds=time.monotonic() - started,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  artifacts={name: sha256(destination / name) for name in names},
                  outer_predictions_computed=job['stage'] == 'outer')
    write_json(destination / 'result.json', report)
    return report
