"""Execute one declared job and verify immutable completion artifacts."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import resource
import time

import pandas as pd

from .baselines import score_front
from .metrics import front_metrics, select_configuration
from .models import fit_model, predict_scores
from .pilot import sha256, utc, write_json
from .training_data import prepare_training


def completed_job(run: Path, job: dict) -> tuple[dict, Path] | None:
    """A successful marker is usable only while every recorded artifact agrees."""
    directory = run / 'jobs' / job['id']
    successes = []
    for path in sorted(directory.glob('attempt-*/result.json')):
        report = json.loads(path.read_text())
        if report.get('status') != 'complete':
            continue
        if report.get('job') != job or not report.get('artifacts'):
            raise ValueError(f'Completion identity or artifacts differ: {path}')
        expected = ({'model.cbm', 'model.json'} if job.get('stage') == 'deployment' else
                    {'metrics.csv', 'predictions.csv'})
        if 'stage' in job and set(report['artifacts']) != expected:
            raise ValueError(f'Incomplete artifact inventory: {path}')
        for relative, expected in report['artifacts'].items():
            artifact = Path(relative)
            if artifact.is_absolute() or '..' in artifact.parts:
                raise ValueError(f'Invalid artifact path: {relative}')
            source = path.parent / artifact
            if not source.is_file() or sha256(source) != expected:
                raise ValueError(f'Completed artifact changed or is missing: {source}')
        successes.append((report, path.parent))
    if len(successes) > 1:
        raise ValueError(f'Multiple completed attempts for {job["id"]}')
    return successes[0] if successes else None


def select_job_config(contract: dict, job: dict, plan: dict, run: Path):
    if job['config'] is not None:
        return job['config'], None
    by_config = defaultdict(list)
    for dependency in job['dependencies']:
        upstream = plan[dependency]
        completed = completed_job(run, upstream)
        if completed is None:
            raise ValueError(f'Incomplete configuration dependency: {dependency}')
        _, directory = completed
        metrics = pd.read_csv(directory / 'metrics.csv')
        if set(metrics['front_id']) != set(upstream['evaluation_front_ids']):
            raise ValueError('Inner validation coverage differs from its planned fold')
        by_config[upstream['config_id']].append(metrics)
    grid = contract['split_contract']['grid']
    if set(by_config) != {c['id'] for c in grid} or any(len(v) != 3 for v in by_config.values()):
        raise ValueError('Configuration selection requires all six configurations and three folds')
    candidates = {key: pd.concat(parts, ignore_index=True) for key, parts in by_config.items()}
    if any(set(frame['front_id']) != set(job['train_front_ids']) for frame in candidates.values()):
        raise ValueError('Inner validation must cover exactly the final training complement')
    winner, evidence = select_configuration(candidates, contract['split_contract']['topology_by_front'])
    return next(c for c in grid if c['id'] == winner), dict(selected=winner, candidates=evidence,
                                                         dependencies=job['dependencies'])


def load_evaluation(root: Path, contract: dict, job: dict, features: list[str]) -> pd.DataFrame:
    source = root / contract['split_contract']['cases'][job['case']]['data_path']
    columns = list(dict.fromkeys(['front_id', 'item_id', 'AUPR', *features]))
    chunks = [part.loc[part['front_id'].isin(job['evaluation_front_ids'])]
              for part in pd.read_csv(source, usecols=columns, chunksize=10000)]
    data = pd.concat(chunks, ignore_index=True).sort_values(['front_id', 'item_id']).reset_index(drop=True)
    if set(data['front_id']) != set(job['evaluation_front_ids']):
        raise ValueError('Evaluation front coverage differs from the plan')
    return data


def execute_job(root: Path, contract: dict, job: dict, plan: dict, run: Path,
                destination: Path, threads: int) -> dict:
    """Write the completion marker last; failed/interrupted attempts remain separate."""
    start = time.monotonic()
    report = dict(job=job, status='running', started_at_utc=utc())
    artifacts = []

    def csv(name, frame):
        frame.to_csv(destination / name, index=False)
        artifacts.append(name)

    case = contract['cases'][job['case']]
    features = case['feature_sets'][job['feature_set']]
    if job['stage'] == 'baselines':
        frame = load_evaluation(root, contract, job, case['objective_columns'])
        metrics, predictions, diagnostics = [], [], []
        for front_id, group in frame.groupby('front_id', sort=True):
            scores, diagnostic = score_front(group, case['objective_columns'], case['objective_directions'],
                                             include_oracle=True)
            diagnostics.append(dict(front_id=int(front_id), **diagnostic))
            prediction = group[['front_id', 'item_id']].copy()
            for method, score in scores.items():
                prediction[method] = score
                metrics.append(front_metrics(group, score).assign(method=method))
            predictions.append(prediction)
        csv('metrics.csv', pd.concat(metrics, ignore_index=True))
        csv('predictions.csv', pd.concat(predictions, ignore_index=True))
        report['knee_diagnostics'] = diagnostics
    else:
        params, selection = select_job_config(contract, job, plan, run)
        prepared = prepare_training(root, job['case'], job['train_front_ids'],
                                    {int(k): v for k, v in contract['split_contract']['topology_by_front'].items()},
                                    features)
        report['preparation_and_selection_seconds'] = time.monotonic() - start
        fit_start = time.monotonic()
        model = fit_model(prepared, job['arm'], params, job['seed'], threads)
        report.update(fit_seconds=time.monotonic() - fit_start, parameters=params,
                      effective_parameters=model.get_all_params(), training_data=prepared['report'],
                      configuration_selection=selection, tree_count=int(model.tree_count_))
        if job['evaluation_front_ids']:
            scoring_start = time.monotonic()
            frame = load_evaluation(root, contract, job, features)
            scores = predict_scores(model, job['arm'], frame, features, threads)
            csv('predictions.csv', frame[['front_id', 'item_id']].assign(score=scores))
            csv('metrics.csv', front_metrics(frame, scores))
            report['evaluation_seconds'] = time.monotonic() - scoring_start
        if job['stage'] == 'deployment':
            model.save_model(str(destination / 'model.cbm'))
            artifacts.append('model.cbm')
            write_json(destination / 'model.json', dict(case=job['case'], arm=job['arm'],
                       feature_columns=features, parameters=params, seed=job['seed']))
            artifacts.append('model.json')
    report.update(status='complete', ended_at_utc=utc(), total_seconds=time.monotonic() - start,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  artifacts={name: sha256(destination / name) for name in artifacts})
    write_json(destination / 'result.json', report)
    return report
