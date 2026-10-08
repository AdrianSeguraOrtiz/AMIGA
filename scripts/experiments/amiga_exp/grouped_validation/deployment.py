"""Score new fronts without quality labels using the fixed application selectors."""
from pathlib import Path

from catboost import CatBoostClassifier, CatBoostRanker, CatBoostRegressor
import pandas as pd

from .baselines import score_front
from .execution import completed_job
from .models import predict_scores
from .pilot import sha256, write_json
from .runner import read_run, verify_sources
from .training_data import _integer_ids

APPLICATION_RULES = ('objective__reducenonessentialsinteractions', 'objective_mean_rank',
                     'objective_topsis', 'objective__metricdistribution', 'objective_knee')


def score_deployment(run: Path, case: str, data: Path, output: Path):
    manifest, contract, jobs = read_run(run)
    verify_sources(Path(manifest['repo_root']), contract)
    if case not in contract['cases']:
        raise ValueError(f'Unknown case: {case}')
    info = contract['cases'][case]
    if not set(APPLICATION_RULES) <= set(info['baseline_ids']):
        raise ValueError('The nine application selectors require the BIO-INSIGHT objectives')
    columns = ['front_id', 'item_id', *info['feature_sets']['full']]
    frame = pd.read_csv(data, usecols=columns)
    for name in ('front_id', 'item_id'):
        frame[name] = _integer_ids(frame[name], name)
    if frame.empty or frame.duplicated(['front_id', 'item_id']).any():
        raise ValueError('Input requires nonempty fronts with unique candidate IDs')
    result = frame[['front_id', 'item_id']].copy()
    artifacts = {}
    for job in jobs:
        if job['case'] != case or job['stage'] != 'deployment':
            continue
        completed = completed_job(run, job)
        if completed is None:
            raise ValueError(f'Missing deployment model: {job["arm"]}')
        _, directory = completed
        cls = (CatBoostRanker if job['arm'] == 'ltr_catboost' else
               CatBoostClassifier if job['arm'] == 'clf_top20' else CatBoostRegressor)
        model = cls()
        model.load_model(str(directory / 'model.cbm'))
        result[job['arm']] = predict_scores(model, job['arm'], frame, info['feature_sets']['full'], manifest['threads'])
        artifacts[job['arm']] = sha256(directory / 'model.cbm')
    for _, group in frame.groupby('front_id', sort=True):
        scores, _ = score_front(group, info['objective_columns'], info['objective_directions'])
        for rule in APPLICATION_RULES:
            result.loc[group.index, rule] = scores[rule]
    output.mkdir(parents=True, exist_ok=False)
    result.to_csv(output / 'candidate_scores.csv', index=False)
    write_json(output / 'manifest.json', dict(case=case, input_sha256=sha256(data), models=artifacts,
               contract_sha256=sha256(run / 'contract.json'), selectors=list(result.columns[2:]),
               quality_labels_used=False, artifact_sha256=sha256(output / 'candidate_scores.csv')))
    return dict(output=str(output), rows=len(result), selectors=len(result.columns) - 2)
