"""Inner-selection summaries; deliberately no outer performance estimates."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.experiments.amiga_exp.grouped_validation.metrics import topology_metrics
from scripts.experiments.amiga_exp.grouped_validation.pilot import sha256, write_json
from .selection_execution import completed_job
from .runner import read_run, verify_sources


def choose(rows, *, features=False):
    """Use the frozen lexicographic rule, including deterministic near-ties."""
    if not rows:
        raise ValueError('No selection candidates')
    return min(rows, key=lambda r: (round(r['mean_regret5'], 12),
                                    round(r['mean_regret1'], 12),
                                    r['n_features'] if features else 0, r['candidate']))


def describe(metrics, mapping):
    values = topology_metrics(metrics, mapping)
    return dict(mean_regret5=float(values['Regret@5'].mean()),
                mean_regret1=float(values['Regret@1'].mean()),
                std_regret5=float(values['Regret@5'].std(ddof=1)),
                n_topologies=len(values), n_fronts=len(metrics))


def validate_metrics(metrics, contract, job):
    outer = contract['split_contract']['outer_folds'][job['outer_fold']]
    fractions = contract['fractions'] if job['stage'] == 'phase3' else [1.0]
    if set(metrics['fraction']) != set(fractions):
        raise ValueError('Unexpected feature fractions')
    for _, part in metrics.groupby('fraction'):
        if part['front_id'].duplicated().any() or set(part['front_id']) != set(outer['train_front_ids']):
            raise ValueError('Inner validation coverage differs')
        for inner in outer['inner_folds']:
            observed = part.loc[part['inner_fold'] == inner['fold'], 'front_id']
            if set(observed) != set(inner['validation_front_ids']):
                raise ValueError('Inner fold assignment differs')


def summarize(run, output, *, figures=True):
    run, output = Path(run).resolve(), Path(output).resolve()
    manifest, contract, jobs = read_run(run)
    verify_sources(Path(manifest['repo_root']), contract)
    state = json.loads((run / 'state.json').read_text())
    if state['status'] != 'complete' or contract['mode'] != 'selection':
        raise ValueError('A complete phase 1–3 run is required')
    if output.exists():
        raise ValueError('Use a new summary directory')
    mapping = contract['split_contract']['topology_by_front']
    rows, paths, configs, provenance, fit_audit = [], {}, {}, {}, []
    for job in jobs:
        complete = completed_job(run, job)
        if complete is None:
            raise ValueError(f'Incomplete job: {job["id"]}')
        _, directory = complete
        metrics = pd.read_csv(directory / 'metrics.csv', float_precision='round_trip')
        validate_metrics(metrics, contract, job)
        from scripts.experiments.amiga_exp.outer_evaluation.summary import audit_predictions
        from .outer_execution import load_rows
        from .models import TARGET
        if complete[0].get('classification_policy') != TARGET:
            raise ValueError('Classification target provenance differs')
        predicted = pd.read_csv(directory / 'predictions.csv', float_precision='round_trip')
        labels = load_rows(Path(manifest['repo_root']), contract, job['case'],
                           contract['split_contract']['outer_folds'][job['outer_fold']]['train_front_ids'], [])
        for fraction, part in metrics.groupby('fraction'):
            scores = predicted.loc[predicted['fraction'] == fraction, ['front_id', 'item_id', 'score']]
            audit_predictions(labels, scores, part, ['score'])
        info = json.loads((directory / 'selection.json').read_text())
        if info['family'] != job['family'] or info['arm'] != job['arm'] or info['label'] != 'rank_dense':
            raise ValueError('Selection model identity differs')
        outer = contract['split_contract']['outer_folds'][job['outer_fold']]
        if job['stage'] == 'phase2':
            if info['config'] != job['config']:
                raise ValueError('Tuning configuration changed')
            groups = [(r['inner_fold'], r['training'], [r]) for r in info['fit_reports']]
        else:
            saved_paths = json.loads((directory / 'feature_paths.json').read_text())
            groups = [(r['inner_fold'], r['training'], r['path']) for r in saved_paths]
        if sorted(g[0] for g in groups) != list(range(3)):
            raise ValueError('Fitting inner-fold coverage differs')
        for inner_id, training, fitted in groups:
            if (training['selected_front_ids'] != sorted(outer['inner_folds'][inner_id]['train_front_ids'])
                    or training.get('classification_policy') != TARGET):
                raise ValueError('Fitting training scope/target differs')
            expected_fits = 4 if job['stage'] == 'phase3' else 1
            if len(fitted) != expected_fits:
                raise ValueError('Fitting feature-path budget differs')
            for fit in fitted:
                if (fit['requested_iterations'] != 3000 or fit['early_stopping']
                        or not 1 <= fit['actual_iterations'] <= 3000
                        or fit.get('classification_policy') != TARGET):
                    raise ValueError('Selection iteration/target budget differs')
                fit_audit.append(dict(job_id=job['id'], inner_fold=inner_id,
                                      requested_iterations=3000, actual_iterations=fit['actual_iterations']))
        configs[job['id']] = info
        provenance[job['id']] = sha256(directory / 'result.json')
        for fraction, part in metrics.groupby('fraction'):
            if part['n_features'].nunique() != 1:
                raise ValueError('Feature budget differs between inner folds')
            rows.append(dict(job=job['id'], case=job['case'], outer_fold=job['outer_fold'],
                             stage=job['stage'], family=job['family'], arm=job['arm'],
                             label=info['label'], config_id=info['config'].get('id', 'reference'),
                             parameters=json.dumps(info['config'], sort_keys=True),
                             fraction=float(fraction), n_features=int(part['n_features'].iloc[0]),
                             candidate=f'{job["id"]}/features-{fraction:g}',
                             **describe(part, mapping)))
        if job['stage'] == 'phase3':
            paths[job['id']] = json.loads((directory / 'feature_paths.json').read_text())
    table = pd.DataFrame(rows)
    table['selected_within_family'] = False
    grouping = ['case', 'outer_fold', 'stage', 'family', 'arm']
    for keys, part in table.groupby(grouping):
        winner = choose(part.to_dict('records'), features=keys[2] == 'phase3')
        table.loc[table['candidate'] == winner['candidate'], 'selected_within_family'] = True
    procedures, selected_masks = [], []
    phase3 = table[table['stage'] == 'phase3']
    for keys, part in phase3.groupby(['case', 'outer_fold', 'arm']):
        winner = choose(part.to_dict('records'), features=True)
        info = configs[winner['job']]
        procedures.append(dict(case=keys[0], outer_fold=int(keys[1]), arm=keys[2],
                               family=winner['family'], label=info['label'], config=info['config'],
                               fraction=winner['fraction'], n_features=winner['n_features'],
                               selected_candidate=winner['candidate'],
                               inner_mean_regret5=winner['mean_regret5'],
                               outer_feature_mask='must_be_relearned_on_outer_training_only'))
    for row in phase3[phase3['selected_within_family']].to_dict('records'):
        features = contract['split_contract']['cases'][row['case']]['feature_columns']
        for inner in paths[row['job']]:
            path = next(p for p in inner['path'] if p['fraction'] == row['fraction'])
            for feature in features:
                selected_masks.append(dict(case=row['case'], outer_fold=row['outer_fold'],
                                           inner_fold=inner['inner_fold'], family=row['family'],
                                           arm=row['arm'], fraction=row['fraction'], feature=feature,
                                           selected=feature in path['features']))
    output.mkdir(parents=True)
    from .spec import counts
    if len(fit_audit) != counts(contract)['total_fits']:
        raise ValueError('Selection total fitting budget differs')
    table.to_csv(output / 'selection_candidates.csv', index=False)
    pd.DataFrame(fit_audit).to_csv(output / 'fit_audit.csv', index=False)
    masks = pd.DataFrame(selected_masks)
    masks.to_csv(output / 'selected_inner_masks.csv', index=False)
    stability = masks.groupby(['case', 'family', 'arm', 'feature'])['selected'].agg(
        selected_frequency='mean', number_of_fits='count').reset_index()
    stability.to_csv(output / 'feature_stability.csv', index=False)
    write_json(output / 'selected_procedures.json', procedures)
    if figures:
        raise ValueError('Inner selection exports tables; final comparison generates figures')
    report = dict(status='complete', classification_policy=dict(contract['classification_policy']), scope='inner_validation_for_selection_only',
                  audit=dict(completed_jobs=len(jobs), completed_fits=len(fit_audit),
                             predictions_and_metrics_recomputed=True),
                  outer_evaluation_performed=False, selected_procedures=len(procedures),
                  contract_sha256=sha256(run / 'contract.json'),
                  dependency_result_hashes=provenance,
                  artifacts={p.relative_to(output).as_posix(): sha256(p)
                             for p in sorted(output.rglob('*')) if p.is_file()},
                  limitations=['Selection scores are optimistic development diagnostics.',
                               'Overlapping inner training masks are not independent replicates.',
                               'Feature identities must be relearned inside each outer training set.'])
    write_json(output / 'summary_manifest.json', report)
    return report
