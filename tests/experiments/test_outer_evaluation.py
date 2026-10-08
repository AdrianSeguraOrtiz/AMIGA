"""Phase-4 isolation, immutable selection reuse and complete paired evaluation."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from test_grouped_validation_contract import fixture_repo, REPO_ROOT
from test_sequential_selection import toy_frame, tiny_params
from scripts.experiments.amiga_exp.grouped_validation.baselines import baseline_ids
from scripts.experiments.amiga_exp.grouped_validation.metrics import front_metrics, METRICS
from scripts.experiments.amiga_exp.grouped_validation.pilot import sha256, write_json
from scripts.experiments.amiga_exp.sequential_selection import models, spec as selection_spec
from scripts.experiments.amiga_exp.outer_evaluation import spec, execution, summary, runner


@pytest.fixture(scope='module')
def phase4_repo(fixture_repo, tmp_path_factory):
    root = tmp_path_factory.mktemp('phase4-repo')
    shutil.copytree(fixture_repo, root, dirs_exist_ok=True)
    for package in ('grouped_validation', 'sequential_selection', 'outer_evaluation'):
        relative = f'scripts/experiments/amiga_exp/{package}'
        shutil.copytree(REPO_ROOT / relative, root / relative, dirs_exist_ok=True)
    for relative in (spec.DOC, 'scripts/experiments/amiga_exp/decision_baselines.py'):
        shutil.copyfile(REPO_ROOT / relative, root / relative)
    return root


@pytest.fixture(scope='module')
def contract(phase4_repo):
    selection = selection_spec.build_contract(phase4_repo)
    base = selection['split_contract']
    procedures, cases = [], {}
    for case in base['cases']:
        for fold in range(5):
            for arm in models.ARMS:
                family = models.FAMILIES[fold % 3]
                fraction = (.5 if (case, fold, arm) == ('BIO-INSIGHT', 4, 'reg_normalized') else
                            .25 if (case, fold, arm) == ('MO-GENECI', 0, 'ranking') else 1.0)
                procedures.append(dict(case=case, outer_fold=fold, arm=arm, family=family,
                    config=selection['grids'][family][0], label='rank_dense', fraction=fraction,
                    n_features=int(np.ceil(len(base['cases'][case]['feature_columns']) * fraction)),
                    selected_candidate=f'{case}/outer-{fold}/phase3/{family}/{arm}/features-{fraction:g}'))
        features = json.loads((phase4_repo / f'docs/experiments/contracts/{case}_feature_columns.json').read_text())
        cases[case] = {k: features[k] for k in ('objective_columns', 'objective_directions')}
        cases[case]['baseline_ids'] = baseline_ids(features['objective_columns'])
    upstream = dict(run='experiments/selection/run', summary='experiments/selection/summary')
    for directory, name, value in ((upstream['run'], 'contract.json', selection),
                                   (upstream['summary'], 'selected_procedures.json', procedures)):
        (phase4_repo / directory).mkdir(parents=True)
        write_json(phase4_repo / directory / name, value)
    sources = dict(selection['source_hashes'])
    paths = [spec.DOC, 'scripts/experiments/amiga_exp/decision_baselines.py',
             upstream['run'] + '/contract.json', upstream['summary'] + '/selected_procedures.json']
    paths += [p.relative_to(phase4_repo).as_posix() for p in (phase4_repo / spec.PACKAGE).glob('*.py')]
    sources.update({p: sha256(phase4_repo / p) for p in paths})
    return dict(schema_version=1, workflow='sequential_outer_evaluation', status='frozen_outer_evaluation',
        selection_contract=selection, split_contract=base, procedures=procedures, cases=cases,
        arms=list(models.ARMS), seeds=[1201, 1202, 1203, 1204, 1205],
        feature_mask_policy=deepcopy(spec.MASK_POLICY), statistics=deepcopy(spec.STATISTICS),
        failures=deepcopy(spec.FAILURES), source_hashes=sources, upstream=upstream)


def test_plan_preserves_splits_selections_and_all_five_seeds(contract, phase4_repo):
    spec.verify_sources(phase4_repo, contract)
    assert spec.counts(contract) == dict(jobs=204, final_fits=200, mask_fits=5, total_fits=205,
                                       jobs_by_stage={'mask': 2, 'baselines': 2, 'outer': 200})
    jobs = spec.build_plan(contract)
    seen = set()
    for job in jobs:
        assert set(job['dependencies']) <= seen
        seen.add(job['id'])
        if job['stage'] == 'baselines':
            continue
        outer = contract['split_contract']['outer_folds'][job['outer_fold']]
        assert job['train_front_ids'] == outer['train_front_ids']
        if job['stage'] == 'outer':
            assert job['evaluation_front_ids'] == outer['test_front_ids']
            assert job['seed'] in contract['seeds']
        else:
            assert job['evaluation_front_ids'] == [] and job['seed'] == 1101
    for p in contract['procedures']:
        own = [j for j in jobs if j['stage'] == 'outer' and j['procedure'] == p]
        assert {j['seed'] for j in own} == set(contract['seeds'])


@pytest.mark.parametrize('mutation', ['seed', 'mask_seed', 'fraction', 'configuration', 'duplicate', 'fold'])
def test_contract_rejects_policy_drift(contract, mutation):
    bad = deepcopy(contract)
    if mutation == 'seed':
        bad['seeds'][0] = 99
    elif mutation == 'mask_seed':
        bad['feature_mask_policy']['training_seed'] = 1201
    elif mutation == 'fraction':
        bad['procedures'][0]['fraction'] = .5
    elif mutation == 'configuration':
        bad['procedures'][0]['config']['iterations'] = 100
    elif mutation == 'duplicate':
        bad['procedures'][-1] = deepcopy(bad['procedures'][0])
    else:
        bad['split_contract']['outer_folds'][0]['train_front_ids'].append(
            bad['split_contract']['outer_folds'][0]['test_front_ids'][0])
    with pytest.raises(ValueError):
        spec.validate_contract(bad)


def test_valid_but_different_selection_is_rejected_by_provenance(contract, phase4_repo):
    bad = deepcopy(contract)
    p = bad['procedures'][0]
    p['config'] = bad['selection_contract']['grids'][p['family']][1]
    with pytest.raises(ValueError, match='saved inner selection'):
        spec.verify_sources(phase4_repo, bad)


def test_dry_run_and_resumption_keep_definitions(contract, phase4_repo, tmp_path):
    path = tmp_path / 'contract.json'
    spec.freeze(path, contract)
    run = tmp_path / 'run'
    report = runner.run_evaluation(path, run, root=phase4_repo, jobs=1, threads=1, dry_run=True)
    assert report['planned_fits'] == 205 and report['completed_jobs'] == 0
    assert not (run / 'jobs').exists()
    second = runner.run_evaluation(path, run, root=phase4_repo, jobs=1, threads=1, dry_run=True, resume=True)
    assert second == report
    with pytest.raises(ValueError, match='resource'):
        runner.run_evaluation(path, run, root=phase4_repo, jobs=1, threads=2, dry_run=True, resume=True)
    plan = json.loads((run / 'plan.json').read_text())
    plan[0]['seed'] = 999
    write_json(run / 'plan.json', plan)
    with pytest.raises(ValueError, match='Run definition changed'):
        runner.read_run(run)


@pytest.mark.parametrize('family', models.FAMILIES)
def test_relearned_mask_equals_original_recursive_path(family):
    frame = toy_frame()
    data = models.prepare(frame, 'BIO-INSIGHT', {i: str(i) for i in range(1, 9)}, [f'x{i}' for i in range(8)])
    p = dict(family=family, arm='ranking', config=tiny_params(family), fraction=.25, n_features=2)
    features, paths, importance = execution.learn_mask(data, p, threads=1)
    expected, _ = models.feature_path(data, family, p['arm'], p['config'], seed=1101, threads=1)
    assert features == expected[-1]['features']
    assert len(paths) == 3  # No unnecessary fit on the terminal representation.
    assert all(row['sampling']['sampling_seed'] == 1501 for row in paths)
    assert set(importance['fraction']) == {1., .75, .5}


@pytest.mark.parametrize('family', models.FAMILIES)
def test_outer_predictions_ignore_heldout_quality_and_use_final_seed(family, tmp_path, monkeypatch):
    frame = toy_frame()
    source = tmp_path / 'data.csv'
    frame.to_csv(source, index=False)
    features = [f'x{i}' for i in range(8)]
    c = dict(split_contract=dict(cases={'BIO-INSIGHT': dict(data_path='data.csv', feature_columns=features)},
                                 topology_by_front={str(i): str(i) for i in range(1, 9)}))
    p = dict(family=family, arm='ranking', config=tiny_params(family), fraction=1., n_features=8, label='rank_dense')
    job = dict(id='test', case='BIO-INSIGHT', stage='outer', arm='ranking', procedure=p,
               train_front_ids=list(range(1, 7)), evaluation_front_ids=[7, 8], seed=1203, dependencies=[])
    fit_calls, loads = [], []
    native_fit, native_load = models.fit, execution.load_rows

    def fit(data, family, arm, params, *, seed, threads):
        fit_calls.append((set(data['group_id']), seed))
        return native_fit(data, family, arm, params, seed=seed, threads=threads)

    def load(root, contract, case, fronts, columns, *, labels=True):
        loads.append((set(fronts), labels))
        return native_load(root, contract, case, fronts, columns, labels=labels)

    monkeypatch.setattr(models, 'fit', fit)
    monkeypatch.setattr(execution, 'load_rows', load)
    destinations = []
    for number in range(2):
        destination = tmp_path / f'run-{number}' / 'jobs' / 'test' / 'attempt-001'
        destination.mkdir(parents=True)
        execution.execute_job(tmp_path, c, job, {}, tmp_path, destination, 1)
        destinations.append(destination)
        frame.loc[frame['front_id'].isin([7, 8]), 'AUPR'] = 1 - frame.loc[frame['front_id'].isin([7, 8]), 'AUPR']
        frame.to_csv(source, index=False)
    assert fit_calls == [(set(range(1, 7)), 1203)] * 2
    assert loads[:3] == [(set(range(1, 7)), True), ({7, 8}, False), ({7, 8}, True)]
    pd.testing.assert_frame_equal(pd.read_csv(destinations[0] / 'predictions.csv'),
                                  pd.read_csv(destinations[1] / 'predictions.csv'))
    assert not pd.read_csv(destinations[0] / 'metrics.csv').equals(pd.read_csv(destinations[1] / 'metrics.csv'))
    run = destinations[0].parents[2]
    assert execution.completed_job(run, job) is not None
    with (destinations[0] / 'predictions.csv').open('a') as handle:
        handle.write('\n')
    with pytest.raises(ValueError, match='artifact changed'):
        execution.completed_job(run, job)


def synthetic_metrics(contract):
    rows = []
    for job in spec.build_plan(contract):
        if job['stage'] == 'mask':
            continue
        methods = [job['arm']] if job['stage'] == 'outer' else contract['cases'][job['case']]['baseline_ids']
        for method in methods:
            for front in job['evaluation_front_ids']:
                value = .01 + front / 10000
                if job['stage'] == 'outer':
                    value += (job['seed'] - 1201) / 1000
                if method == 'reg_aupr':
                    value += .02
                rows.append(dict(case=job['case'], stage=job['stage'], method=method, front_id=front,
                                 seed=job['seed'], outer_fold=job['outer_fold'], **{m: value for m in METRICS}))
    return pd.DataFrame(rows)


def test_summary_averages_seeds_then_topologies_and_corrects_six_tests(contract):
    long = synthetic_metrics(contract)
    tables = summary.central_tables(long, contract)
    assert len(tables['primary_tests.csv']) == 6
    assert (tables['paired_differences.csv']['n_topologies'] == 87).all()
    front = tables['front_metrics.csv'].query('method == "ranking"').iloc[0]
    assert front['Regret@5'] == pytest.approx(.012 + front['front_id'] / 10000)
    paired = tables['paired_differences.csv'].query('comparator == "reg_aupr"')
    assert paired['mean_difference'].to_numpy() == pytest.approx(np.full(len(paired), -.02))
    tied = tables['primary_tests.csv'].query('comparator == "reg_normalized"')
    assert (tied['p_value'] == 1).all() and (tied['p_holm'] == 1).all()
    central = tables['central_summary.csv'].query('method == "ranking" and aggregation == "topology_macro"')
    topology = tables['topology_metrics.csv'].query('method == "ranking"')
    for case in contract['cases']:
        assert central.loc[central['case'] == case, 'Regret@5'].iloc[0] == pytest.approx(
            topology.loc[topology['case'] == case, 'Regret@5'].mean())


@pytest.mark.parametrize('mutation', ['missing_seed', 'wrong_seed', 'wrong_fold', 'missing_comparator'])
def test_summary_rejects_incomplete_or_reassigned_results(contract, mutation):
    data = synthetic_metrics(contract)
    if mutation == 'missing_seed':
        data = data.drop(data.index[data['stage'] == 'outer'][0])
    elif mutation == 'wrong_seed':
        data.loc[data['seed'] == 1201, 'seed'] = 99
    elif mutation == 'wrong_fold':
        data.loc[data['outer_fold'] == 0, 'outer_fold'] = 1
    else:
        data = data.loc[data['method'] != 'objective_knee']
    with pytest.raises(ValueError):
        summary.central_tables(data, contract)


def test_prediction_audit_detects_wrong_metrics_missing_candidates_and_ties():
    frame = toy_frame()
    pred = frame[['front_id', 'item_id']].assign(score=0.)
    metrics = front_metrics(frame, pred['score'])
    summary.audit_predictions(frame, pred, metrics, ['score'])
    changed = metrics.copy()
    changed.loc[0, 'Regret@5'] += .01
    with pytest.raises(ValueError, match='Metrics differ'):
        summary.audit_predictions(frame, pred, changed, ['score'])
    with pytest.raises(ValueError, match='coverage differs'):
        summary.audit_predictions(frame, pred.iloc[1:], metrics, ['score'])


def test_complete_comparison_figures(contract, tmp_path):
    from scripts.experiments.amiga_exp.outer_evaluation.plots import make_plots
    tables = summary.central_tables(synthetic_metrics(contract), contract)
    make_plots(tables, tmp_path / 'plots')
    assert len(list((tmp_path / 'plots').rglob('*.png'))) == 6
    assert len(list((tmp_path / 'plots').rglob('*.pdf'))) == 6


def test_monitor_follows_live_child(tmp_path):
    from scripts.experiments.amiga_exp.outer_evaluation.commands import inspect_status
    import os
    import time
    run, launch = tmp_path / 'run', tmp_path / 'launch'
    run.mkdir(); launch.mkdir()
    write_json(run / 'state.json', dict(status='running', supervisor_pid=os.getpid(),
               accounted_at_unix=time.time(), completed_jobs=17, total_jobs=204))
    write_json(launch / 'state.json', dict(status='running_evaluation', run=str(run),
               supervisor_pid=os.getpid(), accounted_at_unix=0))
    result = inspect_status(launch)
    assert result['health'] == 'running' and result['evaluation']['completed_jobs'] == 17


def test_complete_summary_checks_saved_predictions_and_mask_dependencies(contract, phase4_repo, tmp_path, monkeypatch):
    run = tmp_path / 'run'
    run.mkdir()
    write_json(run / 'contract.json', contract)
    write_json(run / 'state.json', dict(status='complete'))
    jobs = spec.build_plan(contract)
    lookup = {j['id']: j for j in jobs}
    frame = pd.DataFrame([dict(front_id=f, item_id=i, AUPR=(i + 1) / 10)
                          for f in sorted(map(int, contract['split_contract']['topology_by_front']))
                          for i in range(6)])
    for job in jobs:
        destination = run / 'jobs' / job['id'] / 'attempt-001'
        destination.mkdir(parents=True)
        report = dict(job=job, status='complete', seconds=.1, peak_rss_mib=1.)
        if job['stage'] == 'mask':
            full = contract['split_contract']['cases'][job['case']]['feature_columns']
            paths = [dict(fraction=fraction, requested_iterations=3000, actual_iterations=3000)
                     for fraction in models.FRACTIONS[:job['planned_fits']]]
            write_json(destination / 'mask.json', dict(features=full[:job['procedure']['n_features']],
                       training_seed=1101, train_front_ids=job['train_front_ids'], paths=paths))
            pd.DataFrame(dict(feature=full, centered_mean_abs_shap=1.)).to_csv(destination / 'feature_importance.csv', index=False)
            report['fit_reports'] = paths
            names = ['mask.json', 'feature_importance.csv']
        else:
            labels = frame.loc[frame['front_id'].isin(job['evaluation_front_ids'])].reset_index(drop=True)
            predictions = labels[['front_id', 'item_id']].copy()
            if job['stage'] == 'outer':
                predictions['score'] = np.sin(labels['item_id'] + job['seed'])
                metrics = front_metrics(labels, predictions['score'])
                features, dependency = execution.selected_features(contract, job, lookup, run)
                write_json(destination / 'model.json', dict(procedure=job['procedure'], seed=job['seed'],
                           feature_columns=features, mask_dependency=dependency,
                           train_front_ids=job['train_front_ids'], evaluation_front_ids=job['evaluation_front_ids'],
                           fitting=dict(requested_iterations=3000, actual_iterations=3000)))
                names = ['metrics.csv', 'predictions.csv', 'model.json']
            else:
                tables = []
                for method in contract['cases'][job['case']]['baseline_ids']:
                    predictions[method] = labels['AUPR'] if method == 'oracle' else 0.
                    tables.append(front_metrics(labels, predictions[method]).assign(method=method))
                metrics = pd.concat(tables, ignore_index=True)
                names = ['metrics.csv', 'predictions.csv']
            predictions.to_csv(destination / 'predictions.csv', index=False)
            metrics.to_csv(destination / 'metrics.csv', index=False)
        report['artifacts'] = {name: sha256(destination / name) for name in names}
        write_json(destination / 'result.json', report)
    monkeypatch.setattr(summary, 'read_run', lambda _: (dict(repo_root=str(phase4_repo)), contract, jobs))
    monkeypatch.setattr(summary, 'load_rows', lambda *args: frame)
    result = summary.summarize(run, tmp_path / 'summary', figures=False)
    assert result['audit']['completed_fits'] == 205
    assert result['audit']['actual_iterations'] == 615000
    assert len(result['source_result_hashes']) == 204
    assert len(result['artifacts']) == 11
    # Matching hashes alone cannot bless a wrong metric: the audit recomputes it.
    job = next(j for j in jobs if j['stage'] == 'outer')
    directory = run / 'jobs' / job['id'] / 'attempt-001'
    data = pd.read_csv(directory / 'metrics.csv')
    data.loc[0, 'Regret@5'] += .01
    data.to_csv(directory / 'metrics.csv', index=False)
    report = json.loads((directory / 'result.json').read_text())
    report['artifacts']['metrics.csv'] = sha256(directory / 'metrics.csv')
    write_json(directory / 'result.json', report)
    with pytest.raises(ValueError, match='Metrics differ'):
        summary.summarize(run, tmp_path / 'invalid-summary', figures=False)


@pytest.mark.parametrize('summary_fails', [False, True])
def test_pipeline_completion_requires_successful_summary(contract, phase4_repo, tmp_path, monkeypatch, summary_fails):
    from scripts.experiments.amiga_exp.outer_evaluation import pipeline
    path, launch, run, output = [tmp_path / name for name in ('contract.json', 'launch', 'run', 'summary')]
    write_json(path, contract)
    monkeypatch.setattr(pipeline, 'REPO', phase4_repo)
    calls = []

    def evaluate(contract, destination, **options):
        calls.append(options)
        destination.mkdir(exist_ok=True)
        return dict(status='planned' if options.get('dry_run') else 'complete')

    def summarize(*args):
        if summary_fails:
            raise ValueError('Audit failed')
        return dict(audit={'completed_fits': 205})

    monkeypatch.setattr(pipeline, 'run_evaluation', evaluate)
    monkeypatch.setattr(pipeline, 'summarize', summarize)
    if summary_fails:
        with pytest.raises(ValueError, match='Audit failed'):
            pipeline.run_full(path, launch, run, output, jobs=16, threads=4)
    else:
        pipeline.run_full(path, launch, run, output, jobs=16, threads=4)
    assert calls[0]['dry_run'] and calls[1]['resume']
    state = json.loads((launch / 'state.json').read_text())
    assert state['status'] == ('failed' if summary_fails else 'complete')
