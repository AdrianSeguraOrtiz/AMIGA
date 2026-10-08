"""Scientific edge cases and bounded synthetic end-to-end execution checks."""
from collections import Counter
from copy import deepcopy
from itertools import permutations
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from test_grouped_validation_contract import fixture_repo, REPO_ROOT
from scripts.experiments.amiga_exp.grouped_validation import execution, runner
from scripts.experiments.amiga_exp.grouped_validation import summary as summary_module
from scripts.experiments.amiga_exp.grouped_validation import deployment
from scripts.experiments.amiga_exp.grouped_validation.baselines import knee_scores, score_front
from scripts.experiments.amiga_exp.grouped_validation.metrics import front_metrics, select_configuration
from scripts.experiments.amiga_exp.grouped_validation.models import fit_model
from scripts.experiments.amiga_exp.grouped_validation.pilot import sha256, write_json
from scripts.experiments.amiga_exp.grouped_validation.planning import (
    build_evaluation_contract, build_plan, freeze_contract, FIT_COUNTS,
)
from scripts.experiments.amiga_exp.grouped_validation.summary import holm, paired_difference, summarize


@pytest.fixture(scope='module')
def evaluation_repo(fixture_repo, tmp_path_factory):
    root = tmp_path_factory.mktemp('evaluation-repo')
    shutil.copytree(fixture_repo, root, dirs_exist_ok=True)
    package = 'scripts/experiments/amiga_exp/grouped_validation'
    shutil.copytree(REPO_ROOT / package, root / package, dirs_exist_ok=True)
    for relative in ['scripts/experiments/amiga_exp/decision_baselines.py', 'docs/experiments/grouped-validation.md']:
        shutil.copyfile(REPO_ROOT / relative, root / relative)
    for case in ('BIO-INSIGHT', 'MO-GENECI'):
        path = root / f'experiments/{case}/data/data_104.csv'
        data = pd.read_csv(path)
        data['AUPR'] = data['item_id'] / 3.0
        for i, name in enumerate(c for c in data if c not in ('front_id', 'item_id', 'AUPR')):
            data[name] = data['item_id'] * (i + 1) / 100 + data['front_id'] % 3
        data.to_csv(path, index=False)
        manifest = root / f'docs/experiments/contracts/{case}_data_manifest.json'
        value = json.loads(manifest.read_text())
        value['data_csv_sha256'] = sha256(path)
        manifest.write_text(json.dumps(value))
    return root


@pytest.fixture(scope='module')
def evaluation_contract(evaluation_repo):
    return build_evaluation_contract(evaluation_repo)


@pytest.mark.parametrize('scores', [[0, 0, 0, 0], [2, 1, 1, 0], [3, 2, 1, 0]])
def test_tie_expectations_match_exhaustive_permutations(scores):
    target = np.array([0.1, 0.3, 0.8, 0.8])
    frame = pd.DataFrame(dict(front_id=1, item_id=range(4), AUPR=target))
    result = front_metrics(frame, scores).iloc[0]
    valid = [p for p in permutations(range(4)) if all(scores[p[i]] >= scores[p[i+1]] for i in range(3))]
    for k in (1, 3, 5, 10):
        best = np.array([target[list(p[:k])].max() for p in valid])
        assert result[f'BestAUPR@{k}'] == pytest.approx(best.mean())
        assert result[f'Hit@{k}'] == pytest.approx(np.isclose(best, target.max()).mean())
        assert result[f'Regret@{k}'] == pytest.approx(target.max() - best.mean())
    single = front_metrics(frame.iloc[:1], [0]).iloc[0]
    assert single['Regret@10'] == 0 and single['Hit@10'] == 1


@pytest.mark.parametrize('dimensions', [2, 3, 6])
def test_knee_retains_duplicates_constants_and_dominance(dimensions):
    # A pronounced knee beats either endpoint; duplicate candidates stay tied.
    matrix = np.array([[0, 1], [.2, .2], [1, 0], [.2, .2], [1, 1]])
    matrix = np.pad(matrix, [(0, 0), (0, dimensions - 2)])
    scores, report = knee_scores(pd.DataFrame(matrix))
    assert scores[1] == scores[3] > scores[0] == scores[2] > scores[4]
    assert report['constant_objectives'] == dimensions - 2
    assert report['duplicate_vectors'] == 1
    assert report['dominated_candidates'] == 1
    flat, _ = knee_scores(pd.DataFrame(np.zeros((4, dimensions))))
    assert np.array_equal(flat, np.zeros(4))


def test_borda_equivalence_and_label_free_baselines():
    data = pd.DataFrame(dict(front_id=1, item_id=range(4), a=[0, 2, 1, 1], b=[4, 1, 2, 2]))
    scores, _ = score_front(data, ['a', 'b'], dict(a='minimize', b='maximize'))
    borda = ((len(data) - data['a'].rank(method='average', ascending=True)) +
             (len(data) - data['b'].rank(method='average', ascending=False)))
    assert np.array_equal(pd.Series(scores['objective_mean_rank']).rank(), borda.rank())
    assert 'oracle' not in scores
    assert (scores['random_uniform'] == 0).all()


def test_selection_weights_topologies_and_uses_secondary_ties():
    # Three conditions from the same topology must not outweigh two topologies.
    mapping = {'1':'a', '2':'a', '3':'a', '4':'b', '5':'c'}
    a = pd.DataFrame({'front_id':range(1, 6), 'Regret@5':[0, 0, 0, .6, .6], 'Regret@1':.9})
    b = pd.DataFrame({'front_id':range(1, 6), 'Regret@5':[.5, .5, .5, 0, 0], 'Regret@1':.9})
    assert a['Regret@5'].mean() < b['Regret@5'].mean()
    assert select_configuration({'a':a, 'b':b}, mapping)[0] == 'b'
    assert select_configuration({'a':b, 'b':b.assign(**{'Regret@1':.2})}, mapping)[0] == 'b'
    assert select_configuration({'b':b, 'a':b}, mapping)[0] == 'a'


def test_full_plan_is_group_isolated_and_inner_dependencies_are_complete(evaluation_contract, tmp_path):
    plan = build_plan(evaluation_contract)
    assert len(plan) == 1378
    assert Counter(j['stage'] for j in plan if j['stage'] != 'baselines') == FIT_COUNTS
    mapping = evaluation_contract['split_contract']['topology_by_front']
    known = {j['id']: j for j in plan}
    for job in plan:
        train = {mapping[str(f)] for f in job['train_front_ids']}
        held = {mapping[str(f)] for f in job['evaluation_front_ids']}
        assert not train & held
        for identifier in job['dependencies']:
            upstream = known[identifier]
            assert set(upstream['train_front_ids'] + upstream['evaluation_front_ids']) <= set(job['train_front_ids'])
    path = tmp_path / 'frozen.json'
    freeze_contract(path, evaluation_contract)
    freeze_contract(path, evaluation_contract)
    changed = deepcopy(evaluation_contract)
    changed['source_hashes']['docs/experiments/grouped-validation.md'] = 'a' * 64
    with pytest.raises(ValueError, match='replace'):
        freeze_contract(path, changed)


@pytest.mark.parametrize('arm', ['ltr_catboost', 'reg_aupr', 'reg_normalized', 'clf_top20', None])
def test_real_estimators_and_artifact_integrity(evaluation_contract, evaluation_repo, tmp_path, monkeypatch, arm):
    plan = {j['id']: j for j in build_plan(evaluation_contract)}
    job = next(j for j in plan.values() if j['arm'] == arm and j['stage'] == ('tuning' if arm else 'baselines'))
    def small_fit(prepared, formulation, params, seed, threads):
        assert params['iterations'] == 3000
        return fit_model(prepared, formulation, dict(params, iterations=8), seed, threads)
    monkeypatch.setattr(execution, 'fit_model', small_fit)
    destination = tmp_path / 'jobs' / job['id'] / 'attempt-001'
    destination.mkdir(parents=True)
    execution.execute_job(evaluation_repo, evaluation_contract, job, plan, tmp_path, destination, 1)
    report, _ = execution.completed_job(tmp_path, job)
    assert report['status'] == 'complete'
    predictions = pd.read_csv(destination / 'predictions.csv')
    assert set(predictions['front_id']) == set(job['evaluation_front_ids'])
    assert 'AUPR' not in predictions
    if arm == 'clf_top20':
        assert predictions['score'].between(0, 1).all()
    if arm:
        assert report['tree_count'] == 8  # Synthetic test budget, never the scientific runner.
        assert report['training_data']['selected_front_ids'] == sorted(job['train_front_ids'])
    with (destination / 'metrics.csv').open('a') as handle:
        handle.write('tampered')
    with pytest.raises(ValueError, match='changed'):
        execution.completed_job(tmp_path, job)


def test_dry_run_resume_source_verification_and_incomplete_summary(evaluation_contract, evaluation_repo, tmp_path):
    contract_path, output = tmp_path / 'contract.json', tmp_path / 'run'
    freeze_contract(contract_path, evaluation_contract)
    args = dict(root=evaluation_repo, jobs=1, threads=1, dry_run=True)
    result = runner.run_evaluation(contract_path, output, **args)
    assert result['planned_fits'] == 1376
    assert not (output / 'jobs').exists()
    assert runner.run_evaluation(contract_path, output, resume=True, **args)['completed_jobs'] == 0
    with pytest.raises(ValueError, match='already exists'):
        runner.run_evaluation(contract_path, output, **args)
    with pytest.raises(ValueError, match='every planned job'):
        summarize(output, tmp_path / 'summary')
    with (output / 'plan.json').open('a') as handle:
        handle.write(' ')
    with pytest.raises(ValueError, match='definition changed'):
        runner.run_evaluation(contract_path, output, resume=True, **args)


def test_paired_statistics_zero_differences_and_holm():
    result = paired_difference([.1, .2, .3], [.1, .2, .3], samples=100)
    assert result['ci_low'] == result['ci_high'] == result['mean_difference'] == 0
    assert result['tied'] == 3
    assert holm([.03, .001, .9]).tolist() == pytest.approx([.06, .003, .9])


def test_final_configuration_reads_only_complete_inner_dependencies(evaluation_contract, tmp_path, monkeypatch):
    plan = {j['id']: j for j in build_plan(evaluation_contract)}
    job = next(j for j in plan.values() if j['stage'] == 'outer')
    grid = evaluation_contract['split_contract']['grid']
    seen = []
    def inner_result(run, dependency):
        assert dependency['id'] in job['dependencies']
        assert not set(dependency['evaluation_front_ids']) & set(job['evaluation_front_ids'])
        seen.append(dependency['id'])
        directory = tmp_path / dependency['id']
        directory.mkdir(parents=True, exist_ok=True)
        regret = [c['id'] for c in grid].index(dependency['config_id']) / 10
        pd.DataFrame({'front_id':dependency['evaluation_front_ids'], 'Regret@5':regret,
                      'Regret@1':.5}).to_csv(directory / 'metrics.csv', index=False)
        return {}, directory
    monkeypatch.setattr(execution, 'completed_job', inner_result)
    chosen, evidence = execution.select_job_config(evaluation_contract, job, plan, tmp_path)
    assert chosen == grid[0] and len(seen) == 18 and len(evidence['candidates']) == 6
    incomplete = dict(job, dependencies=job['dependencies'][:-1])
    with pytest.raises(ValueError, match='all six'):
        execution.select_job_config(evaluation_contract, incomplete, plan, tmp_path)


def test_summary_averages_seeds_then_conditions_and_keeps_six_tests(evaluation_contract, tmp_path, monkeypatch):
    run, output = tmp_path / 'run', tmp_path / 'summary'
    run.mkdir()
    write_json(run / 'contract.json', evaluation_contract)
    plan = build_plan(evaluation_contract)
    monkeypatch.setattr(summary_module, 'read_run', lambda path: ({'repo_root':str(REPO_ROOT)}, evaluation_contract, plan))
    monkeypatch.setattr(summary_module, 'verify_sources', lambda *args: None)
    def synthetic_result(path, job):
        directory = run / job['id']
        directory.mkdir(parents=True)
        write_json(directory / 'result.json', {'synthetic':True})
        if job['stage'] in ('outer', 'baselines', 'learning_curve', 'ablation', 'family'):
            methods = evaluation_contract['cases'][job['case']]['baseline_ids'] if job['stage'] == 'baselines' else [job['arm']]
            frames = []
            for method in methods:
                # All seeds contribute: mean offset .02, not the best seed's zero.
                seed_offset = (job['seed'] - 1201) * .01 if job['stage'] == 'outer' else 0
                regret = (0 if method == 'ltr_catboost' else .1) + seed_offset
                frame = pd.DataFrame({'front_id':job['evaluation_front_ids']})
                for metric in summary_module.METRICS:
                    frame[metric] = regret
                if job['stage'] == 'baselines':
                    frame['method'] = method
                frames.append(frame)
            pd.concat(frames).to_csv(directory / 'metrics.csv', index=False)
        return {'total_seconds':.1, 'peak_rss_mib':10}, directory
    monkeypatch.setattr(summary_module, 'completed_job', synthetic_result)
    result = summarize(run, output)
    assert result['primary_tests'] == 6
    table = pd.read_csv(output / 'central_summary.csv')
    assert table.loc[table['method'] == 'ltr_catboost', 'Regret@5'].tolist() == pytest.approx([.02] * 4)
    topo = pd.read_csv(output / 'topology_metrics.csv')
    assert topo.groupby(['case', 'method']).size().eq(87).all()
    pairs = pd.read_csv(output / 'paired_differences.csv')
    primary = pairs.loc[(pairs['comparator'] == 'reg_aupr') & (pairs['metric'] == 'Regret@5')]
    assert primary['mean_difference'].tolist() == pytest.approx([-.1, -.1])
    assert primary['improved'].tolist() == [87, 87]
    extras = pd.read_csv(output / 'supplementary_summary.csv')
    full = extras.loc[(extras['stage'] == 'ablation') & (extras['feature_set'] == 'full')]
    assert full['Regret@5'].tolist() == [0, 0]  # Same seed 1201 reference, not five-seed average.


def test_scheduler_dependencies_failure_and_explicit_retry(tmp_path, monkeypatch):
    # A tiny subprocess surrogate tests scheduling without spending the real budget.
    plan = [dict(id='case/a', dependencies=[]), dict(id='case/b', dependencies=['case/a'])]
    contract = {'failures':{'per_job_timeout_seconds':60, 'total_budget_seconds':60}}
    contract_path = tmp_path / 'contract.json'
    write_json(contract_path, contract)
    monkeypatch.setattr(runner, 'build_plan', lambda c: plan)
    monkeypatch.setattr(runner, 'verify_sources', lambda *args: None)
    monkeypatch.setattr(runner, 'environment', lambda: {})
    launched = []
    fail = [True]
    class Process:
        def __init__(self, arguments, **kwargs):
            identifier = arguments[arguments.index('--job') + 1]
            destination = Path(arguments[arguments.index('--attempt') + 1])
            self.returncode = 1 if identifier == 'case/b' and fail[0] else 0
            launched.append((identifier, destination.name))
            if identifier == 'case/b':
                assert execution.completed_job(tmp_path / 'run', plan[0])
            (destination / 'output.txt').write_text('ok')
            write_json(destination / 'result.json', dict(status='failed' if self.returncode else 'complete',
                       job=next(j for j in plan if j['id'] == identifier),
                       artifacts={'output.txt':sha256(destination / 'output.txt')}))
        def poll(self):
            return self.returncode
    monkeypatch.setattr(runner.subprocess, 'Popen', Process)
    monkeypatch.setattr(runner.time, 'sleep', lambda seconds: None)
    with pytest.raises(RuntimeError, match='Job failed'):
        runner.run_evaluation(contract_path, tmp_path / 'run', jobs=1, threads=1)
    with pytest.raises(ValueError, match='retry-failed'):
        runner.run_evaluation(contract_path, tmp_path / 'run', jobs=1, threads=1, resume=True)
    fail[0] = False
    result = runner.run_evaluation(contract_path, tmp_path / 'run', jobs=1, threads=1, resume=True, retry_failed=True)
    assert result['status'] == 'complete' and result['completed_jobs'] == 2
    assert launched == [('case/a','attempt-001'), ('case/b','attempt-001'), ('case/b','attempt-002')]


def test_saved_deployment_models_score_all_nine_selectors_without_labels(
        evaluation_contract, evaluation_repo, tmp_path, monkeypatch):
    jobs = [j for j in build_plan(evaluation_contract) if j['stage'] == 'deployment' and j['case'] == 'BIO-INSIGHT']
    run = tmp_path / 'run'
    run.mkdir()
    write_json(run / 'contract.json', evaluation_contract)
    monkeypatch.setattr(execution, 'select_job_config', lambda *args: (evaluation_contract['split_contract']['grid'][0], None))
    monkeypatch.setattr(execution, 'fit_model', lambda p, a, c, s, t: fit_model(p, a, dict(c, iterations=8), s, t))
    for job in jobs:
        destination = run / 'jobs' / job['id'] / 'attempt-001'
        destination.mkdir(parents=True)
        execution.execute_job(evaluation_repo, evaluation_contract, job, {}, run, destination, 1)
    monkeypatch.setattr(deployment, 'read_run', lambda *args: ({'repo_root':str(evaluation_repo), 'threads':1}, evaluation_contract, jobs))
    monkeypatch.setattr(deployment, 'verify_sources', lambda *args: None)
    application = pd.read_csv(evaluation_repo / 'experiments/BIO-INSIGHT/data/data_104.csv').head(4).drop(columns='AUPR')
    path = tmp_path / 'unlabeled.csv'
    application.to_csv(path, index=False)
    result = deployment.score_deployment(run, 'BIO-INSIGHT', path, tmp_path / 'scores')
    assert result['selectors'] == 9 and result['rows'] == 4
    scores = pd.read_csv(tmp_path / 'scores/candidate_scores.csv')
    assert 'AUPR' not in scores
    assert scores['clf_top20'].between(0, 1).all()
    assert np.isfinite(scores.to_numpy()).all()


def test_real_worker_subprocess_on_synthetic_fronts(evaluation_contract, evaluation_repo, tmp_path):
    contract_path, output = tmp_path / 'contract.json', tmp_path / 'run'
    freeze_contract(contract_path, evaluation_contract)
    runner.run_evaluation(contract_path, output, root=evaluation_repo, jobs=1, threads=1, dry_run=True)
    job = next(j for j in build_plan(evaluation_contract) if j['stage'] == 'tuning')
    destination = output / 'jobs' / job['id'] / 'attempt-001'
    destination.mkdir(parents=True)
    result = subprocess.run([sys.executable, '-m', runner.MODULE, '--worker', '--run', str(output),
                             '--job', job['id'], '--attempt', str(destination)], cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr or (destination / 'result.json').read_text()
    report, _ = execution.completed_job(output, job)
    assert report['tree_count'] == 3000
    # CatBoost omits runtime thread_count from get_all_params; the run manifest
    # records the value supplied to training, Pool construction and prediction.
    assert json.loads((output / 'manifest.json').read_text())['threads'] == 1
    assert report['effective_parameters']['use_best_model'] is False
    assert report['configuration_selection'] is None
