"""Scientific isolation, native contribution correctness and sequential execution."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from xgboost import DMatrix

from test_grouped_validation_contract import fixture_repo, REPO_ROOT
from scripts.experiments.amiga_exp.sequential_selection import models, spec, execution
from scripts.experiments.amiga_exp.sequential_selection.runner import verify_sources
from scripts.experiments.amiga_exp.sequential_selection.summary import choose, validate_metrics
from scripts.experiments.amiga_exp.sequential_selection.costing import choose_profile


@pytest.fixture(scope='module')
def sequential_repo(fixture_repo, tmp_path_factory):
    root = tmp_path_factory.mktemp('sequential-repo')
    shutil.copytree(fixture_repo, root, dirs_exist_ok=True)
    for package in ('grouped_validation', 'sequential_selection'):
        relative = f'scripts/experiments/amiga_exp/{package}'
        shutil.copytree(REPO_ROOT / relative, root / relative, dirs_exist_ok=True)
    return root


@pytest.fixture(scope='module')
def contract(sequential_repo):
    return spec.build_contract(sequential_repo)


def toy_frame():
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(rng.normal(size=(8*24, 8)), columns=[f'x{i}' for i in range(8)])
    frame['front_id'] = np.repeat(np.arange(1, 9), 24)
    frame['item_id'] = np.tile(np.arange(24), 8)
    frame['AUPR'] = expit(frame['x0'] + .3*frame['x1'])
    return frame


def prepared():
    return models.prepare(toy_frame(), 'BIO-INSIGHT', {i:str(i//2) for i in range(1,9)},
                          [f'x{i}' for i in range(8)])


def tiny_params(family):
    values = spec.reference(family)
    values['iterations'] = 8
    if family == 'LightGBM':
        values.update(num_leaves=7, min_child_samples=5)
    elif family == 'XGBoost':
        values.update(max_depth=2)
    else:
        values.update(depth=2)
    return values


@pytest.mark.parametrize('family,arm', product(models.FAMILIES, models.ARMS))
def test_native_shap_additivity_and_score_semantics(family, arm):
    data = prepared()
    model, report = models.fit(data, family, arm, tiny_params(family), threads=1)
    X = data['X'][:20]
    shap = models.contributions(model, family, X, 1)
    if family == 'CatBoost':
        raw = (model.predict(X, thread_count=1) if arm == 'ranking' else
               model.predict(X, prediction_type='RawFormulaVal', thread_count=1))
    elif family == 'XGBoost':
        raw = model.get_booster().predict(DMatrix(X), output_margin=True)
    else:
        raw = model.booster_.predict(X, raw_score=True, num_threads=1)
    np.testing.assert_allclose(shap.sum(axis=1), raw, rtol=1e-5, atol=1e-6)
    expected = expit(raw) if arm == 'clf_top20' else raw
    np.testing.assert_allclose(models.scores(model, family, arm, X, 1), expected, rtol=1e-5)
    assert report['no_validation_set'] and not report['early_stopping']


@pytest.mark.parametrize('family,label', product(models.FAMILIES, models.LABELS))
def test_every_screening_label_is_accepted_by_each_ranker(family, label):
    data = models.prepare(toy_frame(), 'BIO-INSIGHT', {i:str(i) for i in range(1,9)},
                          [f'x{i}' for i in range(8)], label)
    model, _ = models.fit(data, family, 'ranking', tiny_params(family), threads=1)
    assert np.isfinite(models.scores(model, family, 'ranking', data['X'], 1)).all()


@pytest.mark.parametrize('family', models.FAMILIES)
def test_recursive_masks_deterministic_and_ignore_validation_callback(family):
    data = prepared()
    a, imp = models.feature_path(data, family, 'ranking', tiny_params(family), threads=1,
                                 callback=lambda *args: {'validation_quality': -100})
    b, _ = models.feature_path(data, family, 'ranking', tiny_params(family), threads=1,
                               callback=lambda *args: {'validation_quality': 100})
    assert [p['features'] for p in a] == [p['features'] for p in b]
    assert [p['n_features'] for p in a] == [8, 6, 4, 2]
    for parent, child in zip(a, a[1:]):
        assert set(child['features']) < set(parent['features'])
    assert {'mean_abs_shap', 'centered_mean_abs_shap', 'feature', 'fraction'} <= set(imp)
    assert all(p['sampling']['scope'] == 'training_only' for p in a[:-1])


def test_sampling_and_centering_do_not_use_quality(monkeypatch):
    data = prepared()
    sample = models.sample_rows(data, 5)
    assert len(sample) == 8*5
    changed = deepcopy(data)
    changed['labels']['reg_aupr'] = np.zeros(len(data['X']))
    np.testing.assert_array_equal(sample, models.sample_rows(changed, 5))
    # A feature with a query-constant contribution cannot change within-front order.
    def fake(model, family, X, threads):
        return np.c_[X, np.zeros(len(X))]
    monkeypatch.setattr(models, 'contributions', fake)
    data['X'][:, 0] = np.repeat(np.arange(8), 24)
    importance, _ = models.feature_importance(None, 'LightGBM', data, threads=1)
    row = importance.set_index('feature').loc['x0']
    assert row['centered_mean_abs_shap'] == 0
    assert row['mean_abs_shap'] > 0


def test_plan_counts_scope_and_dependency_graph(contract):
    assert spec.counts(contract) == dict(jobs=4320, total_fits=14040,
        jobs_by_stage={'phase1':240, 'phase2':3960, 'phase3':120},
        fits_by_stage={'phase1':720, 'phase2':11880, 'phase3':1440})
    plan = spec.build_plan(contract)
    seen = set()
    for job in plan:
        assert set(job['dependencies']) <= seen
        seen.add(job['id'])
        if job['stage'] == 'phase2' and job['arm'] == 'ranking':
            assert len(job['dependencies']) == 8
        if job['stage'] == 'phase3':
            assert len(job['dependencies']) == len(contract['grids'][job['family']])
    mapping = contract['split_contract']['topology_by_front']
    for outer in contract['split_contract']['outer_folds']:
        heldout = {mapping[str(i)] for i in outer['test_front_ids']}
        for inner in outer['inner_folds']:
            train = {mapping[str(i)] for i in inner['train_front_ids']}
            valid = {mapping[str(i)] for i in inner['validation_front_ids']}
            assert not train & valid and not heldout & (train | valid)
    assert not contract['outer_evaluation_enabled']


def test_bounded_budget_and_no_silent_source_changes(sequential_repo, tmp_path):
    c = spec.build_contract(sequential_repo, profile='bounded')
    assert spec.counts(c)['total_fits'] == 4320
    assert all(len(g) == 6 for g in c['grids'].values())
    verify_sources(sequential_repo, c)
    root = tmp_path / 'copy'
    shutil.copytree(sequential_repo, root)
    (root / spec.PACKAGE / 'models.py').write_text('# altered source\n')
    with pytest.raises(ValueError, match='Source changed'):
        verify_sources(root, c)


def test_selection_ties_and_cost_gate():
    rows = [dict(candidate='b', mean_regret5=.1, mean_regret1=.2, n_features=20),
            dict(candidate='a', mean_regret5=.1+1e-14, mean_regret1=.2, n_features=30)]
    assert choose(rows)['candidate'] == 'a'
    assert choose(rows, features=True)['candidate'] == 'b'
    assert choose_profile({'projections':{'original':{'projected_hours':70},
                                         'bounded':{'projected_hours':20}}}) == 'original'
    assert choose_profile({'projections':{'original':{'projected_hours':80},
                                         'bounded':{'projected_hours':20}}}) == 'bounded'
    with pytest.raises(ValueError, match='Neither'):
        choose_profile({'projections':{'original':{'projected_hours':80},
                                      'bounded':{'projected_hours':75}}})


def test_sequential_dependency_chain_never_loads_outer_rows(tmp_path, monkeypatch):
    frame = toy_frame()
    features = [f'x{i}' for i in range(8)]
    outer = dict(fold=0, train_front_ids=list(range(1,7)), test_front_ids=[7,8],
                 inner_folds=[dict(fold=i, validation_front_ids=valid,
                                   train_front_ids=sorted(set(range(1,7))-set(valid)))
                              for i,valid in enumerate(([1,2],[3,4],[5,6]))])
    contract = dict(split_contract=dict(topology_by_front={str(i):str(i) for i in range(1,9)},
                                        cases={'BIO-INSIGHT':dict(feature_columns=features)},
                                        outer_folds=[outer]), fractions=list(models.FRACTIONS))
    def load(root, c, case, ids):
        assert set(ids) <= set(outer['train_front_ids'])
        return frame[frame['front_id'].isin(ids)].copy()
    monkeypatch.setattr(execution, 'load_rows', load)
    common = dict(case='BIO-INSIGHT', outer_fold=0, family='CatBoost', arm='ranking')
    plan = {}
    for label in ('rank_dense', 'continuous'):
        job = dict(common, id=f'phase1/{label}', stage='phase1', dependencies=[],
                   config=tiny_params('CatBoost'), label=label)
        plan[job['id']] = job
    phase1 = list(plan)
    for i in range(2):
        job = dict(common, id=f'phase2/cfg-{i}', stage='phase2', dependencies=phase1,
                   config=dict(tiny_params('CatBoost'), id=f'cfg-{i}', depth=i+2), label=None)
        plan[job['id']] = job
    job = dict(common, id='phase3', stage='phase3', dependencies=['phase2/cfg-0','phase2/cfg-1'],
               config=None, label=None)
    plan[job['id']] = job
    for job in plan.values():
        directory = tmp_path / 'jobs' / job['id'] / 'attempt-001'
        directory.mkdir(parents=True)
        report = execution.execute_job(tmp_path, contract, job, plan, tmp_path, directory, 1)
        assert report['status'] == 'complete' and not report['outer_predictions_computed']
        assert execution.completed_job(tmp_path, job)
        metrics = pd.read_csv(directory/'metrics.csv')
        validate_metrics(metrics, contract, job)
        assert set(metrics['front_id']) == set(range(1,7))
    selected = json.loads((directory/'selection.json').read_text())
    assert selected['upstream_selection']['selected_job'] in job['dependencies']
    paths = json.loads((directory/'feature_paths.json').read_text())
    assert len(paths) == 3 and all(len(p['path']) == 4 for p in paths)
    # Summarize the executed dependency chain, with a lightweight manifest fixture.
    from scripts.experiments.amiga_exp.sequential_selection import summary
    contract['mode'] = 'selection'
    (tmp_path/'contract.json').write_text(json.dumps(contract))
    (tmp_path/'state.json').write_text(json.dumps(dict(status='complete')))
    monkeypatch.setattr(summary, 'read_run', lambda run: (dict(repo_root=str(tmp_path)), contract, list(plan.values())))
    monkeypatch.setattr(summary, 'verify_sources', lambda root,c: None)
    output = tmp_path/'summary'
    report = summary.summarize(tmp_path, output, figures=False)
    assert report['selected_procedures'] == 1 and not report['outer_evaluation_performed']
    decision = json.loads((output/'selected_procedures.json').read_text())[0]
    assert decision['family'] == 'CatBoost'
    assert decision['outer_feature_mask'] == 'must_be_relearned_on_outer_training_only'
    masks = pd.read_csv(output/'selected_inner_masks.csv')
    assert masks.groupby('inner_fold')['selected'].sum().tolist() == [decision['n_features']]*3
    with pytest.raises(ValueError, match='new summary directory'):
        summary.summarize(tmp_path, output, figures=False)
    with (directory/'metrics.csv').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='artifact changed'):
        execution.completed_job(tmp_path, job)


def test_selection_figures_export_all_stages_and_csv(tmp_path):
    from scripts.experiments.amiga_exp.sequential_selection.plots import make_plots
    rows = []
    for family, label in product(models.FAMILIES, models.LABELS):
        rows.append(dict(case='SYNTHETIC', outer_fold=0, stage='phase1', family=family,
                         arm='ranking', label=label, mean_regret5=.02+.001*models.LABELS.index(label),
                         selected_within_family=label=='rank_dense'))
    for family, arm in product(models.FAMILIES, models.ARMS):
        for i in range(3):
            rows.append(dict(case='SYNTHETIC', outer_fold=0, stage='phase2', family=family,
                             arm=arm, mean_regret5=.02+.001*i, std_regret5=.03+.002*i,
                             selected_within_family=i==0))
        for i,fraction in enumerate(models.FRACTIONS):
            rows.append(dict(case='SYNTHETIC', outer_fold=0, stage='phase3', family=family,
                             arm=arm, mean_regret5=.02+.001*i, n_features=int(100*fraction),
                             selected_within_family=i==0))
    stability = pd.DataFrame([dict(case='SYNTHETIC', family=f, arm=a, feature=x,
                                   selected_frequency=.5, number_of_fits=15)
                              for f,a,x in product(models.FAMILIES, models.ARMS, ['x0','x1'])])
    make_plots(pd.DataFrame(rows), stability, tmp_path)
    assert len(list(tmp_path.rglob('*.pdf'))) == 4
    assert len(list(tmp_path.rglob('*.png'))) == 4
    assert len(list(tmp_path.rglob('*.csv'))) == 3
    assert all(p.stat().st_size > 1000 for p in tmp_path.rglob('*.pdf'))


@pytest.mark.parametrize('profile', ['original', 'bounded'])
def test_full_pipeline_requires_original_grid(sequential_repo, tmp_path, monkeypatch, profile):
    from scripts.experiments.amiga_exp.sequential_selection import pipeline
    contract_path = tmp_path/'contract.json'
    spec.freeze(contract_path,spec.build_contract(sequential_repo,profile=profile,budget_seconds=144*3600))
    monkeypatch.setattr(pipeline, 'REPO', sequential_repo)
    started = []
    def run(contract_path, output, **kwargs):
        contract = json.loads(contract_path.read_text())
        assert kwargs['jobs'] == 16 and kwargs['threads'] == 4
        if kwargs.get('dry_run'):
            output.mkdir()
            return dict(status='planned')
        started.append(contract['grid_profile'])
        assert contract['failures']['total_budget_seconds'] == 144*3600
        return dict(status='complete')
    monkeypatch.setattr(pipeline, 'run_selection', run)
    monkeypatch.setattr(pipeline, 'summarize', lambda *args: dict(selected_procedures=40))
    args = (contract_path, tmp_path/'launch', tmp_path/'run', tmp_path/'summary')
    if profile == 'original':
        pipeline.run_full(*args,jobs=16,threads=4)
        assert started == ['original']
        assert json.loads((tmp_path/'launch/state.json').read_text())['status'] == 'complete'
    else:
        with pytest.raises(ValueError, match='complete original grids'):
            pipeline.run_full(*args,jobs=16,threads=4)
        assert not started
        assert not (tmp_path/'run').exists()


def test_worker_cpu_sets_are_disjoint_and_obey_quota():
    from scripts.experiments.amiga_exp.sequential_selection.resources import layout
    limits = dict(allowed_cpus=list(range(64)),capacity=64,cpu_quota=None)
    for workers,threads in [(8,8),(16,4),(32,2),(64,1)]:
        policy = layout(workers,threads,limits)
        flattened = [c for group in policy['cpu_sets'] for c in group]
        assert flattened == list(range(64))
        assert len(set(flattened)) == workers*threads
    with pytest.raises(ValueError,match='exceeds'):
        layout(16,8,limits)
    with pytest.raises(ValueError,match='exceeds'):
        layout(8,8,dict(limits,capacity=32,cpu_quota=32.))


def test_worker_affinity_and_thread_environment_are_effective(tmp_path):
    import os,subprocess,sys
    from scripts.experiments.amiga_exp.sequential_selection.resources import worker_command,worker_environment
    if not hasattr(os,'sched_setaffinity'):
        pytest.skip('Linux affinity check')
    selected = sorted(os.sched_getaffinity(0))[:2]
    script = tmp_path/'cpu_probe.py'
    script.write_text('import os,json; print(json.dumps(dict(cpus=sorted(os.sched_getaffinity(0)),omp=os.environ["OMP_NUM_THREADS"],blas=os.environ["OPENBLAS_NUM_THREADS"])))')
    env = dict(worker_environment(2),PYTHONPATH=str(tmp_path))
    result = subprocess.run(worker_command('cpu_probe',[],selected),env=env,check=True,capture_output=True,text=True)
    value = json.loads(result.stdout)
    assert value == dict(cpus=selected,omp='2',blas='1')


def test_runner_accepts_full_cpu_layout_and_freezes_it(sequential_repo,tmp_path):
    from scripts.experiments.amiga_exp.sequential_selection.runner import run_selection,read_run
    from scripts.experiments.amiga_exp.sequential_selection.resources import cpu_limits
    workers = min(16,cpu_limits()['capacity'])
    path = tmp_path/'contract.json'
    spec.freeze(path,spec.build_contract(sequential_repo,budget_seconds=144*3600))
    output = tmp_path/'run'
    result = run_selection(path,output,root=sequential_repo,jobs=workers,threads=1,dry_run=True)
    assert result['planned_fits'] == 14040
    manifest,_,_ = read_run(output)
    assert manifest['resources']['cpu_budget'] == workers
    with pytest.raises(ValueError,match='worker/thread resources'):
        run_selection(path,output,root=sequential_repo,jobs=1 if workers>1 else 2,threads=1,dry_run=True,resume=True)


def test_pipeline_monitor_uses_live_selection_heartbeat(tmp_path):
    import time,os
    from scripts.experiments.amiga_exp.sequential_selection.monitor import status
    child = tmp_path/'run'
    child.mkdir()
    (child/'state.json').write_text(json.dumps(dict(status='running',accounted_at_unix=time.time(),supervisor_pid=os.getpid(),completed_jobs=4)))
    parent = tmp_path/'launch'
    parent.mkdir()
    (parent/'state.json').write_text(json.dumps(dict(status='running_selection',accounted_at_unix=0,supervisor_pid=os.getpid(),run=str(child))))
    value = status(parent)
    assert value['health'] == 'running'
    assert value['selection']['completed_jobs'] == 4
