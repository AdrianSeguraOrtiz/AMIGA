"""Fixed recipes, training-size isolation, original real-case scope and deposits."""
import hashlib
import json
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from test_sequential_selection import toy_frame, tiny_params
from scripts.experiments.amiga_exp.sequential_selection import models
from scripts.experiments.amiga_exp.grouped_validation.contract import _partitions, _front_ids, _learning_subsets
from scripts.experiments.amiga_exp.supplementary import spec, execution, archive, deployment
from scripts.experiments.amiga_exp.supplementary.deployment import load_native, prediction_costs
from scripts.experiments.amiga_exp.supplementary.summary import aggregate_learning
from scripts.experiments.amiga_exp.grouped_validation.metrics import METRICS
from scripts.experiments.amiga_exp.supplementary.pipeline import atomic_stage


def test_learning_summary_gives_equal_topology_weights_after_seed_and_subset_means():
    mapping=base_splits()['topology_by_front']
    rows=[]
    for front,topology in mapping.items():
        for seed,subset in product((1201,1202),(1301,1302,1303)):
            value=(int(topology,16)%2)*.6+(seed-1201)*.02+(subset-1301)*.03
            rows.append(dict(case='BIO-INSIGHT',method='ranking',training_size='10',
                             outer_fold=0,subset_seed=subset,seed=seed,front_id=int(front),
                             **{metric:value for metric in METRICS}))
    curves,fronts,subsets=aggregate_learning(pd.DataFrame(rows),mapping)
    expected=sum((i%2)*.6+.01+.03 for i in range(87))/87
    np.testing.assert_allclose(curves['mean'],expected,rtol=1e-12)
    assert len(fronts)==104 and len(subsets)==104*3
    # Front weighting would count the repeated conditions differently.
    assert not np.isclose(fronts['Regret@5'].mean(),expected)


def test_atomic_stage_recovers_missing_checkpoint_and_detects_changed_outputs(tmp_path):
    calls=[]
    def generate(path):
        calls.append(1)
        path.mkdir()
        item=path/'data.csv'
        item.write_text('a\n1\n')
        (path/'manifest.json').write_text(json.dumps(dict(status='complete',artifacts={
            item.name:hashlib.sha256(item.read_bytes()).hexdigest()})))
    atomic_stage(tmp_path,'summary',generate)
    atomic_stage(tmp_path,'summary',generate)
    assert calls==[1]
    (tmp_path/'summary/data.csv').write_text('changed')
    with pytest.raises(ValueError,match='mismatch'): atomic_stage(tmp_path,'summary',generate)


def test_completion_receipt_requires_scientific_outputs(tmp_path):
    job=dict(id='a/selection',stage='selection',context_id='deployment',arms=['ranking'])
    path=tmp_path/'jobs'/job['id']/'attempt-1'
    path.mkdir(parents=True)
    artifact=path/'unrelated.txt'
    artifact.write_text('irrelevant')
    (path/'result.json').write_text(json.dumps(dict(status='complete',job=job,artifacts={
        artifact.name:hashlib.sha256(artifact.read_bytes()).hexdigest()})))
    with pytest.raises(ValueError,match='omits'): execution.completed_job(tmp_path,job)


def base_splits():
    mapping={str(i+1):f'{i:064x}' for i in range(87)}
    # Conditions of one topology must stay together at every training size.
    mapping.update({str(i+88):mapping[str(i+1)] for i in range(17)})
    outer=[]
    for fold,test in enumerate(_partitions(sorted(set(mapping.values())),20260910,5)):
        train=sorted(set(mapping.values())-set(test))
        outer.append(dict(fold=fold,train_topology_ids=train,test_topology_ids=test,
                          learning_subsets=_learning_subsets(mapping,train)))
    return dict(topology_by_front=mapping,outer_folds=outer,seeds={'deployment_split':20260916})


def test_deposit_detects_tampering_and_restores_without_overwriting(tmp_path):
    folder=tmp_path/'deposit'
    folder.mkdir()
    inventory=archive.pack(folder/'inputs.tar.xz',{'experiments/example/input.csv':b'a,b\n1,2\n'})
    manifest=dict(schema_version=1,status='complete',archives={'inputs.tar.xz':inventory})
    (folder/'manifest.json').write_text(json.dumps(manifest))
    assert archive.verify_archive(folder)==manifest
    root=tmp_path/'restore'
    assert archive.restore(folder,root)==1
    assert archive.restore(folder,root)==0
    (root/'experiments/example/input.csv').write_text('different')
    with pytest.raises(ValueError,match='Refusing'): archive.restore(folder,root)
    (folder/'inputs.tar.xz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='identity'): archive.verify_archive(folder)


@pytest.mark.parametrize('path',['../escape','/absolute','x/../escape','a\\b'])
def test_deposit_rejects_escaping_names(tmp_path,path):
    with pytest.raises(ValueError,match='Unsafe'): archive.pack(tmp_path/'bad.tar.xz',{path:b'x'})


def test_deposit_rejects_symlink_destinations_before_any_write(tmp_path):
    folder=tmp_path/'deposit'
    folder.mkdir()
    inventory=archive.pack(folder/'inputs.tar.xz',{'experiments/x.csv':b'a\n'})
    (folder/'manifest.json').write_text(json.dumps(dict(schema_version=1,status='complete',archives={'inputs.tar.xz':inventory})))
    root=tmp_path/'root'
    root.mkdir()
    elsewhere=tmp_path/'elsewhere'
    elsewhere.mkdir()
    (root/'experiments').symlink_to(elsewhere,target_is_directory=True)
    with pytest.raises(ValueError,match='escapes'): archive.restore(folder,root)
    assert not list(elsewhere.iterdir())


def test_cli_exposes_monitoring_and_safe_restore():
    from scripts.experiments.amiga_exp.cli import app
    result=CliRunner().invoke(app,['supplement','--help'])
    assert result.exit_code==0
    for word in ['freeze','run','status','archive','restore-archive','verify-archive']:
        assert word in result.stdout


def test_fixed_refits_ignore_unavailable_labels_and_never_reselect_columns(tmp_path, monkeypatch):
    frame = toy_frame()
    features = ['x0', 'x3']
    scope = dict(id='learning/fold-0/size-4/subset-1301', kind='learning', training_size=4,
                 train_front_ids=[1, 2, 3, 4], test_front_ids=[7, 8])
    procedure = dict(case='BIO-INSIGHT', outer_fold=0, arm='ranking', family='LightGBM',
                     label='rank_dense', config=tiny_params('LightGBM'), fraction=.25, n_features=2)
    recipe = dict(procedure=procedure, feature_columns=features, source_model_metadata='original/model.json')
    c = dict(contexts=[scope], final_seeds=list(range(1201, 1206)),
        original=dict(split_contract=dict(topology_by_front={str(i): str(i) for i in range(1, 9)})))
    job = dict(id='BIO-INSIGHT/'+scope['id']+'/final', stage='final', case='BIO-INSIGHT',
               context_id=scope['id'], arms=['ranking'], recipe=recipe, planned_fits=5)
    calls = []
    original_fit = models.fit
    def fit(data, family, arm, params, **kwargs):
        assert data['feature_names'] == features
        assert set(data['group_id']) == {1, 2, 3, 4}
        assert params == procedure['config'] and family == 'LightGBM' and arm == 'ranking'
        calls.append(kwargs['seed'])
        return original_fit(data, family, arm, params, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail('Fixed-model curve must not run feature selection')
    monkeypatch.setattr(models, 'fit', fit)
    monkeypatch.setattr(models, 'feature_importance', forbidden)
    monkeypatch.setattr(models, 'feature_path', forbidden)
    def execute(name):
        run = tmp_path/name
        dest = run/'jobs'/job['id']/'attempt-001'
        dest.mkdir(parents=True)
        reads = []
        def load(root, original, case, ids, cols, *, labels=True):
            assert set(ids) in ({1, 2, 3, 4}, {7, 8})
            if ids == [7, 8] and labels:
                reads.append(1)
                assert len(list(dest.glob('*.predictions.csv.gz'))) == len(reads)
            return frame.loc[frame.front_id.isin(ids), ['front_id', 'item_id', *cols,
                             *(['AUPR'] if labels else [])]].copy().reset_index(drop=True)
        monkeypatch.setattr(execution, 'load_rows', load)
        execution.execute_job(tmp_path, c, job, {}, run, dest, 1)
        assert execution.completed_job(run, job) is not None
        assert len(reads) == 5
        return dest
    first = execute('first')
    frame.loc[frame.front_id.isin([5, 6, 7, 8]), 'AUPR'] = 1-frame.loc[frame.front_id.isin([5, 6, 7, 8]), 'AUPR']
    second = execute('second')
    assert calls == list(range(1201, 1206))*2
    for seed in range(1201, 1206):
        name = f'ranking-seed-{seed}.predictions.csv.gz'
        assert (first/name).read_bytes() == (second/name).read_bytes()
        info = json.loads((second/f'ranking-seed-{seed}.model.json').read_text())
        assert info['feature_columns'] == features and not info['selection_repeated']
    assert set(pd.read_csv(second/'metrics.csv').seed) == set(range(1201, 1206))


def test_learning_contexts_are_nested_and_never_use_outer_groups():
    base = base_splits()
    scopes = spec.contexts(base)
    assert len(scopes) == 46
    mapping = base['topology_by_front']
    for scope in scopes:
        train = {mapping[str(f)] for f in scope['train_front_ids']}
        test = {mapping[str(f)] for f in scope['test_front_ids']}
        assert not train & test and len(train) == scope['training_size']
        assert set(scope['train_front_ids']) == set(_front_ids(mapping, sorted(train)))
        assert 'inner_folds' not in scope
    for fold, seed in product(range(5), (1301, 1302, 1303)):
        subsets = [set(s['train_front_ids']) for s in scopes if s['outer_fold'] == fold and s['subset_seed'] == seed]
        assert subsets[0] < subsets[1] < subsets[2]


@pytest.mark.parametrize('family', models.FAMILIES)
def test_native_ranker_round_trip_and_prepared_front_timing(family, tmp_path):
    features = [f'x{i}' for i in range(8)]
    data = models.prepare(toy_frame(), 'BIO-INSIGHT', {i: str(i) for i in range(1, 9)}, features)
    model, _ = models.fit(data, family, 'ranking', tiny_params(family), threads=1)
    expected = models.scores(model, family, 'ranking', data['X'], 1)
    path = execution.save_model(model, family, tmp_path/'model')
    info = dict(procedure=dict(family=family, arm='ranking'), model_file=path.name,
                model_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    loaded = load_native(tmp_path, info)
    actual, timings = prediction_costs(loaded, family, data['X'], np.arange(len(data['X'])), 1)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert len(timings) == 5 and set(timings.method) == {'AMIGA'}
    np.testing.assert_allclose(timings.total_seconds, timings.prediction_seconds+timings.ranking_seconds)
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='identity'):
        load_native(tmp_path, info)


def test_deployment_choice_uses_only_recorded_inner_validation():
    recipes = [dict(procedure=dict(case='BIO-INSIGHT', outer_fold=i, inner_mean_regret5=value))
               for i, value in enumerate((.2, .1, .1, .3, .4))]
    assert spec.deployment_recipe(recipes) == recipes[1]
    altered = deepcopy(recipes)
    for i, r in enumerate(altered):
        r['outer_test_regret'] = 1/(i+1)
        r['tcga_support'] = i
    assert spec.deployment_recipe(altered)['procedure'] == recipes[1]['procedure']


def test_real_case_keeps_original_five_selectors_and_fixed_evidence(tmp_path, monkeypatch):
    from scripts.experiments.amiga_exp.real_world_validation import OBJECTIVE_COLUMNS, REPORTED_SOURCES, REPORTED_SELECTOR_IDS
    case = tmp_path/'case'
    (case/'amiga').mkdir(parents=True)
    evidence_dir = case/'validation/amiga_exp_reported'
    evidence_dir.mkdir(parents=True)
    grns = case/'bioinsight/input/lists'
    grns.mkdir(parents=True)
    (grns/'GRN_TOY.csv').write_text('TF1,G1,0.9\nTF1,G2,0.8\n')
    frame = toy_frame().query('front_id == 1').drop(columns='AUPR').reset_index(drop=True)
    for i, objective in enumerate(OBJECTIVE_COLUMNS):
        frame[objective] = np.arange(len(frame))*(i+1)/len(frame)
    frame['GRN_TOY.csv'] = 1.0
    frame.to_csv(case/'amiga/data_real.csv', index=False)
    evidence = pd.DataFrame([dict(resource=s['resource'], source='TF1', target='G1') for s in REPORTED_SOURCES])
    evidence.to_csv(evidence_dir/'reported_external_tf_target_evidence.csv', index=False)
    directory = tmp_path/'model'
    directory.mkdir()
    features = [f'x{i}' for i in range(8)]
    data = models.prepare(toy_frame(), 'BIO-INSIGHT', {i: str(i) for i in range(1, 9)}, features)
    model, _ = models.fit(data, 'LightGBM', 'ranking', tiny_params('LightGBM'), threads=1)
    native = execution.save_model(model, 'LightGBM', directory/'ranking')
    info = dict(procedure=dict(family='LightGBM', arm='ranking'), feature_columns=features,
                model_file=native.name, model_sha256=hashlib.sha256(native.read_bytes()).hexdigest())
    (directory/'ranking-seed-1201.model.json').write_text(json.dumps(info))
    c = dict(deployment_seed=1201, deployment_policy=spec.DEPLOYMENT_POLICY,
             prediction_cost_scope='prepared-front prediction and ranking only')
    monkeypatch.setattr(deployment, 'read_run', lambda run: (dict(repo_root=str(tmp_path), threads=1), c, [dict(context_id='deployment')]))
    monkeypatch.setattr(deployment, 'verify_sources', lambda *args: None)
    monkeypatch.setattr(deployment, 'completed_job', lambda *args: ({}, directory))
    output = tmp_path/'application'
    result = deployment.apply(tmp_path/'run', case, output)
    assert result['selectors'] == list(REPORTED_SELECTOR_IDS)
    assert len(pd.read_csv(output/'real_world_source_support_top1.csv')) == 5
    assert len(pd.read_csv(output/'prediction_costs.csv')) == 5
    pd.testing.assert_frame_equal(pd.read_csv(output/'source_evidence_snapshot.csv'), evidence)
    assert (case/'amiga/data_real.csv').read_bytes() == frame.to_csv(index=False).encode()


def test_plan_has_only_fixed_amiga_refits_and_rejects_comparators():
    base = base_splits()
    base['cases'] = {case: dict(feature_columns=['x0']) for case in ('BIO-INSIGHT', 'MO-GENECI')}
    recipes = [dict(procedure=dict(case=case, outer_fold=fold, arm='ranking', n_features=1,
                                  inner_mean_regret5=.01+fold*.001),
                    feature_columns=['x0'], source_model_metadata=f'{case}/{fold}/model.json')
               for case in base['cases'] for fold in range(5)]
    c = dict(schema_version=2, workflow='fixed_amiga_supplement', status='frozen',
        original=dict(split_contract=base, procedures=[r['procedure'] for r in recipes], source_hashes={}),
        contexts=spec.contexts(base), recipes=recipes, deployment_recipe=spec.deployment_recipe(recipes),
        deployment_policy=spec.DEPLOYMENT_POLICY, learning_methods=['ranking'], deployment_methods=['ranking'],
        final_seeds=list(range(1201, 1206)), deployment_seed=1201, full_endpoint_summary=spec.SUMMARY,
        source_hashes={}, failures=dict(total_budget_seconds=518400, per_job_timeout_seconds=43200, automatic_retries=0))
    plan = spec.build_plan(c)
    assert len(plan) == 91 and sum(j['planned_fits'] for j in plan) == 451
    assert all(j['stage'] == 'final' and j['arms'] == ['ranking'] and not j['dependencies'] for j in plan)
    assert sum(j['context_id'] == 'deployment' for j in plan) == 1
    changed = deepcopy(c)
    changed['learning_methods'].append('reg_aupr')
    with pytest.raises(ValueError, match='fixed-AMIGA'):
        spec.build_plan(changed)
    changed = deepcopy(c)
    changed['recipes'][0]['feature_columns'] = []
    with pytest.raises(ValueError, match='Fixed recipe'):
        spec.build_plan(changed)
