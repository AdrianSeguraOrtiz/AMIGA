"""Subset isolation, exact selection, native deployment and deposit integrity."""
import hashlib
import json
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from test_sequential_selection import toy_frame,tiny_params
from scripts.experiments.amiga_exp.sequential_selection import models
from scripts.experiments.amiga_exp.grouped_validation.contract import _partitions,_front_ids,_learning_subsets
from scripts.experiments.amiga_exp.sequential_selection.summary import choose
from scripts.experiments.amiga_exp.supplementary import spec,selection,execution,archive
from scripts.experiments.amiga_exp.supplementary.deployment import load_native,stable_choice
from scripts.experiments.amiga_exp.supplementary.summary import aggregate_learning
from scripts.experiments.amiga_exp.grouped_validation.metrics import METRICS
from scripts.experiments.amiga_exp.supplementary.pipeline import atomic_stage


def test_full_inner_selection_is_insulated_from_unavailable_labels(tmp_path,monkeypatch):
    frame=toy_frame()
    features=[f'x{i}' for i in range(8)]
    scope=dict(id='learning/fold-0/size-6/subset-1301',train_front_ids=list(range(1,7)),
               test_front_ids=[7,8],inner_folds=[dict(fold=i,validation_front_ids=valid,
               train_front_ids=sorted(set(range(1,7))-set(valid)))
               for i,valid in enumerate(([1,2],[3,4],[5,6]))])
    mapping={str(i):str(i) for i in range(1,9)}
    params=dict(tiny_params('LightGBM'),id='tiny')
    contract=dict(contexts=[scope],original=dict(labels=list(models.LABELS),
        references={'LightGBM':params},grids={'LightGBM':[params]},
        split_contract=dict(topology_by_front=mapping,cases={'BIO-INSIGHT':dict(feature_columns=features)})))
    job=dict(context_id=scope['id'],case='BIO-INSIGHT',family='LightGBM',
             arms=['ranking','reg_aupr'],planned_fits=54)
    loaded=[]
    def restricted(root,original,case,ids):
        loaded.append(set(ids))
        assert set(ids).isdisjoint({7,8})
        return frame[frame.front_id.isin(ids)].copy()
    monkeypatch.setattr(selection,'load_rows',restricted)
    first=tmp_path/'first'
    second=tmp_path/'second'
    first.mkdir()
    second.mkdir()
    selection.select(tmp_path,contract,job,first,1)
    # Altering unavailable test labels cannot alter any inner decision.
    frame.loc[frame.front_id.isin([7,8]),'AUPR']=1-frame.loc[frame.front_id.isin([7,8]),'AUPR']
    selection.select(tmp_path,contract,job,second,1)
    assert loaded==[set(range(1,7))]*2
    assert (first/'selected_procedures.json').read_bytes()==(second/'selected_procedures.json').read_bytes()
    candidates=pd.read_csv(first/'selection_candidates.csv')
    for procedure in json.loads((first/'selected_procedures.json').read_text()):
        expected=choose(candidates[(candidates.stage=='phase3')&(candidates.arm==procedure['arm'])].to_dict('records'),features=True)
        assert expected['candidate']==procedure['candidate']
    progress=json.loads((first/'progress.json').read_text())
    assert progress['completed_fits']==progress['planned_fits']==54


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


def test_final_fit_writes_predictions_before_requesting_test_quality(tmp_path,monkeypatch):
    frame=toy_frame()
    features=[f'x{i}' for i in range(8)]
    scope=dict(id='learning/fold-0/size-6/subset-1301',kind='learning',training_size=6,
               train_front_ids=list(range(1,7)),test_front_ids=[7,8])
    c=dict(contexts=[scope],final_seeds=list(range(1201,1206)),
        original=dict(split_contract=dict(topology_by_front={str(i):str(i) for i in range(1,9)},
                        cases={'BIO-INSIGHT':dict(feature_columns=features)})))
    job=dict(id='BIO-INSIGHT/'+scope['id']+'/final',stage='final',case='BIO-INSIGHT',
             context_id=scope['id'],arms=['ranking','reg_aupr'],dependencies=['dep'])
    dep=tmp_path/'dependency'
    dep.mkdir()
    procedures=[dict(arm=arm,family='LightGBM',label='rank_dense',config=tiny_params('LightGBM'),
        fraction=1.,n_features=8,candidate=arm,mean_regret5=.1,mean_regret1=.2) for arm in job['arms']]
    (dep/'selected_procedures.json').write_text(json.dumps(procedures))
    (dep/'result.json').write_text('{}')
    monkeypatch.setattr(execution,'completed_job',lambda *args:({},dep))
    dest=tmp_path/'final'
    dest.mkdir()
    label_reads=[]
    def restricted(root,original,case,ids,cols,labels=True):
        if ids==[7,8] and labels:
            label_reads.append(1)
            assert len(list(dest.glob('*.predictions.csv.gz')))==len(label_reads)
        columns=['front_id','item_id',*cols,*(['AUPR'] if labels else [])]
        return frame.loc[frame.front_id.isin(ids),columns].copy().reset_index(drop=True)
    monkeypatch.setattr(execution,'load_rows',restricted)
    result=execution.execute_job(tmp_path,c,job,{'dep':{}},tmp_path,dest,1)
    assert result['status']=='complete' and len(label_reads)==10
    recorded=pd.read_csv(dest/'metrics.csv')
    assert len(recorded)==2*2*5
    assert set(recorded.seed)==set(range(1201,1206))


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


def test_learning_contexts_are_nested_and_never_use_outer_groups():
    base=base_splits()
    contexts=spec.contexts(base)
    assert len(contexts)==46
    mapping=base['topology_by_front']
    for scope in contexts:
        train={mapping[str(f)] for f in scope['train_front_ids']}
        test={mapping[str(f)] for f in scope['test_front_ids']}
        assert not train & test and len(train)==scope['training_size']
        assert set(scope['train_front_ids'])==set(_front_ids(mapping,sorted(train)))
        observed=[]
        for inner in scope['inner_folds']:
            a={mapping[str(f)] for f in inner['train_front_ids']}
            b={mapping[str(f)] for f in inner['validation_front_ids']}
            assert not a & b and a | b==train and not (a|b)&test
            observed.extend(inner['validation_front_ids'])
        assert sorted(observed)==scope['train_front_ids']
    for fold,seed in product(range(5),(1301,1302,1303)):
        subsets=[set(s['train_front_ids']) for s in contexts if s['outer_fold']==fold and s['subset_seed']==seed]
        assert subsets[0]<subsets[1]<subsets[2]


@pytest.mark.parametrize('family,arm',product(models.FAMILIES,spec.METHODS))
def test_native_model_round_trip_preserves_predictions(family,arm,tmp_path):
    frame=toy_frame()
    features=[f'x{i}' for i in range(8)]
    library=selection.adapter(arm)
    data=library.prepare(frame,'BIO-INSIGHT',{i:str(i) for i in range(1,9)},features)
    model,_=library.fit(data,family,arm,tiny_params(family),threads=1)
    expected=library.scores(model,family,arm,data['X'],1)
    path=execution.save_model(model,family,tmp_path/'model')
    info=dict(procedure=dict(family=family,arm=arm),model_file=path.name,
              model_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    loaded=load_native(tmp_path,info)
    np.testing.assert_allclose(library.scores(loaded,family,arm,data['X'],1),expected,rtol=1e-12,atol=1e-12)
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='identity'): load_native(tmp_path,info)


@pytest.mark.parametrize('family',models.FAMILIES)
def test_relearned_mask_matches_original_recursive_path(family):
    frame=toy_frame()
    features=[f'x{i}' for i in range(8)]
    data=models.prepare(frame,'BIO-INSIGHT',{i:str(i) for i in range(1,9)},features)
    params=tiny_params(family)
    paths,_=models.feature_path(data,family,'ranking',params,threads=1)
    for fraction in (1.,.75,.5,.25):
        procedure=dict(family=family,arm='ranking',config=params,fraction=fraction,
                       n_features=int(np.ceil(8*fraction)))
        mask,_=execution.learn_mask(data,procedure,models,1)
        assert mask==next(p['features'] for p in paths if p['fraction']==fraction)


def test_real_choice_uses_item_identity_for_ties_and_rejects_nonfinite():
    frame=pd.DataFrame({'item_id':[8,2,5]})
    assert stable_choice(frame,[1.,1.,0.])==(1,2)
    with pytest.raises(ValueError): stable_choice(frame,[1.,np.nan,0.])


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
