"""Target semantics, comparable budgets, isolation and paired evidence coverage."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from test_grouped_validation_contract import REPO_ROOT, fixture_repo
from test_outer_evaluation import phase4_repo, contract as previous_contract, synthetic_metrics
from test_sequential_selection import toy_frame, tiny_params
from scripts.experiments.amiga_exp.grouped_validation.pilot import write_json, sha256
from scripts.experiments.amiga_exp.sequential_selection import models as original_models
from scripts.experiments.amiga_exp.top5_classification import models, spec, outer_execution, summary


@pytest.fixture(scope='module')
def contract(previous_contract,phase4_repo):
    package=Path(spec.PACKAGE)
    shutil.copytree(REPO_ROOT/package,phase4_repo/package,dirs_exist_ok=True)
    shutil.copyfile(REPO_ROOT/spec.DOC,phase4_repo/spec.DOC)
    run,output=phase4_repo/spec.PREVIOUS_RUN,phase4_repo/spec.PREVIOUS_SUMMARY
    run.mkdir(parents=True,exist_ok=True)
    output.mkdir(parents=True,exist_ok=True)
    for name,value in [('contract.json',previous_contract),('manifest.json',{}),('plan.json',[])]:
        write_json(run/name,value)
    synthetic_metrics(previous_contract).to_csv(output/'metrics_long.csv',index=False)
    (output/'central_summary.csv').write_text('placeholder\n')
    write_json(run/'state.json',dict(status='complete'))
    write_json(output/'manifest.json',dict(status='complete',contract_sha256=sha256(run/'contract.json'),
        source_result_hashes={},artifacts={name:sha256(output/name) for name in ('metrics_long.csv','central_summary.csv')}))
    return spec._selection_contract(phase4_repo,previous_contract)


@pytest.mark.parametrize('n,k',[(20,1),(21,2),(100,5),(113,6),(300,15)])
def test_five_percent_is_rounded_per_front(n,k):
    labels,report=models.binary_labels(np.arange(n,dtype=float),np.ones(n,dtype=int))
    assert labels.sum()==k
    assert np.array_equal(labels,np.r_[np.zeros(n-k),np.ones(k)])
    assert report[0]['requested_top_k']==k


def test_ties_are_included_and_minimum_fallback_matches_control():
    y=np.r_[np.arange(94),np.repeat(99.,6)]
    labels,report=models.binary_labels(y,np.ones(100,dtype=int))
    assert labels.sum()==6 and report[0]['requested_top_k']==5
    labels,report=models.binary_labels(np.r_[np.zeros(20),1.],np.ones(21,dtype=int))
    assert labels.sum()==1 and report[0]['minimum_threshold_fallback']
    with pytest.raises(ValueError,match='two classes'):
        models.binary_labels(np.zeros(30),np.ones(30,dtype=int))


def test_preparation_changes_only_binary_target_and_preserves_weights_features():
    frame=toy_frame()
    features=[f'x{i}' for i in range(8)]
    mapping={i:str(i//2) for i in range(1,9)}
    old=original_models.prepare(frame,'BIO-INSIGHT',mapping,features)
    new=models.prepare(frame,'BIO-INSIGHT',mapping,features)
    for key in ('X','group_id','point_weight','group_weight','item_id'):
        np.testing.assert_array_equal(old[key],new[key])
    assert 'clf_top20' not in new['labels']
    assert sum(new['labels'][models.ARM])<sum(old['labels']['clf_top20'])
    assert new['report']['classification_policy']==models.TARGET


def test_original_grid_budget_and_topology_partitions_are_retained(contract,phase4_repo):
    spec.verify_sources(phase4_repo,contract)
    assert spec.counts(contract)==dict(jobs=1020,total_fits=3330,
        jobs_by_stage={'phase2':990,'phase3':30},fits_by_stage={'phase2':2970,'phase3':360})
    assert contract['split_contract']==contract['parent_contract']['split_contract']
    assert contract['grids']==contract['parent_contract']['selection_contract']['grids']
    plan=spec.build_plan(contract)
    assert {j['arm'] for j in plan}=={models.ARM}
    assert all(j['stage']!='phase1' for j in plan)
    assert all(not j['dependencies'] for j in plan if j['stage']=='phase2')


@pytest.mark.parametrize('field,value',[('classification_policy',dict(positive_fraction=.20)),
    ('seeds',[1201]),('statistics',dict(primary_metric='Hit@5')),('fractions',[1.])])
def test_contract_rejects_policy_drift(contract,field,value):
    changed=deepcopy(contract)
    changed[field]=value
    with pytest.raises(ValueError):
        spec.validate_contract(changed)


def outer_contract(contract):
    c=deepcopy(contract)
    c.update(mode='outer',status='frozen_outer_evaluation',selection_contract=deepcopy(contract))
    procedures=[]
    for case in c['cases']:
        for fold in range(5):
            family=models.FAMILIES[fold%3]
            procedures.append(dict(case=case,outer_fold=fold,arm=models.ARM,family=family,
                config=c['grids'][family][0],label='rank_dense',fraction=1.,
                n_features=len(c['split_contract']['cases'][case]['feature_columns']),
                selected_candidate=f'{case}/outer-{fold}/phase3/{family}/{models.ARM}/features-1'))
    c['procedures']=procedures
    return c


def test_outer_plan_contains_all_seeds_and_never_uses_inner_masks(contract):
    c=outer_contract(contract)
    plan=spec.build_plan(c)
    assert spec.counts(c)['total_fits']==50
    assert len(plan)==50
    mapping=c['split_contract']['topology_by_front']
    for job in plan:
        assert not job['dependencies']
        assert job['seed'] in c['seeds']
        assert not ({mapping[str(f)] for f in job['train_front_ids']} & {mapping[str(f)] for f in job['evaluation_front_ids']})


@pytest.mark.parametrize('family',models.FAMILIES)
def test_native_classifiers_ignore_heldout_quality_and_save_scores_first(family,tmp_path,monkeypatch):
    frame=toy_frame()
    frame.to_csv(tmp_path/'data.csv',index=False)
    features=[f'x{i}' for i in range(8)]
    c=dict(split_contract=dict(cases={'BIO-INSIGHT':dict(data_path='data.csv',feature_columns=features)},
                              topology_by_front={str(i):str(i) for i in range(1,9)}))
    p=dict(family=family,arm=models.ARM,config=tiny_params(family),fraction=1.,n_features=8,label='rank_dense')
    job=dict(id='test',case='BIO-INSIGHT',stage='outer',arm=models.ARM,procedure=p,
             train_front_ids=list(range(1,7)),evaluation_front_ids=[7,8],seed=1203,dependencies=[])
    native=outer_execution.load_rows
    events=[]
    def load(root,contract,case,fronts,columns,*,labels=True):
        if set(fronts)=={7,8} and labels:
            assert (destination/'predictions.csv').exists()
        events.append((set(fronts),labels))
        return native(root,contract,case,fronts,columns,labels=labels)
    monkeypatch.setattr(outer_execution,'load_rows',load)
    outputs=[]
    for i in range(2):
        destination=tmp_path/f'run-{i}'/'jobs/test/attempt-001'
        destination.mkdir(parents=True)
        report=outer_execution.execute_job(tmp_path,c,job,{},tmp_path,destination,1)
        assert report['classification_policy']==models.TARGET
        assert report['training']['classification_policy']==models.TARGET
        assert report['fitting']['classification_policy']==models.TARGET
        outputs.append(pd.read_csv(destination/'predictions.csv'))
        frame.loc[frame['front_id'].isin([7,8]),'AUPR']=1-frame.loc[frame['front_id'].isin([7,8]),'AUPR']
        frame.to_csv(tmp_path/'data.csv',index=False)
    pd.testing.assert_frame_equal(*outputs)
    assert events[:3]==[(set(range(1,7)),True),({7,8},False),({7,8},True)]


@pytest.mark.parametrize('family',models.FAMILIES)
def test_relearned_mask_matches_training_only_recursive_path(family):
    data=models.prepare(toy_frame(),'BIO-INSIGHT',{i:str(i) for i in range(1,9)},[f'x{i}' for i in range(8)])
    p=dict(family=family,arm=models.ARM,config=tiny_params(family),fraction=.25,n_features=2)
    features,paths,_=outer_execution.learn_mask(data,p,threads=1)
    expected,_=models.feature_path(data,family,models.ARM,p['config'],threads=1)
    assert features==expected[-1]['features']
    assert len(paths)==3


def test_central_summary_averages_seed_metrics_and_preserves_both_targets(contract,tmp_path):
    c=outer_contract(contract)
    long=synthetic_metrics(c['parent_contract'])
    new=long.loc[long['method']=='clf_top20'].copy()
    new['method']=models.ARM
    new['Regret@5']*=1.1
    new['Hit@5']*=.9
    tables=summary.central_tables(pd.concat([long,new],ignore_index=True),c)
    central=tables['central_summary.csv']
    for case in c['cases']:
        rows=central.loc[(central['case']==case)&(central['aggregation']=='topology_macro')].set_index('method')
        assert rows.loc[models.ARM,'Regret@5']==pytest.approx(1.1*rows.loc['clf_top20','Regret@5'])
    assert len(tables['paired_differences.csv'])==16
    from scripts.experiments.amiga_exp.top5_classification.plots import make_plots
    make_plots(tables,tmp_path/'plots')
    assert len(list((tmp_path/'plots').rglob('*.pdf')))==4


@pytest.mark.parametrize('mutation',['missing_seed','wrong_fold','duplicate_candidate_method'])
def test_central_summary_rejects_incomplete_or_misassigned_evidence(contract,mutation):
    c=outer_contract(contract)
    long=synthetic_metrics(c['parent_contract'])
    new=long.loc[long['method']=='clf_top20'].copy()
    new['method']=models.ARM
    if mutation=='missing_seed':
        new=new.iloc[1:]
    elif mutation=='wrong_fold':
        new.iloc[0,new.columns.get_loc('outer_fold')]=4-new.iloc[0]['outer_fold']
    else:
        new=pd.concat([new,new.iloc[:1]],ignore_index=True)
    with pytest.raises(ValueError):
        summary.central_tables(pd.concat([long,new],ignore_index=True),c)


def test_complete_summary_recomputes_saved_scores_and_checks_all_jobs(contract,phase4_repo,tmp_path,monkeypatch):
    from scripts.experiments.amiga_exp.grouped_validation.metrics import front_metrics
    from scripts.experiments.amiga_exp.top5_classification import runner
    c=outer_contract(contract)
    native_load=outer_execution.load_rows
    def load(*args,**kwargs):
        frame=native_load(*args,**kwargs)
        if 'AUPR' in frame:
            frame['AUPR']=.1+frame.groupby('front_id')['item_id'].rank(pct=True)*.4
        return frame
    monkeypatch.setattr(outer_execution,'load_rows',load)
    jobs=spec.build_plan(c)
    run,output=tmp_path/'outer-run',tmp_path/'summary'
    run.mkdir()
    write_json(run/'state.json',dict(status='complete'))
    write_json(run/'contract.json',c)
    for job in jobs:
        directory=run/'jobs'/job['id']/'attempt-001'
        directory.mkdir(parents=True)
        frame=outer_execution.load_rows(phase4_repo,c,job['case'],job['evaluation_front_ids'],[])
        scores=np.arange(len(frame),dtype=float)
        frame[['front_id','item_id']].assign(score=scores).to_csv(directory/'predictions.csv',index=False)
        front_metrics(frame,scores).to_csv(directory/'metrics.csv',index=False)
        fitting=dict(requested_iterations=3000,actual_iterations=3000,early_stopping=False,classification_policy=models.TARGET)
        features=c['split_contract']['cases'][job['case']]['feature_columns']
        info=dict(procedure=job['procedure'],seed=job['seed'],feature_columns=features,mask_dependency=None,
                  train_front_ids=job['train_front_ids'],evaluation_front_ids=job['evaluation_front_ids'],fitting=fitting)
        write_json(directory/'model.json',info)
        write_json(directory/'result.json',dict(status='complete',job=job,classification_policy=models.TARGET,
            training=dict(selected_front_ids=sorted(job['train_front_ids']),classification_policy=models.TARGET),
            fitting=fitting,seconds=1.,peak_rss_mib=1.,
            artifacts={n:sha256(directory/n) for n in ('predictions.csv','metrics.csv','model.json')}))
    monkeypatch.setattr(runner,'read_run',lambda path:(dict(repo_root=str(phase4_repo)),c,jobs))
    report=summary.summarize(run,output,figures=False)
    assert report['audit']['completed_jobs']==50 and report['audit']['completed_fits']==50
    assert summary.verify_summary(run,output)['status']=='complete'
    directory=run/'jobs'/jobs[0]['id']/'attempt-001'
    metrics=pd.read_csv(directory/'metrics.csv')
    metrics['Regret@5']+=.1
    metrics.to_csv(directory/'metrics.csv',index=False)
    result=json.loads((directory/'result.json').read_text())
    result['artifacts']['metrics.csv']=sha256(directory/'metrics.csv')
    write_json(directory/'result.json',result)
    with pytest.raises(ValueError,match='Metrics differ'):
        summary.summarize(run,tmp_path/'bad-summary',figures=False)


def test_resume_preserves_frozen_plan_and_worker_layout(contract,phase4_repo,tmp_path):
    from scripts.experiments.amiga_exp.top5_classification import runner
    path,output=tmp_path/'contract.json',tmp_path/'run'
    write_json(path,contract)
    a=runner.run_selection(path,output,root=phase4_repo,jobs=1,threads=1,dry_run=True)
    b=runner.run_selection(path,output,root=phase4_repo,jobs=1,threads=1,dry_run=True,resume=True)
    assert a['planned_jobs']==b['planned_jobs']==1020
    assert runner.read_run(output)[2]==spec.build_plan(contract)
    with pytest.raises(ValueError,match='worker/thread'):
        runner.run_selection(path,output,root=phase4_repo,jobs=2,threads=1,dry_run=True,resume=True)


def test_inner_summary_and_outer_freeze_reconstruct_choices_from_audited_predictions(contract,phase4_repo,monkeypatch):
    from scripts.experiments.amiga_exp.grouped_validation.metrics import front_metrics
    from scripts.experiments.amiga_exp.top5_classification import runner,selection_summary
    jobs=[]
    for job in spec.build_plan(contract):
        if job['stage']=='phase3' or job['config']['id']=='cfg-000':
            job=deepcopy(job)
            if job['stage']=='phase3':
                job['dependencies']=job['dependencies'][:1]
            jobs.append(job)
    run=phase4_repo/'experiments/top5-test/selection-run'
    output=phase4_repo/'experiments/top5-test/selection-summary'
    run.mkdir(parents=True)
    write_json(run/'contract.json',contract)
    write_json(run/'plan.json',jobs)
    write_json(run/'manifest.json',{})
    write_json(run/'state.json',dict(status='complete'))
    native=outer_execution.load_rows
    def load(*args,**kwargs):
        frame=native(*args,**kwargs)
        if 'AUPR' in frame:
            frame['AUPR']=.1+frame.groupby('front_id')['item_id'].rank(pct=True)*.4
        return frame
    monkeypatch.setattr(outer_execution,'load_rows',load)
    fitting=dict(requested_iterations=3000,actual_iterations=3000,early_stopping=False,classification_policy=models.TARGET)
    for job in jobs:
        directory=run/'jobs'/job['id']/'attempt-001'
        directory.mkdir(parents=True)
        outer=contract['split_contract']['outer_folds'][job['outer_fold']]
        fractions=contract['fractions'] if job['stage']=='phase3' else [1.]
        columns=contract['split_contract']['cases'][job['case']]['feature_columns']
        metrics,predicted,fit_reports,paths=[],[],[],[]
        for inner in outer['inner_folds']:
            frame=load(phase4_repo,contract,job['case'],inner['validation_front_ids'],[])
            scores=np.arange(len(frame),dtype=float)
            training=dict(selected_front_ids=sorted(inner['train_front_ids']),classification_policy=models.TARGET)
            path=[]
            for fraction in fractions:
                n_features=int(np.ceil(len(columns)*fraction))
                metrics.append(front_metrics(frame,scores).assign(inner_fold=inner['fold'],fraction=fraction,n_features=n_features))
                predicted.append(frame[['front_id','item_id']].assign(score=scores,inner_fold=inner['fold'],fraction=fraction))
                path.append(dict(fraction=fraction,n_features=n_features,features=columns[:n_features],**fitting))
            fit_reports.append(dict(inner_fold=inner['fold'],training=training,**fitting))
            paths.append(dict(inner_fold=inner['fold'],training=training,path=path))
        pd.concat(metrics,ignore_index=True).to_csv(directory/'metrics.csv',index=False)
        pd.concat(predicted,ignore_index=True).to_csv(directory/'predictions.csv',index=False)
        config=job['config'] or contract['grids'][job['family']][0]
        write_json(directory/'selection.json',dict(family=job['family'],arm=models.ARM,label='rank_dense',config=config,fit_reports=fit_reports))
        names=['metrics.csv','predictions.csv','selection.json']
        if job['stage']=='phase3':
            write_json(directory/'feature_paths.json',paths)
            pd.DataFrame(dict(feature=columns,centered_mean_abs_shap=1.)).to_csv(directory/'feature_importance.csv',index=False)
            names+=['feature_paths.json','feature_importance.csv']
        write_json(directory/'result.json',dict(status='complete',job=job,classification_policy=models.TARGET,
            artifacts={n:sha256(directory/n) for n in names}))
    monkeypatch.setattr(runner,'read_run',lambda path:(dict(repo_root=str(phase4_repo)),contract,jobs))
    monkeypatch.setattr(selection_summary,'read_run',lambda path:(dict(repo_root=str(phase4_repo)),contract,jobs))
    native_counts=spec.counts
    monkeypatch.setattr(spec,'counts',lambda c:dict(jobs=60,total_fits=450) if c['mode']=='selection' else native_counts(c))
    report=selection_summary.summarize(run,output,figures=False)
    assert report['audit']['completed_fits']==450 and report['selected_procedures']==10
    outer=spec.outer_contract(run,output,root=phase4_repo)
    assert len(outer['procedures'])==10
    assert all(p['fraction']==.25 for p in outer['procedures'])
    assert native_counts(outer)['total_fits']==80


@pytest.mark.parametrize('fail_summary',[False,True])
def test_pipeline_reaches_complete_only_after_both_stages_and_final_audit(contract,phase4_repo,tmp_path,monkeypatch,fail_summary):
    from scripts.experiments.amiga_exp.top5_classification import pipeline,selection_summary
    path=tmp_path/'contract.json'
    write_json(path,contract)
    calls=[]
    def run(path,output,**kwargs):
        output.mkdir(parents=True,exist_ok=True)
        calls.append((output.name,kwargs.get('dry_run',False)))
        return dict(status='planned' if kwargs.get('dry_run') else 'complete')
    def inner(run,output,**kwargs):
        output.mkdir()
        return dict(status='complete')
    def final(run,output,**kwargs):
        if fail_summary:
            raise ValueError('Synthetic audit rejection')
        output.mkdir()
        return dict(status='complete',audit=dict(completed_fits=50,predictions_and_metrics_recomputed=True,all_five_seeds_present=True))
    monkeypatch.setattr(pipeline,'REPO',phase4_repo)
    monkeypatch.setattr(pipeline,'run_selection',run)
    monkeypatch.setattr(pipeline,'outer_contract',lambda *a,**kw:outer_contract(contract))
    monkeypatch.setattr(selection_summary,'summarize',inner)
    monkeypatch.setattr(summary,'summarize',final)
    launch,work=tmp_path/'launch',tmp_path/'work'
    if fail_summary:
        with pytest.raises(ValueError,match='Synthetic audit rejection'):
            pipeline.run_pipeline(path,launch,work,jobs=1,threads=1)
        assert json.loads((launch/'state.json').read_text())['status']=='failed'
    else:
        result=pipeline.run_pipeline(path,launch,work,jobs=1,threads=1)
        assert result['status']=='complete' and result['selection_fits']==3330
    assert calls==[('selection-run',True),('selection-run',False),('outer-run',True),('outer-run',False)]
