"""Native model loading, label-free scoring and fixed-snapshot contextual support."""
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.experiments.amiga_exp.grouped_validation.baselines import score_front
from scripts.experiments.amiga_exp.grouped_validation.pilot import sha256, write_json
from scripts.experiments.amiga_exp.real_world_validation import (
    Selector, OBJECTIVE_COLUMNS, OBJECTIVE_DIRECTIONS, REPORTED_SOURCES,
    infer_grn_dir, reconstruct_selected_networks, build_reported_source_support_table,
)
from .execution import completed_job
from .selection import adapter
from .runner import read_run, verify_sources

RULES=('objective__reducenonessentialsinteractions','objective_mean_rank',
       'objective_topsis','objective__metricdistribution','objective_knee')
NAMES={'ranking':'AMIGA','reg_aupr':'Direct AUPR regression','clf_top05':'Classification top-5%',
       'clf_top10':'Classification top-10%','clf_top20':'Classification top-20%',
       RULES[0]:'ReduceNEI',RULES[1]:'Mean rank',RULES[2]:'TOPSIS',RULES[3]:'Metric dist.',RULES[4]:'Knee'}


def load_native(directory,info):
    from catboost import CatBoostClassifier, CatBoostRanker, CatBoostRegressor
    from lightgbm import Booster
    from xgboost import XGBClassifier, XGBRanker, XGBRegressor
    path=Path(directory)/info['model_file']
    if sha256(path)!=info['model_sha256']: raise ValueError('Native model identity differs')
    family,arm=info['procedure']['family'],info['procedure']['arm']
    if family=='LightGBM': return SimpleNamespace(booster_=Booster(model_file=str(path)))
    classification=arm.startswith('clf_')
    cls=((CatBoostRanker if arm=='ranking' else CatBoostClassifier if classification else CatBoostRegressor)
         if family=='CatBoost' else (XGBRanker if arm=='ranking' else XGBClassifier if classification else XGBRegressor))
    model=cls()
    model.load_model(str(path))
    return model


def stable_choice(frame,values):
    scores=np.asarray(values,dtype=float)
    if scores.shape!=(len(frame),) or not np.isfinite(scores).all():
        raise ValueError('Invalid real-front scores')
    order=np.lexsort((frame.item_id.to_numpy(),-scores))
    return int(order[0]),int(np.sum(scores==scores[order[0]]))


def apply(run,case_dir,output):
    run,case_dir,output=map(lambda p:Path(p).resolve(),(run,case_dir,output))
    manifest,c,jobs=read_run(run)
    root=Path(manifest['repo_root'])
    verify_sources(root,c)
    if output.exists(): raise ValueError('Use a new application output directory')
    job=next(j for j in jobs if j['case']=='BIO-INSIGHT' and j['context_id']=='deployment' and j['stage']=='final')
    complete=completed_job(run,job)
    if complete is None: raise ValueError('Deployment selection and fitting are incomplete')
    _,directory=complete
    features=c['original']['split_contract']['cases']['BIO-INSIGHT']['feature_columns']
    source=case_dir/'amiga/data_real.csv'
    evidence_path=case_dir/'validation/amiga_exp_reported/reported_external_tf_target_evidence.csv'
    inputs={p:sha256(p) for p in [source,evidence_path,*sorted(infer_grn_dir(case_dir).glob('GRN_*.csv'))]}
    frame=pd.read_csv(source,usecols=['front_id','item_id',*features]).sort_values(['front_id','item_id']).reset_index(drop=True)
    if frame.front_id.nunique()!=1 or frame.duplicated(['front_id','item_id']).any() or frame.empty:
        raise ValueError('Expected one nonempty real front with unique candidate identifiers')
    output.mkdir(parents=True)
    scores=frame[['front_id','item_id']].copy()
    timing,models=[],{}
    for arm in c['deployment_methods']:
        info=json.loads((directory/f'{arm}-seed-{c["deployment_seed"]}.model.json').read_text())
        start=time.monotonic()
        model=load_native(directory,info)
        load_seconds=time.monotonic()-start
        cols=info['feature_columns']
        library=adapter(arm)
        values=library.scores(model,info['procedure']['family'],arm,frame[cols].to_numpy(),manifest['threads'])
        for repeat in range(5):
            start=time.monotonic()
            again=library.scores(model,info['procedure']['family'],arm,frame[cols].to_numpy(),manifest['threads'])
            np.testing.assert_allclose(values,again,rtol=1e-12,atol=1e-12)
            timing.append(dict(method=arm,repeat=repeat,rows=len(frame),features=len(cols),
                               scoring_seconds=time.monotonic()-start,model_load_seconds=load_seconds))
        scores[arm]=values
        models[arm]=dict(model_sha256=info['model_sha256'],metadata_sha256=sha256(directory/f'{arm}-seed-{c["deployment_seed"]}.model.json'))
    rules,_=score_front(frame,list(OBJECTIVE_COLUMNS),OBJECTIVE_DIRECTIONS)
    for rule in RULES: scores[rule]=rules[rule]
    scores.to_csv(output/'candidate_scores.csv',index=False)
    pd.DataFrame(timing).to_csv(output/'scoring_costs.csv',index=False)
    ranked=frame.copy()
    ranked['score']=scores.ranking
    ranked=ranked.sort_values(['score','item_id'],ascending=[False,True],kind='mergesort')
    ranked['rank_in_front']=np.arange(1,len(ranked)+1)
    ranked.to_csv(output/'ranked_real.csv',index=False)
    selectors=[]
    rows=[]
    for method in [*c['deployment_methods'],*RULES]:
        index,ties=stable_choice(frame,scores[method])
        candidate=int(frame.iloc[index].item_id)
        selectors.append(Selector(method,'learned' if method in models else 'objective',index,candidate,float(scores.iloc[index][method])))
        rows.append(dict(method=method,display_name=NAMES[method],candidate=candidate,selector_score=float(scores.iloc[index][method]),top_score_ties=ties,
                         tie_rule='smallest item_id among tied maximum scores'))
    pd.DataFrame(rows).to_csv(output/'selected_candidates.csv',index=False)
    networks_dir=output/'selected_networks'
    networks_dir.mkdir()
    network_paths=reconstruct_selected_networks(ranked=frame,selectors=selectors,grn_dir=infer_grn_dir(case_dir),networks_dir=networks_dir)
    # Reuse the identical historical resource snapshot for every new selector.
    # Cistrome coverage is restricted to the TFs present in this frozen snapshot.
    evidence=pd.read_csv(evidence_path)
    expected={s['resource'] for s in REPORTED_SOURCES}
    if set(evidence.resource)!=expected or evidence[['source','target','resource']].isna().any().any():
        raise ValueError('Incomplete frozen evidence resources')
    support=build_reported_source_support_table(selectors=selectors,network_paths=network_paths,evidence=evidence)
    support['method']=support.selector_id.map(NAMES)
    support.to_csv(output/'source_support_top1.csv',index=False)
    evidence.to_csv(output/'source_evidence_snapshot.csv',index=False)
    resources=[]
    for resource,part in evidence.groupby('resource'):
        resources.append(dict(resource=resource,pairs=len(part),source_genes=part.source.nunique(),target_genes=part.target.nunique()))
    write_json(output/'resource_coverage.json',resources)
    if any(sha256(p)!=digest for p,digest in inputs.items()):
        raise ValueError('Application inputs changed while scoring or reconstructing networks')
    result=dict(status='complete',quality_labels_used=False,model_selection='three grouped folds across all 87 benchmark topologies; full original grids',
                seed=c['deployment_seed'],models=models,selectors=[r['method'] for r in rows],
                evidence_scope='Fixed existing resource snapshot; incomplete contextual support, not biological accuracy',
                cistrome_scope='Only source TFs in the recorded snapshot; absent pairs are not verified negatives',
                resource_cutoffs=list(REPORTED_SOURCES),inputs_sha256={str(p.relative_to(root)):d for p,d in inputs.items()},
                artifacts={p.relative_to(output).as_posix():sha256(p) for p in sorted(output.rglob('*')) if p.is_file()})
    write_json(output/'manifest.json',result)
    return result
