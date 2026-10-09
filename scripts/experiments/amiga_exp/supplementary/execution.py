"""Checksummed subset selection, final held-out fits and native deployment models."""
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from scripts.experiments.amiga_exp.grouped_validation.metrics import front_metrics
from scripts.experiments.amiga_exp.grouped_validation.pilot import sha256, utc, write_json
from scripts.experiments.amiga_exp.outer_evaluation.execution import load_rows
from scripts.experiments.amiga_exp.sequential_selection.summary import choose
from .selection import adapter, select


def completed_job(run, job):
    successful=[]
    for path in sorted((Path(run)/'jobs'/job['id']).glob('attempt-*/result.json')):
        report=json.loads(path.read_text())
        if report.get('status')!='complete':
            continue
        if report.get('job')!=job or not report.get('artifacts'):
            raise ValueError('Completion identity or inventory differs')
        required={'selected_procedures.json','fit_reports.json'}
        if job['stage']=='selection':
            required.update({'selection_candidates.csv','progress.json'})
        elif job['context_id'].startswith('learning/'):
            required.add('metrics.csv')
            for arm in job['arms']:
                for seed in range(1201,1206):
                    required.update({f'{arm}-seed-{seed}.predictions.csv.gz',
                                     f'{arm}-seed-{seed}.model.json'})
        else:
            for arm in job['arms']:
                required.add(f'{arm}-seed-1201.model.json')
        if not required.issubset(report['artifacts']):
            raise ValueError('Completion inventory omits required scientific artifacts')
        for name,digest in report['artifacts'].items():
            p=Path(name)
            if p.is_absolute() or '..' in p.parts or sha256(path.parent/p)!=digest:
                raise ValueError(f'Completed artifact changed: {path.parent/p}')
        if job['stage']=='final' and job['context_id']=='deployment':
            for arm in job['arms']:
                info=json.loads((path.parent/f'{arm}-seed-1201.model.json').read_text())
                if report['artifacts'].get(info['model_file'])!=info['model_sha256']:
                    raise ValueError('Completion inventory omits the native deployment model')
        successful.append((report,path.parent))
    if len(successful)>1:
        raise ValueError('Multiple successful attempts')
    return successful[0] if successful else None


def learn_mask(prepared, procedure, library, threads):
    import math
    original=list(prepared['feature_names'])
    current=list(range(len(original)))
    stop=(1.,.75,.5,.25).index(procedure['fraction'])
    reports=[]
    for child in (.75,.5,.25)[:stop]:
        data=dict(prepared,X=prepared['X'][:,current],feature_names=[original[i] for i in current])
        model,report=library.fit(data,procedure['family'],procedure['arm'],procedure['config'],seed=1101,threads=threads)
        importance,_=library.feature_importance(model,procedure['family'],data,threads=threads)
        retained=set(importance.sort_values(['centered_mean_abs_shap','feature'],ascending=[False,True]).feature.iloc[:math.ceil(len(original)*child)])
        current=[i for i in current if original[i] in retained]
        reports.append(report)
    if len(current)!=procedure['n_features']:
        raise ValueError('Relearned mask count differs')
    return [original[i] for i in current],reports


def save_model(model,family,prefix):
    path=Path(str(prefix)+({'CatBoost':'.cbm','LightGBM':'.txt','XGBoost':'.json'}[family]))
    if family=='LightGBM': model.booster_.save_model(str(path))
    else: model.save_model(str(path))
    return path


def execute_job(root,c,job,plan,run,destination,threads):
    start=time.monotonic()
    if job['stage']=='selection':
        extra=select(root,c,job,destination,threads)
    else:
        scope=next(s for s in c['contexts'] if s['id']==job['context_id'])
        candidates=[]
        dependencies={}
        for identifier in job['dependencies']:
            completed=completed_job(run,plan[identifier])
            if completed is None: raise ValueError('Missing family-selection dependency')
            _,directory=completed
            candidates.extend(json.loads((directory/'selected_procedures.json').read_text()))
            dependencies[identifier]=sha256(directory/'result.json')
        selected=[choose([p for p in candidates if p['arm']==arm],features=True) for arm in job['arms']]
        write_json(destination/'selected_procedures.json',selected)
        predictions,metrics,reports=[],[],[]
        original=c['original']
        full=original['split_contract']['cases'][job['case']]['feature_columns']
        mapping={int(k):v for k,v in original['split_contract']['topology_by_front'].items()}
        train=load_rows(root,original,job['case'],scope['train_front_ids'],full)
        seeds=c['final_seeds'] if scope['kind']=='learning' else [c['deployment_seed']]
        for procedure in selected:
            arm=procedure['arm']
            library=adapter(arm)
            prepared=library.prepare(train,job['case'],mapping,full,procedure['label'])
            features,mask_reports=learn_mask(prepared,procedure,library,threads)
            index=[full.index(f) for f in features]
            reduced=dict(prepared,X=prepared['X'][:,index],feature_names=features)
            for seed in seeds:
                before=time.monotonic()
                model,fitting=library.fit(reduced,procedure['family'],arm,procedure['config'],seed=seed,threads=threads)
                fit_seconds=time.monotonic()-before
                info=dict(procedure=procedure,feature_columns=features,seed=seed,
                          train_front_ids=scope['train_front_ids'],mask_reports=mask_reports,
                          fitting=fitting,fit_seconds=fit_seconds,training=prepared['report'])
                if scope['kind']=='learning':
                    evaluation=load_rows(root,original,job['case'],scope['test_front_ids'],features,labels=False)
                    before=time.monotonic()
                    score=library.scores(model,procedure['family'],arm,evaluation[features].to_numpy(),threads)
                    info['scoring_seconds']=time.monotonic()-before
                    prediction=evaluation[['front_id','item_id']].assign(method=arm,seed=seed,score=score)
                    prediction.to_csv(destination/f'{arm}-seed-{seed}.predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
                    labels=load_rows(root,original,job['case'],scope['test_front_ids'],[],labels=True)
                    if not labels[['front_id','item_id']].equals(evaluation[['front_id','item_id']]):
                        raise ValueError('Prediction identifiers and quality labels differ')
                    metrics.append(front_metrics(labels,score).assign(method=arm,seed=seed))
                else:
                    path=save_model(model,procedure['family'],destination/arm)
                    info['model_file']=path.name
                    info['model_sha256']=sha256(path)
                write_json(destination/f'{arm}-seed-{seed}.model.json',info)
                reports.append(dict(method=arm,seed=seed,family=procedure['family'],fit_seconds=fit_seconds,
                                    scoring_seconds=info.get('scoring_seconds'),**fitting))
        if metrics: pd.concat(metrics,ignore_index=True).to_csv(destination/'metrics.csv',index=False)
        write_json(destination/'fit_reports.json',reports)
        extra=dict(dependency_result_hashes=dependencies,scope=scope['kind'],training_size=scope['training_size'])
    artifacts={p.relative_to(destination).as_posix():sha256(p) for p in sorted(destination.rglob('*')) if p.is_file() and p.name!='worker.log'}
    report=dict(status='complete',job=job,ended_at_utc=utc(),seconds=time.monotonic()-start,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                artifacts=artifacts,**extra)
    write_json(destination/'result.json',report)
    return report
