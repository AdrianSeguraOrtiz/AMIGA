"""Execute selection jobs without accessing outer held-out outcomes."""
from __future__ import annotations

import json
from pathlib import Path
import resource
import time

import pandas as pd

from scripts.experiments.amiga_exp.grouped_validation.metrics import front_metrics, select_configuration
from scripts.experiments.amiga_exp.grouped_validation.pilot import sha256, utc, write_json
from . import models


def completed_job(run, job):
    successful=[]
    expected=({'pilot.json'} if job['stage']=='pilot' else
              {'metrics.csv','predictions.csv','selection.json'} |
              ({'feature_importance.csv','feature_paths.json'} if job['stage']=='phase3' else set()))
    for path in sorted((Path(run)/'jobs'/job['id']).glob('attempt-*/result.json')):
        report=json.loads(path.read_text())
        if report.get('status')!='complete':
            continue
        if report.get('job')!=job or set(report.get('artifacts',{}))!=expected:
            raise ValueError(f'Completion identity/inventory differs: {path}')
        for name,digest in report['artifacts'].items():
            if not (path.parent/name).is_file() or sha256(path.parent/name)!=digest:
                raise ValueError(f'Completed artifact changed: {path.parent/name}')
        successful.append((report,path.parent))
    if len(successful)>1:
        raise ValueError('Multiple successful attempts')
    return successful[0] if successful else None


def load_rows(root,c,case,front_ids):
    info=c['split_contract']['cases'][case]
    columns=['front_id','item_id','AUPR',*info['feature_columns']]
    parts=[chunk.loc[chunk['front_id'].isin(front_ids)] for chunk in
           pd.read_csv(Path(root)/info['data_path'],usecols=columns,chunksize=10000)]
    result=pd.concat(parts,ignore_index=True).sort_values(['front_id','item_id']).reset_index(drop=True)
    if set(result['front_id'])!=set(front_ids) or result.duplicated(['front_id','item_id']).any():
        raise ValueError('Input front/candidate coverage differs')
    return result


def selected_dependency(run,c,job,plan):
    candidates={}
    directories={}
    for name in job['dependencies']:
        upstream=plan[name]
        complete=completed_job(run,upstream)
        if complete is None:
            raise ValueError(f'Incomplete dependency: {name}')
        _,directory=complete
        metrics=pd.read_csv(directory/'metrics.csv')
        outer=c['split_contract']['outer_folds'][job['outer_fold']]
        if set(metrics['front_id'])!=set(outer['train_front_ids']):
            raise ValueError('Selection dependencies must cover the outer training complement')
        candidates[name]=metrics
        directories[name]=directory
    winner,evidence=select_configuration(candidates,c['split_contract']['topology_by_front'])
    chosen=json.loads((directories[winner]/'selection.json').read_text())
    return chosen,dict(selected_job=winner,candidates=evidence,
                       dependency_result_hashes={name:sha256(directory/'result.json')
                                                 for name,directory in directories.items()})


def execute_job(root,c,job,plan,run,destination,threads):
    started=time.monotonic()
    report=dict(job=job,status='running',started_at_utc=utc(), classification_policy=dict(models.TARGET))
    mapping={int(k):v for k,v in c['split_contract']['topology_by_front'].items()}
    features=c['split_contract']['cases'][job['case']]['feature_columns']
    outer=c['split_contract']['outer_folds'][job['outer_fold']]
    family,arm=job['family'],job['arm']
    if job['stage']=='pilot':
        frame=load_rows(root,c,job['case'],job['train_front_ids'])
        prepared=models.prepare(frame,job['case'],mapping,features,job['label'])
        paths,_=models.feature_path(prepared,family,arm,job['config'],threads=threads)
        # Shapes/additivity are covered by synthetic tests; no evaluation metrics.
        write_json(destination/'pilot.json',dict(paths=paths,training=prepared['report'],
                   external_predictions_computed=False,quality_metrics_computed=False))
        names=['pilot.json']
    else:
        config=job['config']
        label=job['label'] or 'rank_dense'
        evidence=None
        if job['dependencies']:
            chosen,evidence=selected_dependency(run,c,job,plan)
            label=chosen['label']
            if job['stage']=='phase3':
                config=chosen['config']
        if config is None:
            raise ValueError('Missing selected configuration')
        # Load only the current outer-training complement, never its held-out rows.
        frame=load_rows(root,c,job['case'],outer['train_front_ids'])
        metrics,predictions,paths,importances,fit_reports=[],[],[],[],[]
        for inner in outer['inner_folds']:
            train=frame.loc[frame['front_id'].isin(inner['train_front_ids'])].copy()
            valid=frame.loc[frame['front_id'].isin(inner['validation_front_ids'])].copy()
            if {mapping[f] for f in train['front_id']} & {mapping[f] for f in valid['front_id']}:
                raise ValueError('A topology crosses the inner split')
            prepared=models.prepare(train,job['case'],mapping,features,label)

            def evaluate(model,data,fraction):
                score=models.scores(model,family,arm,valid[data['feature_names']].to_numpy(dtype=float),threads)
                metrics.append(front_metrics(valid,score).assign(inner_fold=inner['fold'],
                                                                fraction=fraction,n_features=len(data['feature_names'])))
                predictions.append(valid[['front_id','item_id']].assign(score=score,
                                         inner_fold=inner['fold'],fraction=fraction))

            if job['stage']=='phase3':
                path,importance=models.feature_path(prepared,family,arm,config,threads=threads,callback=evaluate)
                paths.append(dict(inner_fold=inner['fold'],path=path,training=prepared['report']))
                importances.append(importance.assign(inner_fold=inner['fold']))
            else:
                before=time.monotonic()
                model,fit_report=models.fit(prepared,family,arm,config,threads=threads)
                fit_reports.append(dict(inner_fold=inner['fold'],fit_seconds=time.monotonic()-before,
                                        training=prepared['report'],**fit_report))
                evaluate(model,prepared,1.0)
        pd.concat(metrics,ignore_index=True).to_csv(destination/'metrics.csv',index=False)
        pd.concat(predictions,ignore_index=True).to_csv(destination/'predictions.csv',index=False)
        write_json(destination/'selection.json',dict(family=family,arm=arm,label=label,
                   config=config,upstream_selection=evidence,fit_reports=fit_reports,
                   evaluation_scope='inner_validation_for_selection_only'))
        names=['metrics.csv','predictions.csv','selection.json']
        if job['stage']=='phase3':
            pd.concat(importances,ignore_index=True).to_csv(destination/'feature_importance.csv',index=False)
            write_json(destination/'feature_paths.json',paths)
            names+=['feature_importance.csv','feature_paths.json']
    report.update(status='complete',ended_at_utc=utc(),seconds=time.monotonic()-started,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  artifacts={name:sha256(destination/name) for name in names},
                  outer_predictions_computed=False)
    write_json(destination/'result.json',report)
    return report
