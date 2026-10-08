"""Audit five-seed held-out scores and retain both classification definitions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiments.amiga_exp.grouped_validation.metrics import METRICS, topology_metrics
from scripts.experiments.amiga_exp.grouped_validation.summary import paired_difference
from scripts.experiments.amiga_exp.grouped_validation.pilot import sha256, write_json
from scripts.experiments.amiga_exp.outer_evaluation.summary import audit_predictions
from .models import ARM, TARGET
from .spec import counts, verify_artifacts, verify_sources


def central_tables(long, c):
    base, rows, fronts, topologies, paired, differences = c['split_contract'], [], [], [], [], []
    if not np.isfinite(long[list(METRICS)].to_numpy()).all():
        raise ValueError('Nonfinite metric evidence')
    for case in sorted(c['cases']):
        methods = {}
        data = long.loc[long['case'] == case]
        expected = set(c['parent_contract']['arms']) | {ARM} | set(c['cases'][case]['baseline_ids'])
        if set(data['method']) != expected:
            raise ValueError('Comparator coverage differs')
        for method in sorted(expected):
            part = data.loc[data['method'] == method]
            learned = method in set(c['parent_contract']['arms']) | {ARM}
            if set(part['front_id']) != set(map(int,base['topology_by_front'])):
                raise ValueError('Front coverage differs')
            if not (part.groupby('front_id').size() == (5 if learned else 1)).all():
                raise ValueError('Seed observation coverage differs')
            if learned:
                if part.duplicated(['front_id','seed']).any() or any(set(g['seed']) != set(c['seeds']) for _,g in part.groupby('front_id')):
                    raise ValueError('Missing or duplicated final seeds')
                for outer in base['outer_folds']:
                    if set(part.loc[part['outer_fold']==outer['fold'],'front_id']) != set(outer['test_front_ids']):
                        raise ValueError('Outer assignment differs')
            average = part.groupby('front_id',as_index=False)[list(METRICS)].mean()
            topo = topology_metrics(average, base['topology_by_front'])
            methods[method] = topo.set_index('topology_id')
            fronts.append(average.assign(case=case,method=method))
            topologies.append(topo.assign(case=case,method=method))
            for scope, table in [('topology_macro',topo),('front_macro',average)]:
                rows.append(dict(case=case,method=method,aggregation=scope,**table[list(METRICS)].mean().to_dict()))
        new = methods[ARM]
        for comparator in c['statistics']['comparisons']:
            other = methods[comparator].reindex(new.index)
            for metric in [c['statistics']['primary_metric'],*c['statistics']['secondary_metrics']]:
                evidence = paired_difference(new[metric],other[metric],seed=c['statistics']['bootstrap_seed'],
                                             samples=c['statistics']['bootstrap_samples'])
                if metric.startswith('Hit@'):
                    evidence['improved'],evidence['worsened'] = evidence['worsened'],evidence['improved']
                paired.append(dict(case=case,reference=ARM,comparator=comparator,metric=metric,**evidence))
                delta = new[metric]-other[metric]
                differences.extend(dict(case=case,reference=ARM,comparator=comparator,metric=metric,
                                        topology_id=t,difference=float(v)) for t,v in delta.items())
    return {'central_summary.csv':pd.DataFrame(rows),'front_metrics.csv':pd.concat(fronts,ignore_index=True),
            'topology_metrics.csv':pd.concat(topologies,ignore_index=True),
            'paired_differences.csv':pd.DataFrame(paired),'topology_differences.csv':pd.DataFrame(differences)}


def summarize(run, output, *, figures=True):
    from .runner import read_run
    from .outer_execution import completed_job,load_rows,selected_features
    run,output=Path(run).resolve(),Path(output).resolve()
    manifest,c,jobs=read_run(run)
    root=Path(manifest['repo_root'])
    verify_sources(root,c)
    if c['mode']!='outer' or json.loads((run/'state.json').read_text())['status']!='complete' or output.exists():
        raise ValueError('A complete outer run and new summary destination are required')
    reference_run,reference_summary=root/c['reference']['run'],root/c['reference']['summary']
    verify_artifacts(reference_run,reference_summary)
    labels={case:load_rows(root,c,case,sorted(map(int,c['split_contract']['topology_by_front'])),[]) for case in c['cases']}
    sources,parts,fits,timing,masks={ },[],[],[],[]
    plan={j['id']:j for j in jobs}
    for job in jobs:
        done=completed_job(run,job)
        if done is None:
            raise ValueError('Incomplete final job')
        report,directory=done
        if report.get('classification_policy')!=TARGET or report['training'].get('classification_policy')!=TARGET:
            raise ValueError('Final classification target provenance differs')
        if report['training']['selected_front_ids']!=sorted(job['train_front_ids']):
            raise ValueError('Training scope differs')
        sources[(directory/'result.json').relative_to(run).as_posix()]=sha256(directory/'result.json')
        timing.append(dict(job_id=job['id'],stage=job['stage'],seconds=report['seconds'],peak_rss_mib=report['peak_rss_mib']))
        if job['stage']=='mask':
            for fit in report['fit_reports']:
                fits.append(dict(job_id=job['id'],stage='mask',fraction=fit['fraction'],
                                 requested_iterations=fit['requested_iterations'],actual_iterations=fit['actual_iterations']))
            continue
        predictions=pd.read_csv(directory/'predictions.csv',float_precision='round_trip')
        metrics=pd.read_csv(directory/'metrics.csv',float_precision='round_trip')
        frame=labels[job['case']].loc[labels[job['case']]['front_id'].isin(job['evaluation_front_ids'])]
        audit_predictions(frame,predictions,metrics,['score'])
        info=json.loads((directory/'model.json').read_text())
        features,dependency=selected_features(c,job,plan,run)
        if (info['procedure']!=job['procedure'] or info['seed']!=job['seed'] or info['feature_columns']!=features
                or info['mask_dependency']!=dependency or info['train_front_ids']!=job['train_front_ids']
                or info['evaluation_front_ids']!=job['evaluation_front_ids'] or info['fitting']!=report['fitting']
                or info['fitting']['classification_policy']!=TARGET or info['fitting']['early_stopping']):
            raise ValueError('Final model metadata differs from the plan')
        fits.append(dict(job_id=job['id'],stage='outer',fraction=job['procedure']['fraction'],
                         requested_iterations=info['fitting']['requested_iterations'],actual_iterations=info['fitting']['actual_iterations']))
        parts.append(metrics.assign(case=job['case'],method=ARM,seed=job['seed'],outer_fold=job['outer_fold'],stage='outer'))
        if job['seed']==c['seeds'][0]:
            masks.extend(dict(case=job['case'],outer_fold=job['outer_fold'],feature=f,selected=f in features)
                         for f in c['split_contract']['cases'][job['case']]['feature_columns'])
    if len(fits)!=counts(c)['total_fits'] or any(x['requested_iterations']!=3000 for x in fits):
        raise ValueError('Fit budget differs')
    previous=pd.read_csv(reference_summary/'metrics_long.csv',float_precision='round_trip')
    long=pd.concat([previous,*parts],ignore_index=True)
    tables=central_tables(long,c)
    tables.update({'metrics_long.csv':long,'fit_audit.csv':pd.DataFrame(fits),'runtimes.csv':pd.DataFrame(timing),
                   'outer_feature_masks.csv':pd.DataFrame(masks)})
    output.mkdir(parents=True)
    for name,table in tables.items():
        table.to_csv(output/name,index=False)
    if figures:
        from .plots import make_plots
        make_plots(tables,output/'plots')
    report=dict(status='complete',evidence_role='exploratory_sensitivity',scope='outer_held_out_topologies',
                contract_sha256=sha256(run/'contract.json'),counts=counts(c),source_result_hashes=sources,
                classification_policy=TARGET,statistics=c['statistics'],reference=c['reference'],
                audit=dict(completed_jobs=len(jobs),completed_fits=len(fits),final_fits=50,
                           actual_iterations=sum(x['actual_iterations'] for x in fits),
                           predictions_and_metrics_recomputed=True,all_five_seeds_present=True),
                limitations=['Reused benchmarks; exploratory sensitivity, not fresh independent confirmation.',
                             'Bootstrap intervals condition on selected procedures, fitted models and shared partitions.'],
                artifacts={p.relative_to(output).as_posix():sha256(p) for p in sorted(output.rglob('*')) if p.is_file()})
    write_json(output/'manifest.json',report)
    return report


def verify_summary(run,output):
    from .runner import read_run
    from .outer_execution import completed_job
    run,output=Path(run).resolve(),Path(output).resolve()
    manifest,c,jobs=read_run(run)
    verify_sources(Path(manifest['repo_root']),c)
    report=verify_artifacts(run,output)
    if report.get('counts')!=counts(c) or report.get('classification_policy')!=TARGET or report.get('statistics')!=c['statistics']:
        raise ValueError('Final summary policy differs')
    actual={}
    for job in jobs:
        done=completed_job(run,job)
        if done is None:
            raise ValueError('Final summary refers to incomplete jobs')
        actual[(done[1]/'result.json').relative_to(run).as_posix()]=sha256(done[1]/'result.json')
    if actual!=report['source_result_hashes'] or report['audit']['completed_fits']!=counts(c)['total_fits'] or not report['audit']['all_five_seeds_present']:
        raise ValueError('Final summary job coverage differs')
    return report
