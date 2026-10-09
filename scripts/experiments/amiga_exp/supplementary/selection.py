"""Full original selection rules, restricted to each explicitly labelled subset."""
import time

import pandas as pd

from scripts.experiments.amiga_exp.grouped_validation.metrics import front_metrics
from scripts.experiments.amiga_exp.grouped_validation.pilot import utc, write_json
from scripts.experiments.amiga_exp.sequential_selection import models as trees
from scripts.experiments.amiga_exp.sequential_selection.execution import load_rows
from scripts.experiments.amiga_exp.sequential_selection.summary import choose, describe
from scripts.experiments.amiga_exp.top5_classification import models as top5
from scripts.experiments.amiga_exp.top10_classification import models as top10


def adapter(arm):
    return top5 if arm=='clf_top05' else top10 if arm=='clf_top10' else trees


def select(root, contract, job, destination, threads):
    scope=next(s for s in contract['contexts'] if s['id']==job['context_id'])
    original=contract['original']
    mapping=original['split_contract']['topology_by_front']
    features=original['split_contract']['cases'][job['case']]['feature_columns']
    data=load_rows(root,original,job['case'],scope['train_front_ids'])
    family=job['family']
    rows, fitting=[],[]
    prepared_cache={}
    configs={}
    candidate_dir=destination/'candidates'
    candidate_dir.mkdir()
    observed_fits=0

    def evaluate(arm,label,params,stage,identifier):
        library=adapter(arm)
        metrics,predictions=[],[]
        for fold in scope['inner_folds']:
            key=(arm,label,fold['fold'])
            if key not in prepared_cache:
                train=data[data.front_id.isin(fold['train_front_ids'])].copy()
                prepared_cache[key]=library.prepare(train,job['case'],{int(k):v for k,v in mapping.items()},features,label)
            prepared=prepared_cache[key]
            valid=data[data.front_id.isin(fold['validation_front_ids'])].copy()

            def score(model,training,fraction):
                nonlocal observed_fits
                values=library.scores(model,family,arm,valid[training['feature_names']].to_numpy(),threads)
                metrics.append(front_metrics(valid,values).assign(inner_fold=fold['fold'],fraction=fraction,n_features=len(training['feature_names'])))
                predictions.append(valid[['front_id','item_id']].assign(score=values,inner_fold=fold['fold'],fraction=fraction))
                observed_fits+=1
                write_json(destination/'progress.json',dict(status='running',updated_at_utc=utc(),
                    completed_fits=observed_fits,planned_fits=job['planned_fits'],
                    candidate=identifier,inner_fold=fold['fold'],fraction=fraction))

            start=time.monotonic()
            if stage=='phase3':
                path,_=library.feature_path(prepared,family,arm,params,threads=threads,callback=score)
                fitting.extend(dict(candidate=identifier,inner_fold=fold['fold'],**p) for p in path)
            else:
                model,report=library.fit(prepared,family,arm,params,threads=threads)
                fitting.append(dict(candidate=identifier,inner_fold=fold['fold'],fit_seconds=time.monotonic()-start,**report))
                score(model,prepared,1.)
        metric=pd.concat(metrics,ignore_index=True)
        pred=pd.concat(predictions,ignore_index=True)
        stem=identifier.replace('/','__')
        metric.to_csv(candidate_dir/f'{stem}.metrics.csv',index=False)
        pred.to_csv(candidate_dir/f'{stem}.predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        for fraction,part in metric.groupby('fraction'):
            if part.front_id.duplicated().any() or set(part.front_id)!=set(scope['train_front_ids']):
                raise ValueError('Inner predictions do not cover the labelled subset exactly once')
            record=dict(candidate=f'{identifier}/features-{fraction:g}',stage=stage,arm=arm,family=family,
                        label=label,config_id=params.get('id','reference'),fraction=float(fraction),
                        n_features=int(part.n_features.iloc[0]),**describe(part,mapping))
            rows.append(record)
            configs[record['candidate']]=dict(params)

    for label in original['labels']:
        evaluate('ranking',label,original['references'][family],'phase1',f'phase1/{family}/{label}')
    label=choose([r for r in rows if r['stage']=='phase1'])['label']
    for arm in job['arms']:
        arm_label=label if arm=='ranking' else 'rank_dense'
        for params in original['grids'][family]:
            evaluate(arm,arm_label,params,'phase2',f"phase2/{family}/{arm}/{params['id']}")
        winner=choose([r for r in rows if r['stage']=='phase2' and r['arm']==arm])
        evaluate(arm,arm_label,configs[winner['candidate']],'phase3',f'phase3/{family}/{arm}')
    winners=[]
    for arm in job['arms']:
        winner=choose([r for r in rows if r['stage']=='phase3' and r['arm']==arm],features=True)
        winners.append(dict(case=job['case'],arm=arm,family=family,label=winner['label'],
                            config=configs[winner['candidate']],fraction=winner['fraction'],
                            n_features=winner['n_features'],candidate=winner['candidate'],
                            mean_regret5=winner['mean_regret5'],mean_regret1=winner['mean_regret1']))
    pd.DataFrame(rows).to_csv(destination/'selection_candidates.csv',index=False)
    write_json(destination/'selected_procedures.json',winners)
    write_json(destination/'fit_reports.json',fitting)
    if observed_fits!=job['planned_fits']:
        raise ValueError('Completed inner fitting budget differs from the complete plan')
    write_json(destination/'progress.json',dict(status='complete',completed_fits=observed_fits,
                                               planned_fits=job['planned_fits'],updated_at_utc=utc()))
    return dict(inner_model_fits=len(fitting),outer_labels_used_for_selection=False,
                train_front_ids=scope['train_front_ids'])
