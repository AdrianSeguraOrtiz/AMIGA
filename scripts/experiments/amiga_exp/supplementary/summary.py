"""Recompute subset predictions and describe label-scarcity without new selection."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiments.amiga_exp.grouped_validation.metrics import METRICS, front_metrics, topology_metrics
from scripts.experiments.amiga_exp.grouped_validation.pilot import sha256, write_json
from scripts.experiments.amiga_exp.outer_evaluation.execution import load_rows
from scripts.experiments.amiga_exp.reporting.supervised import verify_manifest
from .execution import completed_job
from .runner import read_run, verify_sources


def aggregate_learning(long,mapping):
    # First average final estimator seeds, then the three nested subset repetitions.
    keys=['case','method','training_size','outer_fold','subset_seed','front_id']
    by_subset=long.groupby(keys,dropna=False)[list(METRICS)].mean().reset_index()
    fronts=by_subset.groupby(['case','method','training_size','front_id'])[list(METRICS)].mean().reset_index()
    rows=[]
    for (case,method,size),part in fronts.groupby(['case','method','training_size']):
        topo=topology_metrics(part,mapping)
        if len(topo)!=87 or len(part)!=104:
            raise ValueError('Every learning size must cover all 87 evaluation topologies and 104 fronts')
        for metric in METRICS:
            values=topo[metric].to_numpy()
            rng=np.random.default_rng(1401)
            resampled=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(axis=1)
            low,high=np.quantile(resampled,[.025,.975])
            rows.append(dict(case=case,method=method,training_size=size,metric=metric,
                             mean=float(values.mean()),ci_low=float(low),ci_high=float(high),
                             n_topologies=len(values),n_fronts=len(part)))
    return pd.DataFrame(rows),fronts,by_subset


def summarize(run,output):
    run,output=Path(run).resolve(),Path(output).resolve()
    manifest,c,jobs=read_run(run)
    root=Path(manifest['repo_root'])
    verify_sources(root,c)
    if json.loads((run/'state.json').read_text())['status']!='complete' or output.exists():
        raise ValueError('Require a complete run and a new summary destination')
    rows,times,selections,sources=[],[],[],{}
    for job in jobs:
        result=completed_job(run,job)
        if result is None: raise ValueError('Incomplete supplementary job')
        report,directory=result
        sources[job['id']]=sha256(directory/'result.json')
        times.append(dict(job_id=job['id'],case=job['case'],stage=job['stage'],
                          seconds=report['seconds'],peak_rss_mib=report['peak_rss_mib']))
        if job['stage']!='final': continue
        scope=next(x for x in c['contexts'] if x['id']==job['context_id'])
        procedures=json.loads((directory/'selected_procedures.json').read_text())
        selections.extend(dict(context_id=scope['id'],**p) for p in procedures)
        if scope['kind']!='learning': continue
        recorded=pd.read_csv(directory/'metrics.csv')
        labels=load_rows(root,c['original'],job['case'],scope['test_front_ids'],[],labels=True)
        for arm in job['arms']:
            for seed in c['final_seeds']:
                pred=pd.read_csv(directory/f'{arm}-seed-{seed}.predictions.csv.gz')
                if not labels[['front_id','item_id']].equals(pred[['front_id','item_id']]):
                    raise ValueError('Saved held-out candidate coverage differs')
                recalculated=front_metrics(labels,pred.score)
                previous=recorded[(recorded.method==arm)&(recorded.seed==seed)].sort_values('front_id')
                np.testing.assert_allclose(recalculated[list(METRICS)],previous[list(METRICS)],rtol=1e-12,atol=1e-12)
                rows.append(recalculated.assign(case=job['case'],method=arm,seed=seed,
                                                training_size=str(scope['training_size']),
                                                outer_fold=scope['outer_fold'],subset_seed=scope['subset_seed']))
    verify_manifest(root/c['full_endpoint_summary']/'manifest.json')
    full=pd.read_csv(root/c['full_endpoint_summary']/'metrics_long.csv')
    full=full[full.method.isin(c['learning_methods'])].copy()
    if len(full)!=2*2*104*5 or full.duplicated(['case','method','front_id','seed']).any():
        raise ValueError('Full-size endpoint must contain every final seed and front')
    full['training_size']='full'
    full['subset_seed']=np.nan
    rows.append(full)
    long=pd.concat(rows,ignore_index=True)
    curves,fronts,subsets=aggregate_learning(long,c['original']['split_contract']['topology_by_front'])
    output.mkdir(parents=True)
    long.to_csv(output/'learning_metrics_long.csv',index=False)
    curves.to_csv(output/'learning_curves.csv',index=False)
    fronts.to_csv(output/'learning_front_metrics.csv',index=False)
    subsets.to_csv(output/'learning_subset_metrics.csv',index=False)
    pd.DataFrame(times).to_csv(output/'runtimes.csv',index=False)
    write_json(output/'selected_procedures.json',selections)
    plot_curves(curves,output)
    result=dict(status='complete',workflow='supplementary_evaluation',
                contract_sha256=sha256(run/'contract.json'),sources=sources,
                scope=c['learning_scope'],aggregation=c['aggregation'],intervals=c['intervals'],
                final_seeds=c['final_seeds'],deployment_seed=c['deployment_seed'],
                original_full_endpoint_reused=True,external_confirmation=False,
                artifacts={p.relative_to(output).as_posix():sha256(p) for p in sorted(output.rglob('*')) if p.is_file()})
    write_json(output/'manifest.json',result)
    return result


def plot_curves(curves,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'ranking':'#2e9d78','reg_aupr':'#7561b4'}
    names={'ranking':'AMIGA','reg_aupr':'Direct AUPR regression'}
    for case,part in curves.groupby('case'):
        fig,axes=plt.subplots(1,2,figsize=(10,4.2))
        for ax,metric in zip(axes,('Regret@5','Hit@5')):
            for method in colors:
                data=part[(part.metric==metric)&(part.method==method)].set_index('training_size').loc[['10','20','40','full']]
                x=np.array([10,20,40,69.6])
                ax.plot(x,data['mean'].to_numpy(),marker='o',color=colors[method],label=names[method],linewidth=2)
                ax.fill_between(x,data.ci_low.to_numpy(),data.ci_high.to_numpy(),color=colors[method],alpha=.15)
            ax.set_xticks([10,20,40,69.6],['10','20','40','All (69–70)'])
            ax.set_xlabel('Labelled training topologies')
            ax.set_ylabel(metric)
            ax.grid(alpha=.2)
        axes[0].legend(frameon=False)
        fig.suptitle(case+' — full selection repeated at each labelled training size')
        fig.text(.5,.02,'Five outer folds; three nested subsets at 10/20/40; five final seeds. Bands: conditional topology bootstrap 95% intervals.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.05,1,.94))
        for suffix in ('pdf','png'): fig.savefig(output/f'{case}-learning-curves.{suffix}',dpi=180,bbox_inches='tight')
        plt.close(fig)
