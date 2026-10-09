"""Isolated empirical feature-preparation measurements and recorded fitting costs."""
import argparse
import json
from pathlib import Path
import random
import resource
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd

from scripts.experiments.amiga_exp.grouped_validation.pilot import sha256, write_json
from scripts.experiments.amiga_exp.sequential_selection.resources import layout, worker_command, worker_environment
from scripts.experiments.amiga_exp.real_world_validation import infer_grn_dir

CASE='experiments/BIO-INSIGHT/real-world/tcga_brca'


def measure(root,size,repeat,output):
    from amiga.core.main import extract_expression_features,extract_grnet_features
    from amiga.utils import row_weights_from_front,weighted_confidence
    case=Path(root)/CASE
    np.random.seed(1501)
    random.seed(1501)
    start=time.monotonic()
    expression_path=case/'data/tcga_brca_primary_tumor_log2tpm_top500_variable.csv'
    expression_digest=sha256(expression_path)
    expression=pd.read_csv(expression_path,index_col=0).iloc[:size]
    io_seconds=time.monotonic()-start
    before=time.monotonic()
    expression_features=extract_expression_features(expression).metrics
    expression_seconds=time.monotonic()-before
    # Candidate 1 is fixed before inspecting any ranking or biological support.
    front=pd.read_csv(case/'amiga/front_real.csv').sort_values('item_id').iloc[0]
    weights=row_weights_from_front(front)
    grn=infer_grn_dir(case)
    genes=set(expression.index.astype(str))
    source_hashes={str(expression_path):expression_digest}
    front_path=case/'amiga/front_real.csv'
    source_hashes[str(front_path)]=sha256(front_path)
    with tempfile.TemporaryDirectory(prefix='amiga-cost-') as temporary:
        specs=[]
        preparation=time.monotonic()
        for name,weight in weights.items():
            path=grn/name
            source_hashes[str(path)]=sha256(path)
            edges=pd.read_csv(path,header=None,names=['Source','Target','Confidence'])
            selected=edges[edges.Source.astype(str).isin(genes)&edges.Target.astype(str).isin(genes)]
            target=Path(temporary)/name
            selected.to_csv(target,index=False,header=False)
            specs.append(f'{weight}*{target}')
        subset_preparation_seconds=time.monotonic()-preparation
        before=time.monotonic()
        network=weighted_confidence(specs)
        if not len(network) or len(expression)!=size:
            raise ValueError('Feature benchmark requires the requested gene set and a nonempty consensus')
        consensus_seconds=time.monotonic()-before
        before=time.monotonic()
        network_features=extract_grnet_features(network).metrics
        network_seconds=time.monotonic()-before
    if any(sha256(Path(p))!=digest for p,digest in source_hashes.items()):
        raise ValueError('Feature benchmark inputs changed while measuring')
    result=dict(size=size,repeat=repeat,genes=len(expression),samples=expression.shape[1],
                candidate=int(front.item_id),network_edges=len(network),
                expression_read_seconds=io_seconds,expression_feature_seconds=expression_seconds,
                restricted_input_preparation_seconds=subset_preparation_seconds,
                consensus_seconds=consensus_seconds,network_feature_seconds=network_seconds,
                expression_feature_count=len(expression_features),network_feature_count=len(network_features),
                process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                total_seconds=time.monotonic()-start,inputs_sha256=source_hashes)
    write_json(output,result)


def profile(root,output,threads=4):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    resources=layout(1,threads)
    results=[]
    for size in (100,250,500):
        for repeat in range(3):
            path=output/f'features-{size}-repeat-{repeat}.json'
            command=worker_command('scripts.experiments.amiga_exp.supplementary.costs',
                ['--root',str(root),'--size',size,'--repeat',repeat,'--output',str(path)],resources['cpu_sets'][0])
            with (output/f'features-{size}-repeat-{repeat}.log').open('w') as stream:
                subprocess.run(command,cwd=root,env=worker_environment(threads),stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=7200)
            result=json.loads(path.read_text())
            results.append({k:v for k,v in result.items() if k!='inputs_sha256'})
    pd.DataFrame(results).to_csv(output/'feature_costs.csv',index=False)
    metadata=dict(status='complete',resources=resources,scope='One prespecified candidate per gene size, three isolated-process repetitions',
        extrapolation='A full front needs expression features once plus consensus/network features for every candidate; full-front time is not measured here',
        excluded_work='Evolutionary front generation and upstream network inference',
        memory='Absolute per-process maximum RSS including loading and restricted-input construction; not incremental stage memory',
        warm_cache='Repeated reads may use the operating-system cache',
        artifacts={p.relative_to(output).as_posix():sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    write_json(output/'manifest.json',metadata)
    return metadata


def summarize_fitting_costs(root,output):
    # Completed executions retain their original complete result sets.
    roots=['experiments/sequential-selection/runs/full-001',
           'experiments/outer-evaluation/runs/evaluation-001',
           'experiments/top5-classification/full-001/selection-run',
           'experiments/top5-classification/full-001/outer-run',
           'experiments/top10-classification/full-001/selection-run',
           'experiments/top10-classification/full-001/outer-run']
    methods={'ranking','reg_aupr','clf_top05','clf_top10','clf_top20'}
    rows=[]
    for relative in roots:
        run=Path(root)/relative
        plan=json.loads((run/'plan.json').read_text())
        for job in plan:
            arm=job.get('arm')
            if arm not in methods and job['stage'] not in ('baselines',): continue
            results=[]
            for path in (run/'jobs'/job['id']).glob('attempt-*/result.json'):
                report=json.loads(path.read_text())
                if report.get('status')=='complete': results.append((path,report))
            if len(results)!=1: raise ValueError('Need exactly one complete recorded timing per job')
            path,report=results[0]
            rows.append(dict(run=relative,job_id=job['id'],case=job['case'],method=arm or 'objective_selectors',
                             stage=job['stage'],seconds=report['seconds'],peak_rss_mib=report['peak_rss_mib'],
                             result_sha256=sha256(path)))
    table=pd.DataFrame(rows)
    table.to_csv(Path(output)/'recorded_fitting_costs.csv',index=False)
    summary=table.groupby(['case','method','stage']).agg(jobs=('job_id','size'),
        worker_seconds=('seconds','sum'),median_job_seconds=('seconds','median'),
        peak_process_rss_mib=('peak_rss_mib','max')).reset_index()
    summary.to_csv(Path(output)/'recorded_fitting_cost_summary.csv',index=False)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--size',type=int,choices=(100,250,500),required=True)
    parser.add_argument('--repeat',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    measure(args.root,args.size,args.repeat,args.output)
