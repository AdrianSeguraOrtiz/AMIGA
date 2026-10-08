"""Paired descriptive summaries from complete grouped out-of-fold predictions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .execution import completed_job
from .metrics import METRICS, topology_metrics
from .pilot import sha256, write_json
from .runner import read_run, verify_sources


def paired_difference(reference, alternative, *, seed=1401, samples=10000):
    """Negative mean differences favor the reference (both metrics are regret)."""
    delta = np.asarray(reference, dtype=float) - np.asarray(alternative, dtype=float)
    if delta.ndim != 1 or not len(delta) or not np.isfinite(delta).all():
        raise ValueError('Paired differences must be finite and nonempty')
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), size=(samples, len(delta)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return dict(mean_difference=float(delta.mean()), ci_low=float(low), ci_high=float(high),
                improved=int((delta < 0).sum()), tied=int((delta == 0).sum()),
                worsened=int((delta > 0).sum()), n_topologies=len(delta))


def holm(pvalues):
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values, kind='stable')
    corrected = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    result = np.empty_like(values)
    result[order] = corrected
    return result


def summarize(run: Path, output: Path):
    run, output = Path(run).resolve(), Path(output).resolve()
    manifest, contract, jobs = read_run(run)
    verify_sources(Path(manifest['repo_root']), contract)
    mapping = contract['split_contract']['topology_by_front']
    tables, timings, sources = [], [], {}
    for job in jobs:
        completed = completed_job(run, job)
        if completed is None:
            raise ValueError(f'Summary requires every planned job, including {job["id"]}')
        report, directory = completed
        sources[str((directory / 'result.json').relative_to(run))] = sha256(directory / 'result.json')
        timings.append(dict(job_id=job['id'], stage=job['stage'], case=job['case'],
                            seconds=report['total_seconds'], peak_rss_mib=report['peak_rss_mib']))
        if job['stage'] not in ('outer', 'baselines', 'learning_curve', 'ablation', 'family'):
            continue
        data = pd.read_csv(directory / 'metrics.csv')
        if set(data['front_id']) != set(job['evaluation_front_ids']):
            raise ValueError('Metric coverage differs from planned evaluation')
        data['case'], data['stage'] = job['case'], job['stage']
        if job['arm']:
            data['method'] = job['arm']
        for field in ('seed', 'outer_fold', 'feature_set', 'training_size', 'n_training_topologies', 'subset_seed', 'family'):
            data[field] = job.get(field)
        tables.append(data)
    long = pd.concat(tables, ignore_index=True)
    central = long.loc[long['stage'].isin(['outer', 'baselines'])]
    summaries, topology_rows, fronts, differences, tests = [], [], [], [], []
    for case, case_data in central.groupby('case', sort=True):
        methods = {}
        for method, data in case_data.groupby('method', sort=True):
            # Average metric values over seeds, never scores over seeds.
            averaged = data.groupby('front_id', as_index=False)[list(METRICS)].mean()
            if len(averaged) != len(mapping):
                raise ValueError('Central comparison must cover all benchmark fronts')
            expected = 5 if method in contract['split_contract']['arms'] else 1
            if not (data.groupby('front_id').size() == expected).all():
                raise ValueError('Unexpected seed replication in the central comparison')
            topology = topology_metrics(averaged, mapping)
            methods[method] = topology.set_index('topology_id')
            fronts.append(averaged.assign(case=case, method=method))
            topology_rows.append(topology.assign(case=case, method=method))
            for scope, table in (('topology_macro', topology), ('front_macro', averaged)):
                summaries.append(dict(case=case, method=method, aggregation=scope,
                                      **table[list(METRICS)].mean().to_dict()))
        reference = methods['ltr_catboost']
        for method, table in methods.items():
            if method in ('ltr_catboost', 'oracle'):
                continue
            table = table.reindex(reference.index)
            for metric in ('Regret@5', 'Regret@1'):
                values = paired_difference(reference[metric], table[metric])
                differences.append(dict(case=case, comparator=method, metric=metric, **values))
                if metric == 'Regret@5' and method in contract['split_contract']['arms']:
                    delta = (reference[metric] - table[metric]).to_numpy()
                    pvalue = (1.0 if np.all(delta == 0) else
                              float(wilcoxon(delta, zero_method='pratt', correction=True,
                                             alternative='two-sided', method='approx').pvalue))
                    tests.append(dict(case=case, comparator=method, metric=metric, p_value=pvalue))
    if len(tests) != 6:
        raise ValueError('Exactly six prespecified primary tests are required')
    for row, corrected in zip(tests, holm([r['p_value'] for r in tests])):
        row['p_holm'] = float(corrected)
    # Supplementary blocks retain their identifiers. Average subset repetitions
    # within fronts before topology averaging; keep all raw rows for diagnostics.
    extra = []
    for stage, keys in [('learning_curve', ['method', 'training_size']),
                        ('ablation', ['feature_set']), ('family', ['method', 'family'])]:
        selected = long.loc[long['stage'] == stage].copy()
        if stage == 'ablation':
            full = long.loc[(long['stage'] == 'outer') & (long['method'] == 'ltr_catboost') & (long['seed'] == 1201)].copy()
            selected = pd.concat([selected, full], ignore_index=True)
        for labels, group in selected.groupby(['case', *keys], dropna=False, sort=True):
            averaged = group.groupby('front_id', as_index=False)[list(METRICS)].mean()
            topo = topology_metrics(averaged, mapping)
            extra.append(dict(stage=stage, **dict(zip(['case', *keys], labels)),
                              n_fronts=len(averaged), n_topologies=len(topo),
                              **topo[list(METRICS)].mean().to_dict()))
    diagnostics = []
    for (case, method, fold, seed), data in long.loc[long['stage'] == 'outer'].groupby(
            ['case', 'method', 'outer_fold', 'seed']):
        topo = topology_metrics(data, mapping)
        diagnostics.append(dict(case=case, method=method, outer_fold=int(fold), seed=int(seed),
                                n_topologies=len(topo), **topo[list(METRICS)].mean().to_dict()))
    output.mkdir(parents=True, exist_ok=False)
    outputs = {'metrics_long.csv': long, 'central_summary.csv': pd.DataFrame(summaries),
               'front_metrics.csv': pd.concat(fronts, ignore_index=True),
               'topology_metrics.csv': pd.concat(topology_rows, ignore_index=True),
               'paired_differences.csv': pd.DataFrame(differences), 'primary_tests.csv': pd.DataFrame(tests),
               'supplementary_summary.csv': pd.DataFrame(extra),
               'fold_seed_diagnostics.csv': pd.DataFrame(diagnostics), 'runtimes.csv': pd.DataFrame(timings)}
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False)
    write_json(output / 'manifest.json', dict(contract_sha256=sha256(run / 'contract.json'),
               sources=sources, statistics=contract['statistics'],
               difference_direction='LTR minus comparator; negative regret differences favor LTR',
               interval_limit='Conditional on fitted models and partitions; internal exploratory evaluation',
               artifacts={name: sha256(output / name) for name in outputs}))
    return dict(output=str(output), primary_tests=len(tests), tables=len(outputs))
