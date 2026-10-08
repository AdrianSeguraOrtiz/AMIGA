"""Audit complete out-of-fold predictions and produce prespecified comparisons."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from scripts.experiments.amiga_exp.grouped_validation.metrics import METRICS, front_metrics, topology_metrics
from scripts.experiments.amiga_exp.grouped_validation.pilot import sha256, write_json
from scripts.experiments.amiga_exp.grouped_validation.summary import paired_difference, holm
from .execution import completed_job, load_rows, selected_features
from .runner import read_run
from .spec import counts, verify_sources


def audit_predictions(frame, predictions, metrics, methods):
    """Recompute all front metrics, including score ties, from saved predictions."""
    ids = ['front_id', 'item_id']
    if predictions.duplicated(ids).any():
        raise ValueError('Duplicate prediction identities')
    predictions = predictions.sort_values(ids).reset_index(drop=True)
    frame = frame.sort_values(ids).reset_index(drop=True)
    if not predictions[ids].equals(frame[ids]):
        raise ValueError('Prediction candidate coverage differs from held-out data')
    if set(predictions) != set(ids + list(methods)):
        raise ValueError('Prediction method coverage differs')
    for method in methods:
        observed = metrics if method == 'score' else metrics.loc[metrics['method'] == method]
        observed = observed.sort_values('front_id').reset_index(drop=True)
        expected = front_metrics(frame, predictions[method].to_numpy())
        if (observed['front_id'].duplicated().any() or
                not observed[['front_id', 'n_items']].equals(expected[['front_id', 'n_items']])):
            raise ValueError('Metric front/candidate coverage differs')
        if not np.allclose(observed[list(METRICS)], expected[list(METRICS)], rtol=1e-10, atol=1e-12):
            raise ValueError('Metrics differ from saved prediction scores')


def central_tables(long, contract):
    """Average metrics over all seeds, conditions, then equally over topologies."""
    mapping = contract['split_contract']['topology_by_front']
    summaries, fronts, topologies, paired, deltas, tests, diagnostics = [], [], [], [], [], [], []
    for case in sorted(contract['cases']):
        case_data = long.loc[long['case'] == case]
        expected_methods = set(contract['arms']) | set(contract['cases'][case]['baseline_ids'])
        if set(case_data['method']) != expected_methods:
            raise ValueError('Central comparison method coverage differs')
        methods = {}
        for method in sorted(expected_methods):
            data = case_data.loc[case_data['method'] == method]
            learned = method in contract['arms']
            expected_rows = len(contract['seeds']) if learned else 1
            if (set(data['front_id']) != set(map(int, mapping)) or
                    not (data.groupby('front_id').size() == expected_rows).all()):
                raise ValueError('Missing or duplicate central front/seed observations')
            if learned:
                if any(set(group['seed']) != set(contract['seeds']) for _, group in data.groupby('front_id')):
                    raise ValueError('Final seed identities differ')
                for fold in contract['split_contract']['outer_folds']:
                    assigned = set(data.loc[data['outer_fold'] == fold['fold'], 'front_id'])
                    if assigned != set(fold['test_front_ids']):
                        raise ValueError('Outer test assignment differs')
                for (fold, seed), part in data.groupby(['outer_fold', 'seed']):
                    topo = topology_metrics(part, mapping)
                    diagnostics.append(dict(case=case, method=method, outer_fold=int(fold), seed=int(seed),
                                            n_topologies=len(topo), **topo[list(METRICS)].mean().to_dict()))
            averaged = data.groupby('front_id', as_index=False)[list(METRICS)].mean()
            topo = topology_metrics(averaged, mapping)
            methods[method] = topo.set_index('topology_id')
            fronts.append(averaged.assign(case=case, method=method))
            topologies.append(topo.assign(case=case, method=method))
            for scope, table in (('topology_macro', topo), ('front_macro', averaged)):
                summaries.append(dict(case=case, method=method, aggregation=scope,
                                      **table[list(METRICS)].mean().to_dict()))
        reference = methods['ranking']
        for method, table in methods.items():
            if method in ('ranking', 'oracle'):
                continue
            table = table.reindex(reference.index)
            for metric in ('Regret@5', 'Regret@1'):
                values = paired_difference(reference[metric], table[metric],
                                           seed=contract['statistics']['bootstrap_seed'],
                                           samples=contract['statistics']['bootstrap_samples'])
                paired.append(dict(case=case, comparator=method, metric=metric, **values))
                delta = (reference[metric] - table[metric]).to_numpy()
                deltas.extend(dict(case=case, comparator=method, metric=metric, topology_id=topology,
                                   difference=float(d)) for topology, d in zip(reference.index, delta))
                if metric == 'Regret@5' and method in contract['arms']:
                    p = (1.0 if np.all(delta == 0) else float(wilcoxon(
                        delta, zero_method='pratt', correction=True, alternative='two-sided', method='approx').pvalue))
                    tests.append(dict(case=case, comparator=method, metric=metric, p_value=p))
    if len(tests) != 6:
        raise ValueError('Exactly six primary supervised comparisons are required')
    for row, corrected in zip(tests, holm([r['p_value'] for r in tests])):
        row['p_holm'] = float(corrected)
    return {'central_summary.csv': pd.DataFrame(summaries),
            'front_metrics.csv': pd.concat(fronts, ignore_index=True),
            'topology_metrics.csv': pd.concat(topologies, ignore_index=True),
            'paired_differences.csv': pd.DataFrame(paired),
            'topology_differences.csv': pd.DataFrame(deltas),
            'primary_tests.csv': pd.DataFrame(tests),
            'fold_seed_diagnostics.csv': pd.DataFrame(diagnostics)}


def summarize(run, output, *, figures=True):
    run, output = Path(run).resolve(), Path(output).resolve()
    manifest, contract, jobs = read_run(run)
    root = Path(manifest['repo_root'])
    verify_sources(root, contract)
    if json.loads((run / 'state.json').read_text())['status'] != 'complete':
        raise ValueError('All phase-4 jobs must finish before summarizing')
    if output.exists():
        raise ValueError('Use a new summary directory')
    base = contract['split_contract']
    labels = {case: load_rows(root, contract, case, sorted(map(int, base['topology_by_front'])), [])
              for case in contract['cases']}
    tables, timings, masks, sources, fits = [], [], [], {}, []
    plan = {j['id']: j for j in jobs}
    for job in jobs:
        complete = completed_job(run, job)
        if complete is None:
            raise ValueError(f'Incomplete phase-4 job: {job["id"]}')
        report, directory = complete
        sources[str((directory / 'result.json').relative_to(run))] = sha256(directory / 'result.json')
        timings.append(dict(job_id=job['id'], case=job['case'], stage=job['stage'],
                            seconds=report['seconds'], peak_rss_mib=report['peak_rss_mib']))
        if job['stage'] == 'mask':
            for row in report['fit_reports']:
                fits.append(dict(job_id=job['id'], stage='mask', fraction=row['fraction'],
                                 requested_iterations=row['requested_iterations'],
                                 actual_iterations=row['actual_iterations']))
            continue
        frame = labels[job['case']].loc[labels[job['case']]['front_id'].isin(job['evaluation_front_ids'])]
        predictions = pd.read_csv(directory / 'predictions.csv', float_precision='round_trip')
        metrics = pd.read_csv(directory / 'metrics.csv', float_precision='round_trip')
        methods = ['score'] if job['stage'] == 'outer' else contract['cases'][job['case']]['baseline_ids']
        audit_predictions(frame, predictions, metrics, methods)
        if job['stage'] == 'outer':
            metrics['method'] = job['arm']
            info = json.loads((directory / 'model.json').read_text())
            features, dependency = selected_features(contract, job, plan, run)
            if (info['procedure'] != job['procedure'] or info['seed'] != job['seed']
                    or info['feature_columns'] != features or info['mask_dependency'] != dependency
                    or info['train_front_ids'] != job['train_front_ids']
                    or info['evaluation_front_ids'] != job['evaluation_front_ids']):
                raise ValueError('Final model metadata differs from the plan')
            fits.append(dict(job_id=job['id'], stage='outer', fraction=job['procedure']['fraction'],
                             requested_iterations=info['fitting']['requested_iterations'],
                             actual_iterations=info['fitting']['actual_iterations']))
            if job['seed'] == contract['seeds'][0]:
                for feature in base['cases'][job['case']]['feature_columns']:
                    masks.append(dict(case=job['case'], outer_fold=job['outer_fold'], arm=job['arm'],
                                      family=job['procedure']['family'], feature=feature, selected=feature in features))
        elif set(metrics['method']) != set(methods):
            raise ValueError('Baseline metric method coverage differs')
        tables.append(metrics.assign(case=job['case'], stage=job['stage'], seed=job['seed'],
                                     outer_fold=job['outer_fold']))
    if len(fits) != counts(contract)['total_fits'] or any(r['requested_iterations'] != 3000 for r in fits):
        raise ValueError('Final fit count or iteration budget differs')
    long = pd.concat(tables, ignore_index=True)
    outputs = central_tables(long, contract)
    outputs.update({'metrics_long.csv': long, 'runtimes.csv': pd.DataFrame(timings),
                    'outer_feature_masks.csv': pd.DataFrame(masks), 'fit_audit.csv': pd.DataFrame(fits)})
    output.mkdir(parents=True)
    for name, table in outputs.items():
        table.to_csv(output / name, index=False)
    if figures:
        from .plots import make_plots
        make_plots(outputs, output / 'plots')
    report = dict(status='complete', scope='outer_held_out_topologies',
        contract_sha256=sha256(run / 'contract.json'), counts=counts(contract),
        source_result_hashes=sources, statistics=contract['statistics'],
        audit=dict(predictions_and_metrics_recomputed=True, all_five_seeds_present=True,
                   completed_fits=len(fits), actual_iterations=sum(r['actual_iterations'] for r in fits)),
        difference_direction='ranking minus comparator; negative regret differences favor ranking',
        limitations=['Reused benchmark data; internal grouped cross-validation, not a new external dataset.',
                     'Bootstrap intervals condition on the fitted models and evaluated partitions.',
                     'Overlapping training sets limit independence; tests are exploratory.',
                     'Five seeds vary final training, conditional on selected procedures and learned masks.'],
        artifacts={p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()})
    write_json(output / 'manifest.json', report)
    return report
