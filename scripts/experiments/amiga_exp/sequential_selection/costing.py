"""Quality-blind runtime estimates from completed training-only pilot jobs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.experiments.amiga_exp.grouped_validation.pilot import sha256
from .execution import completed_job
from .runner import read_run, verify_sources
from .spec import FAMILIES, ARMS, LABELS, grid


def project(pilot_run):
    pilot_run = Path(pilot_run).resolve()
    manifest, contract, jobs = read_run(pilot_run)
    state = json.loads((pilot_run / 'state.json').read_text())
    if contract['mode'] != 'pilot' or state['status'] != 'complete':
        raise ValueError('Costing requires a complete training-only pilot')
    root = Path(manifest['repo_root'])
    snapshot = pilot_run / 'source-snapshot'
    verify_sources(snapshot if snapshot.exists() else root, contract)
    base = contract['split_contract']
    timings, results = {}, {}
    for job in jobs:
        completed = completed_job(pilot_run, job)
        if completed is None:
            raise ValueError('Pilot job is incomplete')
        result, directory = completed
        data = json.loads((directory / 'pilot.json').read_text())
        if data['external_predictions_computed'] or data['quality_metrics_computed']:
            raise ValueError('A quality-blind pilot is required')
        path = data['paths']
        timings[(job['case'], job['family'], job['arm'])] = dict(
            full_fit_seconds=path[0]['fit_seconds'], path_seconds=sum(
                p['fit_seconds'] + p.get('shap_seconds', 0.) for p in path),
            training_rows=data['training']['input_rows'],
            overhead_seconds=max(10., result['seconds'] - sum(
                p['fit_seconds'] + p.get('shap_seconds', 0.) for p in path)))
        results[job['id']] = sha256(directory / 'result.json')
    projections = {}
    for profile in ('original', 'bounded'):
        seconds = dict(phase1=0., phase2=0., phase3=0.)
        for case, info in base['cases'].items():
            counts = pd.read_csv(root / info['data_path'], usecols=['front_id'])['front_id'].value_counts()
            largest = max(int(counts.reindex(inner['train_front_ids']).sum())
                          for outer in base['outer_folds'] for inner in outer['inner_folds'])
            n_outer = len(base['outer_folds'])
            n_inner = len(base['outer_folds'][0]['inner_folds'])
            for family in FAMILIES:
                for arm in ARMS:
                    timing = timings[(case, family, arm)]
                    scale = max(1., largest / timing['training_rows'])
                    overhead = timing['overhead_seconds']
                    full, path = timing['full_fit_seconds'] * scale, timing['path_seconds'] * scale
                    if arm == 'ranking':
                        seconds['phase1'] += n_outer * len(LABELS) * (n_inner * full * 2/3 + overhead)
                    seconds['phase2'] += n_outer * len(grid(family, profile)) * (n_inner * full + overhead)
                    seconds['phase3'] += n_outer * (n_inner * path + overhead)
        hours = sum(seconds.values()) * 1.25 / (manifest['workers'] * .8) / 3600
        projections[profile] = dict(projected_hours=hours, serial_seconds_by_stage=seconds)
    return dict(pilot_elapsed_seconds=state['elapsed_seconds'], projections=projections,
                pilot_manifest_sha256=sha256(pilot_run / 'manifest.json'),
                pilot_result_hashes=results,
                assumptions=dict(cost_point='high_complexity_grid_endpoint',
                                 row_scaling='largest_inner_training_set',
                                 phase1_iterations_ratio=2/3, overhead_margin=1.25,
                                 parallel_efficiency=.8, workers=manifest['workers']),
                caveat='Runtime estimate, not a guaranteed completion time.')


def choose_profile(projection, maximum_hours=72.):
    """Prefer the original grid; use a predeclared smaller grid on runtime only."""
    for profile in ('original', 'bounded'):
        if projection['projections'][profile]['projected_hours'] <= maximum_hours:
            return profile
    raise ValueError('Neither frozen grid fits the runtime target; no quality jobs were started')
