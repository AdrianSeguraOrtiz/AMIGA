"""Freeze phase-4 inputs and policies before producing held-out predictions."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile

from scripts.experiments.amiga_exp.grouped_validation.baselines import baseline_ids
from scripts.experiments.amiga_exp.grouped_validation.pilot import sha256, utc, write_json
from scripts.experiments.amiga_exp.grouped_validation.planning import STATISTICS
from scripts.experiments.amiga_exp.sequential_selection import models
from scripts.experiments.amiga_exp.sequential_selection.runner import read_run, verify_sources as verify_selection_sources
from scripts.experiments.amiga_exp.sequential_selection.spec import validate_contract as validate_selection
from scripts.experiments.amiga_exp.sequential_selection.summary import summarize as summarize_selection

REPO = Path(__file__).resolve().parents[4]
PACKAGE = 'scripts/experiments/amiga_exp/outer_evaluation'
DOC = 'docs/experiments/outer-evaluation.md'
MASK_POLICY = dict(training_seed=1101, sampling_seed=1501, max_rows_per_front=32,
                   method='recursive_centered_training_tree_shap',
                   fractions=[1.0, .75, .5, .25],
                   scope='outer_training_only', shared_across_final_seeds=True,
                   fit_only_parents_of_selected_fraction=True)
FAILURES = dict(total_budget_seconds=518400, per_job_timeout_seconds=7200, automatic_retries=0)


def verify_selection(run, summary, root):
    """Reconstruct the selection from every completed job, never from test scores."""
    run, summary, root = Path(run).resolve(), Path(summary).resolve(), Path(root).resolve()
    manifest, selection, plan = read_run(run)
    if Path(manifest['repo_root']) != root:
        raise ValueError('Selection repository differs')
    verify_selection_sources(root, selection)
    if selection['mode'] != 'selection' or selection['grid_profile'] != 'original':
        raise ValueError('Phase 4 requires the completed original-grid selection')
    saved = json.loads((summary / 'summary_manifest.json').read_text())
    if saved['contract_sha256'] != sha256(run / 'contract.json'):
        raise ValueError('Selection summary belongs to another contract')
    for name, digest in saved['artifacts'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256(summary / name) != digest:
            raise ValueError(f'Selection summary artifact changed: {name}')
    # The existing summarizer checks all 4,320 job identities and artifact hashes,
    # verifies inner coverage, and reapplies the frozen tie-breaking rule.
    with tempfile.TemporaryDirectory(prefix='amiga-outer-selection-audit-') as temporary:
        target = Path(temporary) / 'summary'
        rebuilt = summarize_selection(run, target, figures=False)
        procedures = json.loads((target / 'selected_procedures.json').read_text())
    if (rebuilt['dependency_result_hashes'] != saved['dependency_result_hashes'] or
            procedures != json.loads((summary / 'selected_procedures.json').read_text())):
        raise ValueError('Saved selections differ from the completed inner experiment')
    return selection, procedures, dict(verified_at_utc=utc(), complete_jobs=len(plan),
        selected_procedures=len(procedures), recomputed_from_inner_predictions=True)


def build_contract(run, summary, root=REPO):
    root, run, summary = [Path(p).resolve() for p in (root, run, summary)]
    selection, procedures, audit = verify_selection(run, summary, root)
    cases = {}
    for case in selection['split_contract']['cases']:
        info = json.loads((root / f'docs/experiments/contracts/{case}_feature_columns.json').read_text())
        cases[case] = {key: info[key] for key in ('objective_columns', 'objective_directions')}
        cases[case]['baseline_ids'] = baseline_ids(info['objective_columns'])
    paths = set(selection['source_hashes'])
    for package in (PACKAGE, 'scripts/experiments/amiga_exp/grouped_validation'):
        paths.update(p.relative_to(root).as_posix() for p in (root / package).glob('*.py'))
    paths.update([DOC, 'scripts/experiments/amiga_exp/decision_baselines.py'])
    for folder, names in ((run, ('manifest.json', 'contract.json', 'plan.json')),
                          (summary, ('summary_manifest.json', 'selected_procedures.json'))):
        paths.update((folder / name).relative_to(root).as_posix() for name in names)
    contract = dict(schema_version=1, workflow='sequential_outer_evaluation',
        status='frozen_outer_evaluation', selection_contract=selection,
        split_contract=selection['split_contract'], procedures=procedures,
        arms=list(models.ARMS), seeds=[1201, 1202, 1203, 1204, 1205], cases=cases,
        feature_mask_policy=deepcopy(MASK_POLICY), statistics=deepcopy(STATISTICS),
        failures=deepcopy(FAILURES), upstream_audit=audit,
        upstream=dict(run=run.relative_to(root).as_posix(), summary=summary.relative_to(root).as_posix()),
        source_hashes={p: sha256(root / p) for p in sorted(paths)})
    validate_contract(contract)
    return contract


def validate_contract(contract):
    if (contract.get('schema_version') != 1 or contract.get('workflow') != 'sequential_outer_evaluation'
            or contract.get('status') != 'frozen_outer_evaluation'):
        raise ValueError('A frozen phase-4 contract is required')
    selection = contract['selection_contract']
    validate_selection(selection)
    if selection['mode'] != 'selection' or selection['grid_profile'] != 'original':
        raise ValueError('Selection must use the full original grids')
    base = contract['split_contract']
    if base != selection['split_contract']:
        raise ValueError('Outer partitions changed after selection')
    for key, expected in [('arms', list(models.ARMS)), ('seeds', base['seeds']['final']),
                          ('feature_mask_policy', MASK_POLICY), ('statistics', STATISTICS),
                          ('failures', FAILURES)]:
        if contract.get(key) != expected:
            raise ValueError(f'Phase-4 policy differs: {key}')
    keys = set()
    for procedure in contract['procedures']:
        case, fold, arm = (procedure[k] for k in ('case', 'outer_fold', 'arm'))
        key = (case, fold, arm)
        if key in keys or case not in base['cases'] or fold not in range(5) or arm not in models.ARMS:
            raise ValueError('Duplicate or unknown selected procedure')
        keys.add(key)
        family, fraction = procedure['family'], procedure['fraction']
        if family not in models.FAMILIES or fraction not in models.FRACTIONS:
            raise ValueError('Unknown selected family or feature fraction')
        if procedure['config'] not in selection['grids'][family] or procedure['label'] not in models.LABELS:
            raise ValueError('Selected parameters or labels were not in the search')
        if procedure['n_features'] != math.ceil(len(base['cases'][case]['feature_columns']) * fraction):
            raise ValueError('Selected feature count differs from its fraction')
        expected = f'{case}/outer-{fold}/phase3/{family}/{arm}/features-{fraction:g}'
        if procedure['selected_candidate'] != expected:
            raise ValueError('Selected candidate identity differs')
    expected = {(c, f, a) for c in base['cases'] for f in range(5) for a in models.ARMS}
    if keys != expected or set(contract['cases']) != set(base['cases']):
        raise ValueError('Incomplete selected procedure coverage')
    for case, info in contract['cases'].items():
        if (not info['objective_columns'] or len(set(info['objective_columns'])) != len(info['objective_columns'])
                or not set(info['objective_columns']) <= set(base['cases'][case]['feature_columns'])
                or info['baseline_ids'] != baseline_ids(info['objective_columns'])
                or any(info['objective_directions'].get(o) not in ('minimize', 'maximize') for o in info['objective_columns'])):
            raise ValueError('Invalid objective-only comparators')
    for p, digest in selection['source_hashes'].items():
        if contract['source_hashes'].get(p) != digest:
            raise ValueError('Selection source identity changed')
    required = {DOC, 'scripts/experiments/amiga_exp/decision_baselines.py'}
    required.update(f'{PACKAGE}/{p.name}' for p in Path(__file__).parent.glob('*.py'))
    if not required <= set(contract['source_hashes']):
        raise ValueError('Missing phase-4 source identities')
    for p, digest in contract['source_hashes'].items():
        if Path(p).is_absolute() or '..' in Path(p).parts or len(digest) != 64:
            raise ValueError('Invalid source identity')
    return True


def verify_sources(root, contract):
    validate_contract(contract)
    root = Path(root)
    for path, digest in contract['source_hashes'].items():
        if sha256(root / path) != digest:
            raise ValueError(f'Source changed after phase-4 freeze: {path}')
    summary = root / contract['upstream']['summary']
    if json.loads((summary / 'selected_procedures.json').read_text()) != contract['procedures']:
        raise ValueError('Phase-4 procedures differ from the saved inner selection')
    upstream = root / contract['upstream']['run']
    if json.loads((upstream / 'contract.json').read_text()) != contract['selection_contract']:
        raise ValueError('Upstream selection contract differs')
    for case, info in contract['cases'].items():
        source = json.loads((root / f'docs/experiments/contracts/{case}_feature_columns.json').read_text())
        if any(info[key] != source[key] for key in ('objective_columns', 'objective_directions')):
            raise ValueError('Baseline objectives differ from the feature contract')


def build_plan(contract):
    validate_contract(contract)
    base, jobs = contract['split_contract'], []
    for p in sorted(contract['procedures'], key=lambda p: (p['case'], p['outer_fold'], p['arm'])):
        fold = base['outer_folds'][p['outer_fold']]
        prefix = f'{p["case"]}/outer-{p["outer_fold"]}/{p["arm"]}'
        common = dict(case=p['case'], outer_fold=p['outer_fold'], arm=p['arm'], procedure=p,
                      train_front_ids=fold['train_front_ids'])
        dependencies = []
        if p['fraction'] != 1.0:
            name = f'{prefix}/mask'
            jobs.append(dict(id=name, stage='mask', seed=1101, dependencies=[],
                             evaluation_front_ids=[], planned_fits=models.FRACTIONS.index(p['fraction']), **common))
            dependencies = [name]
        for seed in contract['seeds']:
            jobs.append(dict(id=f'{prefix}/seed-{seed}', stage='outer', seed=seed,
                             dependencies=dependencies, evaluation_front_ids=fold['test_front_ids'],
                             planned_fits=1, **common))
    for case in sorted(contract['cases']):
        jobs.append(dict(id=f'{case}/baselines/all', case=case, stage='baselines',
                         arm=None, seed=None, outer_fold=None, procedure=None, dependencies=[],
                         train_front_ids=[], evaluation_front_ids=sorted(map(int, base['topology_by_front'])),
                         planned_fits=0))
    # Start the shared mask dependencies early to avoid a serial tail.
    jobs.sort(key=lambda j: (j['stage'] != 'mask', j['id']))
    mapping = base['topology_by_front']
    for job in jobs:
        if {mapping[str(f)] for f in job['train_front_ids']} & {mapping[str(f)] for f in job['evaluation_front_ids']}:
            raise ValueError('A topology crosses the outer training/test boundary')
    return jobs


def counts(contract):
    plan = build_plan(contract)
    return dict(jobs=len(plan), final_fits=sum(j['stage'] == 'outer' for j in plan),
                mask_fits=sum(j['planned_fits'] for j in plan if j['stage'] == 'mask'),
                total_fits=sum(j['planned_fits'] for j in plan),
                jobs_by_stage=dict(Counter(j['stage'] for j in plan)))


def freeze(path, contract):
    validate_contract(contract)
    path = Path(path)
    if path.exists():
        raise ValueError('Use a new phase-4 contract path')
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, contract)
