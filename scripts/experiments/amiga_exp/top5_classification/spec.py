"""Freeze a fully retuned top-5% comparator on the existing topology partitions."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path

from scripts.experiments.amiga_exp.grouped_validation.pilot import sha256, utc, write_json
from scripts.experiments.amiga_exp.outer_evaluation import spec as previous_outer
from scripts.experiments.amiga_exp.sequential_selection.spec import grid
from .models import ARM, ARMS, FAMILIES, FRACTIONS, TARGET

REPO = Path(__file__).resolve().parents[4]
PACKAGE = 'scripts/experiments/amiga_exp/top5_classification'
DOC = 'docs/experiments/top5-classification.md'
PREVIOUS_RUN = 'experiments/outer-evaluation/runs/evaluation-001'
PREVIOUS_SUMMARY = 'experiments/outer-evaluation/summaries/evaluation-001'
FAILURES = dict(total_budget_seconds=518400, per_job_timeout_seconds=7200, automatic_retries=0)
STATISTICS = dict(primary_metric='Regret@5', secondary_metrics=['Regret@1', 'Hit@1', 'Hit@5'],
                  comparisons=['ranking', 'clf_top20'], difference_direction='top5 minus reference',
                  bootstrap_samples=10000, bootstrap_seed=1401, tests='none; descriptive sensitivity only')


def verify_artifacts(run, summary, *, manifest_name='manifest.json'):
    run, summary = Path(run), Path(summary)
    if json.loads((run / 'state.json').read_text())['status'] != 'complete':
        raise ValueError('Reference run is incomplete')
    report = json.loads((summary / manifest_name).read_text())
    if report['status'] != 'complete' or report['contract_sha256'] != sha256(run / 'contract.json'):
        raise ValueError('Summary identity or status differs')
    for name, digest in report['artifacts'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256(summary / name) != digest:
            raise ValueError(f'Summary artifact changed: {name}')
    if {p.relative_to(summary).as_posix() for p in summary.rglob('*') if p.is_file()} != set(report['artifacts']) | {manifest_name}:
        raise ValueError('Summary inventory differs')
    hashes = report.get('source_result_hashes', report.get('dependency_result_hashes'))
    for name, digest in hashes.items():
        # Sequential summaries key provenance by job ID; outer summaries use paths.
        if manifest_name == 'summary_manifest.json':
            matches = [p for p in (run / 'jobs' / name).glob('attempt-*/result.json') if sha256(p) == digest]
            if len(matches) != 1:
                raise ValueError('Selection result provenance changed')
        elif Path(name).is_absolute() or '..' in Path(name).parts or sha256(run / name) != digest:
            raise ValueError('Reference result provenance changed')
    return report


def _selection_contract(root, parent):
    root = Path(root)
    paths = set(parent['source_hashes'])
    paths.update(p.relative_to(root).as_posix() for p in (root / PACKAGE).glob('*.py'))
    paths.add(DOC)
    paths.update(f'{PREVIOUS_RUN}/{p}' for p in ('manifest.json', 'contract.json', 'plan.json'))
    paths.update(f'{PREVIOUS_SUMMARY}/{p}' for p in ('manifest.json', 'metrics_long.csv', 'central_summary.csv'))
    result = dict(schema_version=1, workflow='top5_classification', mode='selection',
                  status='frozen_training_only', created_at_utc=utc(), evidence_role='exploratory_sensitivity',
                  parent_contract=deepcopy(parent), split_contract=deepcopy(parent['split_contract']),
                  cases=deepcopy(parent['cases']), families=list(FAMILIES), arms=list(ARMS), labels=['rank_dense'],
                  fractions=list(FRACTIONS), grids={f: grid(f, 'original') for f in FAMILIES},
                  classification_policy=deepcopy(TARGET), selection_seed=1101, seeds=[1201,1202,1203,1204,1205],
                  feature_selection=deepcopy(parent['selection_contract']['feature_selection']),
                  selection_rule=deepcopy(parent['selection_contract']['selection_rule']),
                  statistics=deepcopy(STATISTICS), failures=deepcopy(FAILURES),
                  reference=dict(run=PREVIOUS_RUN, summary=PREVIOUS_SUMMARY),
                  source_hashes={p: sha256(root / p) for p in sorted(paths)})
    validate_contract(result)
    return result


def build_contract(root=REPO):
    root = Path(root).resolve()
    run, summary = root / PREVIOUS_RUN, root / PREVIOUS_SUMMARY
    from scripts.experiments.amiga_exp.outer_evaluation.runner import read_run
    from scripts.experiments.amiga_exp.outer_evaluation.execution import completed_job
    _, parent, jobs = read_run(run)
    previous_outer.verify_sources(root, parent)
    report = verify_artifacts(run, summary)
    if len(jobs) != report['counts']['jobs'] or any(completed_job(run, job) is None for job in jobs):
        raise ValueError('Previous held-out comparator coverage differs')
    return _selection_contract(root, parent)


def validate_contract(c):
    if (c.get('schema_version') != 1 or c.get('workflow') != 'top5_classification'
            or c.get('mode') not in ('selection', 'outer')
            or c.get('status') != ('frozen_training_only' if c['mode'] == 'selection' else 'frozen_outer_evaluation')):
        raise ValueError('Invalid top-5% contract')
    previous_outer.validate_contract(c['parent_contract'])
    parent = c['parent_contract']
    expected = dict(split_contract=parent['split_contract'], cases=parent['cases'],
                    families=list(FAMILIES), arms=list(ARMS), labels=['rank_dense'], fractions=list(FRACTIONS),
                    grids={f: grid(f, 'original') for f in FAMILIES}, classification_policy=TARGET,
                    selection_seed=1101, seeds=[1201,1202,1203,1204,1205], statistics=STATISTICS,
                    feature_selection=parent['selection_contract']['feature_selection'],
                    selection_rule=parent['selection_contract']['selection_rule'], failures=FAILURES,
                    reference=dict(run=PREVIOUS_RUN, summary=PREVIOUS_SUMMARY), evidence_role='exploratory_sensitivity')
    for key, value in expected.items():
        if c.get(key) != value:
            raise ValueError(f'Classification policy changed: {key}')
    for path, digest in parent['source_hashes'].items():
        if c['source_hashes'].get(path) != digest:
            raise ValueError('Previous source identity changed')
    if DOC not in c['source_hashes'] or not any(p.startswith(PACKAGE + '/') for p in c['source_hashes']):
        raise ValueError('Missing classification source snapshot')
    for path, digest in c['source_hashes'].items():
        if Path(path).is_absolute() or '..' in Path(path).parts or len(digest) != 64:
            raise ValueError('Invalid source identity')
    if c['mode'] == 'outer':
        selection = c['selection_contract']
        validate_contract(selection)
        if selection['mode'] != 'selection':
            raise ValueError('Outer stage requires an inner selection contract')
        if any(c[k] != selection[k] for k in expected):
            raise ValueError('Outer policy differs from selection')
        keys = set()
        for p in c['procedures']:
            key = p['case'], p['outer_fold']
            if key in keys or p['arm'] != ARM or p['family'] not in FAMILIES or p['fraction'] not in FRACTIONS:
                raise ValueError('Invalid selected procedure identity')
            keys.add(key)
            if (p['config'] not in c['grids'][p['family']] or p['label'] != 'rank_dense'
                    or p['n_features'] != math.ceil(len(c['split_contract']['cases'][p['case']]['feature_columns']) * p['fraction'])
                    or p['selected_candidate'] != f'{p["case"]}/outer-{p["outer_fold"]}/phase3/{p["family"]}/{ARM}/features-{p["fraction"]:g}'):
                raise ValueError('Selected configuration/fraction differs')
        if keys != {(case, fold) for case in c['cases'] for fold in range(5)}:
            raise ValueError('Ten selected procedures are required')
        if any(c['source_hashes'].get(k) != v for k, v in selection['source_hashes'].items()):
            raise ValueError('Selection source snapshot changed')
    return True


def verify_sources(root, c):
    validate_contract(c)
    for name, digest in c['source_hashes'].items():
        if sha256(Path(root) / name) != digest:
            raise ValueError(f'Source changed after freezing: {name}')


def build_plan(c):
    validate_contract(c)
    jobs, base = [], c['split_contract']
    if c['mode'] == 'selection':
        for case in sorted(c['cases']):
            for outer in base['outer_folds']:
                for family in FAMILIES:
                    tuning = []
                    prefix = f'{case}/outer-{outer["fold"]}'
                    for config in c['grids'][family]:
                        name = f'{prefix}/phase2/{family}/{ARM}/{config["id"]}'
                        tuning.append(name)
                        jobs.append(dict(id=name, case=case, outer_fold=outer['fold'], family=family, arm=ARM,
                                         stage='phase2', config=config, label='rank_dense', dependencies=[], planned_fits=3))
                    jobs.append(dict(id=f'{prefix}/phase3/{family}/{ARM}', case=case, outer_fold=outer['fold'],
                                     family=family, arm=ARM, stage='phase3', config=None, label=None,
                                     dependencies=tuning, planned_fits=12))
    else:
        for p in sorted(c['procedures'], key=lambda x:(x['case'],x['outer_fold'])):
            outer = base['outer_folds'][p['outer_fold']]
            prefix = f'{p["case"]}/outer-{p["outer_fold"]}/{ARM}'
            common = dict(case=p['case'], outer_fold=p['outer_fold'], arm=ARM, procedure=p,
                          train_front_ids=outer['train_front_ids'])
            dependencies = []
            if p['fraction'] != 1.0:
                name = f'{prefix}/mask'
                jobs.append(dict(id=name, stage='mask', seed=1101, dependencies=[], evaluation_front_ids=[],
                                 planned_fits=FRACTIONS.index(p['fraction']), **common))
                dependencies = [name]
            for seed in c['seeds']:
                jobs.append(dict(id=f'{prefix}/seed-{seed}', stage='outer', seed=seed, dependencies=dependencies,
                                 evaluation_front_ids=outer['test_front_ids'], planned_fits=1, **common))
        jobs.sort(key=lambda j:(j['stage'] != 'mask',j['id']))
    return jobs


def counts(c):
    jobs = build_plan(c)
    return dict(jobs=len(jobs), total_fits=sum(j['planned_fits'] for j in jobs),
                jobs_by_stage=dict(Counter(j['stage'] for j in jobs)),
                fits_by_stage={s:sum(j['planned_fits'] for j in jobs if j['stage']==s)
                               for s in sorted({j['stage'] for j in jobs})})


def freeze(path, c):
    validate_contract(c)
    path = Path(path)
    if path.exists():
        raise ValueError('Use a new contract destination')
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, c)


def outer_contract(selection_run, selection_summary, root=REPO):
    from .runner import read_run
    from .selection_summary import summarize
    from tempfile import TemporaryDirectory
    root, run, summary = Path(root).resolve(), Path(selection_run).resolve(), Path(selection_summary).resolve()
    _, selection, _ = read_run(run)
    verify_sources(root, selection)
    saved = verify_artifacts(run, summary, manifest_name='summary_manifest.json')
    with TemporaryDirectory(prefix='amiga-top5-selection-audit-') as temporary:
        target = Path(temporary) / 'summary'
        audited = summarize(run, target, figures=False)
        procedures = json.loads((target / 'selected_procedures.json').read_text())
    if audited['dependency_result_hashes'] != saved['dependency_result_hashes'] or procedures != json.loads((summary / 'selected_procedures.json').read_text()):
        raise ValueError('Saved selections differ from recomputed inner choices')
    c = deepcopy(selection)
    c.update(mode='outer', status='frozen_outer_evaluation', created_at_utc=utc(),
             selection_contract=selection, procedures=procedures,
             upstream=dict(run=run.relative_to(root).as_posix(), summary=summary.relative_to(root).as_posix()))
    paths = [(run / p).relative_to(root).as_posix() for p in ('contract.json','plan.json','manifest.json')]
    paths += [(summary / p).relative_to(root).as_posix() for p in ('summary_manifest.json','selected_procedures.json')]
    c['source_hashes'].update({p:sha256(root/p) for p in paths})
    validate_contract(c)
    return c
