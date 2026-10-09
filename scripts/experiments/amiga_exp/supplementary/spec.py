"""Freeze fixed AMIGA recipes from completed phase 4; never search again."""
from copy import deepcopy
import json
from pathlib import Path

from scripts.experiments.amiga_exp.grouped_validation.contract import _front_ids
from scripts.experiments.amiga_exp.grouped_validation.pilot import sha256, write_json
from scripts.experiments.amiga_exp.outer_evaluation.execution import completed_job
from scripts.experiments.amiga_exp.outer_evaluation.spec import verify_sources as verify_outer
from scripts.experiments.amiga_exp.reporting.supervised import verify_manifest

REPO = Path(__file__).resolve().parents[4]
SOURCE = 'experiments/sequential-selection/runs/full-001/contract.json'
OUTER = 'experiments/outer-evaluation/runs/evaluation-001'
SUMMARY = 'experiments/top10-classification/full-001/outer-summary'
METHODS = LEARNING_METHODS = ('ranking',)
DEPLOYMENT_POLICY = 'lowest stored inner-validation Regret@5 among BIO-INSIGHT phase-4 recipes; outer-fold tie break'


def contexts(base):
    result = []
    for outer in base['outer_folds']:
        for subset in outer['learning_subsets']:
            if subset['seed'] is None:
                continue
            result.append(dict(id=f"learning/fold-{outer['fold']}/size-{subset['size']}/subset-{subset['seed']}",
                kind='learning', outer_fold=outer['fold'], training_size=subset['size'],
                subset_seed=subset['seed'],
                train_front_ids=_front_ids(base['topology_by_front'], subset['topology_ids']),
                test_front_ids=_front_ids(base['topology_by_front'], outer['test_topology_ids'])))
    result.append(dict(id='deployment', kind='deployment', outer_fold=None,
        training_size=len(set(base['topology_by_front'].values())), subset_seed=None,
        train_front_ids=sorted(map(int, base['topology_by_front'])), test_front_ids=[]))
    return result


def deployment_recipe(recipes):
    """Reuse an existing choice using inner validation only, without any fitting."""
    candidates = [r for r in recipes if r['procedure']['case'] == 'BIO-INSIGHT']
    return deepcopy(min(candidates, key=lambda r: (
        r['procedure']['inner_mean_regret5'], r['procedure']['outer_fold'])))


def build_contract(root=REPO):
    root = Path(root).resolve()
    run = root / OUTER
    original = json.loads((run / 'contract.json').read_text())
    verify_outer(root, original)
    if json.loads((run / 'state.json').read_text())['status'] != 'complete':
        raise ValueError('Completed phase 4 is required')
    plan = json.loads((run / 'plan.json').read_text())
    hashes = dict(original['source_hashes'])
    for name in ('contract.json', 'plan.json', 'state.json', 'manifest.json'):
        hashes[f'{OUTER}/{name}'] = sha256(run / name)
    recipes = []
    for job in plan:
        if job['stage'] != 'outer' or job['arm'] != 'ranking' or job['seed'] != 1201:
            continue
        result = completed_job(run, job)
        if result is None:
            raise ValueError('Missing completed AMIGA fit in phase 4')
        _, directory = result
        path = directory / 'model.json'
        info = json.loads(path.read_text())
        if info['procedure'] != job['procedure'] or info['train_front_ids'] != job['train_front_ids']:
            raise ValueError('Phase-4 model metadata differs from its job')
        for p in (path, directory / 'result.json'):
            hashes[p.relative_to(root).as_posix()] = sha256(p)
        recipes.append(dict(procedure=info['procedure'], feature_columns=info['feature_columns'],
                            source_model_metadata=path.relative_to(root).as_posix()))
    _, identities = verify_manifest(root / SUMMARY / 'manifest.json')
    hashes.update({p.relative_to(root).as_posix(): d for p, d in identities.items()})
    for folder in ('scripts/experiments/amiga_exp/supplementary', 'amiga'):
        hashes.update({p.relative_to(root).as_posix(): sha256(p) for p in (root / folder).rglob('*.py')})
    for name in ('scripts/experiments/amiga_exp/real_world_validation.py',
                 'scripts/experiments/amiga_exp/version.py', 'pyproject.toml', 'poetry.lock'):
        hashes[name] = sha256(root / name)
    case = root / 'experiments/BIO-INSIGHT/real-world/tcga_brca'
    inputs = [case / 'amiga/data_real.csv',
              case / 'validation/amiga_exp_reported/reported_external_tf_target_evidence.csv',
              *sorted((case / 'bioinsight').glob('*/lists/GRN_*.csv'))]
    hashes.update({p.relative_to(root).as_posix(): sha256(p) for p in inputs})
    result = dict(schema_version=2, workflow='fixed_amiga_supplement', status='frozen',
        original=original, recipes=recipes, deployment_recipe=deployment_recipe(recipes),
        deployment_policy=DEPLOYMENT_POLICY, contexts=contexts(original['split_contract']),
        learning_methods=['ranking'], deployment_methods=['ranking'], final_seeds=list(range(1201, 1206)),
        deployment_seed=1201, full_endpoint_summary=SUMMARY, source_hashes=hashes,
        failures=dict(total_budget_seconds=518400, per_job_timeout_seconds=43200, automatic_retries=0),
        learning_scope='AMIGA data-size sensitivity conditional on fixed phase-4 recipes and exact columns; no reselection',
        aggregation='mean seeds within subset/front; mean subsets within front; mean conditions within topology; equal topology weights',
        intervals='10000 topology bootstrap resamples, seed 1401; conditional descriptive intervals',
        prediction_cost_scope='prepared-front prediction and ranking only; no feature preparation or training')
    validate_contract(result)
    return result


def validate_contract(c):
    if (c.get('schema_version') != 2 or c.get('workflow') != 'fixed_amiga_supplement'
            or c.get('status') != 'frozen' or c.get('learning_methods') != ['ranking']
            or c.get('deployment_methods') != ['ranking']
            or c.get('final_seeds') != list(range(1201, 1206)) or c.get('deployment_seed') != 1201):
        raise ValueError('Expected a frozen fixed-AMIGA contract; extended contracts are not supported')
    base = c['original']['split_contract']
    if c['contexts'] != contexts(base):
        raise ValueError('Training subsets or outer partitions differ')
    expected = {(case, fold) for case in base['cases'] for fold in range(5)}
    observed = set()
    for recipe in c['recipes']:
        p = recipe['procedure']
        key = (p['case'], p['outer_fold'])
        columns = recipe['feature_columns']
        full = base['cases'][p['case']]['feature_columns']
        if (key in observed or p['arm'] != 'ranking' or p not in c['original']['procedures']
                or len(columns) != p['n_features'] or columns != [f for f in full if f in columns]):
            raise ValueError('Fixed recipe differs from the completed phase-4 choice')
        observed.add(key)
    if observed != expected or c['deployment_recipe'] != deployment_recipe(c['recipes']):
        raise ValueError('Incomplete fixed recipes or changed deployment choice')
    if c['deployment_policy'] != DEPLOYMENT_POLICY or c['full_endpoint_summary'] != SUMMARY:
        raise ValueError('Deployment policy or full-size reference changed')
    if c['failures'] != dict(total_budget_seconds=518400, per_job_timeout_seconds=43200, automatic_retries=0):
        raise ValueError('Technical failure policy changed')
    for p, d in c['original']['source_hashes'].items():
        if c['source_hashes'].get(p) != d:
            raise ValueError('Original source identity changed')
    for p, d in c['source_hashes'].items():
        if Path(p).is_absolute() or '..' in Path(p).parts or not isinstance(d, str) or len(d) != 64:
            raise ValueError('Invalid portable source identity')
    return True


def build_plan(c):
    validate_contract(c)
    recipes = {(r['procedure']['case'], r['procedure']['outer_fold']): r for r in c['recipes']}
    jobs = []
    for case in c['original']['split_contract']['cases']:
        for scope in c['contexts']:
            deployment = scope['kind'] == 'deployment'
            if deployment and case != 'BIO-INSIGHT':
                continue
            jobs.append(dict(id=f"{case}/{scope['id']}/final", case=case, stage='final',
                context_id=scope['id'], arms=['ranking'], dependencies=[],
                recipe=deepcopy(c['deployment_recipe'] if deployment else recipes[case, scope['outer_fold']]),
                planned_fits=1 if deployment else len(c['final_seeds'])))
    return sorted(jobs, key=lambda j: (j['context_id'] != 'deployment', j['id']))


def freeze(path, contract):
    validate_contract(contract)
    path = Path(path)
    if path.exists():
        raise ValueError('Use a new frozen contract path')
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, contract)
