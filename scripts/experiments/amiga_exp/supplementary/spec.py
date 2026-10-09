"""Frozen full-grid selection at each labelled training size and for deployment."""
import json
from pathlib import Path

import numpy as np

from scripts.experiments.amiga_exp.grouped_validation.contract import _partitions, _front_ids
from scripts.experiments.amiga_exp.grouped_validation.pilot import sha256, write_json
from scripts.experiments.amiga_exp.sequential_selection.spec import validate_contract as validate_original
from scripts.experiments.amiga_exp.reporting.supervised import verify_manifest

REPO = Path(__file__).resolve().parents[4]
SOURCE = 'experiments/sequential-selection/runs/full-001/contract.json'
SUMMARY = 'experiments/top10-classification/full-001/outer-summary'
METHODS = ('ranking', 'reg_aupr', 'clf_top05', 'clf_top10', 'clf_top20')
LEARNING_METHODS = ('ranking', 'reg_aupr')


def contexts(base):
    result = []
    mapping = base['topology_by_front']
    for outer in base['outer_folds']:
        for subset in outer['learning_subsets']:
            if subset['seed'] is None:
                continue
            seed = int(np.random.SeedSequence([20260916, outer['fold'], subset['seed'], subset['size']]).generate_state(1)[0])
            result.append(context(f"learning/fold-{outer['fold']}/size-{subset['size']}/subset-{subset['seed']}",
                                  'learning', mapping, subset['topology_ids'],
                                  outer['test_topology_ids'], seed,
                                  outer_fold=outer['fold'], training_size=subset['size'], subset_seed=subset['seed']))
    result.append(context('deployment', 'deployment', mapping, sorted(set(mapping.values())),
                          [], base['seeds']['deployment_split'], outer_fold=None,
                          training_size=87, subset_seed=None))
    return result


def context(identifier, kind, mapping, training, testing, seed, **metadata):
    folds = []
    for i, validation in enumerate(_partitions(training, seed, 3)):
        folds.append(dict(fold=i, train_front_ids=_front_ids(mapping, sorted(set(training)-set(validation))),
                          validation_front_ids=_front_ids(mapping, validation)))
    return dict(id=identifier, kind=kind, train_front_ids=_front_ids(mapping, training),
                test_front_ids=_front_ids(mapping, testing), inner_folds=folds,
                inner_split_seed=seed, **metadata)


def build_contract(root=REPO):
    root = Path(root)
    original = json.loads((root/SOURCE).read_text())
    validate_original(original)
    _, identities = verify_manifest(root/SUMMARY/'manifest.json')
    hashes = dict(original['source_hashes'])
    hashes[SOURCE] = sha256(root/SOURCE)
    hashes.update({str(p.relative_to(root)):d for p,d in identities.items()})
    for folder in ('scripts/experiments/amiga_exp/supplementary',
                   'scripts/experiments/amiga_exp/top5_classification',
                   'scripts/experiments/amiga_exp/top10_classification'):
        hashes.update({p.relative_to(root).as_posix():sha256(p) for p in (root/folder).glob('*.py')})
    hashes['scripts/experiments/amiga_exp/outer_evaluation/execution.py'] = sha256(root/'scripts/experiments/amiga_exp/outer_evaluation/execution.py')
    for name in ('scripts/experiments/amiga_exp/real_world_validation.py',
                 'scripts/experiments/amiga_exp/version.py','pyproject.toml','poetry.lock'):
        hashes[name]=sha256(root/name)
    hashes.update({p.relative_to(root).as_posix():sha256(p) for p in (root/'amiga').rglob('*.py')})
    case=root/'experiments/BIO-INSIGHT/real-world/tcga_brca'
    application_inputs=[case/'amiga/data_real.csv',case/'amiga/front_real.csv',
                        case/'data/tcga_brca_primary_tumor_log2tpm_top500_variable.csv',
                        case/'validation/amiga_exp_reported/reported_external_tf_target_evidence.csv',
                        *sorted((case/'bioinsight').glob('*/lists/GRN_*.csv'))]
    hashes.update({p.relative_to(root).as_posix():sha256(p) for p in application_inputs})
    result = dict(schema_version=1, workflow='supplementary_evaluation', status='frozen',
                  source_contract=SOURCE, original=original, contexts=contexts(original['split_contract']),
                  learning_methods=list(LEARNING_METHODS), deployment_methods=list(METHODS),
                  final_seeds=[1201,1202,1203,1204,1205], deployment_seed=1201,
                  full_endpoint_summary=SUMMARY, source_hashes=hashes,
                  failures=dict(total_budget_seconds=518400, per_job_timeout_seconds=43200, automatic_retries=0),
                  learning_scope='Repeat full labels/parameters/features/family selection using only available labelled groups',
                  aggregation='mean seeds within subset/front; mean subsets within front; mean conditions within topology; equal topology weights',
                  intervals='10000 paired topology bootstrap resamples, seed 1401; conditional descriptive intervals',
                  untouched_external_confirmation=False)
    validate_contract(result)
    return result


def validate_contract(c):
    if (c.get('schema_version')!=1 or c.get('workflow')!='supplementary_evaluation' or c.get('status')!='frozen'
            or c.get('learning_methods')!=list(LEARNING_METHODS) or c.get('deployment_methods')!=list(METHODS)
            or c.get('final_seeds')!=list(range(1201,1206)) or c.get('deployment_seed')!=1201):
        raise ValueError('Invalid supplementary policy')
    validate_original(c['original'])
    if c['contexts']!=contexts(c['original']['split_contract']):
        raise ValueError('Training sizes, subsets or grouped inner partitions differ')
    if c['failures']!=dict(total_budget_seconds=518400,per_job_timeout_seconds=43200,automatic_retries=0):
        raise ValueError('Invalid technical failure policy')
    if c['full_endpoint_summary']!=SUMMARY or c['source_contract']!=SOURCE:
        raise ValueError('Full-size reference changed')
    for p,d in c['original']['source_hashes'].items():
        if c['source_hashes'].get(p)!=d:
            raise ValueError('Original source identity changed')
    for p,d in c['source_hashes'].items():
        if Path(p).is_absolute() or '..' in Path(p).parts or not isinstance(d,str) or len(d)!=64:
            raise ValueError('Invalid portable source identity')
    return True


def build_plan(c):
    validate_contract(c)
    jobs=[]
    for case in c['original']['split_contract']['cases']:
        # Start deployment early: its complete-grid jobs form the longest branch.
        for scope in sorted(c['contexts'],key=lambda x:(x['kind']!='deployment',-x['training_size'],x['id'])):
            arms=c['learning_methods'] if scope['kind']=='learning' else c['deployment_methods']
            dependencies=[]
            for family in ('CatBoost','XGBoost','LightGBM'):
                identifier=f"{case}/{scope['id']}/select/{family}"
                fits=(8*3+len(c['original']['grids'][family])*3*len(arms)+12*len(arms))
                jobs.append(dict(id=identifier,case=case,stage='selection',context_id=scope['id'],
                                 family=family,arms=arms,dependencies=[],planned_fits=fits))
                dependencies.append(identifier)
            seeds=c['final_seeds'] if scope['kind']=='learning' else [c['deployment_seed']]
            jobs.append(dict(id=f"{case}/{scope['id']}/final",case=case,stage='final',
                             context_id=scope['id'],arms=arms,dependencies=dependencies,
                             planned_fits=len(arms)*len(seeds),maximum_mask_parent_fits=3*len(arms)))
    priorities={s['id']:(s['kind']!='deployment',-s['training_size'],s['id']) for s in c['contexts']}
    return sorted(jobs,key=lambda j:(priorities[j['context_id']],j['case'],j['stage']=='final',j.get('family','')))


def freeze(path, contract):
    validate_contract(contract)
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=contract:
            raise ValueError('Refusing to replace a different frozen contract')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        write_json(path,contract)
