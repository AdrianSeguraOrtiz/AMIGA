"""Executable specification for sequential labels, parameters and columns."""
from __future__ import annotations

from collections import Counter
from itertools import product
import json
from pathlib import Path

from scripts.experiments.amiga_exp.grouped_validation.contract import build_contract as base_contract, validate_contract as validate_base
from scripts.experiments.amiga_exp.grouped_validation.pilot import sha256, write_json
from .models import ARMS, FAMILIES, LABELS, FRACTIONS

PACKAGE = 'scripts/experiments/amiga_exp/sequential_selection'
REPO = Path(__file__).resolve().parents[4]


def grid(family, profile='original'):
    """Original family-specific grids, shared by the four formulations."""
    if profile not in ('original', 'bounded'):
        raise ValueError('Unknown grid profile')
    if family == 'LightGBM':
        values = [dict(num_leaves=n, min_child_samples=m, learning_rate=r,
                       subsample=.8, subsample_freq=1, colsample_bytree=.8, iterations=3000)
                  for n, m, r in product((31,63,127), (30,50,100), (.03,.05,.1))]
        if profile == 'bounded':
            values = [v for v in values if v['min_child_samples']==50 and v['learning_rate'] != .05]
    elif family == 'XGBoost':
        values = [dict(max_depth=d, subsample=s, min_child_weight=m, learning_rate=r,
                       colsample_bytree=.8, iterations=3000)
                  for d, s, m, r in product((4,6,8), (.8,1.0), (1,5,10), (.03,.05))]
        if profile == 'bounded':
            values = [v for v in values if v['subsample']==.8 and v['min_child_weight']==5]
    elif family == 'CatBoost':
        values = [dict(depth=d, l2_leaf_reg=l, learning_rate=r, iterations=3000)
                  for d,l,r in product((4,6,8), (3,5,7,10), (.03,.05,.1))]
        if profile == 'bounded':
            values = [v for v in values if v['l2_leaf_reg']==3 and v['learning_rate'] != .05]
    else:
        raise ValueError('Unknown family')
    return [dict(id=f'cfg-{i:03d}', **v) for i,v in enumerate(values)]


def reference(family):
    if family == 'LightGBM':
        return dict(num_leaves=63, min_child_samples=50, learning_rate=.05,
                    subsample=.8, subsample_freq=1, colsample_bytree=.8, iterations=2000)
    if family == 'XGBoost':
        return dict(max_depth=6, min_child_weight=5, learning_rate=.05,
                    subsample=.8, colsample_bytree=.8, iterations=2000)
    return dict(depth=6, l2_leaf_reg=5, learning_rate=.05, iterations=2000)


def endpoint(family):
    if family == 'LightGBM':
        return dict(num_leaves=127, min_child_samples=30, learning_rate=.03,
                    subsample=.8, subsample_freq=1, colsample_bytree=.8, iterations=3000)
    if family == 'XGBoost':
        return dict(max_depth=8, min_child_weight=1, learning_rate=.03,
                    subsample=1., colsample_bytree=.8, iterations=3000)
    return dict(depth=8, l2_leaf_reg=3, learning_rate=.03, iterations=3000)


def build_contract(root=REPO, *, mode='selection', profile='original', budget_seconds=None):
    root = Path(root).resolve()
    base = base_contract(root)
    paths = set(base['source_hashes'])
    paths.update(p.relative_to(root).as_posix() for p in (root/PACKAGE).glob('*.py'))
    paths.update(['scripts/experiments/amiga_exp/grouped_validation/training_data.py',
                  'scripts/experiments/amiga_exp/grouped_validation/metrics.py',
                  'scripts/experiments/amiga_exp/grouped_validation/pilot.py'])
    doc = root/'docs/experiments/sequential-selection.md'
    if doc.exists():
        paths.add(doc.relative_to(root).as_posix())
    if (root/'scripts/experiments/amiga_exp/plots.py').exists():
        paths.add('scripts/experiments/amiga_exp/plots.py')
    result = dict(schema_version=1, workflow='sequential_selection', mode=mode,
                  status='frozen_training_only', split_contract=base,
                  families=list(FAMILIES), arms=list(ARMS), labels=list(LABELS),
                  control_labels_selectable=True,
                  fractions=list(FRACTIONS), grid_profile=profile,
                  grids={f:grid(f,profile) for f in FAMILIES},
                  references={f:reference(f) for f in FAMILIES},
                  feature_selection=dict(method='recursive_centered_training_tree_shap',
                                         max_rows_per_front=32, sampling_seed=1501,
                                         hyperparameters='fixed_phase_2_winner',
                                         order='full_75_50_25', training_seed=1101),
                  selection_rule=['topology_mean_Regret@5', 'topology_mean_Regret@1',
                                  'fewer_features_for_phase3', 'stable_id'],
                  outer_evaluation_enabled=False,
                  failures=dict(total_budget_seconds=budget_seconds or (7200 if mode=='pilot' else 518400),
                                per_job_timeout_seconds=7200 if mode=='selection' else 1800,
                                automatic_retries=0),
                  source_hashes={p:sha256(root/p) for p in sorted(paths)})
    validate_contract(result)
    return result


def validate_contract(c):
    if c.get('schema_version') != 1 or c.get('workflow') != 'sequential_selection':
        raise ValueError('Invalid sequential contract')
    if c.get('mode') not in ('pilot','selection') or c.get('status') != 'frozen_training_only':
        raise ValueError('Invalid execution scope')
    validate_base(c['split_contract'])
    expected = dict(families=list(FAMILIES), arms=list(ARMS), labels=list(LABELS),
                    fractions=list(FRACTIONS), outer_evaluation_enabled=False,
                    control_labels_selectable=True,
                    references={f:reference(f) for f in FAMILIES},
                    grids={f:grid(f,c['grid_profile']) for f in FAMILIES},
                    feature_selection=dict(method='recursive_centered_training_tree_shap',
                                           max_rows_per_front=32, sampling_seed=1501,
                                           hyperparameters='fixed_phase_2_winner',
                                           order='full_75_50_25', training_seed=1101),
                    selection_rule=['topology_mean_Regret@5','topology_mean_Regret@1',
                                    'fewer_features_for_phase3','stable_id'])
    for key,value in expected.items():
        if c.get(key)!=value:
            raise ValueError(f'Frozen policy differs: {key}')
    if not 1 <= c['failures']['total_budget_seconds'] <= (7200 if c['mode']=='pilot' else 518400):
        raise ValueError('Invalid cumulative resource budget')
    if c['failures']['automatic_retries'] != 0 or c['failures']['per_job_timeout_seconds'] != (1800 if c['mode']=='pilot' else 7200):
        raise ValueError('Invalid failure policy')
    for p,digest in c['split_contract']['source_hashes'].items():
        if c['source_hashes'].get(p)!=digest:
            raise ValueError('Base source identity differs')
    for p,digest in c['source_hashes'].items():
        if Path(p).is_absolute() or '..' in Path(p).parts or len(digest)!=64:
            raise ValueError('Invalid source identity')
    return True


def build_plan(c):
    validate_contract(c)
    base=c['split_contract']
    jobs=[]
    if c['mode']=='pilot':
        train=base['outer_folds'][0]['inner_folds'][0]['train_front_ids']
        for case, family, arm in product(base['cases'],FAMILIES,ARMS):
            jobs.append(dict(id=f'{case}/pilot/{family}/{arm}', case=case, stage='pilot',
                             family=family, arm=arm, outer_fold=0, dependencies=[],
                             train_front_ids=train, evaluation_front_ids=[], config=endpoint(family),
                             label='rank_dense', planned_fits=4))
        return jobs
    for case in base['cases']:
        for outer in base['outer_folds']:
            prefix=f'{case}/outer-{outer["fold"]}'
            for family in FAMILIES:
                screen=[]
                for label in LABELS:
                    name=f'{prefix}/phase1/{family}/{label}'
                    jobs.append(dict(id=name, case=case, stage='phase1', family=family,
                                     arm='ranking', outer_fold=outer['fold'], dependencies=[],
                                     config=c['references'][family], label=label, planned_fits=3))
                    screen.append(name)
                for arm in ARMS:
                    tuning=[]
                    for config in c['grids'][family]:
                        name=f'{prefix}/phase2/{family}/{arm}/{config["id"]}'
                        jobs.append(dict(id=name, case=case, stage='phase2', family=family,
                                         arm=arm, outer_fold=outer['fold'],
                                         dependencies=screen if arm=='ranking' else [],
                                         config=config, label=None, planned_fits=3))
                        tuning.append(name)
                    jobs.append(dict(id=f'{prefix}/phase3/{family}/{arm}', case=case, stage='phase3',
                                     family=family, arm=arm, outer_fold=outer['fold'],
                                     dependencies=tuning, config=None, label=None, planned_fits=12))
    return jobs


def counts(c):
    jobs=build_plan(c)
    return dict(jobs=len(jobs), jobs_by_stage=dict(Counter(j['stage'] for j in jobs)),
                fits_by_stage={s:sum(j['planned_fits'] for j in jobs if j['stage']==s)
                               for s in sorted({j['stage'] for j in jobs})},
                total_fits=sum(j['planned_fits'] for j in jobs))


def freeze(path, contract):
    validate_contract(contract)
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=contract:
            raise ValueError('Refusing to replace a different contract')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    write_json(path,contract)
