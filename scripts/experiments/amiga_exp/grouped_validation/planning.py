"""Frozen evaluation specification and dependency-aware job planning."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from .baselines import baseline_ids
from .contract import build_contract, validate_contract, _partitions, _front_ids
from .pilot import sha256

FAMILIES = ('BioGrid', 'InSilico', 'GRNdb', 'dream4', 'eipo-modular', 'scale-free')
ABLATIONS = ('objectives_only', 'technique_weights_only', 'expression_only', 'network_only',
             'no_objectives', 'no_technique_weights', 'no_expression', 'no_network')
FIXED_CONFIG = dict(depth=6, learning_rate=0.03, l2_leaf_reg=3, iterations=3000, use_best_model=False)
FIT_COUNTS = dict(tuning=720, outer=200, learning_curve=200, ablation=80,
                  family=24, deployment_tuning=144, deployment=8)
STATISTICS = dict(primary='Regret@5', secondary='Regret@1', bootstrap_samples=10000,
                  bootstrap_seed=1401, interval='conditional_topology_percentile_95',
                  test='two_sided_wilcoxon_pratt_asymptotic_continuity',
                  multiplicity='Holm_jointly_over_six_supervised_comparisons',
                  secondary_heuristic_tests=False, interpretation='exploratory_internal_evaluation')
FAILURES = dict(policy='stop_on_failure_no_outcome_exclusions', automatic_retries=0,
                resume='verify_contract_sources_environment_and_completed_artifacts',
                explicit_retry='new_attempt_preserving_previous_attempts',
                per_job_timeout_seconds=3600, total_budget_seconds=345600)


def build_evaluation_contract(repo_root: Path) -> dict:
    root = Path(repo_root).resolve()
    base = build_contract(root)
    metadata = json.loads((root / 'docs/experiments/groups/topology_groups.json').read_text())
    cases = {}
    for name in base['cases']:
        features = json.loads((root / f'docs/experiments/contracts/{name}_feature_columns.json').read_text())
        cases[name] = {k: features[k] for k in ('feature_sets', 'objective_columns', 'objective_directions')}
        cases[name]['baseline_ids'] = baseline_ids(features['objective_columns'])
    paths = set(base['source_hashes'])
    paths.update(p.relative_to(root).as_posix() for p in
                 (root / 'scripts/experiments/amiga_exp/grouped_validation').glob('*.py'))
    paths.update(['scripts/experiments/amiga_exp/decision_baselines.py',
                  'docs/experiments/grouped-validation.md'])
    result = dict(schema_version=2, status='frozen_evaluation', split_contract=base,
                  cases=cases, family_by_front={str(r['front_id']): r['family'] for r in metadata['fronts']},
                  families=list(FAMILIES), ablations=list(ABLATIONS),
                  fixed_config=FIXED_CONFIG.copy(), statistics=STATISTICS.copy(), failures=FAILURES.copy(),
                  planned_fit_counts=FIT_COUNTS.copy(),
                  source_hashes={p: sha256(root / p) for p in sorted(paths)})
    validate_evaluation_contract(result)
    return result


def validate_evaluation_contract(contract: dict) -> dict:
    if contract.get('schema_version') != 2 or contract.get('status') != 'frozen_evaluation':
        raise ValueError('A frozen grouped evaluation contract is required')
    validate_contract(contract['split_contract'])
    for key, expected in [('families', list(FAMILIES)), ('ablations', list(ABLATIONS)),
                          ('fixed_config', FIXED_CONFIG), ('statistics', STATISTICS),
                          ('failures', FAILURES), ('planned_fit_counts', FIT_COUNTS)]:
        if contract.get(key) != expected:
            raise ValueError(f'Evaluation policy differs from the specification: {key}')
    base = contract['split_contract']
    if set(contract['cases']) != set(base['cases']) or set(contract['family_by_front']) != set(base['topology_by_front']):
        raise ValueError('Evaluation case or family coverage differs')
    if not set(FAMILIES) <= set(contract['family_by_front'].values()):
        raise ValueError('A planned family exclusion has no benchmark fronts')
    for case, info in contract['cases'].items():
        if info['feature_sets']['full'] != base['cases'][case]['feature_columns']:
            raise ValueError('Full predictors differ from the split contract')
        for subset in ABLATIONS:
            columns = info['feature_sets'][subset]
            if not columns or len(columns) != len(set(columns)) or not set(columns) <= set(info['feature_sets']['full']):
                raise ValueError('Invalid ablation feature set')
        if info['baseline_ids'] != baseline_ids(info['objective_columns']):
            raise ValueError('Baseline list differs from objective definitions')
        if not set(info['objective_columns']) <= set(info['feature_sets']['full']):
            raise ValueError('Objective column is not a predictor')
        if any(info['objective_directions'].get(o) not in ('minimize', 'maximize') for o in info['objective_columns']):
            raise ValueError('Invalid objective direction')
    for path, digest in base['source_hashes'].items():
        if contract['source_hashes'].get(path) != digest:
            raise ValueError('Source identity differs from split contract')
    required = {f'scripts/experiments/amiga_exp/grouped_validation/{p.name}'
                for p in Path(__file__).parent.glob('*.py')}
    required.update(['scripts/experiments/amiga_exp/decision_baselines.py',
                     'docs/experiments/grouped-validation.md'])
    if not required <= set(contract['source_hashes']):
        raise ValueError('Evaluation source hashes are incomplete')
    for path, digest in contract['source_hashes'].items():
        if Path(path).is_absolute() or '..' in Path(path).parts or len(digest) != 64:
            raise ValueError('Invalid source hash entry')
    return dict(passed=True, planned_fits=sum(FIT_COUNTS.values()), cases=len(contract['cases']))


def freeze_contract(path: Path, contract: dict) -> None:
    validate_evaluation_contract(contract)
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError('Refusing to replace a different frozen contract; choose a new path')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(json.dumps(contract, sort_keys=True, indent=2, allow_nan=False) + '\n')


def build_plan(contract: dict) -> list[dict]:
    validate_evaluation_contract(contract)
    base = contract['split_contract']
    mapping, grid = base['topology_by_front'], base['grid']
    all_fronts, all_topologies = sorted(map(int, mapping)), sorted(set(mapping.values()))
    jobs = []

    def add(case, stage, name, train, evaluation, arm=None, config=None, seed=1101,
            dependencies=(), feature_set='full', **extra):
        job = dict(id=f'{case}/{stage}/{name}', case=case, stage=stage, arm=arm,
                   train_front_ids=list(train), evaluation_front_ids=list(evaluation),
                   config=config, seed=seed, feature_set=feature_set,
                   dependencies=list(dependencies), **extra)
        jobs.append(job)
        return job['id']

    def tune(case, stage, prefix, folds, arm):
        return [add(case, stage, f'{prefix}/{arm}/{cfg["id"]}/inner-{inner["fold"]}',
                    inner['train_front_ids'], inner['validation_front_ids'], arm, cfg,
                    config_id=cfg['id'], inner_fold=inner['fold'])
                for cfg in grid for inner in folds]

    for case in contract['cases']:
        add(case, 'baselines', 'all', [], all_fronts)
        for outer in base['outer_folds']:
            fold = outer['fold']
            for arm in base['arms']:
                dependencies = tune(case, 'tuning', f'outer-{fold}', outer['inner_folds'], arm)
                for seed in base['seeds']['final']:
                    add(case, 'outer', f'{fold}/{arm}/seed-{seed}', outer['train_front_ids'],
                        outer['test_front_ids'], arm, seed=seed, dependencies=dependencies, outer_fold=fold)
                if arm == 'ltr_catboost':
                    for subset in ABLATIONS:
                        add(case, 'ablation', f'{fold}/{subset}', outer['train_front_ids'], outer['test_front_ids'],
                            arm, seed=1201, dependencies=dependencies, feature_set=subset, outer_fold=fold)
            for arm in ('ltr_catboost', 'reg_normalized'):
                for subset in outer['learning_subsets']:
                    size = 'full' if subset['seed'] is None else str(subset['size'])
                    add(case, 'learning_curve', f'{fold}/{arm}/size-{size}/subset-{subset["seed"]}',
                        subset['front_ids'], outer['test_front_ids'], arm, FIXED_CONFIG.copy(), 1201,
                        outer_fold=fold, training_size=size, n_training_topologies=subset['size'],
                        subset_seed=subset['seed'])
        for family in FAMILIES:
            evaluation = [f for f in all_fronts if contract['family_by_front'][str(f)] == family]
            held = {mapping[str(f)] for f in evaluation}
            train = [f for f in all_fronts if mapping[str(f)] not in held]
            for arm in ('ltr_catboost', 'reg_normalized'):
                add(case, 'family', f'{family}/{arm}', train, evaluation, arm, FIXED_CONFIG.copy(), 1201, family=family)
        deployment_folds = []
        for i, valid in enumerate(_partitions(all_topologies, base['seeds']['deployment_split'], 3)):
            deployment_folds.append(dict(fold=i, train_front_ids=_front_ids(mapping, sorted(set(all_topologies) - set(valid))),
                                         validation_front_ids=_front_ids(mapping, valid)))
        for arm in base['arms']:
            dependencies = tune(case, 'deployment_tuning', 'all', deployment_folds, arm)
            add(case, 'deployment', arm, all_fronts, [], arm, seed=1201, dependencies=dependencies)
    counts = Counter(j['stage'] for j in jobs if j['stage'] != 'baselines')
    if dict(counts) != FIT_COUNTS or len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('Unexpected job counts or duplicate job IDs')
    known = {j['id']: j for j in jobs}
    for job in jobs:
        train, evaluation = set(job['train_front_ids']), set(job['evaluation_front_ids'])
        if {mapping[str(f)] for f in train} & {mapping[str(f)] for f in evaluation}:
            raise ValueError(f'Topology leakage in planned job {job["id"]}')
        if job['dependencies']:
            covered = set()
            for dependency in job['dependencies']:
                upstream = known[dependency]
                if upstream['case'] != job['case'] or upstream['arm'] != job['arm']:
                    raise ValueError('Configuration dependency has a different case/formulation')
                if not (set(upstream['train_front_ids']) | set(upstream['evaluation_front_ids'])) <= train:
                    raise ValueError('Configuration selection uses data outside training')
                covered.update(upstream['evaluation_front_ids'])
            if covered != train:
                raise ValueError('Inner validation does not cover the complete training complement')
    return jobs
