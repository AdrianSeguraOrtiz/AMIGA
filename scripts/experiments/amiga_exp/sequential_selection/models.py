"""Common bounded adapters and training-only tree-contribution selection."""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, CatBoostRegressor, Pool
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
from xgboost import DMatrix, XGBClassifier, XGBRanker, XGBRegressor

from amiga.selection.learn2rank import LabelMode, build_labels
from scripts.experiments.amiga_exp.grouped_validation.training_data import prepare_frame

FAMILIES = ('LightGBM', 'XGBoost', 'CatBoost')
ARMS = ('ranking', 'reg_aupr', 'reg_normalized', 'clf_top20')
FRACTIONS = (1.0, .75, .5, .25)
LABELS = ('rank_dense', 'rank_avg', 'continuous', 'quantiles_q5',
          'quantiles_q10', 'quantiles_q15', 'reversed', 'shuffled')


def prepare(data, case, mapping, features, label='rank_dense', seed=1101):
    """Receive training rows only; all label transformations are local to them."""
    result = prepare_frame(data, case, mapping, features)
    frame = pd.DataFrame(dict(front_id=result['group_id'], item_id=result['item_id'],
                              AUPR=result['labels']['reg_aupr']))
    mode, quantiles = (('quantiles', int(label.split('_q')[1])) if label.startswith('quantiles_q')
                       else (label, 20))
    result['labels']['ranking'] = build_labels(frame, 'front_id', 'AUPR', LabelMode(mode),
                                               quantiles, random_state=seed)
    if not np.isfinite(result['labels']['ranking']).all():
        raise ValueError('Nonfinite ranking relevance')
    result['report']['rank_label_mode'] = label
    result['report']['label_seed'] = seed
    return result


def rank_labels(values, family):
    """Use training-only linear integer gains in libraries requiring integers."""
    values = np.asarray(values)
    if family == 'CatBoost':
        return values
    if not np.issubdtype(values.dtype, np.integer):
        if values.min() < 0 or values.max() > 1:
            raise ValueError('Floating relevance must lie in [0, 1]')
        return np.floor(values * 255).astype(int)
    if values.min() < 0:
        raise ValueError('Ranking relevance must be nonnegative')
    return values.astype(int)


def fit(data, family, arm, params, *, seed=1101, threads=8):
    if family not in FAMILIES or arm not in ARMS:
        raise ValueError('Unknown family/formulation')
    params = {k: v for k, v in params.items() if k != 'id'}
    iterations = int(params.pop('iterations'))
    if iterations < 1 or not 1 <= threads <= 8:
        raise ValueError('Invalid training resources')
    X = data['X']
    y = rank_labels(data['labels'][arm], family) if arm == 'ranking' else data['labels'][arm]
    ids = data['group_id']
    _, starts, counts = np.unique(ids, return_index=True, return_counts=True)
    if not np.array_equal(np.repeat(ids[starts], counts), ids):
        raise ValueError('Training queries must be contiguous and sorted')
    if family == 'CatBoost':
        estimator = (CatBoostRanker(loss_function='YetiRank') if arm == 'ranking' else
                     CatBoostClassifier(loss_function='Logloss') if arm == 'clf_top20' else
                     CatBoostRegressor(loss_function='RMSE'))
        estimator.set_params(**params, iterations=iterations, random_seed=seed,
                             thread_count=threads, task_type='CPU', use_best_model=False,
                             allow_writing_files=False, verbose=False)
        extra = (dict(group_id=ids, group_weight=data['group_weight']) if arm == 'ranking'
                 else dict(weight=data['point_weight']))
        estimator.fit(Pool(X, label=y, thread_count=threads, **extra))
        actual = estimator.tree_count_
    elif family == 'XGBoost':
        cls = XGBRanker if arm == 'ranking' else XGBClassifier if arm == 'clf_top20' else XGBRegressor
        objective = 'rank:ndcg' if arm == 'ranking' else 'binary:logistic' if arm == 'clf_top20' else 'reg:squarederror'
        extra = dict(ndcg_exp_gain=False) if arm == 'ranking' else {}
        estimator = cls(**params, **extra, objective=objective, n_estimators=iterations,
                        random_state=seed, n_jobs=threads, tree_method='hist', device='cpu',
                        verbosity=0)
        extra = (dict(group=counts, sample_weight=data['group_weight'][starts]) if arm == 'ranking'
                 else dict(sample_weight=data['point_weight']))
        estimator.fit(X, y, verbose=False, **extra)
        actual = estimator.get_booster().num_boosted_rounds()
    else:
        cls = LGBMRanker if arm == 'ranking' else LGBMClassifier if arm == 'clf_top20' else LGBMRegressor
        objective = 'lambdarank' if arm == 'ranking' else 'binary' if arm == 'clf_top20' else 'regression'
        extra = dict(label_gain=list(range(int(y.max()) + 1))) if arm == 'ranking' else {}
        estimator = cls(**params, **extra, objective=objective, n_estimators=iterations,
                        random_state=seed, n_jobs=threads, deterministic=True,
                        force_col_wise=True, verbosity=-1)
        # LightGBM exposes row weights; repeat the query weight over its rows.
        extra = (dict(group=counts, sample_weight=data['group_weight']) if arm == 'ranking'
                 else dict(sample_weight=data['point_weight']))
        estimator.fit(X, y, **extra)
        actual = estimator.booster_.current_iteration()
    # LightGBM may exhaust admissible splits: record it, without treating it as
    # validation-driven early stopping or excluding the resulting configuration.
    if actual > iterations or (family != 'LightGBM' and actual != iterations):
        raise ValueError('Estimator violated the requested iteration budget')
    return estimator, dict(requested_iterations=iterations, actual_iterations=int(actual),
                          no_validation_set=True, early_stopping=False,
                          fewer_iterations_due_to_no_splits=bool(actual < iterations))


def scores(model, family, arm, X, threads=8):
    X = np.asarray(X, dtype=float)
    if family == 'CatBoost':
        result = (model.predict_proba(X, thread_count=threads)[:, 1] if arm == 'clf_top20'
                  else model.predict(X, thread_count=threads))
    elif family == 'LightGBM':
        result = model.booster_.predict(X, num_threads=threads)
    else:
        result = model.predict_proba(X)[:, 1] if arm == 'clf_top20' else model.predict(X)
    result = np.asarray(result, dtype=float)
    if result.shape != (len(X),) or not np.isfinite(result).all():
        raise ValueError('Invalid prediction scores')
    return result


def contributions(model, family, X, threads=8):
    """Return native tree SHAP contributions, including the last bias column."""
    if family == 'CatBoost':
        values = model.get_feature_importance(Pool(X, thread_count=threads), type='ShapValues',
                                             shap_calc_type='Regular', thread_count=threads)
    elif family == 'LightGBM':
        values = model.booster_.predict(X, pred_contrib=True, num_threads=threads)
    else:
        values = model.get_booster().predict(DMatrix(X, nthread=threads), pred_contribs=True)
    values = np.asarray(values, dtype=float)
    if values.shape != (len(X), X.shape[1] + 1) or not np.isfinite(values).all():
        raise ValueError('Invalid SHAP contribution matrix')
    return values


def sample_rows(data, maximum=32, seed=1501):
    """Uniform deterministic sampling within training queries, without labels."""
    selected = []
    for front in np.unique(data['group_id']):
        indices = np.flatnonzero(data['group_id'] == front)
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(front)]))
        selected.extend(sorted(rng.choice(indices, min(maximum, len(indices)), replace=False).tolist()))
    return np.asarray(selected, dtype=int)


def feature_importance(model, family, data, *, threads=8, maximum=32):
    sampled = sample_rows(data, maximum)
    values = contributions(model, family, data['X'][sampled], threads)[:, :-1]
    gids = data['group_id'][sampled]
    mean_abs, centered, weights = [], [], []
    for front in np.unique(gids):
        mask = gids == front
        block = values[mask]
        mean_abs.append(np.abs(block).mean(axis=0))
        centered.append(np.abs(block - block.mean(axis=0)).mean(axis=0))
        weights.append(float(data['group_weight'][sampled[mask][0]]))
    result = pd.DataFrame(dict(feature=data['feature_names'],
                               mean_abs_shap=np.average(mean_abs, axis=0, weights=weights),
                               centered_mean_abs_shap=np.average(centered, axis=0, weights=weights)))
    return result, dict(sampled_rows=len(sampled), sampled_fronts=len(weights),
                        maximum_rows_per_front=maximum, sampling_seed=1501,
                        scope='training_only', selection_statistic='centered_mean_abs_shap')


def feature_path(prepared, family, arm, params, *, seed=1101, threads=8,
                 fractions=FRACTIONS, callback=None):
    """Fit nested 100/75/50/25% models; SHAP uses training data only.

    callback observes each fitted model (e.g. exports inner-validation metrics)
    but its return value is ignored and cannot guide recursive elimination.
    """
    all_features = list(prepared['feature_names'])
    sizes = [math.ceil(len(all_features) * f) for f in fractions]
    if sizes[0] != len(all_features) or any(b > a for a, b in zip(sizes, sizes[1:])):
        raise ValueError('Feature path must decrease from the full representation')
    current = list(range(len(all_features)))
    reports, importances = [], []
    for position, (fraction, size) in enumerate(zip(fractions, sizes)):
        if len(current) != size:
            raise ValueError('Unexpected feature count')
        data = dict(prepared, X=prepared['X'][:, current],
                    feature_names=[all_features[i] for i in current])
        start = time.monotonic()
        model, fit_report = fit(data, family, arm, params, seed=seed, threads=threads)
        report = dict(fraction=fraction, n_features=size, features=data['feature_names'],
                      fit_seconds=time.monotonic() - start, **fit_report)
        if callback is not None:
            callback(model, data, fraction)
        if position + 1 < len(sizes):
            start = time.monotonic()
            importance, sampling = feature_importance(model, family, data, threads=threads)
            report.update(shap_seconds=time.monotonic() - start, sampling=sampling)
            importances.append(importance.assign(fraction=fraction))
            ordered = importance.sort_values(['centered_mean_abs_shap', 'feature'], ascending=[False, True])
            retained = set(ordered['feature'].iloc[:sizes[position + 1]])
            current = [i for i in current if all_features[i] in retained]
        reports.append(report)
    return reports, pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
