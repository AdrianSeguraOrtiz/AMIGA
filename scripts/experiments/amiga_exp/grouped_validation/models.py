"""Fixed-budget CatBoost adapters shared by grouped experiment jobs."""
from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier, CatBoostRanker, CatBoostRegressor, Pool


def fit_model(prepared, arm, params, seed, threads):
    params = {k: v for k, v in params.items() if k != 'id'}
    params.update(random_seed=int(seed), task_type='CPU', thread_count=threads,
                  allow_writing_files=False, use_best_model=False, verbose=False)
    arguments = dict(data=prepared['X'], label=prepared['labels'][arm],
                     feature_names=prepared['feature_names'], thread_count=threads)
    if arm == 'ltr_catboost':
        pool = Pool(**arguments, group_id=prepared['group_id'], group_weight=prepared['group_weight'])
        model = CatBoostRanker(loss_function='YetiRank', **params)
    else:
        pool = Pool(**arguments, weight=prepared['point_weight'])
        if arm == 'clf_top20':
            model = CatBoostClassifier(loss_function='Logloss', **params)
        elif arm in ('reg_aupr', 'reg_normalized'):
            model = CatBoostRegressor(loss_function='RMSE', **params)
        else:
            raise ValueError(f'Unknown formulation: {arm}')
    model.fit(pool)
    if model.tree_count_ != params['iterations'] or model.get_all_params().get('use_best_model') is not False:
        raise ValueError('Estimator violated the fixed training budget')
    if model.get_all_params().get('od_type') not in (None, 'None'):
        raise ValueError('Early stopping must remain disabled')
    return model


def predict_scores(model, arm, frame, features, threads):
    # Only the declared predictor matrix is ever passed to the estimator.
    matrix = frame[features].to_numpy(dtype=float)
    if np.isinf(matrix).any():
        raise ValueError('Infinite predictors are not supported')
    score = (model.predict_proba(matrix, thread_count=threads)[:, 1] if arm == 'clf_top20'
             else model.predict(matrix, thread_count=threads))
    score = np.asarray(score, dtype=float)
    if score.shape != (len(frame),) or not np.isfinite(score).all():
        raise ValueError('Estimator produced invalid scores')
    return score
