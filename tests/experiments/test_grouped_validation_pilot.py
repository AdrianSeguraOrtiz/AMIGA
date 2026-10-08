"""Check fit compatibility, budget accounting and process cleanup on synthetic data."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRanker, CatBoostRegressor, CatBoostClassifier, Pool

from scripts.experiments.amiga_exp.grouped_validation.pilot import terminate_process_group, timing_projection
from scripts.experiments.amiga_exp.grouped_validation.training_data import prepare_frame


@pytest.mark.parametrize('arm', ['ltr_catboost', 'reg_aupr', 'reg_normalized', 'clf_top20'])
def test_catboost_accepts_contract_weights_and_fixed_tree_count(arm):
    frame = pd.DataFrame([
        dict(front_id=f, item_id=i, AUPR=(i % 6) / 6, x=i, z=f * i)
        for f in range(1, 4) for i in range(1, 13)
    ])
    prepared = prepare_frame(frame, 'BIO-INSIGHT', {1: 'a', 2: 'a', 3: 'b'}, ['x', 'z'])
    pool_args = dict(data=prepared['X'], label=prepared['labels'][arm], thread_count=1)
    params = dict(iterations=12, depth=3, learning_rate=.03, random_seed=1101,
                  thread_count=1, use_best_model=False, verbose=False, allow_writing_files=False)
    if arm == 'ltr_catboost':
        pool = Pool(**pool_args, group_id=prepared['group_id'], group_weight=prepared['group_weight'])
        model = CatBoostRanker(loss_function='YetiRank', **params)
    else:
        pool = Pool(**pool_args, weight=prepared['point_weight'])
        model = (CatBoostClassifier(loss_function='Logloss', **params) if arm == 'clf_top20'
                 else CatBoostRegressor(loss_function='RMSE', **params))
    model.fit(pool)
    assert model.tree_count_ == 12
    assert model.get_all_params()['use_best_model'] is False
    assert model.get_all_params().get('od_type') in (None, 'None')
    assert np.isfinite(model.predict(prepared['X'], thread_count=1)).all()


def test_projection_counts_all_fits_and_refuses_incomplete_pilot():
    results = [dict(status='complete', job=dict(case=c, arm=a, params=dict(depth=d)),
                    training_data=dict(usable_rows=31200), fit_seconds=60,
                    data_preparation_seconds=0, training_prediction_seconds=0)
               for c in ('BIO-INSIGHT', 'MO-GENECI')
               for a in ('ltr_catboost', 'reg_aupr', 'reg_normalized', 'clf_top20')
               for d in (4, 8)]
    report = timing_projection(results, jobs=2)
    assert sum(report['scaled_serial_hours_by_block'].values()) == pytest.approx(1376 / 60)
    assert report['scaled_serial_hours_with_25_percent_margin'] == pytest.approx(1376 / 60 * 1.25)
    assert timing_projection(results[:-1], jobs=2)['status'] == 'incomplete_measurements'


def test_cleanup_stops_separate_worker_and_tolerates_already_exited():
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                               start_new_session=True)
    try:
        terminate_process_group(process)
        assert process.poll() is not None
        terminate_process_group(process)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
