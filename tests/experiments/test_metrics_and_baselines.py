"""Exact score-tie metrics, objective selectors and paired topology statistics."""
from itertools import permutations
import numpy as np
import pandas as pd
import pytest
from scripts.experiments.amiga_exp.grouped_validation.baselines import knee_scores, score_front
from scripts.experiments.amiga_exp.grouped_validation.metrics import front_metrics, select_configuration
from scripts.experiments.amiga_exp.statistics import holm, paired_difference


@pytest.mark.parametrize('scores', [[0, 0, 0, 0], [2, 1, 1, 0], [3, 2, 1, 0]])
def test_tie_expectations_match_exhaustive_permutations(scores):
    target = np.array([0.1, 0.3, 0.8, 0.8])
    frame = pd.DataFrame(dict(front_id=1, item_id=range(4), AUPR=target))
    result = front_metrics(frame, scores).iloc[0]
    valid = [p for p in permutations(range(4)) if all(scores[p[i]] >= scores[p[i+1]] for i in range(3))]
    for k in (1, 3, 5, 10):
        best = np.array([target[list(p[:k])].max() for p in valid])
        assert result[f'BestAUPR@{k}'] == pytest.approx(best.mean())
        assert result[f'Hit@{k}'] == pytest.approx(np.isclose(best, target.max()).mean())
        assert result[f'Regret@{k}'] == pytest.approx(target.max() - best.mean())
    single = front_metrics(frame.iloc[:1], [0]).iloc[0]
    assert single['Regret@10'] == 0 and single['Hit@10'] == 1


@pytest.mark.parametrize('dimensions', [2, 3, 6])
def test_knee_retains_duplicates_constants_and_dominance(dimensions):
    # A pronounced knee beats either endpoint; duplicate candidates stay tied.
    matrix = np.array([[0, 1], [.2, .2], [1, 0], [.2, .2], [1, 1]])
    matrix = np.pad(matrix, [(0, 0), (0, dimensions - 2)])
    scores, report = knee_scores(pd.DataFrame(matrix))
    assert scores[1] == scores[3] > scores[0] == scores[2] > scores[4]
    assert report['constant_objectives'] == dimensions - 2
    assert report['duplicate_vectors'] == 1
    assert report['dominated_candidates'] == 1
    flat, _ = knee_scores(pd.DataFrame(np.zeros((4, dimensions))))
    assert np.array_equal(flat, np.zeros(4))


def test_borda_equivalence_and_label_free_baselines():
    data = pd.DataFrame(dict(front_id=1, item_id=range(4), a=[0, 2, 1, 1], b=[4, 1, 2, 2]))
    scores, _ = score_front(data, ['a', 'b'], dict(a='minimize', b='maximize'))
    borda = ((len(data) - data['a'].rank(method='average', ascending=True)) +
             (len(data) - data['b'].rank(method='average', ascending=False)))
    assert np.array_equal(pd.Series(scores['objective_mean_rank']).rank(), borda.rank())
    assert 'oracle' not in scores
    assert (scores['random_uniform'] == 0).all()


def test_selection_weights_topologies_and_uses_secondary_ties():
    # Three conditions from the same topology must not outweigh two topologies.
    mapping = {'1':'a', '2':'a', '3':'a', '4':'b', '5':'c'}
    a = pd.DataFrame({'front_id':range(1, 6), 'Regret@5':[0, 0, 0, .6, .6], 'Regret@1':.9})
    b = pd.DataFrame({'front_id':range(1, 6), 'Regret@5':[.5, .5, .5, 0, 0], 'Regret@1':.9})
    assert a['Regret@5'].mean() < b['Regret@5'].mean()
    assert select_configuration({'a':a, 'b':b}, mapping)[0] == 'b'
    assert select_configuration({'a':b, 'b':b.assign(**{'Regret@1':.2})}, mapping)[0] == 'b'
    assert select_configuration({'b':b, 'a':b}, mapping)[0] == 'a'


def test_paired_statistics_zero_differences_and_holm():
    result = paired_difference([.1, .2, .3], [.1, .2, .3], samples=100)
    assert result['ci_low'] == result['ci_high'] == result['mean_difference'] == 0
    assert result['tied'] == 3
    assert holm([.03, .001, .9]).tolist() == pytest.approx([.06, .003, .9])
