import numpy as np
import pandas as pd
from operator import add
from functools import reduce
from moseq2_viz.model.util import _get_transitions, calculate_syllable_usage


def test_get_transitions():
    true_labels = [1, 2, 4, 1, 5]
    durs = [3, 4, 2, 6, 7]
    arr = [[x] * y for x, y in zip(true_labels, durs)]
    arr = np.array(reduce(add, arr))

    trans, locs = _get_transitions(arr)

    assert true_labels[1:] == list(trans), 'syllable labels do not match with the transitions'
    assert list(np.diff(locs)) == durs[1:-1], 'syllable locations do not match their durations'


def test_calculate_syllable_usage():
    true_labels = [-5, 1, 3, 1, 2, 4, 1, 5]
    durs = [3, 3, 4, 2, 6, 7, 2, 4]
    arr = [[x] * y for x, y in zip(true_labels, durs)]
    arr = np.array(reduce(add, arr))

    labels_dict = {
        'animal1': arr,
        'animal2': arr
    }

    test_result = {
        1: 6,
        2: 2,
        3: 2,
        4: 2,
        5: 2
    }

    result = calculate_syllable_usage(labels_dict)
    
    assert test_result == result, 'syll usage calculation incorrect for dict'

    df = pd.DataFrame({'syllable': true_labels[1:] + true_labels[1:], 'dur': durs[1:] + durs[1:]})

    result = calculate_syllable_usage(df)

    assert test_result == result, 'syll usage calculation incorrect for dataframe'
