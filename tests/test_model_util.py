import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from operator import add
from functools import reduce
from moseq2_viz.model.util import (
    _get_transitions, calculate_syllable_usage, compress_label_sequence, find_label_transitions,
    get_syllable_statistics, parse_model_results
    )


@pytest.fixture(autouse=True)
def to_test_dir():
    curdir = os.getcwd()
    pth = Path(__file__).parent
    os.chdir(pth)
    yield pth
    os.chdir(curdir)


def make_sequence(lbls, durs):
    arr = [[x] * y for x, y in zip(lbls, durs)]
    return np.array(reduce(add, arr))


def test_get_transitions():
    true_labels = [1, 2, 4, 1, 5]
    durs = [3, 4, 2, 6, 7]
    arr = make_sequence(true_labels, durs)

    trans, locs = _get_transitions(arr)

    assert true_labels[1:] == list(trans), 'syllable labels do not match with the transitions'
    assert list(np.diff(locs)) == durs[1:-1], 'syllable locations do not match their durations'


def test_find_label_transitions():
    lbls = [-5, 1, 3, 1, 4]
    durs = [3, 4, 10, 4, 12]
    arr = make_sequence(lbls, durs)

    inds = find_label_transitions(arr)

    assert list(inds) == list(np.cumsum(durs[:-1])), 'label indices do not align'


def test_compress_label_sequence():
    lbls = [-5, 1, 3, 1, 4]
    durs = [3, 4, 10, 4, 12]
    arr = make_sequence(lbls, durs)

    compressed = compress_label_sequence(arr)

    assert lbls[1:] == list(compressed), 'compressed sequence does not match original'


def test_calculate_syllable_usage():
    true_labels = [-5, 1, 3, 1, 2, 4, 1, 5]
    durs = [3, 3, 4, 2, 6, 7, 2, 4]
    arr = make_sequence(true_labels, durs)

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


def test_get_syllable_statistics(to_test_dir):
    # For now this just tests if there are any function-related errors
    model_results = parse_model_results('my_model.p')
    usages, durations = get_syllable_statistics(model_results['labels'])