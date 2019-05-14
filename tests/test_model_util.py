import numpy as np
from operator import add
from functools import reduce
from moseq2_viz.model.util import _get_transitions


def test_get_transitions():
    true_labels = [1, 2, 4, 1, 5]
    durs = [3, 4, 2, 6, 7]
    arr = [[x] * y for x, y in zip(true_labels, durs)]
    arr = np.array(reduce(add, arr))

    trans, locs = _get_transitions(arr)

    assert true_labels[1:] == list(trans), 'syllable labels do not match with the transitions'
    assert list(np.diff(locs)) == durs[1:-1], 'syllable locations do not match their durations'
