from collections import defaultdict, OrderedDict
from copy import deepcopy
from sklearn.cluster import KMeans
from moseq2_viz.util import h5_to_dict
import numpy as np
import h5py
import pandas as pd
import warnings
import tqdm
import joblib
import os
import pytest

def test_get_transitions():
    # original params: label_sequence, fill_value = -5
    filename =  h5py.File('tests/test_files/_pca/pca_scores.h5', 'r')
    labels = []
    for k,v in filename.items():
        labels = filename[k]

    arr = deepcopy(np.array(labels))
    locs = np.where(arr[1:] != arr[:-1])[0] + 1
    transitions = arr[locs]
    assert (locs != None)
    assert (transitions != None)


def test_whiten_all():
    # original params: pca_scores, center = True
    pca_scores = h5py.File('tests/test_files/_pca/pca_scores.h5', 'r')
    center = True
    print(pca_scores)


    valid_scores = np.concatenate([x[~np.isnan(x).any(axis=1), :] for x in pca_scores.values()])
    mu, cov = valid_scores.mean(axis=0), np.cov(valid_scores, rowvar=False, bias=1)

    L = np.linalg.cholesky(cov)

    if center:
        offset = 0
    else:
        offset = mu

    whitened_scores = deepcopy(pca_scores)

    for k, v in whitened_scores.items():
        whitened_scores[k] = np.linalg.solve(L, (v - mu).T).T + offset

    assert(whitened_scores != None)

def test_whiten_pcs():
    # original params: pca_scores, method='all', center=True
    pytest.fail('not implemented')

def test_normalize_pcs():
    # original params: pca_scores, method='z'
    pytest.fail('not implemented')

def test_retrieve_from_slices():
    # original params: slices, pca_scores, max_dur=60,
#                              max_samples=100, npcs=10, subsampling=None,
#                              remove_offset=False, **kwargs
    pytest.fail('not implemented')

def test_get_transition_matrix():
    # original params: labels, max_syllable=100, normalize='bigram',
#                           smoothing=0.0, combine=False, disable_output=False
    pytest.fail('not implemented')

def test_get_syllable_slices():
    # original params: syllable, labels, label_uuids, index, trim_nans=True
    pytest.fail('not implemented')

def test_get_syllable_statistics():
    # original params: data, fill_value=-5, max_syllable=100, count='usage'
    pytest.fail('not implemented')

def test_labels_to_changepoints():
    # original params: labels, fs=30
    pytest.fail('not implemented')

def test_parse_batch_modeling():
    # original param: filename
    pytest.fail('not implemented')

def test_parse_model_results():
    # original params: model_obj, restart_idx=0, resample_idx=-1,
#                         map_uuid_to_keys=False,
#                         sort_labels_by_usage=False,
#                         count='usage'
    pytest.fail('not implemented')

def test_relabel_by_usage():
    # original params: labels, fill_value=-5, count='usage'
    pytest.fail('not implemented')

def test_results_to_dataframe():
    # original params: model_dict, index_dict, sort=False, count='usage', normalize=True, max_syllable=40,
#                          include_meta=['SessionName', 'SubjectName', 'StartTime']
    pytest.fail('not implemented')

def test_simulate_ar_trajectory():
    # original params: ar_mat, init_points=None, sim_points=100
    pytest.fail('not implemented')

def test_sort_batch_results():
    # original params: data, averaging=True, filenames=None, **kwargs
    pytest.fail('not implemented')