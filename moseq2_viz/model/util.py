from collections import defaultdict, OrderedDict
from copy import deepcopy
from moseq2_viz.util import recursive_find_h5s
import numpy as np
import os


def sort_results(data, averaging=False, **kwargs):

    parameters = np.hstack(kwargs.values())
    param_sets = np.unique(parameters, axis=0)
    param_dict = {k: np.unique(v) for k, v in kwargs.items()}

    param_list = list(param_dict.values())
    new_shape = tuple([len(v) for v in param_list])

    new_matrix = np.zeros(new_shape, dtype=data.dtype)
    new_count = np.zeros(new_shape, dtype=data.dtype)

    for param in param_sets:
        row_matches = np.where((parameters == param).all(axis=1))[0]
        idx = np.zeros((len(param),), dtype='int')

        for i, p in enumerate(param):
            if np.isnan(p):
                idx[i] = -1
            else:
                idx[i] = int(np.where(param_list[i] == p)[0])

        for row in row_matches:
            if (averaging and idx[0] > 0 and idx[1] > 0) and ~np.isnan(data[row]):
                    new_matrix[idx[0], idx[1]] += data[row]
                    new_count[idx[0], idx[1]] += 1
            elif idx[0] > 0 and idx[1] > 0:
                new_matrix[idx[0], idx[1]] = data[row]

    if averaging:
        new_matrix /= new_count

    return new_matrix, param_dict


def get_syllable_statistics(data, fill_value=-5):

    seq_array = np.empty_like(data)
    usages = defaultdict(int)
    durations = defaultdict(list)

    for i, v in np.ndenumerate(data):

        v = np.insert(v, len(v), -10)
        idx = np.where(v[1:] != v[:-1])[0]+1
        seq_array[i] = v[idx][:-1]
        durs = np.diff(idx)

        for s, d in zip(seq_array[i], durs):

            usages[s] += 1
            durations[s].append(d)

    usages = OrderedDict(sorted(usages.items()))
    durations = OrderedDict(sorted(durations.items()))

    return usages, durations


def relabel_by_usage(labels):

    sorted_labels = deepcopy(labels)
    usages, durations = get_syllable_statistics(labels)
    sorting = []

    for w in sorted(usages, key=usages.get, reverse=True):
        sorting.append(w)

    for i, v in np.ndenumerate(labels):
        for j, idx in enumerate(sorting):
            sorted_labels[i][np.where(v == idx)] = j

    return sorted_labels


# return tuples with uuid and syllable indices
def get_syllable_slices(syllable, labels, label_uuids, trim_nans=True,
                        data_dir=os.getcwd(), pca_file=os.path.join(os.getcwd(), '_pca/pca_scores.h5')):

    h5s, dicts, yamls = recursive_find_h5s(data_dir)
    h5_uuids = [d['uuid'] for d in dicts]

    # grab the original indices from the pca file as well...

    if trim_nans:
        with h5py.File(pca_file, 'r') as f:
            score_idx = recursively_load_dict_contents_from_group(f,'scores_idx')


    label_to_h5_idx = [h5_uuids.index(uuid) for uuid in label_uuids]
    sorted_h5s = [h5s[i] for i in label_to_h5_idx]
    syllable_slices = []

    for label_arr, label_uuid, h5 in zip(labels, label_uuids, sorted_h5s):

        if trim_nans:

            idx = score_idx[label_uuid]
            if len(idx) > len(label_arr):
                idx = idx[:len(label_arr)]

            label_arr = label_arr[~np.isnan(idx)]

        match_idx = np.where(label_arr == syllable)[0]
        breakpoints = np.where(np.diff(match_idx) > 1)[0]
        if len(breakpoints) < 1:
            continue

        breakpoints = zip(np.r_[0, breakpoints+1], np.r_[breakpoints, len(breakpoints)-1])

        for i, j in breakpoints:

            syllable_slices.append([(match_idx[i],match_idx[j]), label_uuid, h5])

    return syllable_slices
