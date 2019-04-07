import os
import h5py
from ruamel.yaml import YAML
from cytoolz import curry
from cytoolz.itertoolz import peek, pluck, unique, first, groupby
from cytoolz.dicttoolz import valmap, valfilter, keyfilter, merge_with, dissoc, assoc
import numpy as np
import re


# https://gist.github.com/jaytaylor/3660565
_underscorer1 = re.compile(r'(.)([A-Z][a-z]+)')
_underscorer2 = re.compile(r'([a-z0-9])([A-Z])')


def _load_yaml(yaml_path: str):
    yaml = YAML(typ='safe')
    with open(yaml_path, 'r') as f:
        loaded = yaml.load(f)
    return loaded


def camel_to_snake(s):
    """ Converts CamelCase to snake_case
    """
    subbed = _underscorer1.sub(r'\1_\2', s)
    return _underscorer2.sub(r'\1_\2', subbed).lower()


def check_video_parameters(index: dict) -> dict:
    ''' Iterates through each extraction parameter file to verify extraction parameters
    were the same. If they weren't this function raises a RuntimeError. Otherwise, it
    will return a dictionary containing the following parameters:
        crop_size, fps, max_height, min_height, resolution
    Args:
        index: a `sorted_index` dictionary of extraction parameters
    Returns:
        a dictionary with a subset of the used extraction parameters
    '''

    # define constants
    check_parameters = ['crop_size', 'fps', 'max_height', 'min_height']

    ymls = [v['path'][1] for v in index['files'].values()]

    # load yaml config files when needed
    dicts = map(_load_yaml, ymls)
    # get the parameters key within each dict
    params = pluck('parameters', dicts)

    first_entry, params = peek(params)
    if 'resolution' in first_entry:
        check_parameters += ['resolution']
    
    # filter for only keys in check_parameters
    params = map(curry(keyfilter)(lambda k: k in check_parameters), params)
    # turn lists (in the dict values) into tuples
    params = map(curry(valmap)(lambda x: tuple(x) if isinstance(x, list) else x), params)

    # get unique parameter values
    vid_parameters = merge_with(set, params)

    incorrect_parameters = valfilter(lambda x: len(x) > 1, vid_parameters)

    # if there are multiple values for a parameter, raise error
    if incorrect_parameters:
        raise RuntimeError(f'The following parameters are not equal across extractions: {incorrect_parameters.keys()}')

    # grab the first value in the set
    vid_parameters = valmap(first, vid_parameters)

    # update resolution
    if 'resolution' in vid_parameters:
        vid_parameters['resolution'] = tuple(x + 100 for x in vid_parameters['resolution'])
    else:
        vid_parameters['resolution'] = None

    return vid_parameters


def clean_dict(dct):

    def clean_entry(e):
        if isinstance(e, dict):
            out = clean_dict(e)
        elif isinstance(e, np.ndarray):
            out = e.tolist()
        elif isinstance(e, np.generic):
            out = np.asscalar(e)
        else:
            out = e
        return out

    return valmap(clean_entry, dct)


def _load_h5_to_dict(file: h5py.File, path: str) -> dict:
    ans = {}
    for key, item in file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = _load_h5_to_dict(file, '/'.join([path, key]))
    return ans


def h5_to_dict(h5file, path: str) -> dict:
    '''
    Args:
        h5file (str or h5py.File): file path to the given h5 file or the h5 file handle
        path: path to the base dataset within the h5 file
    Returns:
        a dict with h5 file contents with the same path structure
    '''
    if isinstance(h5file, str):
        with h5py.File(h5file, 'r') as f:
            out = _load_h5_to_dict(f, path)
    elif isinstance(h5file, h5py.File):
        out = _load_h5_to_dict(h5file, path)
    else:
        raise Exception('file input not understood - need h5 file path or file object')
    return out


def load_changepoints(cpfile):
    with h5py.File(cpfile, 'r') as f:
        cps = h5_to_dict(f, 'cps')

    cp_dist = []

    for k, v in cps.items():
        cp_dist.append(np.diff(v.squeeze()))

    return np.concatenate(cp_dist)


def load_timestamps(timestamp_file, col=0):
    """Read timestamps from space delimited text file
    """

    ts = []
    with open(timestamp_file, 'r') as f:
        for line in f:
            cols = line.split()
            ts.append(float(cols[col]))

    return np.array(ts)


def parse_index(index_file: str) -> tuple:
    ''' Load an index file, and use extraction UUIDs as entries in a sorted index.

    Returns:
        a tuple containing the loaded index file, and the index with extraction UUIDs as entries
    '''

    join = os.path.join
    index_dir = os.path.dirname(index_file)

    index = _load_yaml(index_file)
    files = index['files']

    sorted_index = groupby('uuid', files)
    # grab first entry in list
    sorted_index = valmap(first, sorted_index)
    # remove redundant uuid entry
    sorted_index = valmap(lambda d: dissoc(d, 'uuid'), sorted_index)
    # tuple-ize the path entry, join with the index file dirname
    sorted_index = valmap(lambda d: assoc(d, 'path', tuple(join(index_dir, x) for x in d['path'])),
                          sorted_index)

    uuid_sorted = {
        'files': sorted_index,
        'pca_path': join(index_dir, index['pca_path'])
    }

    return index, uuid_sorted


def recursive_find_h5s(root_dir=os.getcwd(),
                       ext='.h5',
                       yaml_string='{}.yaml'):
    """Recursively find h5 files, along with yaml files with the same basename
    """
    dicts = []
    h5s = []
    yamls = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            yaml_file = yaml_string.format(os.path.splitext(file)[0])
            if file.endswith(ext) and os.path.exists(os.path.join(root, yaml_file)):
                with h5py.File(os.path.join(root, file), 'r') as f:
                    if 'frames' not in f.keys():
                        continue
                h5s.append(os.path.join(root, file))
                yamls.append(os.path.join(root, yaml_file))
                dicts.append(read_yaml(os.path.join(root, yaml_file)))

    return h5s, dicts, yamls


def read_yaml(yaml_file):

    yaml = YAML(typ='safe')

    with open(yaml_file, 'r') as f:
        dat = f.read()
        try:
            return_dict = yaml.load(dat)
        except yaml.constructor.ConstructorError:
            return_dict = yaml.load(dat)

    return return_dict


# from https://stackoverflow.com/questions/40084931/taking-subarrays-from-numpy-array-with-given-stride-stepsize/40085052#40085052
# dang this is fast!
def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))
