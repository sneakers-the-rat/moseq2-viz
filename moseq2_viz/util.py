import os
import h5py
import ruamel.yaml as yaml
import numpy as np
import re


# https://gist.github.com/jaytaylor/3660565
_underscorer1 = re.compile(r'(.)([A-Z][a-z]+)')
_underscorer2 = re.compile('([a-z0-9])([A-Z])')


def camel_to_snake(s):
    """Converts CamelCase to snake_case
    """
    subbed = _underscorer1.sub(r'\1_\2', s)
    return _underscorer2.sub(r'\1_\2', subbed).lower()


def check_video_parameters(index):

    ymls = [v['path'][1] for v in index.values()]

    dicts = []

    for yml in ymls:
        with open(yml, 'r') as f:
            dicts.append(yaml.load(f.read(), Loader=yaml.RoundTripLoader))

    check_parameters = ['crop_size', 'fps', 'max_height', 'min_height']

    if 'resolution' in list(dicts[0]['parameters'].keys()):
        check_parameters.append('resolution')

    for chk in check_parameters:
        tmp_list = [dct['parameters'][chk] for dct in dicts]
        if not all(x == tmp_list[0] for x in tmp_list):
            raise RuntimeError('Parameter {} not equal in all extractions'.format(chk))

    vid_parameters = {
        'crop_size': tuple(dicts[0]['parameters']['crop_size']),
        'fps': dicts[0]['parameters']['fps'],
        'max_height': dicts[0]['parameters']['max_height'],
        'min_height': dicts[0]['parameters']['min_height'],
        'resolution': None
    }

    if 'resolution' in check_parameters:
        vid_parameters['resolution'] = tuple([tmp+100 for tmp in dicts[0]['parameters']['resolution']])

    return vid_parameters


def commented_map_to_dict(cmap):

    new_var = dict()

    if type(cmap) is yaml.comments.CommentedMap or type(cmap) is dict:
        for k, v in cmap.items():
            if type(v) is yaml.comments.CommentedMap or type(v) is dict:
                new_var[k] = commented_map_to_dict(v)
            elif type(v) is np.ndarray:
                new_var[k] = v.tolist()
            elif isinstance(v, np.generic):
                new_var[k] = np.asscalar(v)
            else:
                new_var[k] = v

    return new_var


def h5_to_dict(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if type(item) is h5py.Dataset:
            ans[key] = item.value
        elif type(item) is h5py.Group:
            ans[key] = h5_to_dict(h5file, path + key + '/')
    return ans


def load_changepoints(cp_file):
    with h5py.File(cpfile, 'r') as f:
        cps = h5_to_dict(f, 'cps')

    cp_dist = []

    for k, v in cps.items():
        cp_dist.append(np.diff(v.squeeze()))

    return np.concatenate(cp_dist)


def parse_index(index_file, get_metadata=False):

    with open(index_file, 'r') as f:
        index = yaml.load(f.read(), Loader=yaml.RoundTripLoader)

    # sort index by uuids

    # yaml_dir = os.path.dirname(index_file)

    h5s = [idx['path'] for idx in index['files']]
    h5_uuids = [idx['uuid'] for idx in index['files']]
    groups = [idx['group'] for idx in index['files']]
    metadata = [commented_map_to_dict(idx['metadata']) for idx in index['files']]

    sorted_index = {
        'files': {},
        'pca_path': index['pca_path']
    }

    for uuid, h5, group, h5_meta in zip(h5_uuids, h5s, groups, metadata):
        sorted_index['files'][uuid] = {
            'path':  h5,
            'group': group,
            'metadata': h5_meta
        }

    # ymls = ['{}.yaml'.format(os.path.splitext(h5)[0]) for h5 in h5s]

    return index, sorted_index


def recursive_find_h5s(root_dir=os.getcwd(),
                       ext='.h5',
                       yaml_string='{}.yaml'):
    """Recursively find h5 files, along with yaml files with the same basename
    """
    dicts = []
    h5s = []
    yamls = []
    for root, dirs, files in os.walk(root_dir):
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

    with open(yaml_file, 'r') as f:
        dat = f.read()
        try:
            return_dict = yaml.load(dat, Loader=yaml.RoundTripLoader)
        except yaml.constructor.ConstructorError:
            return_dict = yaml.load(dat, Loader=yaml.Loader)

    return return_dict
