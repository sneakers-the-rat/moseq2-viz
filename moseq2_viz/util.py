import os
import h5py
import ruamel.yaml as yaml
import numpy as np

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


def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans


def check_video_parameters(index_file):

    with open(index_file, 'r') as f:
        index = yaml.load(f.read(), Loader=yaml.RoundTripLoader)

    h5s, h5_uuids = zip(*index['files'])
    ymls = ['{}.yaml'.format(os.path.splitext(h5)[0]) for h5 in h5s]

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

    if type(cmap) is yaml.comments.CommentedMap:
        for k, v in cmap.items():
            if type(v) is yaml.comments.CommentedMap:
                new_var[k] = commented_map_to_dict(v)
            else:
                new_var[k] = v

    return new_var


def parse_index(index_file, get_metadata=False):

    with open(index_file, 'r') as f:
        index = yaml.load(f.read(), Loader=yaml.RoundTripLoader)

    yaml_dir = os.path.dirname(index_file)
    index = commented_map_to_dict(index)

    h5s, h5_uuids = zip(*index['files'])
    ymls = ['{}.yaml'.format(os.path.splitext(h5)[0]) for h5 in h5s]

    dicts = []
    has_meta = []

    for yml in ymls:
        with open(os.path.join(yaml_dir, yml), 'r') as f:
            yml_dict = yaml.load(f.read(), Loader=yaml.RoundTripLoader)
            dicts.append(yml_dict)
            has_meta.append('metadata' in list(yml_dict.keys()))

    if get_metadata and all(has_meta):
        metadata = [tmp['metadata'] for tmp in dicts]
    elif get_metadata:
        metadata = [recursively_load_dict_contents_from_group(h5py.File(os.path.join(yaml_dir, h5), 'r'),
                                                              '/metadata/extraction') for h5 in h5s]
    else:
        metadata = None

    return index, h5s, h5_uuids, dicts, metadata
