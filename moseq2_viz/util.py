import os
import h5py
import ruamel.yaml as yaml


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
        print(yml)
        with open(yml, 'r') as f:
            dicts.append(yaml.load(f.read(), Loader=yaml.RoundTripLoader))

    check_parameters = ['crop_size', 'fps']

    for chk in check_parameters:
        tmp_list = [dct['parameters'][chk] for dct in dicts]
        if not all(x == tmp_list[0] for x in tmp_list):
            raise RuntimeError('Parameter {} not equal in all extractions'.format(chk))

    vid_parameters = {
        'crop_size': tuple(dicts[0]['parameters']['crop_size']),
        'fps': dicts[0]['parameters']['fps']
    }

    return vid_parameters
