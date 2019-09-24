from moseq2_viz.util import (recursive_find_h5s, check_video_parameters,
                             parse_index, h5_to_dict, clean_dict)
from moseq2_viz.model.util import (relabel_by_usage, get_syllable_slices,
                                   results_to_dataframe, parse_model_results,
                                   get_transition_matrix, get_syllable_statistics)
from moseq2_viz.viz import (make_crowd_matrix, usage_plot, graph_transition_matrix,
                            scalar_plot, position_plot)
from moseq2_viz.scalars.util import scalars_to_dataframe
from moseq2_viz.io.video import write_frames_preview
from functools import partial
from sys import platform
import click
import os
import ruamel.yaml as yaml
import h5py
import multiprocessing as mp
import numpy as np
import joblib
import tqdm
import warnings
import re
import shutil
import psutil

def copy_h5_metadata_to_yaml_command(input_dir, h5_metadata_path):

    h5s, dicts, yamls = recursive_find_h5s(input_dir)
    to_load = [(tmp, yml, file) for tmp, yml, file in zip(
        dicts, yamls, h5s) if tmp['complete'] and not tmp['skip']]

    # load in all of the h5 files, grab the extraction metadata, reformat to make nice 'n pretty
    # then stage the copy

    for i, tup in tqdm.tqdm(enumerate(to_load), total=len(to_load), desc='Copying data to yamls'):
        with h5py.File(tup[2], 'r') as f:
            tmp = clean_dict(h5_to_dict(f, h5_metadata_path))
            tup[0]['metadata'] = dict(tmp)

        try:
            new_file = '{}_update.yaml'.format(os.path.basename(tup[1]))
            with open(new_file, 'w+') as f:
                yaml.dump(tup[0], f, Dumper=yaml.RoundTripDumper)
            shutil.move(new_file, tup[1])
        except Exception:
            raise Exception
    return True

def generate_index_command(input_dir, pca_file, output_file, filter, all_uuids):

    # gather than h5s and the pca scores file
    # uuids should match keys in the scores file

    h5s, dicts, yamls = recursive_find_h5s(input_dir)
    if not os.path.exists(pca_file) or all_uuids:
        warnings.warn('Will include all files')
        pca_uuids = [dct['uuid'] for dct in dicts]
    else:
        with h5py.File(pca_file, 'r') as f:
            pca_uuids = list(f['scores'].keys())

    file_with_uuids = [(os.path.relpath(h5), os.path.relpath(yml), meta) for h5, yml, meta in
                       zip(h5s, yamls, dicts) if meta['uuid'] in pca_uuids]

    if 'metadata' not in file_with_uuids[0][2]:
        raise RuntimeError('Metadata not present in yaml files, run copy-h5-metadata-to-yaml to update yaml files')

    output_dict = {
        'files': [],
        'pca_path': os.path.relpath(pca_file)
    }

    for i, file_tup in enumerate(file_with_uuids):
        output_dict['files'].append({
            'path': (file_tup[0], file_tup[1]),
            'uuid': file_tup[2]['uuid'],
            'group': 'default'
        })

        output_dict['files'][i]['metadata'] = {}

        for k, v in file_tup[2]['metadata'].items():
            for filt in filter:
                if k == filt[0]:
                    tmp = re.match(filt[1], v)
                    if tmp is not None:
                        v = tmp[0]

            output_dict['files'][i]['metadata'][k] = v

    # write out index yaml

    with open(output_file, 'w') as f:
        yaml.dump(output_dict, f, Dumper=yaml.RoundTripDumper)

    return True