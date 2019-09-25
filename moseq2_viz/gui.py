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

def make_crowd_movies_command(index_file, model_path, max_syllable, max_examples,
                      sort, count, gaussfilter_space, medfilter_space,
                      output_dir, min_height, max_height, raw_size, scale, cmap, dur_clip,
                      legacy_jitter_fix):
    if platform in ['linux', 'linux2']:
        print('Setting CPU affinity to use all CPUs...')
        cpu_count = psutil.cpu_count()
        proc = psutil.Process()
        proc.cpu_affinity(list(range(cpu_count)))

    clean_params = {
        'gaussfilter_space': gaussfilter_space,
        'medfilter_space': medfilter_space
    }

    # need to handle h5 intelligently here...

    if model_path.endswith('.p') or model_path.endswith('.pz'):
        model_fit = parse_model_results(joblib.load(model_path))
        labels = model_fit['labels']

        if 'train_list' in model_fit:
            label_uuids = model_fit['train_list']
        else:
            label_uuids = model_fit['keys']
    elif model_path.endswith('.h5'):
        # load in h5, use index found using another function
        pass
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    info_parameters = ['model_class', 'kappa', 'gamma', 'alpha']
    info_dict = {k: model_fit['model_parameters'][k] for k in info_parameters}

    # convert numpy dtypes to their corresponding primitives
    for k, v in info_dict.items():
        if isinstance(v, (np.ndarray, np.generic)):
            info_dict[k] = info_dict[k].item()

    info_dict['model_path'] = model_path
    info_dict['index_path'] = index_file
    info_file = os.path.join(output_dir, 'info.yaml')

    with open(info_file, 'w+') as f:
        yaml.dump(info_dict, f, Dumper=yaml.RoundTripDumper)

    if sort:
        labels, ordering = relabel_by_usage(labels, count=count)
    else:
        ordering = list(range(max_syllable))

    index, sorted_index = parse_index(index_file)
    vid_parameters = check_video_parameters(sorted_index)

    # uuid in both the labels and the index
    uuid_set = set(label_uuids) & set(sorted_index['files'].keys())

    # make sure the files exist
    uuid_set = [uuid for uuid in uuid_set if os.path.exists(sorted_index['files'][uuid]['path'][0])]

    # harmonize everything...
    labels = [label_arr for label_arr, uuid in zip(labels, label_uuids) if uuid in uuid_set]
    label_uuids = [uuid for uuid in label_uuids if uuid in uuid_set]
    sorted_index['files'] = {k: v for k, v in sorted_index['files'].items() if k in uuid_set}

    if vid_parameters['resolution'] is not None:
        raw_size = vid_parameters['resolution']

    if sort:
        filename_format = 'syllable_sorted-id-{:d} ({})_original-id-{:d}.mp4'
    else:
        filename_format = 'syllable_{:d}.mp4'

    with mp.Pool() as pool:
        slice_fun = partial(get_syllable_slices,
                            labels=labels,
                            label_uuids=label_uuids,
                            index=sorted_index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", tqdm.TqdmSynchronisationWarning)
            slices = list(tqdm.tqdm(pool.imap(slice_fun, range(max_syllable)), total=max_syllable))

        matrix_fun = partial(make_crowd_matrix,
                             nexamples=max_examples,
                             dur_clip=dur_clip,
                             min_height=min_height,
                             crop_size=vid_parameters['crop_size'],
                             raw_size=raw_size,
                             scale=scale,
                             legacy_jitter_fix=legacy_jitter_fix,
                             **clean_params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", tqdm.TqdmSynchronisationWarning)
            crowd_matrices = list(tqdm.tqdm(pool.imap(matrix_fun, slices), total=max_syllable))

        write_fun = partial(write_frames_preview, fps=vid_parameters['fps'], depth_min=min_height,
                            depth_max=max_height, cmap=cmap)
        pool.starmap(write_fun,
                     [(os.path.join(output_dir, filename_format.format(i, count, ordering[i])),
                       crowd_matrix)
                      for i, crowd_matrix in enumerate(crowd_matrices) if crowd_matrix is not None])

def plot_usages_command(index_file, model_fit, sort, count, max_syllable, group, output_file):

    # if the user passes multiple groups, sort and plot against each other
    # relabel by usage across the whole dataset, gather usages per session per group

    # parse the index, parse the model fit, reformat to dataframe, bob's yer uncle

    model_data = parse_model_results(joblib.load(model_fit))
    index, sorted_index = parse_index(index_file)
    df, _ = results_to_dataframe(model_data, sorted_index, max_syllable=max_syllable, sort=sort, count=count)
    plt, _ = usage_plot(df, groups=group, headless=True)
    plt.savefig('{}.png'.format(output_file))
    plt.savefig('{}.pdf'.format(output_file))

def plot_scalar_summary_command(index_file, output_file):

    index, sorted_index = parse_index(index_file)
    scalar_df = scalars_to_dataframe(sorted_index)

    plt_scalars, _ = scalar_plot(scalar_df, headless=True)
    plt_position, _ = position_plot(scalar_df, headless=True)

    plt_scalars.savefig('{}_summary.png'.format(output_file))
    plt_scalars.savefig('{}_summary.pdf'.format(output_file))

    plt_position.savefig('{}_position.png'.format(output_file))
    plt_position.savefig('{}_position.pdf'.format(output_file))

def plot_transition_graph_command(index_file, model_fit, max_syllable, group, output_file,
                          normalize, edge_threshold, usage_threshold, layout,
                          keep_orphans, orphan_weight, arrows, sort, count,
                          edge_scaling, node_scaling, scale_node_by_usage, width_per_group):

    if layout.lower()[:8] == 'graphviz':
        try:
            import pygraphviz
        except ImportError:
            raise ImportError('pygraphviz must be installed to use graphviz layout engines')

    model_data = parse_model_results(joblib.load(model_fit))
    index, sorted_index = parse_index(index_file)

    labels = model_data['labels']

    if sort:
        labels = relabel_by_usage(labels, count=count)[0]

    if 'train_list' in model_data.keys():
        label_uuids = model_data['train_list']
    else:
        label_uuids = model_data['keys']

    label_group = []

    print('Sorting labels...')

    if 'group' in index['files'][0].keys() and len(group) > 0:
        for uuid in label_uuids:
            label_group.append(sorted_index['files'][uuid]['group'])
    # elif 'group' in index['files'][0].keys() and (group is None or len(group) == 0):
    #     for uuid in label_uuids:
    #         label_group.append(sorted_index['files'][uuid]['group'])
    #     group = list(set(label_group))
    else:
        label_group = ['']*len(model_data['labels'])
        group = list(set(label_group))

    print('Computing transition matrices...')

    trans_mats = []
    usages = []
    for plt_group in group:
        use_labels = [lbl for lbl, grp in zip(labels, label_group) if grp == plt_group]
        trans_mats.append(get_transition_matrix(use_labels, normalize=normalize, combine=True, max_syllable=max_syllable))
        usages.append(get_syllable_statistics(use_labels)[0])

    if not scale_node_by_usage:
        usages = None

    print('Creating plot...')

    plt, _, _ = graph_transition_matrix(trans_mats, usages=usages, width_per_group=width_per_group,
                                        edge_threshold=edge_threshold, edge_width_scale=edge_scaling,
                                        difference_edge_width_scale=edge_scaling, keep_orphans=keep_orphans,
                                        orphan_weight=orphan_weight, arrows=arrows, usage_threshold=usage_threshold,
                                        layout=layout, groups=group, usage_scale=node_scaling, headless=True)
    plt.savefig('{}.png'.format(output_file))
    plt.savefig('{}.pdf'.format(output_file))
