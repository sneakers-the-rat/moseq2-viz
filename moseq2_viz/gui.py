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
from tqdm.auto import tqdm
import warnings
import re
import shutil
import psutil

def get_groups_command(index_file):
    with open(index_file, 'r') as f:
        index_data = yaml.safe_load(f)
    f.close()

    groups, uuids = [], []
    subjectNames, sessionNames = [], []
    for f in index_data['files']:
        if f['uuid'] not in uuids:
            uuids.append(f['uuid'])
            groups.append(f['group'])
            subjectNames.append(f['metadata']['SubjectName'])
            sessionNames.append(f['metadata']['SessionName'])

    print('Total number of subjects:', len(set(subjectNames)))
    print('Total number of sessions:', len(set(sessionNames)))
    print('Total number of unique groups:', len(set(groups)))

    for i in range(len(subjectNames)):
        print('Session Name:', sessionNames[i], '; Subject Name:', subjectNames[i], '; group:', groups[i])

def add_group_by_session(index_file, value, group, exact, lowercase, negative):

    key = 'SessionName'
    index = parse_index(index_file)[0]
    h5_uuids = [f['uuid'] for f in index['files']]
    metadata = [f['metadata'] for f in index['files']]

    if type(value) is str:
        value = [value]

    for v in value:
        if exact:
            v = r'\b{}\b'.format(v)
        if lowercase and negative:
            hits = [re.search(v, meta[key].lower()) is None for meta in metadata]
        elif lowercase:
            hits = [re.search(v, meta[key].lower()) is not None for meta in metadata]
        elif negative:
            hits = [re.search(v, meta[key]) is None for meta in metadata]
        else:
            hits = [re.search(v, meta[key]) is not None for meta in metadata]

        for uuid, hit in zip(h5_uuids, hits):
            position = h5_uuids.index(uuid)
            if hit:
                index['files'][position]['group'] = group

    new_index = '{}_update.yaml'.format(os.path.basename(index_file))

    try:
        with open(new_index, 'w+') as f:
            yaml.dump(index, f, Dumper=yaml.RoundTripDumper)
        shutil.move(new_index, index_file)
    except Exception:
        raise Exception

    get_groups_command(index_file)

def add_group_by_subject(index_file, value, group, exact, lowercase, negative):

    key = 'SubjectName'
    index = parse_index(index_file)[0]
    h5_uuids = [f['uuid'] for f in index['files']]
    metadata = [f['metadata'] for f in index['files']]

    if type(value) is str:
        value = [value]

    for v in value:
        if exact:
            v = r'\b{}\b'.format(v)
        if lowercase and negative:
            hits = [re.search(v, meta[key].lower()) is None for meta in metadata]
        elif lowercase:
            hits = [re.search(v, meta[key].lower()) is not None for meta in metadata]
        elif negative:
            hits = [re.search(v, meta[key]) is None for meta in metadata]
        else:
            hits = [re.search(v, meta[key]) is not None for meta in metadata]

        for uuid, hit in zip(h5_uuids, hits):
            position = h5_uuids.index(uuid)
            if hit:
                index['files'][position]['group'] = group

    new_index = '{}_update.yaml'.format(os.path.basename(index_file))

    try:
        with open(new_index, 'w+') as f:
            yaml.dump(index, f, Dumper=yaml.RoundTripDumper)
        shutil.move(new_index, index_file)
    except Exception:
        raise Exception

    get_groups_command(index_file)

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

def make_crowd_movies_command(index_file, model_path, config_file, output_dir, max_syllable, max_examples):

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)


    if platform in ['linux', 'linux2']:
        print('Setting CPU affinity to use all CPUs...')
        cpu_count = psutil.cpu_count()
        proc = psutil.Process()
        proc.cpu_affinity(list(range(cpu_count)))

    clean_params = {
        'gaussfilter_space': config_data['gaussfilter_space'],
        'medfilter_space': config_data['medfilter_space']
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

    if config_data['sort']:
        labels, ordering = relabel_by_usage(labels, count=config_data['count'])
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

    if config_data['sort']:
        filename_format = 'syllable_sorted-id-{:d} ({})_original-id-{:d}.mp4'
    else:
        filename_format = 'syllable_{:d}.mp4'

    with mp.Pool() as pool:
        slice_fun = partial(get_syllable_slices,
                            labels=labels,
                            label_uuids=label_uuids,
                            index=sorted_index)
        with warnings.catch_warnings():
            #warnings.simplefilter("ignore", tqdm.TqdmSynchronisationWarning)
            slices = list(tqdm(pool.imap(slice_fun, range(max_syllable)), total=max_syllable))

        matrix_fun = partial(make_crowd_matrix,
                             nexamples=max_examples,
                             dur_clip=config_data['dur_clip'],
                             min_height=config_data['min_height'],
                             crop_size=vid_parameters['crop_size'],
                             raw_size=config_data['raw_size'],
                             scale=config_data['scale'],
                             legacy_jitter_fix=config_data['legacy_jitter_fix'],
                             **clean_params)
        with warnings.catch_warnings():
            #warnings.simplefilter("ignore", tqdm.TqdmSynchronisationWarning)
            crowd_matrices = list(tqdm(pool.imap(matrix_fun, slices), total=max_syllable))

        write_fun = partial(write_frames_preview, fps=vid_parameters['fps'], depth_min=config_data['min_height'],
                            depth_max=config_data['max_height'], cmap=config_data['cmap'])
        pool.starmap(write_fun,
                     [(os.path.join(output_dir, filename_format.format(i, config_data['count'], ordering[i])),
                       crowd_matrix)
                      for i, crowd_matrix in enumerate(crowd_matrices) if crowd_matrix is not None])

    return 'Successfully generated '+str(max_examples) + ' crowd videos.'

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

    return plt
    #return 'Usage plots successfully completed.'

def plot_scalar_summary_command(index_file, output_file):

    index, sorted_index = parse_index(index_file)
    scalar_df = scalars_to_dataframe(sorted_index)

    plt_scalars, _ = scalar_plot(scalar_df, headless=True)
    plt_position, _ = position_plot(scalar_df, headless=True)

    plt_scalars.savefig('{}_summary.png'.format(output_file))
    plt_scalars.savefig('{}_summary.pdf'.format(output_file))

    plt_position.savefig('{}_position.png'.format(output_file))
    plt_position.savefig('{}_position.pdf'.format(output_file))

    #return plt_scalars, plt_position
    return 'Scalar summary plots successfully completed.'

def plot_transition_graph_command(index_file, model_fit, config_file, max_syllable, group, output_file):

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    if config_data['layout'].lower()[:8] == 'graphviz':
        try:
            import pygraphviz
        except ImportError:
            raise ImportError('pygraphviz must be installed to use graphviz layout engines')

    model_data = parse_model_results(joblib.load(model_fit))
    index, sorted_index = parse_index(index_file)

    labels = model_data['labels']

    if config_data['sort']:
        labels = relabel_by_usage(labels, count=config_data['count'])[0]

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
        trans_mats.append(get_transition_matrix(use_labels, normalize=config_data['normalize'], combine=True, max_syllable=max_syllable))
        usages.append(get_syllable_statistics(use_labels)[0])

    if not config_data['scale_node_by_usage']:
        usages = None

    print('Creating plot...')

    plt, _, _ = graph_transition_matrix(trans_mats, usages=usages, width_per_group=config_data['width_per_group'],
                                        edge_threshold=config_data['edge_threshold'], edge_width_scale=config_data['edge_scaling'],
                                        difference_edge_width_scale=config_data['edge_scaling'], keep_orphans=config_data['keep_orphans'],
                                        orphan_weight=config_data['orphan_weight'], arrows=config_data['arrows'], usage_threshold=config_data['usage_threshold'],
                                        layout=config_data['layout'], groups=group, usage_scale=config_data['node_scaling'], headless=True)
    plt.savefig('{}.png'.format(output_file))
    plt.savefig('{}.pdf'.format(output_file))

    return plt
    #return 'Transition graph(s) successfully generated and saved.'
