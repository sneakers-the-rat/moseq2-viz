from moseq2_viz.util import recursive_find_h5s, check_video_parameters,\
    parse_index
from moseq2_viz.model.util import sort_results, relabel_by_usage, get_syllable_slices,\
    results_to_dataframe, parse_model_results, get_transition_matrix
from moseq2_viz.viz import make_crowd_matrix, usage_plot, graph_transition_matrix
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


if platform == 'linux' or platform == 'linux2':
    os.system('taskset -p 0xff {:d}'.format(os.getpid()))

#TODO: simple way to put extraction metadata into results dataframe for better sorting

orig_init = click.core.Option.__init__


def new_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.show_default = True


click.core.Option.__init__ = new_init


@click.group()
def cli():
    pass


@cli.command(name='generate-index')
@click.option('--input-dir', '-i', type=click.Path(), default=os.getcwd(), help='Directory to find h5 files')
@click.option('--pca-file', '-p', type=click.Path(exists=True), default=os.path.join(os.getcwd(), '_pca/pca_scores.h5'), help='Path to PCA results')
@click.option('--output-file', '-o', type=click.Path(), default=os.path.join(os.getcwd(), 'moseq2-index.yaml'), help="Location for storing index")
def generate_index(input_dir, pca_file, output_file):

    # gather than h5s and the pca scores file

    # uuids should match keys in the scores file

    with h5py.File(pca_file, 'r') as f:
        pca_uuids = list(f['scores'].keys())

    h5s, dicts, yamls = recursive_find_h5s(input_dir)
    file_uuids = [(os.path.relpath(h5), meta['uuid']) for h5, meta in zip(h5s, dicts) if meta['uuid'] in pca_uuids]

    output_dict = {
        'files': file_uuids,
        'pca_path': os.path.relpath(pca_file)
    }

    # write out index yaml

    with open(output_file, 'w') as f:
        yaml.dump(output_dict, f, Dumper=yaml.RoundTripDumper)


@cli.command(name="add-group")
@click.argument('index-file', type=click.Path(exists=True, resolve_path=True))
@click.option('--key', '-k', type=str, default='SubjectName', help='Key to search for value')
@click.option('--value', '-v', type=str, default='Mouse', help='Value to search for')
@click.option('--group', '-g', type=str, default='Group1', help='Group name to map to')
@click.option('--exact', '-e', type=bool, is_flag=True, help='Exact match only')
@click.option('--lowercase', type=bool, is_flag=True, help='Lowercase text filter')
def add_group(index_file, key, value, group, exact, lowercase):

        index, h5s, h5_uuids, dicts, metadata = parse_index(index_file, get_metadata=True)

        if lowercase:
            hits = [re.search(value, meta[key].lower()) is not None for meta in metadata]
        else:
            hits = [re.search(value, meta[key]) is not None for meta in metadata]

        if 'groups' in list(index.keys()):
            group_dict = index['groups']
        else:
            group_dict = {}

        for uuid, hit in zip(h5_uuids, hits):
            if hit:
                group_dict[uuid] = group

        index['groups'] = group_dict

        with open(index_file, 'w+') as f:
            yaml.dump(index, f, Dumper=yaml.RoundTripDumper)


@cli.command(name='make-crowd-movies')
@click.argument('index-file', type=click.Path(exists=True, resolve_path=True))
@click.argument('model-fit', type=click.Path(exists=True, resolve_path=True))
@click.option('--max-syllable', type=int, default=40, help="Index of max syllable to render")
@click.option('--max-examples', '-m', type=int, default=40, help="Number of examples to show")
@click.option('--threads', '-t', type=int, default=-1, help="Number of threads to use for rendering crowd movies")
@click.option('--sort', type=bool, default=True, help="Sort syllables by usage")
@click.option('--output-dir', '-o', type=click.Path(), default=os.path.join(os.getcwd(), 'crowd_movies'), help="Path to store files")
@click.option('--filename-format', type=str, default='syllable_{:d}.mp4', help="Python 3 string format for filenames")
@click.option('--min-height', type=int, default=5, help="Minimum height for scaling videos")
@click.option('--max-height', type=int, default=80, help="Minimum height for scaling videos")
@click.option('--raw-size', type=(int, int), default=(512, 424), help="Size of original videos")
@click.option('--scale', type=float, default=1, help="Scaling from pixel units to mm")
@click.option('--cmap', type=str, default='jet', help="Name of valid Matplotlib colormap for false-coloring images")
def make_crowd_movies(index_file, model_fit, max_syllable, max_examples, threads, sort,
                      output_dir, filename_format, min_height, max_height, raw_size, scale, cmap):

    # need to handle h5 intelligently here...

    if model_fit.endswith('.p') or model_fit.endswith('.pz'):
        model_fit = joblib.load(model_fit)
        labels = model_fit['labels'][0]
        label_array = np.empty((len(labels),), dtype='object')

        for i, label in enumerate(labels):
            label_array[i] = np.squeeze(label)

        labels = label_array

        if 'train_list' in model_fit.keys():
            label_uuids = model_fit['train_list']
        else:
            label_uuids = model_fit['keys']
    elif model_fit.endswith('.h5'):
        # load in h5, use index found using another function
        pass

    if sort:
        labels = relabel_by_usage(labels)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vid_parameters = check_video_parameters(index_file)

    if vid_parameters['resolution'] is not None:
        raw_size = vid_parameters['resolution']

    with mp.Pool() as pool:
        slice_fun = partial(get_syllable_slices,
                            labels=labels,
                            label_uuids=label_uuids,
                            index_file=index_file)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", tqdm.TqdmSynchronisationWarning)
            slices = list(tqdm.tqdm(pool.imap(slice_fun, range(max_syllable)), total=max_syllable))

        matrix_fun = partial(make_crowd_matrix, nexamples=max_examples, dur_clip=None,
                             crop_size=vid_parameters['crop_size'], raw_size=raw_size, scale=scale)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", tqdm.TqdmSynchronisationWarning)
            crowd_matrices = list(tqdm.tqdm(pool.imap(matrix_fun, slices), total=max_syllable))

        write_fun = partial(write_frames_preview, fps=vid_parameters['fps'], depth_min=min_height,
                            depth_max=max_height, cmap=cmap)
        pool.starmap(write_fun, [(os.path.join(output_dir, filename_format.format(i)), crowd_matrix)
                                 for i, crowd_matrix in enumerate(crowd_matrices) if crowd_matrix is not None])


@cli.command(name='plot-usages')
@click.argument('index-file', type=click.Path(exists=True, resolve_path=True))
@click.argument('model-fit', type=click.Path(exists=True, resolve_path=True))
@click.option('--max-syllable', type=int, default=40, help="Index of max syllable to render")
@click.option('--group', type=str, default=None, help="Name of group(s) to show", multiple=True)
@click.option('--output-file', type=click.Path(), default=os.path.join(os.getcwd(), 'usages'), help="Filename to store plot")
def plot_usages(index_file, model_fit, max_syllable, group, output_file):

    # if the user passes multiple groups, sort and plot against each other
    # relabel by usage across the whole dataset, gather usages per session per group

    # parse the index, parse the model fit, reformat to dataframe, bob's yer uncle

    model_data = parse_model_results(joblib.load(model_fit))
    index, _, _, _, _ = parse_index(index_file)
    df, _ = results_to_dataframe(model_data, index, max_syllable=max_syllable, sort=True)
    plt, ax, fig = usage_plot(df, groups=group, headless=True)
    plt.savefig('{}.png'.format(output_file))
    plt.savefig('{}.pdf'.format(output_file))


@cli.command(name='plot-transition-graph')
@click.argument('index-file', type=click.Path(exists=True, resolve_path=True))
@click.argument('model-fit', type=click.Path(exists=True, resolve_path=True))
@click.option('--max-syllable', type=int, default=40, help="Index of max syllable to render")
@click.option('--group', type=str, default=None, help="Name of group(s) to show", multiple=True)
@click.option('--output-file', type=click.Path(), default=os.path.join(os.getcwd(), 'transitions'), help="Filename to store plot")
@click.option('--normalize', type=click.Choice(['bigram', 'rows', 'columns']), default='bigram', help="How to normalize transition probabilities")
@click.option('--edge-threshold', type=float, default=.001, help="Threshold for edges to show")
@click.option('--layout', type=str, default='spring', help="Default networkx layout algorithm")
@click.option('--edge-scaling', type=float, default=250, help="Scale factor from transition probabilities to edge width")
@click.option('--width-per-group', type=float, default=8, help="Width (in inches) for figure canvas per group")
def plot_transition_graph(index_file, model_fit, max_syllable, group, output_file,
                          normalize, edge_threshold, layout, edge_scaling, width_per_group):

    model_data = parse_model_results(joblib.load(model_fit))
    index, _, _, _, _ = parse_index(index_file)

    label_uuids = model_data['train_list']
    label_group = []

    if 'groups' in index.keys() and len(group) > 0:
        for uuid in label_uuids:
            label_group.append(index['groups'][uuid])
    elif 'groups' in index.keys() and (group is None or len(group) == 0):
        for uuid in label_uuids:
            label_group.append(index['groups'][uuid])
        group = list(set(label_group))
    else:
        group = ['']*len(model_data['labels'])
        label_group = group

    trans_mats = []
    for plt_group in group:
        use_labels = [lbl for lbl, grp in zip(model_data['labels'], label_group) if grp == plt_group]
        trans_mats.append(get_transition_matrix(use_labels, normalize=normalize, combine=True, max_syllable=max_syllable))

    plt, fig, ax = graph_transition_matrix(trans_mats, width_per_group=width_per_group,
                                           edge_threshold=edge_threshold, edge_width_scale=edge_scaling,
                                           layout=layout, groups=group, headless=True)
    plt.savefig('{}.png'.format(output_file))
    plt.savefig('{}.pdf'.format(output_file))



# TODO: usages...group comparisons...changepoints...
# function for finding model index in h5 file, then we can pass to other functions and index simply...
# map metadata onto groups
