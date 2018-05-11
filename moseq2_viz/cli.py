from moseq2_viz.util import recursive_find_h5s, check_video_parameters, parse_index
from moseq2_viz.model.util import sort_results, relabel_by_usage, get_syllable_slices
from moseq2_viz.viz import make_crowd_matrix
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


# TODO: usages...group comparisons...changepoints...


# function for finding model index in h5 file, then we can pass to other functions and index simply...

# map metadata onto groups
