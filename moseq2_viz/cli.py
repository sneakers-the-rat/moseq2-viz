from moseq2_viz.util import recursive_find_h5s
import click
import os
import ruamel.yaml as yaml
import h5py


@click.group()
def cli():
    pass


@cli.command(name='generate-index')
@click.option('--input-dir', '-i', type=click.Path(), default=os.getcwd(), help='Directory to find h5 files')
@click.option('--pca-file', '-p', type=click.Path(exists=True), default=os.path.join(os.getcwd(), '_pca/pca.h5'), help='Path to PCA results')
@click.option('--output-file', '-o', type=click.Path, default=os.path.join(os.getcwd(), 'moseq2-index.yaml'), help="Location for storing index")
def generate_index(input_dir, pca_file, output_file):

    # gather than h5s and the pca scores file

    # uuids should match keys in the scores file

    with h5py.File(pca_file, 'r') as f:
        pca_uuids = list(f['scores'].keys())

    h5s, dicts, yamls = recursive_find_h5s(input_dir)
    file_uuids = [(h5, meta['uuid']) for h5, meta in zip(h5s, dicts) if meta['uuid'] in pca_uuids]

    output_dict = {
        'files': file_uuids,
        'pca_path': pca_file
    }

    # write out index yaml

    with open(output_file, 'w') as f:
        yaml.dump(output_dict, f, Dumper=yaml.RoundTripDumper)
