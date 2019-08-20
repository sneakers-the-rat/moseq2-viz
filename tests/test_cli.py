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
import time
import psutil
import pytest
from click.testing import CliRunner
from moseq2_viz.cli import *

def test_add_group():
    input_dir = 'tests/test_files/'
    input_path = os.path.join(input_dir,'test_index.yaml')

    runner = CliRunner()

    group_params = ['-k', 'SubjectName',
                    '-v', 'Mouse',
                    '-g', 'Group1',
                    '-e', # FLAG
                    '-e', # FLAG
                    '--lowercase', # FLAG
                    '-n', # FLAG
                    input_path]

    results = runner.invoke(add_group, group_params)
    assert(not os.path.samefile(os.path.join(input_dir, 'orig.txt'), input_path))
    assert(results.exit_code == 0)

def test_copy_h5_metadata_to_yaml():
    input_dir = 'tests/test_files/_pca/'

    runner = CliRunner()
    results = runner.invoke(copy_h5_metadata_to_yaml, ['--h5-metadata-path', input_dir+'pca_scores.h5',
                                                       '-i', input_dir])

    assert (results.exit_code == 0)

def test_generate_index():
    input_dir = 'tests/test_files/'

    runner = CliRunner()
    results = runner.invoke(generate_index, ['-i', input_dir,
                                             '-p', input_dir+'_pca/pca_scores.h5',
                                             '-o', input_dir+'test_gen_index.yaml'])

    assert(os.path.exists(input_dir+'test_gen_index.yaml'))
    os.remove(input_dir+'test_gen_index.yaml')
    assert (results.exit_code == 0)

def test_plot_scalar_summary():

    input_dir = 'tests/test_files/'
    gen_dir = 'tests/test_files/gen_plots/'
    runner = CliRunner()

    results = runner.invoke(plot_scalar_summary, ['--output-file', gen_dir+'scalar',
                                                  input_dir+'test_index.yaml'])

    #time.sleep(15)
    assert (os.path.exists(gen_dir + 'scalar_position.png'))
    assert (os.path.exists(gen_dir + 'scalar_position.pdf'))
    assert (os.path.exists(gen_dir + 'scalar_summary.png'))
    assert (os.path.exists(gen_dir + 'scalar_summary.pdf'))
    os.remove(gen_dir+'scalar_position.png')
    os.remove(gen_dir + 'scalar_position.pdf')
    os.remove(gen_dir + 'scalar_summary.png')
    os.remove(gen_dir + 'scalar_summary.pdf')
    assert (results.exit_code == 0)

def test_plot_transition_graph():

    input_dir = 'tests/test_files/'
    gen_dir = 'tests/test_files/gen_plots/'
    runner = CliRunner()

    results = runner.invoke(plot_transition_graph, ['--output-file', gen_dir+'transitions',
                                                    input_dir+'test_index.yaml',
                                                    input_dir+'mock_model.p'])
    #time.sleep(15)
    assert(os.path.exists(gen_dir+'transitions.png'))
    assert (os.path.exists(gen_dir + 'transitions.pdf'))
    os.remove(gen_dir + 'transitions.png')
    os.remove(gen_dir + 'transitions.pdf')
    assert (results.exit_code == 0)

def test_plot_usages():
    input_dir = 'tests/test_files/'
    gen_dir = 'tests/test_files/gen_plots/'

    runner = CliRunner()

    results = runner.invoke(plot_usages, ['--output-file', gen_dir+'test_usages',
                                          input_dir+'test_index.yaml',
                                          input_dir+'mock_model.p'])
    #time.sleep(15)
    assert (os.path.exists(gen_dir + 'test_usages.png'))
    assert (os.path.exists(gen_dir + 'test_usages.pdf'))
    os.remove(gen_dir + 'test_usages.png')
    os.remove(gen_dir + 'test_usages.pdf')
    assert (results.exit_code == 0)

def test_make_crowd_movies():
    input_dir = 'tests/test_files/'
    crowd_dir = input_dir+'crowd_movies/'
    max_examples = 40
    runner = CliRunner()

    results = runner.invoke(make_crowd_movies, ['-o', crowd_dir,
                                                '--max-examples', max_examples,
                                                input_dir+'test_index.yaml',
                                                input_dir+'mock_model.p'])
    #time.sleep(15)
    assert(os.path.exists(crowd_dir))
    assert(len([os.listdir(crowd_dir)][0]) == max_examples+1)
    shutil.rmtree(crowd_dir)
    assert (results.exit_code == 0)

