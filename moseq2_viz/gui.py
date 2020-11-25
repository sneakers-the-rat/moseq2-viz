'''

GUI front-end operations. This module contains all the functionality and configurable parameters
users can alter to most accurately process their data.

Note: These functions perform jupyter notebook specific pre-processing, loads in corresponding parameters from the
CLI functions, then call the corresponding wrapper function with the given input parameters.

'''

from os.path import join, exists
import ruamel.yaml as yaml
from .cli import plot_transition_graph, make_crowd_movies
from moseq2_viz.util import read_yaml
from moseq2_viz.helpers.wrappers import add_group_wrapper, plot_syllable_stat_wrapper, \
    plot_scalar_summary_wrapper, plot_transition_graph_wrapper, copy_h5_metadata_to_yaml_wrapper, \
    plot_verbose_pdfs_wrapper, plot_mean_group_position_pdf_wrapper, get_best_fit_model_wrapper, \
    make_crowd_movies_wrapper


def get_groups_command(index_file):
    '''
    Jupyter Notebook to print index file current metadata groupings.

    Parameters
    ----------
    index_file (str): path to index file

    Returns
    -------
    (int): number of unique groups
    '''


    index_data = read_yaml(index_file)

    groups, uuids = [], []
    subjectNames, sessionNames = [], []
    for f in index_data['files']:
        if f['uuid'] not in uuids:
            uuids.append(f['uuid'])
            groups.append(f['group'])
            subjectNames.append(f['metadata']['SubjectName'])
            sessionNames.append(f['metadata']['SessionName'])

    print('Total number of unique subject names:', len(set(subjectNames)))
    print('Total number of unique session names:', len(set(sessionNames)))
    print('Total number of unique groups:', len(set(groups)))

    for i in range(len(subjectNames)):
        print('Session Name:', sessionNames[i], '; Subject Name:', subjectNames[i], '; group:', groups[i])

    return len(set(groups))

def add_group(index_file, by='SessionName', value='default', group='default', exact=False, lowercase=False, negative=False):
    '''
    Updates index file SubjectName group names with user defined group names.

    Parameters
    ----------
    index_file (str): path to index file
    value (str or list): SessionName value(s) to search for and update with the corresponding group(s)
    group (str or list): Respective group name(s) to set corresponding sessions as.
    exact (bool): indicate whether to search for exact match.
    lowercase (bool): indicate whether to convert all searched for names to lowercase.
    negative (bool): whether to update the inverse of the found selection.

    Returns
    -------
    None
    '''

    gui_data = {
        'key': by,
        'exact': exact,
        'lowercase': lowercase,
        'negative': negative
    }

    if isinstance(value, str):
        gui_data.update({
            'value': value,
            'group': group,
        })
        add_group_wrapper(index_file, gui_data)

    elif isinstance(value, list) and isinstance(group, list):
        if len(value) == len(group):
            for v, g in zip(value, group):
                gui_data.update({
                    'value': v,
                    'group': g,
                })
                add_group_wrapper(index_file, gui_data)
        else:
            print('ERROR, did not enter equal number of substring values -> groups.')

def copy_h5_metadata_to_yaml_command(input_dir):
    '''
    Reads h5 metadata from a specified metadata h5 path.

    Parameters
    ----------
    input_dir (str): path to directory containing h5 file

    Returns
    -------
    None
    '''

    copy_h5_metadata_to_yaml_wrapper(input_dir)

def get_best_fit_model(progress_paths, output_file=None, plot_all=False, fps=30, ext='p'):
    '''
    Given a directory containing multiple models, and the path to the pca scores they were trained on,
     this function returns the path to the model that has the closest median syllable duration to that of
     the PC Scores.

    Parameters
    ----------
    progress_paths (dict): Dict containing paths the to model directory and pca scores file
    output_file (str): Optional path to save the comparison plot
    plot_all (bool): Indicates whether to plot all the models' changepoint distributions with the PCs, highlighting
     the best model curve.
    fps (int): Frames per second.

    Returns
    -------
    best_fit_model (str): Path tp best fit model
    '''

    # Check output file path
    if output_file is None:
        output_file = join(progress_paths['plot_path'], 'model_vs_pc_changepoints')

    # Get paths to required parameters
    model_dir = progress_paths['model_session_path']
    if not exists(progress_paths['changepoints_path']):
        changepoint_path = join(progress_paths['pca_dirname'], progress_paths['changepoints_path'] + '.h5')
    else:
        changepoint_path = progress_paths['changepoints_path']
    # Get best fit model and plot requested curves
    best_fit_model, fig = get_best_fit_model_wrapper(model_dir, changepoint_path, output_file,
                                                     plot_all=plot_all, ext=ext, fps=fps)
    fig.show(warn=False)

    return best_fit_model

def make_crowd_movies_command(index_file, model_path, output_dir, config_data=None):
    '''
    Runs CLI function to write crowd movies, due to multiprocessing
    compatibilty issues with Jupyter notebook's scheduler.

    Parameters
    ----------
    index_file (str): path to index file.
    model_path (str): path to fit model.
    output_dir (str): path to directory to save crowd movies in.
    config_data (dict): dictionary conataining all the necessary parameters to generate the crowd movies.
        E.g.: max_syllable: Maximum number of syllables to generate crowd movies for.
              max_example: Maximum number of mouse examples to include in each crowd movie
              specific_syllable: Set to a syllable number to only generate crowd movies of that syllable.
                if value is None, command will generate crowd movies for all syllables with # <= max_syllable.
              separate_by: ['default', 'groups', 'sessions', 'subjects']; If separate_by != 'default', the command
                  will generate a separate crowd movie for each selected grouping per syllable.
                  Resulting in (ngroups * max_syllable) movies.

    Returns
    -------
    (str): Success string.
    '''

    # Get default CLI params
    objs = make_crowd_movies.params
    defaults = {tmp.name: tmp.default for tmp in objs if not tmp.required}

    if config_data is None:
        config_data = defaults
    elif isinstance(config_data, dict):
        config_data = {**defaults, **config_data}
    else:
        raise TypeError('config_data needs to be a dictionary')

    make_crowd_movies_wrapper(index_file, model_path, config_data, output_dir)

def plot_stats_command(model_fit, index_file, output_file, stat='usage', max_syllable=40, count='usage', group=None, sort=True,
                        ordering=None, ctrl_group=None, exp_group=None, colors=None, figsize=(10, 5)):
    '''
    Graph given syllable statistic from fit model data.
    Parameters
    ----------
    model_fit (str): path to fit model.
    index_file (str): path to index file
    output_file (str): name of saved usages graph.
    stat (str): syllable statistic to plot: ['usage', 'speed', 'duration']
    max_syllable (int): max number of syllables to plot.
    count (str): method to calculate syllable usages, either by 'frames' or 'usage'
    group (tuple): groups to include in usage plot. If empty, plots default average of all groups.
    sort (bool): sort by usages.
    ordering (list, range, str, None): order to list syllables. Default is None to graph syllables [0-max_syllable).
     Setting ordering to "m" will graph mutated syllable usage difference between ctrl_group and exp_group.
     None to graph default [0,max_syllable] in order. "usage" to plot descending order of usage values.
    ctrl_group (str): Control group to graph when plotting mutation differences via setting ordering to 'm'.
    exp_group (str): Experimental group to directly compare with control group.
    colors (list): list of colors to serve as the sns palette in the scalar summary. If None, default colors are used.
    figsize (tuple): tuple value of length = 2, representing (columns x rows) of the plotted figure dimensions
    Returns
    -------
    fig (pyplot figure): figure to graph in Jupyter Notebook.
    '''


    fig = plot_syllable_stat_wrapper(model_fit, index_file, output_file, stat=stat, max_syllable=max_syllable,
                                     sort=sort, count=count, group=group, ordering=ordering,
                                     ctrl_group=ctrl_group, exp_group=exp_group, colors=colors, figsize=figsize)

    print(f'{stat} plot successfully generated')
    return fig

def plot_scalar_summary_command(index_file, output_file, show_scalars=['velocity_2d_mm', 'velocity_3d_mm',
                              'height_ave_mm', 'width_mm', 'length_mm'], colors=None, groupby='group'):
    '''
    Creates a scalar summary graph and a position summary graph.

    Parameters
    ----------
    index_file (str): path to index file
    output_file (str): prefix name of scalar summary images
    show_scalars (list): list of scalar variables to plot.
    colors (list): list of colors to serve as the sns palette in the scalar summary
    groupby (str): scalar_df column to group sessions by when graphing scalar and position summaries

    Returns
    -------
    scalar_df (pandas DataFrame): DataFrame containing all of scalar values for debugging.
    '''

    scalar_df = plot_scalar_summary_wrapper(index_file, output_file, groupby=groupby,
                                            colors=colors, show_scalars=show_scalars)
    return scalar_df

def plot_transition_graph_command(index_file, model_fit, config_file, max_syllable, group, output_file):
    '''
    Creates transition graphs given groups to process.

    Parameters
    ----------
    index_file (str): path to index file
    model_fit (str): path to fit model
    config_file (str): path to config file
    max_syllable (int): maximum number of syllables to include in graph
    group (tuple): tuple of names of groups to graph transition graphs for
    output_file (str): name of the transition graph saved image

    Returns
    -------
    fig (pyplot figure): figure to graph in Jupyter Notebook.
    '''

    config_data = read_yaml(config_file)

    # Get default CLI params
    params = {tmp.name: tmp.default for tmp in plot_transition_graph.params if not tmp.required}

    config_data = {**params, **config_data}

    config_data['max_syllable'] = max_syllable
    config_data['group'] = group

    fig = plot_transition_graph_wrapper(index_file, model_fit, config_data, output_file)

    print('Transition graph(s) successfully generated')
    return fig

def plot_mean_group_position_heatmaps_command(index_file, output_file):
    '''
    Plots the average mouse position in a PDF-derived heatmap for each group found in the inputted index file.
    Parameters
    ----------
    index_file (str): path to index file.
    output_file (str): filename for syllable duration graph.

    Returns
    -------
    fig (pyplot figure): figure to graph in Jupyter Notebook.
    '''

    fig = plot_mean_group_position_pdf_wrapper(index_file, output_file)

    return fig

def plot_verbose_position_heatmaps(index_file, output_file):
    '''
    Plots a PDF-derived heatmap of each session found in the index file titled with the session name and group.

    Parameters
    ----------
    index_file (str): path to index file.
    output_file (str): filename for syllable duration graph.

    Returns
    -------
    fig (pyplot figure): figure to graph in Jupyter Notebook.
    '''

    fig = plot_verbose_pdfs_wrapper(index_file, output_file)

    return fig
