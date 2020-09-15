'''

'''

import joblib
import numpy as np
import pandas as pd
from bokeh.io import show
import ruamel.yaml as yaml
from collections import OrderedDict
from moseq2_viz.interactive.widgets import *
from moseq2_viz.util import index_to_dataframe
from ipywidgets import fixed, interactive_output
from IPython.display import display, clear_output
from moseq2_viz.helpers.wrappers import init_wrapper_function
from moseq2_viz.model.util import relabel_by_usage, get_syllable_usages, parse_model_results
from moseq2_viz.interactive.controller import SyllableLabeler, InteractiveSyllableStats, CrowdMovieComparison, InteractiveTransitionGraph

def interactive_group_setting_wrapper(index_filepath):
    '''

    Parameters
    ----------
    index_filepath

    Returns
    -------

    '''
    index_grid = GroupSettingWidgets()

    index_dict, df = index_to_dataframe(index_filepath)
    qgrid_widget = qgrid.show_grid(df[['SessionName', 'SubjectName', 'group', 'uuid']], column_options=index_grid.col_opts,
                                   column_definitions=index_grid.col_defs, show_toolbar=False)

    def update_table(b):
        '''

        Parameters
        ----------
        b

        Returns
        -------

        '''

        index_grid.update_index_button.button_style = 'info'
        index_grid.update_index_button.icon = 'none'

        selected_rows = qgrid_widget.get_selected_df()
        x = selected_rows.index

        for i in x:
            qgrid_widget.edit_cell(i, 'group', index_grid.group_input.value)

    def update_clicked(b):
        '''

        Parameters
        ----------
        b

        Returns
        -------

        '''

        files = index_dict['files']
        meta = [f['metadata'] for f in files]
        meta_cols = pd.DataFrame(meta).columns

        latest_df = qgrid_widget.get_changed_df()
        df.update(latest_df)

        updated_index = {'files': list(df.drop(meta_cols, axis=1).to_dict(orient='index').values()),
                         'pca_path': index_dict['pca_path']}

        with open(index_filepath, 'w+') as f:
            yaml.safe_dump(updated_index, f)

        index_grid.update_index_button.button_style = 'success'
        index_grid.update_index_button.icon = 'check'

    index_grid.update_index_button.on_click(update_clicked)

    index_grid.save_button.on_click(update_table)

    display(index_grid.group_set)
    display(qgrid_widget)

def interactive_syllable_labeler_wrapper(model_path, index_file, crowd_movie_dir, output_file, max_syllables=None):
    '''
    Wrapper function to launch a syllable crowd movie preview and interactive labeling application.

    Parameters
    ----------
    model_path (str): Path to trained model.
    crowd_movie_dir (str): Path to crowd movie directory
    output_file (str): Path to syllable label information file
    max_syllables (int): Maximum number of syllables to preview and label.

    Returns
    -------
    '''

    # Load the model
    model = parse_model_results(joblib.load(model_path))

    # Compute the sorted labels
    model['labels'] = relabel_by_usage(model['labels'], count='usage')[0]

    # Get Maximum number of syllables to include
    if max_syllables == None:
        syllable_usages = get_syllable_usages(model, 'usage')
        cumulative_explanation = 100 * np.cumsum(syllable_usages)
        max_sylls = np.argwhere(cumulative_explanation >= 90)[0][0]
    else:
        max_sylls = max_syllables

    # Make initial syllable information dict
    labeler = SyllableLabeler(model_fit=model, index_file=index_file, max_sylls=max_sylls, save_path=output_file)

    # Populate syllable info dict with relevant syllable information
    labeler.get_crowd_movie_paths(crowd_movie_dir)
    labeler.get_mean_syllable_info()

    labeler.syll_select.options = labeler.syll_info

    # Launch and display interactive API
    output = widgets.interactive_output(labeler.interactive_syllable_labeler, {'syllables': labeler.syll_select})
    display(labeler.syll_select, output)

    def on_syll_change(change):
        '''
        Callback function for when user selects a different syllable number
        from the Dropdown menu

        Parameters
        ----------
        change (ipywidget DropDown select event): User changes current value of DropDownMenu

        Returns
        -------
        '''

        clear_output()
        display(labeler.syll_select, output)

    # Update view when user selects new syllable from DropDownMenu
    output.observe(on_syll_change, names='value')

    # Initialize button callbacks
    labeler.next_button.on_click(labeler.on_next)
    labeler.prev_button.on_click(labeler.on_prev)
    labeler.set_button.on_click(labeler.on_set)

def interactive_syllable_stat_wrapper(index_path, model_path, info_path, max_syllables=None):
    '''
    Wrapper function to launch the interactive syllable statistics API. Users will be able to view different
    syllable statistics, sort them according to their metric of choice, and dynamically group the data to
    view individual sessions or group averages.

    Parameters
    ----------
    index_path (str): Path to index file.
    model_path (str): Path to trained model file.
    info_path (str): Path to syllable information file.
    max_syllables (int): Maximum number of syllables to plot.

    Returns
    -------
    '''

    # Initialize the statistical grapher context
    istat = InteractiveSyllableStats(index_path=index_path, model_path=model_path, info_path=info_path, max_sylls=max_syllables)

    # Compute the syllable dendrogram values
    istat.compute_dendrogram()

    # Plot the Bokeh graph with the currently selected data.
    out = interactive_output(istat.interactive_syll_stats_grapher, {
                                                      'stat': istat.stat_dropdown,
                                                      'sort': istat.sorting_dropdown,
                                                      'groupby': istat.grouping_dropdown,
                                                      'errorbar': istat.errorbar_dropdown,
                                                      'sessions': istat.session_sel,
                                                      'ctrl_group': istat.ctrl_dropdown,
                                                      'exp_group': istat.exp_dropdown
                                                      })


    display(istat.stat_widget_box, out)
    show(istat.cladogram)

def interactive_crowd_movie_comparison_preview(config_data, index_path, model_path, syll_info_path, output_dir):
    '''
    Wrapper function that launches an interactive crowd movie comparison application.
    Uses ipywidgets and Bokeh to facilitate real time user interaction.

    Parameters
    ----------
    config_data (dict): dict containing crowd movie creation parameters
    index_path (str): path to index file with paths to all the extracted sessions
    model_path (str): path to trained model containing syllable labels.
    syll_info_path (str): path to syllable information file containing syllable labels
    output_dir (str): path to directory to store crowd movies

    Returns
    -------
    '''

    with open(syll_info_path, 'r') as f:
        syll_info = yaml.safe_load(f)

    
    index, sorted_index, model_fit = init_wrapper_function(index_file=index_path, model_fit=model_path, output_dir=output_dir)

    cm_compare = CrowdMovieComparison(config_data=config_data, index_path=index_path,
                                      model_path=model_path, syll_info=syll_info, output_dir=output_dir)

    # Set widgets
    cm_compare.cm_syll_select.options = syll_info
    sessions = list(set(model_fit['metadata']['uuids']))
    cm_compare.cm_session_sel.options = [sorted_index['files'][s]['metadata']['SessionName'] for s in sessions]

    cm_compare.get_session_mean_syllable_info_df(model_fit, sorted_index)

    out = interactive_output(cm_compare.crowd_movie_preview, {'syllable': cm_compare.cm_syll_select,
                                                              'groupby': cm_compare.cm_sources_dropdown,
                                                              'nexamples': cm_compare.num_examples})
    display(out)

    cm_compare.cm_session_sel.observe(cm_compare.select_session)
    cm_compare.cm_sources_dropdown.observe(cm_compare.show_session_select)
    cm_compare.cm_trigger_button.on_click(cm_compare.on_click_trigger_button)

def interactive_plot_transition_graph_wrapper(model_path, index_path, info_path):
    '''
    Wrapper function that works as a background process that prepares the data
    for the interactive graphing function.

    Parameters
    ----------
    model_path (str): Path to trained model.
    index_path (str): Path to index file containined trained data metadata.
    info_path (str): Path to user-labeled syllable information file.

    Returns
    -------
    '''

    # Initialize Transition Graph data structure
    i_trans_graph = InteractiveTransitionGraph(model_path=model_path, index_path=index_path, info_path=info_path)

    # Load and store transition graph data
    i_trans_graph.initialize_transition_data()

    # Update threshold range values
    edge_threshold_stds = int(np.max(i_trans_graph.trans_mats)/np.std(i_trans_graph.trans_mats))
    usage_threshold_stds = int(i_trans_graph.df['usage'].max()/i_trans_graph.df['usage'].std()) + 2
    speed_threshold_stds = int(i_trans_graph.df['speed'].max() / i_trans_graph.df['speed'].std()) + 2

    i_trans_graph.edge_thresholder.options = [float('%.3f' % (np.std(i_trans_graph.trans_mats) * i)) for i in range(edge_threshold_stds)]
    i_trans_graph.edge_thresholder.index = (1, edge_threshold_stds-1)

    i_trans_graph.usage_thresholder.options = [float('%.3f' % (i_trans_graph.df['usage'].std() * i)) for i in range(usage_threshold_stds)]
    i_trans_graph.usage_thresholder.index = (0, usage_threshold_stds - 1)

    i_trans_graph.speed_thresholder.options = [float('%.3f' % (i_trans_graph.df['speed'].std() * i)) for i in range(speed_threshold_stds)]
    i_trans_graph.speed_thresholder.index = (0, speed_threshold_stds - 1)

    # Make graphs
    out = interactive_output(i_trans_graph.interactive_transition_graph_helper,
                             {'edge_threshold': i_trans_graph.edge_thresholder,
                              'usage_threshold': i_trans_graph.usage_thresholder,
                              'speed_threshold': i_trans_graph.speed_thresholder,
                              })

    # Display widgets and bokeh network plots
    display(i_trans_graph.thresholding_box, out)