'''

Interactive applications for labeling syllables, plotting syllable stats, comparing crowd movies
    and plotting transition graphs.
Interactivity functionality is facilitated via IPyWidget and Bokeh.

'''

import joblib
import numpy as np
import pandas as pd
from glob import glob
import networkx as nx
from bokeh.io import show
import ruamel.yaml as yaml
from bokeh.layouts import column
from IPython.display import display
from bokeh.models.widgets import Div
from moseq2_viz.util import parse_index
from moseq2_viz.interactive.widgets import *
from sklearn.metrics.pairwise import pairwise_distances
from scipy.cluster.hierarchy import linkage, dendrogram
from moseq2_viz.interactive.view import bokeh_plotting, plot_interactive_transition_graph
from moseq2_viz.model.label_util import get_sorted_syllable_stat_ordering, get_syllable_muteness_ordering
from moseq2_viz.scalars.util import scalars_to_dataframe, compute_session_centroid_speeds, compute_mean_syll_speed
from moseq2_viz.model.util import parse_model_results, results_to_dataframe, get_syllable_usages, relabel_by_usage
from moseq2_viz.model.trans_graph import handle_graph_layout, convert_transition_matrix_to_ebunch, \
    convert_ebunch_to_graph, make_transition_graphs

class SyllableLabeler:
    '''

    '''

    def __init__(self, max_sylls, save_path):
        '''

        Parameters
        ----------
        max_sylls
        save_path
        '''

        self.save_path = save_path
        self.max_sylls = max_sylls
        self.syll_info = {str(i): {'label': '', 'desc': '', 'crowd_movie_path': ''} for i in range(max_sylls)}

    def on_next(self, event):
        '''

        Parameters
        ----------
        event

        Returns
        -------

        '''

        # Updating dict
        self.syll_info[str(syll_select.index)]['label'] = lbl_name_input.value
        self.syll_info[str(syll_select.index)]['desc'] = desc_input.value

        # Updating selection to trigger update
        if syll_select.index != int(list(syll_select.options.keys())[-1]):
            syll_select.index += 1
        else:
            syll_select.index = 0

        # Updating input values with current dict entries
        lbl_name_input.value = self.syll_info[str(syll_select.index)]['label']
        desc_input.value = self.syll_info[str(syll_select.index)]['desc']

    def on_prev(self, event):
        '''

        Parameters
        ----------
        event

        Returns
        -------

        '''

        self.syll_info[str(syll_select.index)]['label'] = lbl_name_input.value
        self.syll_info[str(syll_select.index)]['desc'] = desc_input.value

        if syll_select.index != 0:
            syll_select.index -= 1
        else:
            syll_select.index = int(list(syll_select.options.keys())[-1])

        lbl_name_input.value = self.syll_info[str(syll_select.index)]['label']
        desc_input.value = self.syll_info[str(syll_select.index)]['desc']

    def on_set(self, event):
        '''

        Parameters
        ----------
        event

        Returns
        -------

        '''

        self.syll_info[str(syll_select.index)]['label'] = lbl_name_input.value
        self.syll_info[str(syll_select.index)]['desc'] = desc_input.value

        with open(self.save_path, 'w+') as f:
            yaml.safe_dump(self.syll_info, f)

        set_button.button_type = 'success'

    def interactive_syllable_labeler(self, syllables):
        '''

        Parameters
        ----------
        syllables

        Returns
        -------

        '''

        set_button.button_type = 'primary'

        if len(syllables['label']) > 0:
            lbl_name_input.value = syllables['label']

        if len(syllables['desc']) > 0:
            desc_input.value = syllables['desc']

        # update label
        cm_lbl.text = f'Crowd Movie {syll_select.index + 1}/{len(syll_select.options)}'

        # get movie path
        cm_path = syllables['crowd_movie_path']

        video_div = f'''
                        <h2>{syll_select.index}: {syllables['label']}</h2>
                        <video
                            src="{cm_path}"; alt="{cm_path}"; height="450"; width="450"; preload="true";
                            style="float: left; type: "video/mp4"; margin: 0px 10px 10px 0px;
                            border="2"; autoplay controls loop>
                        </video>
                    '''

        div = Div(text=video_div, style={'width': '100%'})
        out = widgets.Output(height='500px')
        layout = column([div, cm_lbl])

        with out:
            show(layout)

        grid = widgets.GridspecLayout(2, 2)
        grid[0, 0] = out
        grid[0, 1] = data_box
        grid[1, :2] = button_box

        display(grid)

    def get_crowd_movie_paths(self, crowd_movie_dir):
        '''

        Parameters
        ----------
        syll_info
        crowd_movie_dir

        Returns
        -------

        '''

        crowd_movie_paths = [f for f in glob(crowd_movie_dir + '*') if '.mp4' in f]

        for cm in crowd_movie_paths:
            syll_num = cm.split('sorted-id-')[1].split()[0]
            if syll_num in self.syll_info.keys():
                self.syll_info[syll_num]['crowd_movie_path'] = cm

class InteractiveSyllableStats:
    '''

    '''

    def __init__(self, index_path, model_path, info_path, max_sylls):
        '''

        Parameters
        ----------
        index_path
        model_path
        info_path
        max_sylls
        '''

        self.model_path = model_path
        self.info_path = info_path
        self.max_sylls = max_sylls
        self.index_path = index_path
        self.df = None

        self.ar_mats = None
        self.results = None
        self.icoord, self.dcoord = None, None

    def compute_dendrogram(self):
        '''

        Returns
        -------

        '''

        # Get Pairwise distances
        X = pairwise_distances(self.ar_mats, metric='euclidean')
        Z = linkage(X, 'ward')

        # Get Dendogram Metadata
        self.results = dendrogram(Z, distance_sort=True, no_plot=True, get_leaves=True)

        # Get Graph Info
        icoord, dcoord = self.results['icoord'], self.results['dcoord']

        icoord = pd.DataFrame(icoord) - 5
        icoord = icoord * (self.df['syllable'].max() / icoord.max().max())
        self.icoord = icoord.values

        dcoord = pd.DataFrame(dcoord)
        dcoord = dcoord * (self.df['usage'].max() / dcoord.max().max())
        self.dcoord = dcoord.values

    def interactive_stat_helper(self):
        '''

        Returns
        -------

        '''

        with open(self.info_path, 'r') as f:
            syll_info = yaml.safe_load(f)

        info_df = pd.DataFrame(list(syll_info.values()), index=[int(k) for k in list(syll_info.keys())]).sort_index()
        info_df['syllable'] = info_df.index

        model_data = parse_model_results(joblib.load(self.model_path))

        labels, mapping = relabel_by_usage(model_data['labels'], count='usage')

        ar_mats = np.array(model_data['model_parameters']['ar_mat'])
        self.ar_mats = np.reshape(ar_mats, (100, -1))[mapping][:self.max_sylls]

        syllable_usages = get_syllable_usages({'labels': labels}, count='usage')
        cumulative_explanation = 100 * np.cumsum(syllable_usages)
        if self.max_sylls == None:
            self.max_sylls = np.argwhere(cumulative_explanation >= 90)[0][0]

        sorted_index = parse_index(self.index_path)[1]

        # Load scalar Dataframe to compute syllable speeds
        scalar_df = scalars_to_dataframe(sorted_index)

        # Compute a syllable summary Dataframe containing usage-based
        # sorted/relabeled syllable usage and duration information from [0, max_syllable) inclusive
        df, label_df = results_to_dataframe(model_data, sorted_index, count='usage',
                                            max_syllable=self.max_sylls, sort=True, compute_labels=True)

        scalar_df['centroid_speed_mm'] = compute_session_centroid_speeds(scalar_df)
        df = compute_mean_syll_speed(df, scalar_df, label_df, groups=None, max_sylls=self.max_sylls)

        self.df = df.merge(info_df, on='syllable')

    def interactive_syll_stats_grapher(self, df, obj, stat, sort, groupby, sessions, ctrl_group, exp_group):
        '''

        Parameters
        ----------
        df
        obj
        stat
        sort
        groupby
        sessions
        ctrl_group
        exp_group

        Returns
        -------

        '''

        if sort == 'mutation':
            # display Text for groups to input experimental groups
            ordering = get_syllable_muteness_ordering(df, ctrl_group, exp_group, stat=stat)
        elif sort == 'similarity':
            ordering = self.results['leaves']
        elif sort != 'usage':
            ordering, _ = get_sorted_syllable_stat_ordering(df, stat=sort)
        else:
            ordering = range(len(df.syllable.unique()))

        if groupby == 'SessionName':
            session_sel.layout.display = "block"
            df = df[df['SessionName'].isin(session_sel.value)]
        else:
            session_sel.layout.display = "none"

        bokeh_plotting(df, stat, ordering, groupby)

class InteractiveTransitionGraph:
    '''

    Interactive transition graph class used to facilitate interactive graph generation
    and thresholding functionality.

    '''

    def __init__(self, trans_mats, usages, max_sylls, group):
        '''
        Initializes context variables

        Parameters
        ----------
        trans_mats (list of 2D np.arrays): list of transition matrices for each group
        usages (list of OrderedDicts): list of dicts of syllable number keys paired with their counts
        max_sylls (int): maximum number of syllables to include in graph
        group (list): list of unique group names corresponding to trans_mats
        '''

        self.trans_mats = trans_mats
        self.usages = usages
        self.max_sylls = max_sylls
        self.group = group

    def interactive_transition_graph_helper(self, syll_info, trans_mats, edge_threshold, usage_threshold):
        '''

        Helper function that generates all the transition graphs given the currently selected
        thresholding values, then displays them in a Jupyter notebook or web page.

        Parameters
        ----------
        syll_info (dict): Dict of user-labeled syllable information.
        edge_threshold (tuple): Transition probability range to include in graphs.
        usage_threshold (tuple): Syllable usage range to include in graphs.

        Returns
        -------
        '''

        # Get graph node anchors
        usages, anchor, usages_anchor, ngraphs = handle_graph_layout(trans_mats, self.usages, anchor=0)

        weights = trans_mats

        # Create graph with nodes and edges
        ebunch_anchor, orphans = convert_transition_matrix_to_ebunch(
            weights[anchor], trans_mats[anchor], edge_threshold=edge_threshold,
            keep_orphans=True, usages=usages_anchor,
            usage_threshold=usage_threshold, max_syllable=self.max_sylls-1)

        # Get graph anchor
        graph_anchor = convert_ebunch_to_graph(ebunch_anchor)

        pos = nx.circular_layout(graph_anchor, scale=1)

        # make transition graphs
        group_names = self.group.copy()

        # prepare transition graphs
        usages, group_names, _, _, _, graphs = make_transition_graphs(trans_mats,
                                                             self.usages[:len(self.group)],
                                                             self.group,
                                                             group_names,
                                                             usages_anchor,
                                                             pos, ebunch_anchor, edge_threshold,
                                                             difference_threshold=0.0005, orphans=orphans,
                                                             orphan_weight=0, edge_width_scale=100)
        # interactive plot transition graphs
        plot_interactive_transition_graph(graphs, pos, self.group,
                                          group_names, usages,
                                          syll_info, self.max_sylls)
