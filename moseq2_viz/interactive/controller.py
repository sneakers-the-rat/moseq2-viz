'''

'''

import joblib
import numpy as np
import pandas as pd
from glob import glob
from bokeh.io import show
import ruamel.yaml as yaml
from bokeh.layouts import column
from IPython.display import display
from bokeh.models.widgets import Div
from moseq2_viz.util import parse_index
from moseq2_viz.interactive.widgets import *
from moseq2_viz.interactive.view import bokeh_plotting
from sklearn.metrics.pairwise import pairwise_distances
from scipy.cluster.hierarchy import linkage, dendrogram
from moseq2_viz.model.label_util import get_sorted_syllable_stat_ordering, get_syllable_muteness_ordering
from moseq2_viz.scalars.util import scalars_to_dataframe, compute_session_centroid_speeds, compute_mean_syll_scalar
from moseq2_viz.model.util import parse_model_results, results_to_dataframe, get_syllable_usages, relabel_by_usage

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
    Interactive Syllable Statistics grapher class that holds the context for the current
     inputted session.

    '''

    def __init__(self, index_path, model_path, info_path, max_sylls):
        '''
        Initialize the main data inputted into the current context

        Parameters
        ----------
        index_path (str): Path to index file.
        model_path (str): Path to trained model file.
        info_path (str): Path to syllable information file.
        max_sylls (int): Maximum number of syllables to plot.
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
        Computes the pairwise distances between the included model AR-states, and
        generates the graph information to be plotted after the stats.

        Returns
        -------
        '''

        # Get Pairwise distances
        X = pairwise_distances(self.ar_mats, metric='euclidean')
        Z = linkage(X, 'ward')

        # Get Dendrogram Metadata
        self.results = dendrogram(Z, distance_sort=True, no_plot=True, get_leaves=True)

        # Get Graph layout info
        icoord, dcoord = self.results['icoord'], self.results['dcoord']

        icoord = pd.DataFrame(icoord) - 5
        icoord = icoord * (self.df['syllable'].max() / icoord.max().max())
        self.icoord = icoord.values

        dcoord = pd.DataFrame(dcoord)
        dcoord = dcoord * (self.df['usage'].max() / dcoord.max().max())
        self.dcoord = dcoord.values

    def interactive_stat_helper(self):
        '''
        Computes and saves the all the relevant syllable information to be displayed.
         Loads the syllable information dict and merges it with the syllable statistics DataFrame.

        Returns
        -------
        '''

        # Read syllable information dict
        with open(self.info_path, 'r') as f:
            syll_info = yaml.safe_load(f)

        # Getting number of syllables included in the info dict
        max_sylls = len(list(syll_info.keys()))
        for k in range(max_sylls):
            del syll_info[str(k)]['group_info']

        info_df = pd.DataFrame(list(syll_info.values()), index=[int(k) for k in list(syll_info.keys())]).sort_index()
        info_df['syllable'] = info_df.index

        # Load the model
        model_data = parse_model_results(joblib.load(self.model_path))

        # Relabel the models, and get the order mapping
        labels, mapping = relabel_by_usage(model_data['labels'], count='usage')

        # Get max syllables if None is given
        syllable_usages = get_syllable_usages({'labels': labels}, count='usage')
        cumulative_explanation = 100 * np.cumsum(syllable_usages)
        if self.max_sylls == None:
            self.max_sylls = np.argwhere(cumulative_explanation >= 90)[0][0]

        # Read AR matrices and reorder according to the syllable mapping
        ar_mats = np.array(model_data['model_parameters']['ar_mat'])
        self.ar_mats = np.reshape(ar_mats, (100, -1))[mapping][:self.max_sylls]

        # Read index file
        index, sorted_index = parse_index(self.index_path)

        # Load scalar Dataframe to compute syllable speeds
        scalar_df = scalars_to_dataframe(sorted_index)

        # Compute a syllable summary Dataframe containing usage-based
        # sorted/relabeled syllable usage and duration information from [0, max_syllable) inclusive
        df, label_df = results_to_dataframe(model_data, index, count='usage',
                                            max_syllable=self.max_sylls, sort=True, compute_labels=True)

        # Compute centroid speeds
        scalar_df['centroid_speed_mm'] = compute_session_centroid_speeds(scalar_df)

        # Compute and append additional syllable scalar data
        df = compute_mean_syll_scalar(df, scalar_df, label_df, groups=None, max_sylls=self.max_sylls)
        df = compute_mean_syll_scalar(df, scalar_df, label_df, scalar='dist_to_center_px', groups=None, max_sylls=self.max_sylls)

        self.df = df.merge(info_df, on='syllable')

    def interactive_syll_stats_grapher(self, stat, sort, groupby, sessions, ctrl_group, exp_group):
        '''
        Helper function that is responsible for handling ipywidgets interactions and updating the currently
         displayed Bokeh plot.

        Parameters
        ----------
        stat (list or ipywidgets.DropDown): Statistic to plot: ['usage', 'speed', 'distance to center']
        sort (list or ipywidgets.DropDown): Statistic to sort syllables by (in descending order).
            ['usage', 'speed', 'distance to center', 'similarity', 'mutation'].
        groupby (list or ipywidgets.DropDown): Data to plot; either group averages, or individual session data.
        sessions (list or ipywidgets.MultiSelect): List of selected sessions to display data from.
        ctrl_group (str or ipywidgets.DropDown): Name of control group to compute mutation sorting with.
        exp_group (str or ipywidgets.DropDown): Name of comparative group to compute mutation sorting with.

        Returns
        -------
        '''

        # Get current dataFrame to plot
        df = self.df

        # Handle names to query DataFrame with
        if stat == 'distance to center':
            stat = 'dist_to_center'
        if sort == 'distance to center':
            sort = 'dist_to_center'

        # Get selected syllable sorting
        if sort == 'mutation':
            # display Text for groups to input experimental groups
            ordering = get_syllable_muteness_ordering(df, ctrl_group, exp_group, stat=stat)
        elif sort == 'similarity':
            ordering = self.results['leaves']
        elif sort != 'usage':
            ordering, _ = get_sorted_syllable_stat_ordering(df, stat=sort)
        else:
            ordering = range(len(df.syllable.unique()))

        # Handle selective display for whether mutation sort is selected
        if sort == 'mutation':
            mutation_box.layout.display = "block"
        else:
            mutation_box.layout.display = "none"

        # Handle selective display to select included sessions to graph
        if groupby == 'SessionName':
            session_sel.layout.display = "block"
            df = df[df['SessionName'].isin(session_sel.value)]
        else:
            session_sel.layout.display = "none"

        # Create Bokeh plot
        bokeh_plotting(df, stat, ordering, groupby)