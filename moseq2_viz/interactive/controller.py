'''

'''

from glob import glob
from bokeh.io import show
import ruamel.yaml as yaml
from bokeh.layouts import column
from IPython.display import display
from bokeh.models.widgets import Div
from moseq2_viz.util import parse_index
from moseq2_viz.interactive.widgets import *
from moseq2_viz.model.util import results_to_dataframe
from moseq2_viz.scalars.util import scalars_to_dataframe, compute_session_centroid_speeds, compute_mean_syll_speed

class SyllableLabeler:
    '''

    Class that contains functionality for previewing syllable crowd movies and
     user interactions with buttons and menus.

    '''

    def __init__(self, model_fit, index_file, max_sylls, save_path):
        '''
        Initializes class context parameters, reads and creates the syllable information dict.

        Parameters
        ----------
        model_fit (dict): Loaded trained model dict.
        index_file (str): Path to saved index file.
        max_sylls (int): Maximum number of syllables to preview and label.
        save_path (str): Path to save syllable label information dictionary.
        '''

        self.save_path = save_path
        self.max_sylls = max_sylls

        self.model_fit = model_fit
        self.sorted_index = parse_index(index_file)[1]
        self.syll_info = {str(i): {'label': '', 'desc': '', 'crowd_movie_path': ''} for i in range(max_sylls)}

    def on_next(self, event):
        '''
        Callback function to trigger an view update when the user clicks the "Next" button.

        Parameters
        ----------
        event (ipywidgets.ButtonClick): User clicks next button.

        Returns
        -------
        '''

        # Updating dict
        self.syll_info[str(syll_select.index)]['label'] = lbl_name_input.value
        self.syll_info[str(syll_select.index)]['desc'] = desc_input.value

        # Handle cycling through syllable labels
        if syll_select.index != int(list(syll_select.options.keys())[-1]):
            # Updating selection to trigger update
            syll_select.index += 1
        else:
            syll_select.index = 0

        # Updating input values with current dict entries
        lbl_name_input.value = self.syll_info[str(syll_select.index)]['label']
        desc_input.value = self.syll_info[str(syll_select.index)]['desc']

    def on_prev(self, event):
        '''
        Callback function to trigger an view update when the user clicks the "Previous" button.

        Parameters
        ----------
        event (ipywidgets.ButtonClick): User clicks 'previous' button.

        Returns
        -------
        '''

        # Update syllable information dict
        self.syll_info[str(syll_select.index)]['label'] = lbl_name_input.value
        self.syll_info[str(syll_select.index)]['desc'] = desc_input.value

        # Handle cycling through syllable labels
        if syll_select.index != 0:
            # Updating selection to trigger update
            syll_select.index -= 1
        else:
            syll_select.index = int(list(syll_select.options.keys())[-1])

        # Reloading previously inputted text area string values
        lbl_name_input.value = self.syll_info[str(syll_select.index)]['label']
        desc_input.value = self.syll_info[str(syll_select.index)]['desc']

    def on_set(self, event):
        '''
        Callback function to save the dict to syllable information file.

        Parameters
        ----------
        event (ipywidgets.ButtonClick): User clicks the 'Save' button.

        Returns
        -------
        '''

        # Update dict
        self.syll_info[str(syll_select.index)]['label'] = lbl_name_input.value
        self.syll_info[str(syll_select.index)]['desc'] = desc_input.value

        yml = yaml.YAML()
        yml.indent(mapping=3, offset=2)

        # Write to file
        with open(self.save_path, 'w+') as f:
            yml.dump(self.syll_info, f)

        # Update button style
        set_button.button_type = 'success'

    def get_mean_syllable_info(self):
        '''
        Populates syllable information dict with usage and scalar information.

        Returns
        -------
        '''

        # Load scalar Dataframe to compute syllable speeds
        scalar_df = scalars_to_dataframe(self.sorted_index)

        # Compute a syllable summary Dataframe containing usage-based
        # sorted/relabeled syllable usage and duration information from [0, max_syllable) inclusive
        df, label_df = results_to_dataframe(self.model_fit, self.sorted_index, count='usage',
                                            max_syllable=self.max_sylls, sort=True, compute_labels=True)

        # Compute syllable speed
        scalar_df['centroid_speed_mm'] = compute_session_centroid_speeds(scalar_df)
        df = compute_mean_syll_speed(df, scalar_df, label_df, groups=None, max_sylls=self.max_sylls)

        # Get mean syllable info dict
        mean_syll_info = df.groupby('syllable').mean().to_dict()

        # Populate syllable info dict with usage and scalar values
        for syll in range(self.max_sylls):
            self.syll_info[str(syll)]['usage'] = mean_syll_info['usage'][syll]
            self.syll_info[str(syll)]['speed'] = mean_syll_info['speed'][syll]
            self.syll_info[str(syll)]['duration'] = mean_syll_info['duration'][syll]

    def interactive_syllable_labeler(self, syllables):
        '''
        Helper function that facilitates the interactive view. Function will create a Bokeh Div object
        that will display the current video path.

        Parameters
        ----------
        syllables (int or ipywidgets.DropDownMenu): Current syllable to label

        Returns
        -------
        '''

        set_button.button_type = 'primary'

        # Set current widget values
        if len(syllables['label']) > 0:
            lbl_name_input.value = syllables['label']

        if len(syllables['desc']) > 0:
            desc_input.value = syllables['desc']

        # Update label
        cm_lbl.text = f'Crowd Movie {syll_select.index + 1}/{len(syll_select.options)}'
        syll_usage_value_lbl.value = '{:0.3f}'.format(syllables['usage'])
        syll_speed_value_lbl.value = '{:0.3f}'.format(syllables['speed'])
        syll_duration_value_lbl.value = '{:0.3f}'.format(syllables['duration'])

        # Get current movie path
        cm_path = syllables['crowd_movie_path']

        # Create syllable crowd movie HTML div to embed
        video_div = f'''
                        <h2>{syll_select.index}: {syllables['label']}</h2>
                        <video
                            src="{cm_path}"; alt="{cm_path}"; height="450"; width="450"; preload="true";
                            style="float: left; type: "video/mp4"; margin: 0px 10px 10px 0px;
                            border="2"; autoplay controls loop>
                        </video>
                    '''

        # Create embedded HTML Div and view layout
        div = Div(text=video_div, style={'width': '100%'})
        layout = column([div, cm_lbl])

        # Insert Bokeh div into ipywidgets Output widget to display
        out = widgets.Output(height='500px')
        with out:
            show(layout)

        # Create grid layout to display all the widgets
        grid = widgets.GridspecLayout(2, 2)
        grid[0, 0] = out
        grid[0, 1] = data_box
        grid[1, :2] = button_box

        # Display all widgets
        display(grid)

    def get_crowd_movie_paths(self, crowd_movie_dir):
        '''
        Populates the syllable information dict with the respective crowd movie paths.

        Parameters
        ----------
        crowd_movie_dir (str): Path to directory containing all the generated crowd movies

        Returns
        -------
        '''

        # Get movie paths
        crowd_movie_paths = [f for f in glob(crowd_movie_dir + '*') if '.mp4' in f]

        for cm in crowd_movie_paths:
            # Parse paths to get corresponding syllable number
            syll_num = cm.split('sorted-id-')[1].split()[0]
            if syll_num in self.syll_info.keys():
                self.syll_info[syll_num]['crowd_movie_path'] = cm