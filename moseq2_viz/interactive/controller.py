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
        self.syll_info = {str(i): {'label': '', 'desc': '', 'crowd_movie_path': '', 'group_info': {}} for i in range(max_sylls)}

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

        # Get all unique groups in df
        self.groups = df.group.unique()

        # Get grouped DataFrame
        group_df = df.groupby(('group', 'syllable'), as_index=False).mean()

        # Get array of grouped syllable info
        group_dicts = []
        for group in self.groups:
            group_dict = {
                group: group_df[group_df['group'] == group].drop('group', axis=1).reset_index(drop=True).to_dict()}
            group_dicts.append(group_dict)

        for gd in group_dicts:
            group_name = list(gd.keys())[0]
            for syll in range(self.max_sylls):
                self.syll_info[str(syll)]['group_info'][group_name] = {
                    'usage': gd[group_name]['usage'][syll],
                    'speed': gd[group_name]['speed'][syll],
                    'duration': gd[group_name]['duration'][syll]
                }

    def get_group_info_widgets(self, group_name, group_info):
        '''
        Instantiates new syllable information widgets to display
        group_name (str): Name of the widget grouping
        group_info (dict):

        Returns
        -------
        info_box (ipywidgets.VBox):
        '''

        # Syllable label scalar names
        group_lbl = widgets.Label(value="Group Name:")
        syll_usage_lbl = widgets.Label(value="Syllable Usage:")
        syll_speed_lbl = widgets.Label(value="Syllable Speed:")
        syll_duration_lbl = widgets.Label(value="Syllable Duration:")

        # Syllable label scalar values
        group_value_lbl = widgets.Label(value=group_name)
        syll_usage_value_lbl = widgets.Label(value='{:0.3f}'.format(group_info['usage']))
        syll_speed_value_lbl = widgets.Label(value='{:0.3f} mm/s'.format(group_info['speed']))
        syll_duration_value_lbl = widgets.Label(value='{:0.3f} ms'.format(group_info['duration']))

        # Group info widgets into horizontal layout boxes
        group_box = HBox([group_lbl, group_value_lbl])
        usage_box = HBox([syll_usage_lbl, syll_usage_value_lbl])
        speed_box = HBox([syll_speed_lbl, syll_speed_value_lbl])
        duration_box = HBox([syll_duration_lbl, syll_duration_value_lbl])

        # syllable info box
        info_box = VBox([group_box, usage_box, speed_box, duration_box], layout=info_layout)

        return info_box

    def set_group_info_widgets(self, group_info):
        for wid in info_boxes.children:
            if isinstance(wid, widgets.VBox):
                # get group
                group_name = wid.children[0].children[1].value

                # update usage
                wid.children[1].children[1].value = '{:0.3f}'.format(group_info[group_name]['usage'])

                # update speed
                wid.children[2].children[1].value = '{:0.3f} mm/s'.format(group_info[group_name]['speed'])

                # update duration
                wid.children[3].children[1].value = '{:0.3f} ms'.format(group_info[group_name]['duration'])


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

        # Update scalar values
        group_info = self.syll_info[str(syll_select.index)]['group_info']
        self.set_group_info_widgets(group_info)

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
        vid_out = widgets.Output(height='500px')
        with vid_out:
            show(layout)

        # Create grid layout to display all the widgets
        grid = widgets.GridspecLayout(1, 2, height='500px', layout=widgets.Layout(align_items='flex-start'))
        grid[0, 0] = vid_out
        grid[0, 1] = data_box

        # Display all widgets
        display(grid, button_box)

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