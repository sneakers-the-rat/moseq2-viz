'''

'''
import qgrid
import ipywidgets as widgets
from ipywidgets import HBox, VBox
from bokeh.models.widgets import PreText

### Syllable Labeler Widgets
# UI widgets
class SyllableLabelerWidgets:

    def __init__(self):
            
        self.syll_select = widgets.Dropdown(options={}, description='Syllable #:', disabled=False)

        # labels
        self.cm_lbl = PreText(text="Crowd Movie") # current crowd movie number

        self.syll_lbl = widgets.Label(value="Syllable Name") # name user prompt label
        self.desc_lbl = widgets.Label(value="Short Description") # description label

        self.syll_info_lbl = widgets.Label(value="Syllable Info", font_size=24)

        self.syll_usage_value_lbl = widgets.Label(value="")
        self.syll_speed_value_lbl = widgets.Label(value="")
        self.syll_duration_value_lbl = widgets.Label(value="")

        # text input widgets
        self.lbl_name_input = widgets.Text(value='',
                                    placeholder='Syllable Name',
                                    tooltip='2 word name for syllable')

        self.desc_input = widgets.Text(value='',
                                placeholder='Short description of behavior',
                                tooltip='Describe the behavior.',
                                layout=widgets.Layout(height='260px'),
                                disabled=False)

        # buttons
        self.prev_button = widgets.Button(description='Prev', disabled=False, tooltip='Previous Syllable', layout=widgets.Layout(flex='2 1 0', width='auto', height='40px'))
        self.set_button = widgets.Button(description='Save Setting', disabled=False, tooltip='Save current inputs.', button_style='primary', layout=widgets.Layout(flex='3 1 0', width='auto', height='40px'))
        self.next_button = widgets.Button(description='Next', disabled=False, tooltip='Next Syllable', layout=widgets.Layout(flex='2 1 0', width='auto', height='40px'))

        # Box Layouts
        self.label_layout = widgets.Layout(flex_flow='column', height='75%')
        self.input_layout = widgets.Layout(height='200px')

        self.ui_layout = widgets.Layout(flex_flow='row', width='auto', max_height='50px')
        self.data_layout = widgets.Layout(flex_flow='row', max_height='150px', justify_content='space-around',
                                    padding='top', align_content='center', width='auto')
                                    
        self.center_layout = widgets.Layout(display='flex', align_items='center')

        # label box
        self.lbl_box = VBox([self.syll_lbl, self.desc_lbl], layout=self.label_layout)

        # input box
        self.input_box = VBox([self.lbl_name_input, self.desc_input], layout=self.label_layout)

        # syllable info box
        self.info_boxes = VBox([self.syll_info_lbl], layout=self.center_layout)

        self.data_box = VBox([HBox([self.lbl_box, self.input_box], layout=self.data_layout), self.info_boxes])

        # button box
        self.button_box = HBox([self.prev_button, self.set_button, self.next_button], layout=self.ui_layout)

### Syllable Stat Widgets
class SyllableStatWidgets:

    def __init__(self):

        self.layout_hidden = widgets.Layout(display='none')
        self.layout_visible = widgets.Layout(display='block')

        self.stat_dropdown = widgets.Dropdown(options=['usage', 'speed', 'distance to center'], description='Stat to Plot:', disabled=False)

        # add dist to center
        self.sorting_dropdown = widgets.Dropdown(options=['usage', 'speed', 'distance to center', 'similarity', 'difference'], description='Sort Syllables By:', disabled=False)
        self.ctrl_dropdown = widgets.Dropdown(options=[], description='Group 1:', disabled=False)
        self.exp_dropdown = widgets.Dropdown(options=[], description='Group 2:', disabled=False)

        self.grouping_dropdown = widgets.Dropdown(options=['group', 'SessionName'], description='Group Data By:', disabled=False)
        self.session_sel = widgets.SelectMultiple(options=[], description='Sessions to Graph:', layout=self.layout_hidden, disabled=False)

        self.errorbar_dropdown = widgets.Dropdown(options=['SEM', 'STD'], description='Error Bars:', disabled=False)

        ## boxes
        self.stat_box = VBox([self.stat_dropdown, self.errorbar_dropdown])
        self.mutation_box = VBox([self.ctrl_dropdown, self.exp_dropdown])

        self.sorting_box = VBox([self.sorting_dropdown, self.mutation_box])
        self.session_box = VBox([self.grouping_dropdown, self.session_sel])

        self.stat_widget_box = HBox([self.stat_box, self.sorting_box, self.session_box])

### Group Setting Widgets
class GroupSettingWidgets:

    def __init__(self):
        self.col_opts = {
            'editable': False,
            'toolTip': "Not editable"
        }

        self.col_defs = {
            'group': {
                'editable': True,
                'toolTip': 'editable'
            }
        }

        self.group_input = widgets.Text(value='', placeholder='Enter Group Name to Set', description='Desired Group Name', continuous_update=False, disabled=False)
        self.save_button = widgets.Button(description='Set Group', disabled=False, tooltip='Set Group')
        self.update_index_button = widgets.Button(description='Update Index File', disabled=False, tooltip='Save Parameters')

        self.group_set = widgets.HBox([self.group_input, self.save_button, self.update_index_button])
        qgrid.set_grid_option('forceFitColumns', False)
        qgrid.set_grid_option('enableColumnReorder', True)
        qgrid.set_grid_option('highlightSelectedRow', True)
        qgrid.set_grid_option('highlightSelectedCell', False)

### Transition Graph Widgets
class TransitionGraphWidgets:
    
    
    '''
    edge_thresholder = widgets.FloatRangeSlider(value=[0.0025, 1], min=0, max=1, step=0.001, style=style, readout_format='.4f',
                                                description='Edges weights to display', continuous_update=False)
    usage_thresholder = widgets.FloatRangeSlider(value=[0, 1], min=0, max=1, step=0.001, style=style, readout_format='.4f',
                                                description='Usage nodes to display', continuous_update=False)
    speed_thresholder = widgets.FloatRangeSlider(value=[-25, 200], min=-50, max=200, step=1, style=style, readout_format='.1f',
                                                description='Threshold nodes by speed', continuous_update=False)
    '''

    def __init__(self):    
        style = {'description_width': 'initial', 'display':'flex-grow', 'align_items':'stretch'}

        self.color_nodes_button = widgets.Checkbox(value=False, description='Color Nodes by Speed', disabled=False, indent=False)
        
        self.edge_thresholder = widgets.SelectionRangeSlider(options=['tmp'], style=style,
                                                    description='Edges weights to display', continuous_update=False)
        self.usage_thresholder = widgets.SelectionRangeSlider(options=['tmp'], style=style, readout_format='.4f',
                                                    description='Usage nodes to display', continuous_update=False)
        self.speed_thresholder = widgets.SelectionRangeSlider(options=['tmp'], style=style, readout_format='.1f',
                                                    description='Threshold nodes by speed', continuous_update=False)

        self.thresholding_box = HBox([VBox([self.edge_thresholder, self.usage_thresholder]), VBox([self.color_nodes_button, self.speed_thresholder])])

### Crowd Movie Comparison Widgets
class CrowdMovieCompareWidgets:

    def __init__(self):
        style = {'description_width': 'initial'}

        self.label_layout = widgets.Layout(flex_flow='column', max_height='100px')
        self.layout_hidden = widgets.Layout(display='none')
        self.layout_visible = widgets.Layout(display='block')

        self.cm_syll_select = widgets.Dropdown(options=[], description='Syllable #:', disabled=False)
        self.num_examples = widgets.IntSlider(value=20, min=1, max=40, step=1, description='Number of Example Mice:', disabled=False, continuous_update=False, style=style)

        self.cm_sources_dropdown = widgets.Dropdown(options=['group', 'SessionName'], description='Make Crowd Movies From:', disabled=False)
        self.cm_session_sel = widgets.SelectMultiple(options=[], description='Sessions to Graph:', layout=self.layout_hidden, disabled=False)
        self.cm_trigger_button = widgets.Button(description='Generate Movies', disabled=False, tooltip='Make Crowd Movies', layout=self.layout_hidden)

        self.syllable_box = VBox([self.cm_syll_select, self.num_examples])
        self.session_box = VBox([self.cm_sources_dropdown, self.cm_session_sel, self.cm_trigger_button])

        self.widget_box = HBox([self.syllable_box, self.session_box]) # add layout
