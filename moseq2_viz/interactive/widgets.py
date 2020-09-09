'''

'''
import qgrid
import ipywidgets as widgets
from ipywidgets import HBox, VBox
from bokeh.models.widgets import PreText

### Syllable Labeler Widgets
# UI widgets
syll_select = widgets.Dropdown(options={}, description='Syllable #:', disabled=False)

# labels
cm_lbl = PreText(text="Crowd Movie") # current crowd movie number

syll_lbl = widgets.Label(value="Syllable Name") # name user prompt label
desc_lbl = widgets.Label(value="Short Description") # description label

syll_info_lbl = widgets.Label(value="Syllable Info", font_size=24)

syll_usage_value_lbl = widgets.Label(value="")
syll_speed_value_lbl = widgets.Label(value="")
syll_duration_value_lbl = widgets.Label(value="")

# text input widgets
lbl_name_input = widgets.Text(value='',
                              placeholder='Syllable Name',
                              tooltip='2 word name for syllable')

desc_input = widgets.Text(value='',
                          placeholder='Short description of behavior',
                          tooltip='Describe the behavior.',
                          layout=widgets.Layout(height='260px'),
                          disabled=False)

# buttons
prev_button = widgets.Button(description='Prev', disabled=False, tooltip='Previous Syllable', layout=widgets.Layout(flex='2 1 0', width='auto', height='40px'))
set_button = widgets.Button(description='Save Setting', disabled=False, tooltip='Save current inputs.', button_style='primary', layout=widgets.Layout(flex='3 1 0', width='auto', height='40px'))
next_button = widgets.Button(description='Next', disabled=False, tooltip='Next Syllable', layout=widgets.Layout(flex='2 1 0', width='auto', height='40px'))

# Box Layouts
label_layout = widgets.Layout(flex_flow='column', max_height='100px')
input_layout = widgets.Layout(max_height='200px') # vbox

ui_layout = widgets.Layout(flex_flow='row', width='auto', max_height='50px')
data_layout = widgets.Layout(flex_flow='row', justify_content='space-between',
                             align_content='center', width='auto')
info_layout = widgets.Layout(height='auto', flex_flow='column', display='flex',
                             align_items='center', border='solid', width='100%')
center_layout = widgets.Layout(display='flex', align_items='center')

# label box
lbl_box = VBox([syll_lbl, desc_lbl], layout=label_layout)

# input box
input_box = VBox([lbl_name_input, desc_input], layout=label_layout)

# syllable info box
info_boxes = VBox([syll_info_lbl], layout=center_layout)

data_box = VBox([HBox([lbl_box, input_box], layout=data_layout), info_boxes])

# button box
button_box = HBox([prev_button, set_button, next_button], layout=ui_layout)

### Syllable Stat Widgets
## layouts
layout_hidden  = widgets.Layout(display='none')
layout_visible = widgets.Layout(display='block')

stat_dropdown = widgets.Dropdown(options=['usage', 'speed'], description='Stat to Plot:', disabled=False)

# add dist to center
sorting_dropdown = widgets.Dropdown(options=['usage', 'speed', 'similarity', 'mutation'], description='Sort Syllables By:', disabled=False)
ctrl_dropdown = widgets.Dropdown(options=[], description='Control Group:', disabled=False, layout=layout_hidden)
exp_dropdown = widgets.Dropdown(options=[], description='Treatment Group:', disabled=False, layout=layout_hidden)

grouping_dropdown = widgets.Dropdown(options=['group', 'SessionName'], description='Group Data By:', disabled=False)
session_sel = widgets.SelectMultiple(options=[], description='Sessions to Graph:', layout=layout_hidden, disabled=False)

## boxes
mutation_box = VBox([ctrl_dropdown, exp_dropdown])

sorting_box = VBox([sorting_dropdown, mutation_box])
session_box = VBox([grouping_dropdown, session_sel])

stat_widget_box = HBox([stat_dropdown, sorting_box, session_box])


### Group Setting Widgets
col_opts = {
    'editable': False,
    'toolTip': "Not editable"
}

col_defs = {
    'group': {
        'editable': True,
        'toolTip': 'editable'
    }
}

group_input = widgets.Text(value='', placeholder='Enter Group Name to Set', description='Desired Group Name', continuous_update=False, disabled=False)
save_button = widgets.Button(description='Set Group', disabled=False, tooltip='Set Group')
update_index_button = widgets.Button(description='Update Index File', disabled=False, tooltip='Save Parameters')

group_set = widgets.HBox([group_input, save_button, update_index_button])
qgrid.set_grid_option('forceFitColumns', False)
qgrid.set_grid_option('enableColumnReorder', True)
qgrid.set_grid_option('highlightSelectedRow', True)
qgrid.set_grid_option('highlightSelectedCell', False)

### Transition Graph Widgets

style = {'description_width': 'initial', 'display':'flex-grow', 'align_items':'stretch'}
'''
edge_thresholder = widgets.FloatRangeSlider(value=[0.0025, 1], min=0, max=1, step=0.001, style=style, readout_format='.4f',
                                            description='Edges weights to display', continuous_update=False)
usage_thresholder = widgets.FloatRangeSlider(value=[0, 1], min=0, max=1, step=0.001, style=style, readout_format='.4f',
                                             description='Usage nodes to display', continuous_update=False)
speed_thresholder = widgets.FloatRangeSlider(value=[-25, 200], min=-50, max=200, step=1, style=style, readout_format='.1f',
                                             description='Threshold nodes by speed', continuous_update=False)
'''
edge_thresholder = widgets.SelectionRangeSlider(options=['tmp'], style=style,
                                            description='Edges weights to display', continuous_update=False)
usage_thresholder = widgets.SelectionRangeSlider(options=['tmp'], style=style, readout_format='.4f',
                                             description='Usage nodes to display', continuous_update=False)
speed_thresholder = widgets.SelectionRangeSlider(options=['tmp'], style=style, readout_format='.1f',
                                             description='Threshold nodes by speed', continuous_update=False)

thresholding_box = HBox([VBox([edge_thresholder, usage_thresholder]), speed_thresholder])

### Crowd Movie Comparison Widgets
style = {'description_width': 'initial'}

cm_syll_select = widgets.Dropdown(options=[], description='Syllable #:', disabled=False)
num_examples = widgets.IntSlider(value=20, min=1, max=40, step=1, description='Number of Example Mice:', disabled=False, continuous_update=False, style=style)

cm_sources_dropdown = widgets.Dropdown(options=['group', 'SessionName'], description='Make Crowd Movies From:', disabled=False)
cm_session_sel = widgets.SelectMultiple(options=[], description='Sessions to Graph:', layout=layout_hidden, disabled=False)

syllable_box = VBox([syll_select, num_examples])
session_box = VBox([cm_sources_dropdown, session_sel])

widget_box = HBox([syllable_box, session_box]) # add layout
