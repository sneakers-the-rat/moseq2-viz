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

# Layout Boxes
label_layout = widgets.Layout(flex_flow='column', max_height='200px')
input_layout = widgets.Layout(max_height='200px') # vbox

ui_layout = widgets.Layout(flex_flow='row', width='auto', max_height='50px')
data_layout = widgets.Layout(flex_flow='row', justify_content='space-between',
                             align_content='center', max_height='200px', width='auto')

# input box
lbl_box = VBox([syll_lbl, desc_lbl], layout=label_layout)

# input box
input_box = VBox([lbl_name_input, desc_input])
data_box = HBox([lbl_box, input_box], layout=data_layout)

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

widget_box = HBox([stat_dropdown, sorting_box, session_box])


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