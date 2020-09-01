'''

'''

import ipywidgets as widgets
from ipywidgets import HBox, VBox
from bokeh.models.widgets import PreText

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
