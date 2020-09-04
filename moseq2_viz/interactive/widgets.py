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
