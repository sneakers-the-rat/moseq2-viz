import warnings
import itertools
import numpy as np
import networkx as nx
from bokeh.layouts import gridplot
from IPython.display import display
from bokeh.palettes import Spectral4
from bokeh.models.tickers import FixedTicker
from bokeh.palettes import Dark2_5 as palette
from moseq2_viz.interactive.widgets import widget_box
from bokeh.plotting import figure, show, from_networkx
from bokeh.models import (ColumnDataSource, LabelSet, BoxSelectTool, Circle,
                          EdgesAndLinkedNodes, HoverTool, MultiLine,
                          NodesAndLinkedEdges, TapTool, Div)



color_dict = {'b': 'blue',
              'r': 'red',
              'g': 'green',
              'c': 'cyan'}

def graph_dendrogram(obj):
    '''
    Graphs the distance sorted dendrogram representing syllable neighborhoods. Distance sorting
    is computed by processing the respective syllable AR matrices.

    Parameters
    ----------
    obj (InteractiveSyllableStats object): Syllable Stats object containing syllable stat information.

    Returns
    -------
    '''

    ## Cladogram figure
    cladogram = figure(title='Syllable Stat',
                       width=600,
                       height=400,
                       output_backend="webgl")

    # Show syllable info on hover
    cladogram.add_tools(HoverTool(tooltips=[('label', '@labels')]), TapTool(), BoxSelectTool())

    # Get distance sorted label ordering
    labels = list(map(int, obj.results['ivl']))
    sources = []

    # Each (icoord, dcoord) pair represents a single branch in the dendrogram
    for ii, (i, d) in enumerate(zip(obj.icoord, obj.dcoord)):
        d = list(map(lambda x: x, d))
        tmp = [[x, y] for x, y in zip(i, d)]
        lbls = []

        # Attaching a group/neighborhood to the syllables based on the assigned colors
        # from scipy.cluster.dendrogram results output.
        groups = {c: i for i, c in enumerate(list(color_dict.keys()))}

        # Get labels
        for t in tmp:
            if t[1] == 0:
                lbls.append(labels[int(t[0])])
            else:
                lbls.append(f'group {groups[obj.results["color_list"][ii]]}')

        # Set coordinate DataSource
        source = ColumnDataSource(dict(x=i, y=d, labels=lbls))
        sources.append(source)

        # Draw glyphs
        cladogram.line(x='x', y='y', source=source, line_color=color_dict[obj.results['color_list'][ii]])

    # Set x-axis ticks
    cladogram.xaxis.ticker = FixedTicker(ticks=labels)
    cladogram.xaxis.major_label_overrides = {i: str(l) for i, l in enumerate(labels)}

    # Display cladogram
    show(cladogram)


def bokeh_plotting(df, stat, sorting, groupby='group'):
    '''
    Generates a Bokeh plot with interactive tools such as the HoverTool, which displays
    additional syllable information and the associated crowd movie.

    Parameters
    ----------
    df (pd.DataFrame): Mean syllable statistic DataFrame.
    stat (str): Statistic to plot
    sorting (list): List of the current/selected syllable ordering
    groupby (str): Value to group data by. Either by unique group name or session name.

    Returns
    -------
    '''

    tools = 'pan, box_zoom, wheel_zoom, hover, save, reset'

    # Hover tool to display crowd movie
    cm_tooltip = """
        <div>
            <div>
                <video
                    src="@movies" height="260" alt="@movies" width="260"; preload="true";
                    style="float: left; type: "video/mp4"; "margin: 0px 15px 15px 0px;"
                    border="2"; autoplay loop
                ></video>
            </div>
        </div>
        """

    # Instantiate Bokeh figure with the HoverTool data
    p = figure(title='Syllable Stat',
               width=850,
               height=500,
               tools=tools,
               tooltips=[
                         ("syllable", "@number{0}"),
                         ('usage', "@usage{0.000}"),
                         ('speed', "@speed{0.000}"),
                         ('dist. to center', "@dist_to_center{0.000}"),
                         (f'{stat} SEM', '@sem{0.000}'),
                         ('label', '@label'),
                         ('description', '@desc'),
                         ('crowd movie', cm_tooltip)
                         ],
               output_backend="webgl")

    # TODO: allow users to set their own colors
    colors = itertools.cycle(palette)

    # Set grouping variable to plot separately
    if groupby == 'group':
        groups = list(df.group.unique())
    else:
        groups = list(df.SessionName.unique())

    ## Line Plot
    for i, color in zip(range(len(groups)), colors):
        # Get resorted mean syllable data
        aux_df = df[df[groupby] == groups[i]].groupby('syllable', as_index=False).mean().reindex(sorting)

        # Get SEM values
        sem = df.groupby('syllable')[[stat]].sem().reindex(sorting)
        miny = aux_df[stat] - sem[stat]
        maxy = aux_df[stat] + sem[stat]

        errs_x = [(i, i) for i in range(len(aux_df.index))]
        errs_y = [(min_y, max_y) for min_y, max_y in zip(miny, maxy)]

        # Get Labeled Syllable Information
        desc_data = df.groupby(['syllable', 'label', 'desc', 'crowd_movie_path'], as_index=False).mean()[
            ['syllable', 'label', 'desc', 'crowd_movie_path']].reindex(sorting)

        # Pack data into numpy arrays
        labels = desc_data['label'].to_numpy()
        desc = desc_data['desc'].to_numpy()
        cm_paths = desc_data['crowd_movie_path'].to_numpy()

        # stat data source
        source = ColumnDataSource(data=dict(
            x=range(len(aux_df.index)),
            y=aux_df[stat].to_numpy(),
            usage=aux_df['usage'].to_numpy(),
            speed=aux_df['speed'].to_numpy(),
            dist_to_center=aux_df['dist_to_center'].to_numpy(),
            sem=sem[stat].to_numpy(),
            number=sem.index,
            label=labels,
            desc=desc,
            movies=cm_paths,
        ))

        # SEM data source
        err_source = ColumnDataSource(data=dict(
            x=errs_x,
            y=errs_y,
            usage=aux_df['usage'].to_numpy(),
            speed=aux_df['speed'].to_numpy(),
            dist_to_center=aux_df['dist_to_center'].to_numpy(),
            sem=sem[stat].to_numpy(),
            number=sem.index,
            label=labels,
            desc=desc,
            movies=cm_paths,
        ))

        # Draw glyphs
        p.line('x', 'y', source=source, alpha=0.8, muted_alpha=0.1, legend_label=groups[i], color=color)
        p.circle('x', 'y', source=source, alpha=0.8, muted_alpha=0.1, legend_label=groups[i], color=color, size=6)
        p.multi_line('x', 'y', source=err_source, alpha=0.8, muted_alpha=0.1, legend_label=groups[i], color=color)

    # Setting dynamics xticks
    p.xaxis.ticker = FixedTicker(ticks=list(sorting))
    p.xaxis.major_label_overrides = {i: str(l) for i, l in enumerate(list(sorting))}

    # Setting interactive legend
    p.legend.click_policy = "mute"
    p.legend.location = "top_right"

    ## Display
    show(p)

def format_graphs(graphs, group):
    '''
    Formats multiple transition graphs to be stacked in vertical column-order.

    Parameters
    ----------
    graphs (list): list of generated Bokeh figures.
    group (list): list of unique groups

    Returns
    -------
    formatted_plots (2D list): list of lists corresponding to rows of figures being plotted.
    '''

    # formatting plots into grid format
    ncols = len(group)

    tmp, formatted_plots = [], []
    for i, p in enumerate(graphs):
        if len(tmp) < ncols:
            tmp.append(p)
        elif len(tmp) == ncols:
            formatted_plots.append(tmp)
            tmp = [p]
    formatted_plots.append(tmp)

    return formatted_plots

def plot_interactive_transition_graph(graphs, pos, group, group_names, usages, syll_info, entropies, entropy_rates, scalars):
    '''

    Converts the computed networkx transition graphs to Bokeh glyph objects that can be interacted with
    and updated throughout run-time.

    Parameters
    ----------
    graphs (list of nx.DiGraphs): list of created networkx graphs.
    pos (nx.Layout): shared node position coordinates layout object.
    group (list): list of unique group names.
    group_names (list): list of names for all the generated transition graphs + difference graphs
    usages (list of OrdreredDicts): list of OrderedDicts containing syllable usages.
    syll_info (dict): dict of syllable label information to display with HoverTool
    scalars (dict): dict of syllable scalar information to display with HoverTool

    Returns
    -------
    '''

    warnings.filterwarnings('ignore')

    rendered_graphs, plots = [], []

    for i, graph in enumerate(graphs):
        if i > 0:
            pos = nx.circular_layout(graph, scale=1)

        node_indices = [n for n in graph.nodes if n in usages[i].keys()]

        if len(plots) == 0:
            plot = figure(title=f"{group_names[i]}", x_range=(-1.2, 1.2), y_range=(-1.2, 1.2))
        else:
            # Connecting pan-zoom interaction across plots
            plot = figure(title=f"{group_names[i]}", x_range=plots[0].x_range, y_range=plots[0].y_range)

        cm_tooltip = """
            <div>
                <div>
                    <video
                        src="@movies" height="260" alt="@movies" width="260"; preload="true";
                        style="float: left; type: "video/mp4"; "margin: 0px 15px 15px 0px;"
                        border="2"; autoplay loop
                    ></video>
                </div>
            </div>
            """

        # adding interactive tools
        plot.add_tools(HoverTool(tooltips=[('syllable', '@index'),
                                           ('label', '@label'),
                                           ('description', '@desc'),
                                           ('usage', '@usage{0.0000}'),
                                           ('speed', '@speed{0.0000}'),
                                           ('dist. to center', '@dist{0.0000}'),
                                           ('ent. in', '@ent_in{0.000}'),
                                           ('ent. out', '@ent_out{0.000}'),
                                           ('prev state', '@prev'),
                                           ('next state', '@next'),
                                           ('', cm_tooltip)], line_policy='interp'),
                       TapTool(),
                       BoxSelectTool())

        # edge colors for difference graphs
        if i >= len(group):
            edge_color = {e: 'red' if graph.edges()[e]['weight'] > 0 else 'blue' for e in graph.edges()}
            edge_width = {e: graph.edges()[e]['weight'] * 350 for e in graph.edges()}
        else:
            edge_color = {e: 'black' for e in graph.edges()}
            edge_width = {e: graph.edges()[e]['weight'] * 200 for e in graph.edges()}

        # setting edge attributes
        nx.set_edge_attributes(graph, edge_color, "edge_color")
        nx.set_edge_attributes(graph, edge_width, "edge_width")

        # get usages
        group_usage = [usages[i][j] for j in node_indices if j in usages[i].keys()]

        # get speeds
        group_speed = [scalars['speeds'][i][j] for j in node_indices if j in scalars['speeds'][i].keys()]

        # get mean distances to bucket centers
        group_dist = [scalars['dists'][i][j] for j in node_indices if j in scalars['dists'][i].keys()]

        # node colors for difference graphs
        if i >= len(group):
            node_color = {s: 'red' if usages[i][s] > 0 else 'blue' for s in node_indices}
            node_size = {s: max(15., 10 + abs(usages[i][s] * 500)) for s in node_indices}
        else:
            node_color = {s: 'red' for s in node_indices}
            node_size = {s: max(15., abs(usages[i][s] * 500)) for s in node_indices}

        # setting node attributes
        nx.set_node_attributes(graph, node_color, "node_color")
        nx.set_node_attributes(graph, node_size, "node_size")

        # create bokeh-fied networkx transition graph
        graph_renderer = from_networkx(graph, pos, scale=1, center=(0, 0))

        # get node directed neighbors
        prev_states, next_states = [], []

        entropy_in, entropy_out = [], []
        # get average entropy_in and out
        for n in node_indices:
            try:
                # Get predecessor and neighboring states
                pred = np.array(list(graph.predecessors(n)))
                neighbors = np.array(list(graph.neighbors(n)))

                e_ins, e_outs = [], []
                for p in pred:
                    e_in = entropy_rates[i][p][n] + (entropies[i][n] - entropies[i][p])
                    e_ins.append(e_in)

                for nn in neighbors:
                    e_out = entropy_rates[i][n][nn] + (entropies[i][nn] - entropies[i][n])
                    e_outs.append(e_out)

                # Get predecessor and next state transition weights
                pred_weights = [graph.edges()[(p, n)]['weight'] for p in pred]
                next_weights = [graph.edges()[(n, p)]['weight'] for p in neighbors]

                # Get descending order of weights
                pred_sort_idx = np.argsort(pred_weights)[::-1]
                next_sort_idx = np.argsort(next_weights)[::-1]

                # Get transition likelihood-sorted previous and next states
                prev_states.append(pred[pred_sort_idx])
                next_states.append(neighbors[next_sort_idx])

                entropy_in.append(np.nanmean(e_ins))
                entropy_out.append(np.nanmean(e_outs))
            except nx.NetworkXError:
                # handle orphans
                print('missing', group_names[i], n)
                pass

        # getting hovertool info
        labels, descs, cm_paths = [], [], []

        for n in node_indices:
            labels.append(syll_info[str(n)]['label'])
            descs.append(syll_info[str(n)]['desc'])
            cm_paths.append(syll_info[str(n)]['crowd_movie_path'])

        # setting common data source to display via HoverTool
        graph_renderer.node_renderer.data_source.add(node_indices, 'index')
        graph_renderer.node_renderer.data_source.add(labels, 'label')
        graph_renderer.node_renderer.data_source.add(descs, 'desc')
        graph_renderer.node_renderer.data_source.add(cm_paths, 'movies')
        graph_renderer.node_renderer.data_source.add(prev_states, 'prev')
        graph_renderer.node_renderer.data_source.add(next_states, 'next')
        graph_renderer.node_renderer.data_source.add(group_usage, 'usage')
        graph_renderer.node_renderer.data_source.add(group_speed, 'speed')
        graph_renderer.node_renderer.data_source.add(group_dist, 'dist')
        graph_renderer.node_renderer.data_source.add(np.nan_to_num(entropy_in), 'ent_in')
        graph_renderer.node_renderer.data_source.add(np.nan_to_num(entropy_out), 'ent_out')

        # node interactions
        graph_renderer.node_renderer.glyph = Circle(size='node_size', fill_color='white', line_color='node_color')
        graph_renderer.node_renderer.selection_glyph = Circle(size='node_size', fill_color=Spectral4[2])
        graph_renderer.node_renderer.nonselection_glyph = Circle(size='node_size', line_color='node_color', fill_color='white')
        graph_renderer.node_renderer.hover_glyph = Circle(size='node_size', fill_color=Spectral4[1])

        # edge interactions
        graph_renderer.edge_renderer.glyph = MultiLine(line_color='edge_color', line_alpha=0.7,
                                                       line_width='edge_width', line_join='miter')
        graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color='edge_color', line_width='edge_width',
                                                                 line_join='miter', line_alpha=0.8,)
        graph_renderer.edge_renderer.nonselection_glyph = MultiLine(line_color='edge_color', line_alpha=0.0,
                                                                    line_width='edge_width', line_join='miter')
        ## Change line color to match difference colors
        graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)

        # selection policies
        graph_renderer.selection_policy = NodesAndLinkedEdges()
        graph_renderer.inspection_policy = NodesAndLinkedEdges()

        # added rendered graph to plot
        plot.renderers.append(graph_renderer)

        # get node positions
        x, y = zip(*graph_renderer.layout_provider.graph_layout.values())

        # create DataSource for node info
        label_source = ColumnDataSource({'x': x,
                                         'y': y,
                                         'syllable': list(graph.nodes)
                                         })

        # create the LabelSet to render
        labels = LabelSet(x='x', y='y',
                          x_offset=-7, y_offset=-7,
                          text='syllable', source=label_source,
                          text_color='black', text_font_size="12px",
                          background_fill_color=None,
                          render_mode='canvas')

        # render labels
        plot.renderers.append(labels)

        plots.append(plot)
        rendered_graphs.append(graph_renderer)

    # Format grid of transition graphs
    formatted_plots = format_graphs(plots, group)

    # Create Bokeh grid plot object
    gp = gridplot(formatted_plots, plot_width=500, plot_height=500)
    show(gp)

def display_crowd_movies(divs):
    '''
    Crowd movie comparison helper function that displays the widgets and
    embedded HTML divs to a running jupyter notebook cell or HTML webpage.

    Parameters
    ----------
    divs (list of bokeh.models.Div): list of HTML Div objects containing videos to display

    Returns
    -------

    '''

    # Set HTML formats
    movie_table = '''
                    <head>
                    <style>
                        .output {
                            display: contents;
                            height: auto;
                        }
                        .row {
                            display: flex;
                            flex-wrap: wrap;
                            vertical-align: center;
                            width: 900px;
                            text-align: center;
                        }
        
                        .column {
                            width: 50%;
                            text-align: center;
                        }
        
                        .column video {
                          vertical-align: center;
                        }
                    </style>
                    </head>
                    <div class="row"; style="background-color:#ffffff; height:auto;">
                  '''

    # Create div grid
    for i, div in enumerate(divs):
        if (i % 2 == 0) and i > 0:
            # make a new row
            movie_table += '</div>'
            col = f'''
                      <div class="row"; style="background-color:#ffffff; height:auto;">
                          <div class="column">
                              {div}
                          </div>
                    '''
        else:
            # put movie in column
            col = f'''
                      <div class="column">
                          {div}
                      </div>
                    '''
        movie_table += col

    # Close last div
    movie_table += '</div>'

    div2 = Div(text=movie_table)

    # Display
    display(widget_box)
    show(div2)