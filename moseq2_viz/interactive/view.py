import itertools
from bokeh.palettes import Dark2_5 as palette
from bokeh.models.tickers import FixedTicker

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, TapTool, BoxSelectTool

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