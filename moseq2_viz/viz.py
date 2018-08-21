from matplotlib import lines, gridspec
import matplotlib.pyplot as plt
import numpy as np
import h5py
import cv2
import seaborn as sns
import networkx as nx


def convert_ebunch_to_graph(ebunch):

    g = nx.Graph()
    g.add_weighted_edges_from(ebunch)

    return g


def convert_transition_matrix_to_ebunch(weights, transition_matrix, edge_threshold=0, indices=None):

    if indices is None:
        ebunch = [(i[0], i[1], weights[i[0], i[1]]) for i, v in np.ndenumerate(transition_matrix)
                  if np.abs(v) > edge_threshold]
    else:
        ebunch = [(i[0], i[1], weights[i[0], i[1]]) for i in indices]

    return ebunch


def graph_transition_matrix(trans_mats, usages=None, groups=None,
                            edge_threshold=.0025, anchor=0,
                            node_color='w', node_edge_color='r', layout='spring',
                            edge_width_scale=100, node_size=400,
                            width_per_group=8, height=8, headless=False, font_size=12,
                            plot_differences=True, difference_threshold=.0005,
                            difference_edge_width_scale=500, weights=None,
                            usage_scale=1e4, **kwargs):

    if headless:
        plt.switch_backend('agg')

    if weights is None:
        weights = trans_mats

    if type(trans_mats) is np.ndarray and trans_mats.ndim == 2:
        trans_mats = [trans_mats]
    elif type(trans_mats) is list:
        pass
    else:
        raise RuntimeError("Transition matrix must be a numpy array or list of arrays")

    ngraphs = len(trans_mats)

    if anchor > ngraphs:
        print('Setting anchor to 0')
        anchor = 0

    ebunch_anchor = convert_transition_matrix_to_ebunch(
        weights[anchor], trans_mats[anchor], edge_threshold=edge_threshold)
    graph_anchor = convert_ebunch_to_graph(ebunch_anchor)
    nnodes = len(graph_anchor.nodes())

    if layout == 'spring':
        if 'k' not in kwargs.keys():
            kwargs['k'] = 1.5 / np.sqrt(nnodes)
        pos = nx.spring_layout(graph_anchor, **kwargs)
    elif layout == 'circular':
        pos = nx.circular_layout(graph_anchor, **kwargs)
    elif layout == 'spectral':
        pos = nx.spectral_layout(graph_anchor, **kwargs)
    else:
        raise RuntimeError('Did not understand layout type')

    fig, ax = plt.subplots(ngraphs, ngraphs,
                           figsize=(ngraphs*width_per_group,
                                    ngraphs*width_per_group))

    if ngraphs == 1:
        ax = [[ax]]

    for i, tm in enumerate(trans_mats):

        ebunch = convert_transition_matrix_to_ebunch(
            tm, tm, edge_threshold=edge_threshold, indices=ebunch_anchor)
        graph = convert_ebunch_to_graph(ebunch)

        width = [tm[u][v] * edge_width_scale for u, v in graph.edges()]

        if usages is not None:

            usage_total = sum(usages[i].values())
            for k, v in usages[i].items():
                usages[i][k] = v / usage_total
            node_size = [usages[i][k] * usage_scale for k in pos.keys()]

        nx.draw_networkx_nodes(graph, pos,
                               edgecolors=node_edge_color, node_color=node_color,
                               node_size=node_size, ax=ax[i][i])
        nx.draw_networkx_edges(graph, pos, graph.edges(), width=width, ax=ax[i][i])
        if font_size > 0:
            nx.draw_networkx_labels(graph, pos,
                                    {k: k for k in pos.keys()},
                                    font_size=font_size,
                                    ax=ax[i][i])

        if groups is not None:
            ax[i][i].set_title('{}'.format(groups[i]))

    if plot_differences and groups is not None and ngraphs > 1:
        for i, tm in enumerate(trans_mats):
            for j, tm2 in enumerate(trans_mats[i+1:]):
                df = tm2 - tm

                ebunch = convert_transition_matrix_to_ebunch(
                    df, df, edge_threshold=difference_threshold, indices=ebunch_anchor)
                graph = convert_ebunch_to_graph(ebunch)
                weight = [np.abs(graph[u][v]['weight'])*difference_edge_width_scale for u, v in graph.edges()]

                if usages is not None:
                    df_usage = [usages[j + i + 1][k] - usages[i][k] for k in pos.keys()]
                    node_size = list(np.abs(df_usage))
                    node_edge_color = ['r' if x > 0 else 'b' for x in df_usage]

                nx.draw_networkx_nodes(graph, pos, edgecolors=node_edge_color, node_color=node_color,
                                       node_size=node_size, ax=ax[i][j + i + 1], linewidths=1.5)
                colors = []

                for u, v in graph.edges():
                    if graph[u][v]['weight'] > 0:
                        colors.append('r')
                    else:
                        colors.append('b')

                nx.draw_networkx_edges(graph, pos, graph.edges(),
                                       width=weight, edge_color=colors,
                                       ax=ax[i][j + i + 1])

                if font_size > 0:
                    nx.draw_networkx_labels(graph, pos,
                                            {k: k for k in pos.keys()},
                                            font_size=font_size,
                                            ax=ax[i][j + i + 1])

                ax[i][j + 1].set_title('{} - {}'.format(groups[i], groups[j + i + 1]))

    for i in range(len(ax)):
        for j in range(len(ax[i])):
            ax[i][j].axis('off')

    plt.show()

    return fig, ax


def make_crowd_matrix(slices, nexamples=50, pad=30, raw_size=(512, 424),
                      crop_size=(80, 80), dur_clip=1000, offset=(50, 50), scale=1,
                      center=False, rotate=False):

    if rotate and not center:
        raise NotImplementedError('Rotating without centering not supported')

    durs = np.array([i[1]-i[0] for i, j, k in slices])

    if dur_clip is not None:
        idx = np.where(durs < dur_clip)[0]
        use_slices = [_ for i, _ in enumerate(slices) if i in idx]
    else:
        use_slices = slices

    durs = np.array([i[1]-i[0] for i, j, k in use_slices])

    if len(durs) < 1:
        return None

    max_dur = durs.max()

    # original_dtype = h5py.File(use_slices[0][2], 'r')['frames'].dtype

    crowd_matrix = np.zeros((max_dur + pad * 2, raw_size[1], raw_size[0]), dtype='uint8')
    count = 0

    xc0 = crop_size[1] // 2
    yc0 = crop_size[0] // 2

    xc = np.array(list(range(-xc0, +xc0 + 1)), dtype='int16')
    yc = np.array(list(range(-yc0, +yc0 + 1)), dtype='int16')

    for idx, uuid, fname in use_slices:

        # get the frames, combine in a way that's alpha-aware

        h5 = h5py.File(fname, 'r')
        nframes = h5['frames'].shape[0]
        cur_len = idx[1] - idx[0]
        use_idx = (idx[0] - pad, idx[1] + pad + (max_dur - cur_len))

        if use_idx[0] < 0 or use_idx[1] >= nframes:
            continue

        if 'centroid_x' in h5['scalars'].keys():
            use_names = ('scalars/centroid_x', 'scalars/centroid_y')
        elif 'centroid_x_px' in h5['scalars'].keys():
            use_names = ('scalars/centroid_x_px', 'scalars/centroid_y_px')

        centroid_x = h5[use_names[0]][use_idx[0]:use_idx[1]] + offset[0]
        centroid_y = h5[use_names[1]][use_idx[0]:use_idx[1]] + offset[1]

        if center:
            centroid_x -= centroid_x[pad]
            centroid_x += raw_size[0] // 2
            centroid_y -= centroid_y[pad]
            centroid_y += raw_size[1] // 2

        angles = h5['scalars/angle'][use_idx[0]:use_idx[1]]
        frames = (h5['frames'][use_idx[0]:use_idx[1]] / scale).astype('uint8')

        if 'flips' in h5['metadata'].keys():
            flips = h5['metadata/flips'][use_idx[0]:use_idx[1]]
            angles[np.where(flips == True)] -= np.pi
        else:
            flips = np.zeros(angles.shape, dtype='bool')

        angles = np.rad2deg(angles)

        for i in range(len(centroid_x)):

            if np.isnan(centroid_x[i]) or np.isnan(centroid_y[i]):
                continue

            rr = (yc + centroid_y[i]).astype('int16')
            cc = (xc + centroid_x[i]).astype('int16')

            if (np.any(rr < 1)
                or np.any(cc < 1)
                or np.any(rr >= raw_size[1])
                or np.any(cc >= raw_size[0])
                or (rr[-1] - rr[0] != crop_size[0])
                or (cc[-1] - cc[0] != crop_size[1])):
                continue

            rot_mat = cv2.getRotationMatrix2D((xc0, yc0), angles[i], 1)
            # old_frame = crowd_matrix[i][rr[0]:rr[-1],
            #                             cc[0]:cc[-1]]
            old_frame = crowd_matrix[i]
            new_frame = np.zeros_like(old_frame)
            new_frame_clip = frames[i]

            if flips[i]:
                new_frame_clip = np.fliplr(new_frame_clip)

            new_frame_clip = cv2.warpAffine(new_frame_clip.astype('float32'),
                                            rot_mat, crop_size).astype(frames.dtype)

            if i > pad and i < pad + cur_len:
                cv2.circle(new_frame_clip, (xc0, yc0), 3, (255, 255, 255), -1)
            try:
                new_frame[rr[0]:rr[-1], cc[0]:cc[-1]] = new_frame_clip
            except Exception:
                raise Exception

            if rotate:
                rot_mat = cv2.getRotationMatrix2D((raw_size[0] // 2, raw_size[1] // 2),
                                                  -angles[pad] + flips[pad] * 180,
                                                  1)
                new_frame = cv2.warpAffine(new_frame, rot_mat, raw_size).astype(new_frame.dtype)

            new_frame_nz = new_frame > 0
            old_frame_nz = old_frame > 0

            new_frame[np.where(new_frame < 10)] = 0
            old_frame[np.where(old_frame < 10)] = 0

            blend_coords = np.logical_and(new_frame_nz, old_frame_nz)
            overwrite_coords = np.logical_and(new_frame_nz, ~old_frame_nz)

            old_frame[blend_coords] = .5 * old_frame[blend_coords] + .5 * new_frame[blend_coords]
            old_frame[overwrite_coords] = new_frame[overwrite_coords]

            # crowd_matrix[i][rr[0]:rr[-1], cc[0]:cc[-1]] = old_frame
            crowd_matrix[i] = old_frame

        # if rotate:
        #     for i in range(len(crowd_matrix)):
        #         rot_mat = cv2.getRotationMatrix2D((raw_size[0] // 2, raw_size[1] // 2),
        #                                           angles[pad], 1)
        #         crowd_matrix[i] = cv2.warpAffine(crowd_matrix[i],
        #                                          rot_mat,
        #                                          raw_size).astype(crowd_matrix.dtype)

        count += 1

        if count >= nexamples:
            break

    return crowd_matrix


def position_plot(scalar_df, centroid_vars=['centroid_x_mm', 'centroid_y_mm'],
                  sort_vars=['SubjectName', 'uuid'], group_var='group', sz=50,
                  headless=False, **kwargs):

    grouped = scalar_df.groupby([group_var] + sort_vars)

    groups = [grp[0] for grp in grouped.groups]
    uniq_groups = list(set(groups))
    count = [len([grp1 for grp1 in groups if grp1 == grp]) for grp in uniq_groups]

    grouped = scalar_df.groupby(group_var)

    figsize = (np.round(2.5 * len(uniq_groups)), np.round(2.6 * np.max(count)))
    lims = (np.min(scalar_df[centroid_vars].min()), np.max(scalar_df[centroid_vars].max()))

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=np.max(count),
                           ncols=len(uniq_groups),
                           width_ratios=[1] * len(uniq_groups),
                           wspace=0.0,
                           hspace=0.0)

    for i, (name, group) in enumerate(grouped):

        group_session = group.groupby(sort_vars)

        for j, (name2, group2) in enumerate(group_session):

            ax = plt.subplot(gs[j, i])
            ax.plot(group2[centroid_vars[0]],
                    group2[centroid_vars[1]],
                    linewidth=.5,
                    **kwargs)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.axis('off')

            if j == 0:
                ax.set_title(name)

            if i == 0 and j == len(group_session) - 1:
                y_line = lines.Line2D([lims[0], lims[0]],
                                      [lims[0], lims[0] + sz],
                                      color='b',
                                      alpha=1)
                x_line = lines.Line2D([lims[0], lims[0] + sz],
                                      [lims[0], lims[0]],
                                      color='b',
                                      alpha=1)
                y_line.set_clip_on(False)
                x_line.set_clip_on(False)
                ax.add_line(y_line)
                ax.add_line(x_line)
                ax.text(lims[0] - 10, lims[0] - 60, '{} CM'.format(np.round(sz / 10).astype('int')))

            ax.set_aspect('auto')

    return fig, ax


def scalar_plot(scalar_df, sort_vars=['group', 'SubjectName'], group_var='group',
                show_scalars=['velocity_2d_mm', 'velocity_3d_mm',
                              'height_ave_mm', 'width_mm', 'length_mm'],
                headless=False,
                **kwargs):

    if headless:
        plt.switch_backend('agg')

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

    # sort scalars into a neat summary using group_vars

    summary = {
        'Mean': scalar_df.groupby(sort_vars)[show_scalars].mean(),
        'STD': scalar_df.groupby(sort_vars)[show_scalars].std()
    }

    for i, (k, v) in enumerate(summary.items()):
        summary[k].reset_index(level=summary[k].index.names, inplace=True)
        summary[k] = summary[k].melt(id_vars=group_var, value_vars=show_scalars)
        sns.swarmplot(data=summary[k], x='variable', y='value', hue=group_var, ax=ax[i], **kwargs)
        ax[i].set_ylabel(k)
        ax[i].set_xlabel('')

    plt.tight_layout()

    return fig, ax


def usage_plot(usages, groups=None, headless=False, **kwargs):

    # use a Seaborn pointplot, groups map to hue
    # make a useful x-axis to orient the user (which side is which)

    if headless:
        plt.switch_backend('agg')

    if len(groups) == 0:
        groups = None

    if groups is None:
        hue = None
    else:
        hue = 'group'

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.set_style('ticks')

    ax = sns.pointplot(data=usages,
                       x='syllable',
                       y='usage',
                       hue=hue,
                       hue_order=groups,
                       join=False,
                       **kwargs)

    ax.set_xticks([])
    plt.ylabel('P(syllable)')
    plt.xlabel('Syllable (sorted by usage)')

    sns.despine()

    return fig, ax
