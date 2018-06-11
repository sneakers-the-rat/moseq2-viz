from moseq2_viz.model.util import convert_ebunch_to_graph, convert_transition_matrix_to_ebunch
import matplotlib.pyplot as plt
import numpy as np
import h5py
import cv2
import seaborn as sns
import networkx as nx


def make_crowd_matrix(slices, nexamples=50, pad=30, raw_size=(512, 424),
                      crop_size=(80, 80), dur_clip=1000, offset=(50, 50), scale=1):

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

        h5 = h5py.File(fname)
        nframes = h5['frames'].shape[0]
        cur_len = idx[1] - idx[0]
        use_idx = (idx[0] - pad, idx[1] + pad + (max_dur - cur_len))

        if use_idx[0] < 0 or use_idx[1] >= nframes:
            continue

        centroid_x = h5['scalars/centroid_x'][use_idx[0]:use_idx[1]] + offset[0]
        centroid_y = h5['scalars/centroid_y'][use_idx[0]:use_idx[1]] + offset[1]

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

            if np.any(rr < 1) or np.any(cc < 1) or np.any(rr > raw_size[1]) or np.any(cc > raw_size[0]):
                continue

            old_frame = crowd_matrix[i][rr[0]:rr[-1],
                                        cc[0]:cc[-1]]
            new_frame = frames[i]

            if flips[i]:
                new_frame = np.fliplr(new_frame)

            rot_mat = cv2.getRotationMatrix2D((xc0, yc0), angles[i], 1)
            new_frame = cv2.warpAffine(new_frame.astype('float32'),
                                       rot_mat, crop_size).astype(frames.dtype)

            if i > pad and i < pad + cur_len:
                cv2.circle(new_frame, (xc0, yc0), 3, (255, 255, 255), -1)

            new_frame_nz = new_frame > 0
            old_frame_nz = old_frame > 0

            new_frame[np.where(new_frame < 10)] = 0
            old_frame[np.where(old_frame < 10)] = 0

            blend_coords = np.logical_and(new_frame_nz, old_frame_nz)
            overwrite_coords = np.logical_and(new_frame_nz, ~old_frame_nz)

            old_frame[blend_coords] = .5 * old_frame[blend_coords] + .5 * new_frame[blend_coords]
            old_frame[overwrite_coords] = new_frame[overwrite_coords]

            crowd_matrix[i][rr[0]:rr[-1], cc[0]:cc[-1]] = old_frame

        count += 1

        if count >= nexamples:
            break

    return crowd_matrix


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

    return plt, ax, fig


def graph_transition_matrix(trans_mats, groups=None, edge_threshold=.0025, anchor=0,
                            layout='spring', edge_width_scale=100, node_size=400,
                            width_per_group=8, height=8, headless=False, font_size=12,
                            **kwargs):

    if headless:
        plt.switch_backend('agg')

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
        trans_mats[anchor], edge_threshold=edge_threshold)
    graph_anchor = convert_ebunch_to_graph(ebunch_anchor)
    if layout == 'spring':
        pos = nx.spring_layout(graph_anchor, **kwargs)
    elif layout == 'circular':
        pos = nx.circular_layout(graph_anchor, **kwargs)
    elif layout == 'spectral':
        pos = nx.spectral_layout(graph_anchor, **kwargs)
    else:
        raise RuntimeError('Did not understand layout type')

    fig, ax = plt.subplots(1, ngraphs, figsize=(ngraphs*width_per_group,
                                                height))

    if ngraphs == 1:
        ax = [ax]

    for i, tm in enumerate(trans_mats):
        ebunch = convert_transition_matrix_to_ebunch(
            tm, edge_threshold=edge_threshold, indices=ebunch_anchor)
        graph = convert_ebunch_to_graph(ebunch)

        weight = [graph[u][v]['weight']*edge_width_scale for u, v in graph.edges()]

        nx.draw_networkx_nodes(graph, pos, node_size=node_size, ax=ax[i])
        nx.draw_networkx_edges(graph, pos, ebunch, width=weight, ax=ax[i])
        if font_size > 0:
            nx.draw_networkx_labels(graph, pos,
                                    {k: k for k in pos.keys()},
                                    font_size=font_size,
                                    ax=ax[i])

        ax[i].axis('off')
        if groups is not None:
            ax[i].set_title('{}'.format(groups[i]))

    plt.show()

    return plt, ax, fig
