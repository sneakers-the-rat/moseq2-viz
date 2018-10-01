import numpy as np
from moseq2_viz.model.util import whiten_pcs, parse_model_results, simulate_ar_trajectory, _get_transitions
from moseq2_viz.util import strided_app, h5_to_dict
from moseq2_viz.scalars.util import get_scalar_map, get_scalar_triggered_average
from scipy.spatial.distance import squareform, pdist


def get_behavioral_distance(index, model_file, whiten='all',
                            distances=['ar[init]', 'scalars'], max_syllable=None,
                            dist_options={'scalars': {'nlags': 10, 'zscore': True},
                                          'ar': {'sim_points': 10}},
                            sort_labels_by_usage=True, count='usage'):

    dist_dict = {}

    if 'ar' not in dist_options.keys():
        dist_options['ar'] = {}

    if 'scalars' not in dist_options.keys():
        dist_options['scalars'] = {}

    model_fit = parse_model_results(model_file,
                                    map_uuid_to_keys=True,
                                    sort_labels_by_usage=sort_labels_by_usage,
                                    count=count)

    if max_syllable is None:
        max_syllable = -np.inf
        for lbl in model_fit['labels'].values():
            if lbl.max() > max_syllable:
                max_syllable = lbl.max() + 1

    for dist in distances:
        if dist.lower() == 'ar[init]':
            ar_mat = model_fit['model_parameters']['ar_mat']
            npcs = ar_mat[0].shape[0]
            nlags = ar_mat[0].shape[1] // npcs

            scores = h5_to_dict(index['pca_path'], 'scores')
            for k, v in scores.items():
                scores[k] = scores[k][:, :npcs]

            scores = whiten_pcs(scores, whiten)
            init = get_init_points(scores, model_fit['labels'],
                                   nlags=nlags, npcs=npcs, max_syllable=max_syllable)

            dist_dict['ar[init]'] = get_behavioral_distance_ar(ar_mat,
                                                               init_point=init,
                                                               **dist_options['ar'],
                                                               max_syllable=max_syllable)
        elif dist.lower() == 'scalars':
            scalar_map = get_scalar_map(index)
            scalar_ave = get_scalar_triggered_average(scalar_map,
                                                      model_fit['labels'],
                                                      max_syllable=max_syllable,
                                                      **dist_options['scalars'])

            for k, v in scalar_ave.items():
                key = 'scalar[{}]'.format(k)
                v = v[:, dist_options['scalars']['nlags']:]
                dist_dict[key] = squareform(pdist(v, 'correlation'))

    return dist_dict



def get_behavioral_distance_ar(ar_mat, init_point=None, sim_points=10, max_syllable=40):

    dist_mat = np.zeros((max_syllable, max_syllable), dtype='float32')
    npcs = ar_mat[0].shape[0]

    if init_point is None:
        init_point = [None] * max_syllable

    ar_traj = np.zeros((max_syllable, sim_points * npcs), dtype='float32')

    for i in range(max_syllable):
        ar_traj[i] = simulate_ar_trajectory(ar_mat[i], init_point[i], sim_points=sim_points).ravel()

    ar_dist = squareform(pdist(ar_traj, 'correlation'))

    return ar_dist


def get_init_points(pca_scores, model_labels, max_syllable=40, nlags=3, npcs=10):

    # cumulative average of PCs for nlags

    win = int(nlags * 2 + 1)

    if np.mod(win, 2) == 0:
        win = win + 1

    # grab the windows where 0=syllable onset

    syll_average = []
    count = np.zeros((max_syllable, ), dtype='int')

    for i in range(max_syllable):
        syll_average.append(np.zeros((win, npcs), dtype='float32'))

    for k, v in pca_scores.items():

        if k not in model_labels.keys():
            continue

        labels = model_labels[k]
        seq_array, locs = _get_transitions(labels)

        padded_scores = np.pad(v,((win // 2, win // 2), (0,0)),
                               'constant', constant_values = np.nan)

        for i in range(max_syllable):
            hits = locs[np.where(seq_array == i)[0]]

            if len(hits) < 1:
                continue

            count[i] += len(hits)
            for j in range(npcs):
                win_scores = strided_app(padded_scores[:, j], win, 1)
                syll_average[i][:, j] += np.nansum(win_scores[hits, :], axis=0)

    for i in range(max_syllable):
        syll_average[i] /= count[i].astype('float')

    return syll_average
