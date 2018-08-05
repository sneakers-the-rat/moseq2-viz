import numpy as np
from moseq2_viz.model.util import whiten_pcs, parse_model_results, simulate_ar_trajectory
from moseq2_viz.util import strided_app, h5_to_dict
from scipy.spatial.distance import squareform, pdist


def get_behavioral_distance(index, model_file, distances=['ar[init]']):

    dist_dict = {}

    model_fit = parse_model_results(model_file, map_uuid_to_keys=True, sort_labels_by_usage=True)

    for dist in distances:
        if dist.lower() == 'ar[init]':
            ar_mat = model_fit['model_parameters']['ar_mat']
            npcs = ar_mat[0].shape[0]
            nlags = ar_mat[0].shape[1] // npcs

            scores = h5_to_dict(index['pca_path'], 'scores')
            for k, v in scores.items():
                scores[k] = scores[k][:, :npcs]

            scores = whiten_pcs(scores, 'all')
            init = get_init_points(scores, model_fit['labels'],
                                   nlags=nlags, npcs=npcs)

            dist_dict['ar[init]'] = get_behavioral_distance_ar(ar_mat, init)

    return dist_dict



def get_behavioral_distance_ar(ar_mat, init_point=None, sim_points=10, max_syllable=40):

    dist_mat = np.zeros((max_syllable, max_syllable), dtype='float32')
    npcs = ar_mat[0].shape[0]

    if init_point is None:
        init_point = ['None'] * max_syllable

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
    count = np.zeros((max_syllable, ), dtype='int16')

    for i in range(max_syllable):
        syll_average.append(np.zeros((win, npcs), dtype='float32'))

    for k, v in pca_scores.items():

        labels = model_labels[k]
        padded_scores = np.pad(v,((win // 2, win // 2), (0,0)),
                               'constant', constant_values = np.nan)

        for i in range(max_syllable):
            hits = np.where(labels == i)[0]

            if len(hits) == 0:
                continue

            count[i] += len(hits)
            for j in range(npcs):
                win_scores = strided_app(padded_scores[:, j], win, 1)
                syll_average[i][:, j] += np.nansum(win_scores[hits, :], axis=0)

    for i in range(max_syllable):
        syll_average[i] /= count[i]

    return syll_average
