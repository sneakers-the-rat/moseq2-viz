import tqdm
import h5py
import os
import pandas as pd
import numpy as np
import warnings
from moseq2_viz.util import h5_to_dict, strided_app, load_timestamps, read_yaml
from moseq2_viz.model.util import parse_model_results


# http://stackoverflow.com/questions/17832238/kinect-intrinsic-parameters-from-field-of-view/18199938#18199938
# http://www.imaginativeuniversal.com/blog/post/2014/03/05/quick-reference-kinect-1-vs-kinect-2.aspx
# http://smeenk.com/kinect-field-of-view-comparison/
def convert_pxs_to_mm(coords, resolution=(512, 424), field_of_view=(70.6, 60), true_depth=673.1):
    """Converts x, y coordinates in pixel space to mm
    """
    cx = resolution[0] // 2
    cy = resolution[1] // 2

    xhat = coords[:, 0] - cx
    yhat = coords[:, 1] - cy

    fw = resolution[0] / (2 * np.deg2rad(field_of_view[0] / 2))
    fh = resolution[1] / (2 * np.deg2rad(field_of_view[1] / 2))

    new_coords = np.zeros_like(coords)
    new_coords[:, 0] = true_depth * xhat / fw
    new_coords[:, 1] = true_depth * yhat / fh

    return new_coords


def convert_legacy_scalars(old_features, force=False, true_depth=673.1):
    """Converts scalars in the legacy format to the new format, with explicit units.
    Args:
        old_features (str, h5 group, or dictionary of scalars): filename, h5 group, or dictionary of scalar values
        true_depth (float):  true depth of the floor relative to the camera (673.1 mm by default)

    Returns:
        features (dict): dictionary of scalar values
    """

    if type(old_features) is h5py.Group and 'centroid_x' in old_features.keys():
        print('Loading scalars from h5 dataset')
        feature_dict = {}
        for k, v in old_features.items():
            feature_dict[k] = v.value

        old_features = feature_dict

    if (type(old_features) is str or type(old_features) is np.str_) and os.path.exists(old_features):
        print('Loading scalars from file')
        with h5py.File(old_features, 'r') as f:
            feature_dict = {}
            for k, v in f['scalars'].items():
                feature_dict[k] = v.value

        old_features = feature_dict

    if 'centroid_x_mm' in old_features.keys() and force:
        centroid = np.hstack((old_features['centroid_x_px'][:, None],
                              old_features['centroid_y_px'][:, None]))
        nframes = len(old_features['centroid_x_mm'])

    elif not force:
        print('Features already converted')
        return None
    else:
        centroid = np.hstack((old_features['centroid_x'][:, None],
                              old_features['centroid_y'][:, None]))
        nframes = len(old_features['centroid_x'])

    features = {
        'centroid_x_px': np.zeros((nframes,), 'float32'),
        'centroid_y_px': np.zeros((nframes,), 'float32'),
        'velocity_2d_px': np.zeros((nframes,), 'float32'),
        'velocity_3d_px': np.zeros((nframes,), 'float32'),
        'width_px': np.zeros((nframes,), 'float32'),
        'length_px': np.zeros((nframes,), 'float32'),
        'area_px': np.zeros((nframes,)),
        'centroid_x_mm': np.zeros((nframes,), 'float32'),
        'centroid_y_mm': np.zeros((nframes,), 'float32'),
        'velocity_2d_mm': np.zeros((nframes,), 'float32'),
        'velocity_3d_mm': np.zeros((nframes,), 'float32'),
        'width_mm': np.zeros((nframes,), 'float32'),
        'length_mm': np.zeros((nframes,), 'float32'),
        'area_mm': np.zeros((nframes,)),
        'height_ave_mm': np.zeros((nframes,), 'float32'),
        'angle': np.zeros((nframes,), 'float32'),
        'velocity_theta': np.zeros((nframes,)),
    }

    centroid_mm = convert_pxs_to_mm(centroid, true_depth=true_depth)
    centroid_mm_shift = convert_pxs_to_mm(centroid + 1, true_depth=true_depth)

    px_to_mm = np.abs(centroid_mm_shift - centroid_mm)

    features['centroid_x_px'] = centroid[:, 0]
    features['centroid_y_px'] = centroid[:, 1]

    features['centroid_x_mm'] = centroid_mm[:, 0]
    features['centroid_y_mm'] = centroid_mm[:, 1]

    # based on the centroid of the mouse, get the mm_to_px conversion

    if 'width_px' in old_features.keys():
        features['width_px'] = old_features['width_px']
    else:
        features['width_px'] = old_features['width']

    if 'length_px' in old_features.keys():
        features['length_px'] = old_features['length_px']
    else:
        features['length_px'] = old_features['length']

    if 'area_px' in old_features.keys():
        features['area_px'] = old_features['area_px']
    else:
        features['area_px'] = old_features['area']

    if 'height_ave_mm' in old_features.keys():
        features['height_ave_mm'] = old_features['height_ave_mm']
    else:
        features['height_ave_mm'] = old_features['height_ave']

    features['width_mm'] = features['width_px'] * px_to_mm[:, 1]
    features['length_mm'] = features['length_px'] * px_to_mm[:, 0]
    features['area_mm'] = features['area_px'] * px_to_mm.mean(axis=1)

    features['angle'] = old_features['angle']

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        vel_x = np.diff(np.concatenate((features['centroid_x_px'][:1], features['centroid_x_px'])))
        vel_y = np.diff(np.concatenate((features['centroid_y_px'][:1], features['centroid_y_px'])))
        vel_z = np.diff(np.concatenate((features['height_ave_mm'][:1], features['height_ave_mm'])))

        features['velocity_2d_px'] = np.hypot(vel_x, vel_y)
        features['velocity_3d_px'] = np.sqrt(
            np.square(vel_x)+np.square(vel_y)+np.square(vel_z))

        vel_x = np.diff(np.concatenate((features['centroid_x_mm'][:1], features['centroid_x_mm'])))
        vel_y = np.diff(np.concatenate((features['centroid_y_mm'][:1], features['centroid_y_mm'])))

        features['velocity_2d_mm'] = np.hypot(vel_x, vel_y)
        features['velocity_3d_mm'] = np.sqrt(
            np.square(vel_x)+np.square(vel_y)+np.square(vel_z))

        features['velocity_theta'] = np.arctan2(vel_y, vel_x)

    return features


def get_scalar_map(index, fill_nans=True):

    scalar_map = {}
    score_idx = h5_to_dict(index['pca_path'], 'scores_idx')

    for uuid, v in index['files'].items():

        scalars = convert_legacy_scalars(h5_to_dict(v['path'][0], 'scalars'))
        idx = score_idx[uuid]
        scalar_map[uuid] = {}

        for k, v_scl in scalars.items():
            if fill_nans:
                scalar_map[uuid][k] = np.zeros((len(idx), ), dtype='float32')
                scalar_map[uuid][k][:] = np.nan
                scalar_map[uuid][k][~np.isnan(idx)] = v_scl
            else:
                scalar_map[uuid][k] = v_scl

    return scalar_map


def get_scalar_triggered_average(scalar_map, model_labels, max_syllable=40, nlags=20,
                                 include_keys=['velocity_2d_mm', 'velocity_3d_mm', 'width_mm',
                                             'length_mm', 'height_ave_mm'],
                                 zscore=False):

    win = int(nlags * 2 + 1)

    # cumulative average of PCs for nlags

    if np.mod(win, 2) == 0:
        win = win + 1

    # cumulative average of PCs for nlags
    # grab the windows where 0=syllable onset

    syll_average = {}
    count = np.zeros((max_syllable, ), dtype='int16')

    for scalar in include_keys:
        syll_average[scalar] = np.zeros((max_syllable, win), dtype='float32')

    for k, v in scalar_map.items():

        labels = model_labels[k]

        for i in range(max_syllable):
            hits = np.where(labels == i)[0]

            if len(hits) == 0:
                continue

            count[i] += len(hits)

            for scalar in include_keys:
                if zscore:
                    use_scalar = (v[scalar] - np.nanmean(v[scalar]))  / np.nanstd(v[scalar])
                else:
                    use_scalar = v[scalar]
                padded_scores = np.pad(use_scalar, (win // 2, win // 2),
                                   'constant', constant_values = np.nan)
                win_scores = strided_app(padded_scores, win, 1)
                syll_average[scalar][i] += np.nansum(win_scores[hits, :], axis=0)

    for i in range(max_syllable):
        for scalar in include_keys:
            syll_average[scalar][i] /= count[i]

    return syll_average


def scalars_to_dataframe(index, include_keys=['SessionName', 'SubjectName', 'StartTime'],
                         include_model=None, sort_labels_by_usage=False, disable_output=False,
                         include_feedback=None, force_conversion=True):

    scalar_dict = {}

    # loop through files, load scalars
    # TODO: checks for legacy scalars

    uuids = list(index['files'].keys())
    dset = h5_to_dict(h5py.File(index['files'][uuids[0]]['path'][0], 'r'), 'scalars')

    tmp = convert_legacy_scalars(dset, force=force_conversion)

    if tmp is not None:
        dset = tmp

    scalar_names = list(dset.keys())

    for scalar in scalar_names:
        scalar_dict[scalar] = []

    for key in include_keys:
        scalar_dict[key] = []

    include_labels = False
    if include_model is not None and os.path.exists(include_model):
        labels = parse_model_results(include_model,
                                     sort_labels_by_usage=sort_labels_by_usage)['labels']
        scalar_dict['model_label'] = []
        label_idx = h5_to_dict(index['pca_path'], 'scores_idx')

        for uuid, lbl in labels.items():
            labels[uuid] = lbl[~np.isnan(label_idx[uuid])]

        include_labels = True

    scalar_dict['group'] = []
    scalar_dict['uuid'] = []

    if include_feedback:
        scalar_dict['feedback_status'] = []

    for k, v in tqdm.tqdm(index['files'].items(), disable=disable_output):
        dset = h5_to_dict(h5py.File(v['path'][0], 'r'), 'scalars')
        timestamps = h5py.File(v['path'][0], 'r')['metadata/timestamps'].value
        parameters = read_yaml(v['path'][1])['parameters']

        if include_feedback:
            feedback_path = os.path.join(os.path.dirname(parameters['input_file']),
                                         'feedback_ts.txt')
            if os.path.exists(feedback_path):
                feedback_ts = load_timestamps(feedback_path, 0)
                feedback_status = load_timestamps(feedback_path, 1)
            else:
                raise RuntimeError('Could not find feedback file {}'.format(feedback_path))

        tmp = convert_legacy_scalars(dset, force=force_conversion)

        if tmp is not None:
            dset = tmp

        nframes = len(dset[scalar_names[0]])

        for scalar in scalar_names:
            scalar_dict[scalar].append(dset[scalar])

        for key in include_keys:
            for i in range(nframes):
                scalar_dict[key].append(v['metadata'][key])

        for i in range(nframes):
            scalar_dict['group'].append(v['group'])
            scalar_dict['uuid'].append(k)

        if include_feedback and feedback_ts is not None:

            for ts in timestamps:
                hit = np.where(ts.astype('int32') == feedback_ts.astype('int32'))[0]
                if len(hit) > 0:
                    scalar_dict['feedback_status'].append(feedback_status[hit])
                else:
                    scalar_dict['feedback_status'].append(-1)

        if include_labels:
            if k in labels.keys():
                for lbl in labels[k]:
                    scalar_dict['model_label'].append(lbl)
            else:
                for i in range(nframes):
                    scalar_dict['model_label'].append(np.nan)

    for scalar in scalar_names:
        scalar_dict[scalar] = np.concatenate(scalar_dict[scalar])

    scalar_df = pd.DataFrame(scalar_dict)

    return scalar_df
