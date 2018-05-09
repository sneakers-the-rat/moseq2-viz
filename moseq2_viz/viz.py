import numpy as np
import h5py
import cv2


def make_crowd_matrix(slices, nexamples=50, pad=30, raw_size=(424, 512),
                      crop_size=(80, 80), dtype='int8', dur_clip=None, offset=(100, 100)):

    durs = np.array([i[1]-i[0] for i, j, k in slices])

    if dur_clip is not None:
        idx = np.where(durs > dur_clip)[0]
        use_slices = [_ for i, _ in enumerate(slices) if i in idx]
    else:
        use_slices = slices

    durs = np.array([i[1]-i[0] for i, j, k in use_slices])
    max_dur = durs.max()
    crowd_matrix = np.zeros((max_dur + pad * 2, raw_size[0], raw_size[1]), dtype=dtype)
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
        frames = h5['frames'][use_idx[0]:use_idx[1]]

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
                cv2.circle(new_frame, (40, 40), 3, (255, 255, 255), -1)

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
