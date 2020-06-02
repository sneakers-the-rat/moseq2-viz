import os
import cv2
import warnings
import subprocess
import numpy as np
from tqdm.auto import tqdm
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
from moseq2_viz.viz import make_crowd_matrix
from moseq2_viz.model.util import get_syllable_slices

def write_crowd_movies(sorted_index, config_data, filename_format, vid_parameters, clean_params, ordering, \
                       labels, label_uuids, max_syllable, max_examples, output_dir):
    '''
    Creates syllable slices for crowd movies and writes them to files.

    Parameters
    ----------
    sorted_index (dict): dictionary of sorted index data.
    config_data (dict): dictionary of visualization parameters.
    filename_format (str): string format that denotes the saved crowd movie file names.
    vid_parameters (dict): dictionary of video writing parameters
    clean_params (dict): dictionary of image filtering parameters
    ordering (list): ordering for the new mapping of the relabeled syllable usages.
    labels (numpy ndarray): list of syllable usages
    label_uuids (list): list of session uuids each series of labels belongs to.
    max_syllable (int): maximum number of syllables to create movies for
    max_examples (int): maximum number of mice to include in a crowd movie
    output_dir (str): path directory where all the movies are written.

    Returns
    -------
    None
    '''

    with mp.Pool() as pool:
        slice_fun = partial(get_syllable_slices,
                            labels=labels,
                            label_uuids=label_uuids,
                            index=sorted_index)
        with warnings.catch_warnings():
            slices = list(tqdm(pool.imap(slice_fun, range(max_syllable)), total=max_syllable, desc='Getting Syllable Slices'))

        matrix_fun = partial(make_crowd_matrix,
                             nexamples=max_examples,
                             dur_clip=config_data['dur_clip'],
                             min_height=config_data['min_height'],
                             crop_size=vid_parameters['crop_size'],
                             raw_size=config_data['raw_size'],
                             scale=config_data['scale'],
                             legacy_jitter_fix=config_data['legacy_jitter_fix'],
                             **clean_params)

        with warnings.catch_warnings():
            crowd_matrices = list(tqdm(pool.imap(matrix_fun, slices), total=max_syllable, desc='Getting Crowd Matrices'))

        write_fun = partial(write_frames_preview, fps=vid_parameters['fps'], depth_min=config_data['min_height'],
                            depth_max=config_data['max_height'], cmap=config_data['cmap'])
        pool.starmap(write_fun,
                     [(os.path.join(output_dir, filename_format.format(i, config_data['count'], ordering[i])),
                       crowd_matrix)
                      for i, crowd_matrix in tqdm(enumerate(crowd_matrices), total=max_syllable, desc='Writing Movies') if crowd_matrix is not None])



def write_frames_preview(filename, frames=np.empty((0,)), threads=6,
                         fps=30, pixel_format='rgb24',
                         codec='h264', slices=24, slicecrc=1,
                         frame_size=None, depth_min=0, depth_max=80,
                         get_cmd=False, cmap='jet', text=None, text_scale=1,
                         text_thickness=2, pipe=None, close_pipe=True, progress_bar=True):
    '''
    Writes out a false-colored mp4 video.
    [Duplicate from moseq2-extract]

    Parameters
    ----------
    filename (str):
    frames (3D numpy array): num_frames * r * c
    threads (int): number of threads to write file
    fps (int): frames per second
    pixel_format (str): ffmpeg image formatting flag.
    codec (str): ffmpeg image encoding flag.
    slices (int): number of slices per thread.
    slicecrc (int): check integrity of slices.
    frame_size (tuple): image dimensions
    depth_min (int): minimum mouse distance from bucket floor
    depth_max (int): maximum mouse distance from bucket floor
    get_cmd (bool): return ffmpeg command instead of executing the command in python.
    cmap (str): color map selection.
    text (range(num_frames): display frame number in output video.
    text_scale (int): text size.
    text_thickness (int): text thickness.
    pipe (subProcess.Pipe object): if not None, indicates that there are more frames to be written.
    close_pipe (bool): indicates whether video is done writing, and to close pipe to file-stream.
    progress_bar (bool): display progress bar.

    Returns
    -------
    (subProcess.Pipe object): if there are more slices/chunks to write to, otherwise None.
    '''

    if not np.mod(frames.shape[1], 2) == 0:
        frames = np.pad(frames, ((0, 0), (0, 1), (0, 0)), 'constant', constant_values=0)

    if not np.mod(frames.shape[2], 2) == 0:
        frames = np.pad(frames, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=0)

    if not frame_size and type(frames) is np.ndarray:
        frame_size = '{0:d}x{1:d}'.format(frames.shape[2], frames.shape[1])
    elif not frame_size and type(frames) is tuple:
        frame_size = '{0:d}x{1:d}'.format(frames[0], frames[1])

    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    txt_pos = (5, frames.shape[-1] - 40)

    command = ['ffmpeg',
               '-y',
               '-loglevel', 'fatal',
               '-threads', str(threads),
               '-framerate', str(fps),
               '-f', 'rawvideo',
               '-s', frame_size,
               '-pix_fmt', pixel_format,
               '-i', '-',
               '-an',
               '-vcodec', codec,
               '-slices', str(slices),
               '-slicecrc', str(slicecrc),
               '-r', str(fps),
               '-pix_fmt', 'yuv420p',
               filename]

    if get_cmd:
        return command

    if not pipe:
        pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    use_cmap = plt.get_cmap(cmap)

    for i in tqdm(range(frames.shape[0]), desc="Writing frames", disable=~progress_bar):
        disp_img = frames[i, :].copy().astype('float32')
        disp_img = (disp_img-depth_min)/(depth_max-depth_min)
        disp_img[disp_img < 0] = 0
        disp_img[disp_img > 1] = 1
        disp_img = np.delete(use_cmap(disp_img), 3, 2)*255
        if text is not None:
            disp_img = cv2.putText(disp_img, text, txt_pos, font,
                                   text_scale, white, text_thickness, cv2.LINE_AA)
        pipe.stdin.write(disp_img.astype('uint8').tostring())

    if close_pipe:
        pipe.stdin.close()
        return None
    else:
        return pipe
