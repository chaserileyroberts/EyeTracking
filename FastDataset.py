import tensorflow as tf
from misc import loadmat
import numpy as np
import random
import Preprocess

def read_mat_files(file_name):
    file_name = file_name.decode('utf-8')
    data = loadmat(file_name)
    return (data['face'].astype(np.float32),
            data['eye_left'].astype(np.float32),
            data['eye_right'].astype(np.float32),
            data['gaze'].astype(np.float32),
            data['pts'].astype(np.float32))

def make_fast_dataset(mat_files):
  """Makes the Fast Data Runner
  Args:
    mat_files: List of paths to the 'data.mat' files.
      WARNING!!!! WILL RANDOMLY SHUFFLE THE LIST!!!
    batch_size: Size of the batch to use for training.
    buffer_size: Size of the buffer to use for the random shuffle.
    repeat: Whether the iterator should repeat.
    preprocess: The function used to preprocess the inputs.
        Default to identity.

  Returns:  
    dataset: The dataset object
  """
  random.shuffle(mat_files)
  filenames = (tf.contrib.data.Dataset.from_tensor_slices(mat_files)
    .shuffle(buffer_size=50)
    .take(50)
    .repeat())
  dataset = filenames.flat_map(
      lambda file_name:
      tf.contrib.data.Dataset.from_tensor_slices(
          tuple(tf.py_func(
                read_mat_files, [file_name],
                # Face,    Left,     Right,     Gaze         Pts
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]))))
  return dataset

def get_eyetracking_tensors(mat_files, batch_size, eval_loop=False):
    """Get the iterated tensors for the eyetracking data
    Args:
      mat_files: list of ".mat" files.
      batch_size: Batch size for sampling.
      eval_loop: Whether this is an eval_loop. 
          If True, no input shuffling is done.
    Returns:
      face_tensor: Face image.
      left_eye_tensor: Left eye image.
      right_eye_tensor: Right eye image.
      gaze_tensor: Gaze 2D vector.
      face_pts_tensor: Face pts vector.
    """
    dataset = make_fast_dataset(mat_files)
    # TODO(Chase): Read in multiple files.
    dataset = dataset.map(Preprocess.gaze_images_preprocess)
    if not eval_loop:
      dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    (face_tensor,
    left_eye_tensor,
    right_eye_tensor,
    gaze,
    pts) = iterator.get_next()
    face_tensor.set_shape((None, 128, 128, 3))
    left_eye_tensor.set_shape((None, 36, 60, 3))
    right_eye_tensor.set_shape((None, 36, 60, 3))
    gaze.set_shape((None, 2))
    pts.set_shape((None, 102))
    return (face_tensor,
            left_eye_tensor,
            right_eye_tensor,
            gaze,
            pts) 