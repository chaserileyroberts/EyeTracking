import tensorflow as tf
from misc import loadmat
import numpy as np

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
    batch_size: Size of the batch to use for training.
    buffer_size: Size of the buffer to use for the random shuffle.
    repeat: Whether the iterator should repeat.
    preprocess: The function used to preprocess the inputs.
        Default to identity.

  Returns:  
    dataset: The dataset object
  """
  filenames = tf.contrib.data.Dataset.from_tensor_slices(mat_files)
  dataset = filenames.flat_map(
      lambda file_name:
      tf.contrib.data.Dataset.from_tensor_slices(
          tuple(tf.py_func(
                read_mat_files, [file_name],
                # Face,    Left,     Right,     Gaze         Pts
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]))))
  return dataset