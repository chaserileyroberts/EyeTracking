from __future__ import print_function


import tensorflow as tf
import EyeConvnet
import numpy as np
from misc import loadmat
import Preprocess


slim = tf.contrib.slim

def read_mat_files(file_name):
  file_name = file_name.decode('utf-8')
  data = loadmat(file_name)
  return (data['face'], 
          data['eye_left'],
          data['eye_right'], 
          data['gaze'].astype(np.float32))

class Trainer():
  """Trains and evalualuates the EyeConvnet model"""
  def __init__(self, mat_files, batch_size=32, save_dest=None, 
              eval_loop=False):
    """Trainer object.
    Args:
      mat_filess: List of file names to the '.mat' data files.
      batch_size: Batch size to use for training.
      save_dest: Where to save the saved models and tensorboard events.
      eval_loop: Boolean. Determins whether this Trainer is actually an 
        evaluator.
    """
    # TODO(Chase): This may not work beyond some large files. 
    # We'll test and debug later if that is the case.
    

    filenames = tf.contrib.data.Dataset.from_tensor_slices(mat_files)
    dataset = filenames.flat_map(lambda file_name:
        tf.contrib.data.Dataset.from_tensor_slices(
            tuple(tf.py_func(
                read_mat_files, [file_name], 
                # Face,    Left,     Right,     Gaze
                [tf.uint8, tf.uint8, tf.uint8, tf.float32]))))
    # TODO(Chase): Read in multiple files.
    dataset = dataset.map(Preprocess.gaze_images_preprocess)
    if not eval_loop:
      dataset = dataset.shuffle(buffer_size=100000)
    dataset = dataset.batch(batch_size)
    if not eval_loop:
      dataset = dataset.repeat()
    self.iterator = dataset.make_initializable_iterator()
    (self.face_tensor, 
    self.left_eye_tensor, 
    self.right_eye_tensor,
    self.gaze) = self.iterator.get_next()
    self.gaze = self.gaze / (2900, 1600) # Dimensions of the monitor.
    self.face_tensor.set_shape((None, 128, 128, 3))
    self.left_eye_tensor.set_shape((None, 36, 60, 3))
    self.right_eye_tensor.set_shape((None, 36, 60, 3))
    self.gaze.set_shape((None, 2))
    tf.summary.image(
        "face", tf.cast((self.face_tensor + 1) * 127.5, tf.uint8))
    tf.summary.image(
        "left", tf.cast((self.left_eye_tensor + 1) * 127.5, tf.uint8))
    tf.summary.image(
        "right", tf.cast((self.right_eye_tensor + 1) * 127.5, tf.uint8))
    self.save_dest = save_dest
    self.model = EyeConvnet.EyeConvnet(
        True,
        self.face_tensor,
        self.left_eye_tensor,
        self.right_eye_tensor)
    self.opt = tf.train.AdamOptimizer()
    self.loss = tf.losses.mean_squared_error(self.gaze, self.model.prediction)
    tf.summary.scalar("loss", self.loss)

  def train(self, training_steps=100000, restore=None):
    """Trains the EyeConvnet Model.
    Args:
      training_steps: Number of training steps. Defaults to 100,000
      restore: Possible checkpoint to restore from. If None, then nothing is
          restored.
    """
    # TODO(Chase):
    # Should the optimizer and loss be in the model code?
    # TODO(Chase): Include validation testing during training.
    if restore is not None:
      raise NotImplementedError("Restore is not implemented")
    train_op = slim.learning.create_train_op(self.loss, self.opt)
    init_op = tf.group(self.iterator.initializer, 
                       tf.global_variables_initializer())
    slim.learning.train(
        train_op,
        self.save_dest,
        number_of_steps=training_steps,
        init_op=init_op,
        save_summaries_secs=10)

  def eval(self, eval_secs):
    """ Runs the eval loop
    Args:
      eval_secs: How often to run the eval loop.

    """
    slim.evaluation.evaluation_loop(
    'local',
    self.save_dest,
    self.save_dest,
    eval_interval_secs=eval_secs)