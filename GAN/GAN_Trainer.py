import tensorflow as tf
import numpy as np 
from GAN.FeatureForcing import FFGAN as FeatureForcing
import FastDataset

slim = tf.contrib.slim


class GAN_Trainer:
  def __init__(self, mat_files, batch_size, save_dest):
    """GAN Training of the feature forcing model.
    Args:
      mat_files: list of ".mat" files for training
      batch_size: Size of the batch for training.
      save_dest: Where to save the models and events.
          Default to None
    """
    self.z_noise = tf.random_uniform([batch_size, 128], minval=-1, maxval=1)
    (self.face_tensor,
    self.left_eye_tensor,
    self.right_eye_tensor,
    self.gaze,
    self.pts) = FastDataset.get_eyetracking_tensors(
        mat_files, batch_size=batch_size, eval_loop=False)
    self.model = FeatureForcing(self.face_tensor, self.z_noise)
    self.save_dest = save_dest

  def train(self, training_steps=100000):
    print(training_steps)
    self.init_op = tf.global_variables_initializer()
    if self.save_dest is not None:
      saver = tf.train.Saver(
        keep_checkpoint_every_n_hours=0.01)  # every 12 minutes.
    else:
      saver = None
    sess = tf.Session()
    sess.run(self.init_op)
    merged = tf.summary.merge_all()
    slim.learning.train(
      self.model.train_descrim, #gan_train_op,
      self.save_dest,
      number_of_steps=training_steps,
      init_op=self.init_op,
      saver=saver,
      save_summaries_secs=60,
      save_interval_secs=360)
    