import tensorflow as tf
import numpy as np
from mnist import MNIST
from feature_forcing import FFGAN
slim = tf.contrib.slim


class MnistGanTrainer:
  def __init__(self, batch_size, save_dest):
    mndata =  MNIST('./')
    images, labels = mndata.load_training()
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset.shuffle(1000)
    dataset.repeat()
    dataset.batch(batch_size)
    itr = dataset.make_one_shot_iterator()
    (self.img, self.lbl) = itr.get_next()
    self.img = tf.reshape(self.img, [-1, 28, 28, 1])
    self.z_noise = tf.random_uniform([batch_size, 128], minval=-1, maxval=1)
    self.model = FFGAN(self.img, self.z_noise)
    self.save_dest = save_dest

  def train(self, training_steps=100000):
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
    

if __name__ == '__main__':
  model = MnistGanTrainer(1, "./mnist_events")
  model.train(100000)