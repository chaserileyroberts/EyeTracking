import tensorflow as tf
import EyeConvnet
from misc import loadmat
import Preprocess


class Trainer():
  def __init__(self, sess, mat_file, batch_size=32, save_dest=None):
    """Trainer object.
    Args:
      mat_files: File names to the '.mat' data file.
        TODO(Chase): Add multi file support.
      batch_size: Batch size to use for training.
      save_dest: Where to save the saved models and tensorboard events.
    """ 
    # TODO(Chase): Read in all of the .mat files, make queues for the data. 
    data = loadmat(mat_file)
    dataset = tf.contrib.data.Dataset.from_tensor_slices(
        (data['face'], data['eye_left'], data['eye_right'], data['gaze']))
    dataset = dataset.map(Preprocess.gaze_images_preprocess)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(32)
    dataset = dataset.repeat()
    self.iterator = dataset.make_initializable_iterator()
    (self.face_tensor, 
     self.left_eye_tensor, 
     self.right_eye_tensor,
     self.gaze) = self.iterator.get_next()
    self.sess = sess
    self.save_dest = save_dest
    self.model = EyeConvnet.EyeConvnet(
        True,
        self.face_tensor,
        self.left_eye_tensor,
        self.right_eye_tensor)

  def train(self, training_steps=100000, restore=None):
    """Trains the EyeConvnet Model.
    Args:
      training_steps: Number of training steps. Defaults to 100,000
      restore: Possible checkpoint to restore from. If None, then nothing is
          restored.
    """
    # TODO(Chase):
    # This is super slow and does a ton of unnessissary copying. 
    # To fix this, we need to transfer the .mat files into tfrecords.
    # But this will work for now.
    # TODO(Chase):
    # Should the optimizer and loss be in the model code?
    opt = tf.train.AdamOptimizer()
    loss = tf.losses.mean_squared_error(self.gaze, self.model.prediction)
    train = opt.minimize(loss)
    self.sess.run(self.iterator.initializer)
    self.sess.run(tf.global_variables_initializer())
    for i in xrange(training_steps):
      self.sess.run(train)    
  
