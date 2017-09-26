import tensorflow as tf
import EyeConvnet
from misc import loadmat
import Preprocess


class Trainer():
  def __init__(self, sess, mat_files, batch_size=32, save_dest=None):
    """Trainer object.
    Args:
      mat_filess: List of file names to the '.mat' data files.
      batch_size: Batch size to use for training.
      save_dest: Where to save the saved models and tensorboard events.
    """
    # TODO(Chase): This may not work beyond some large files. 
    # We'll test and debug latter if that is the case.
    dataset = None
    for i,file_name in enumerate(mat_files):
      print "Reading file", file_name
      data = loadmat(file_name)
      tmp_dataset = tf.contrib.data.Dataset.from_tensor_slices(
          (data['face'], data['eye_left'], data['eye_right'], data['gaze']))
      if i == 0: # initial concatenation. 
        dataset = tmp_dataset
      else:
        dataset = dataset.concatenate(tmp_dataset)
    # TODO(Chase): Read in multiple files.
    dataset = dataset.map(Preprocess.gaze_images_preprocess)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
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
    # Should the optimizer and loss be in the model code?
    # TODO(Chase): Include validation testing during training.
    opt = tf.train.AdamOptimizer()
    loss = tf.losses.mean_squared_error(self.gaze, self.model.prediction)
    tf.summary.scalar("loss", loss)
    train = opt.minimize(loss)
    self.sess.run(self.iterator.initializer)
    self.sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    if restore:
      saver.restore(self.sess, restore)
    if self.save_dest:
      writer = tf.summary.FileWriter(self.save_dest, self.sess.graph)
    for i in xrange(training_steps):
      print "On training_step", i
      summary, _ = self.sess.run([merged, train])
      if self.save_dest:    
        writer.add_summary(summary, global_step=i)
      if i % 100 == 0 and self.save_dest:
        saver.save(self.sess, self.save_dest + "/model", global_step=i)
