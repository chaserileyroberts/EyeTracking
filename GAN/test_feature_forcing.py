import pytest
import tensorflow as tf
import GAN.FeatureForcing as FF


def setup():
  tf.reset_default_graph()
  tf.set_random_seed(0)

def test_ff_sanity():
  img = tf.placeholder(tf.float32, (None, 100, 100, 3))
  z_noise = tf.placeholder(tf.float32, (None, 100))
  FF.FFGAN(img, z_noise)

def test_encoding_shape():
  for i in range(100, 500, 25):
    tf.reset_default_graph()
    img = tf.placeholder(tf.float32, (None, 100, 100, 3))
    z_noise = tf.placeholder(tf.float32, (None, i))
    model = FF.FFGAN(img, z_noise)
    assert int(model.encoding.shape[1]) == i
    assert len(model.encoding.shape) == 2

def test_correct_vars_train():
  pass