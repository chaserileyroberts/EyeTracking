import pytest
import tensorflow as tf
import GAN.FeatureForcing as FF
import numpy as np

def setup():
  tf.reset_default_graph()
  tf.set_random_seed(0)

def test_ff_sanity():
  img = tf.placeholder(tf.float32, (None, 128, 128, 3))
  z_noise = tf.placeholder(tf.float32, (None, 128))
  FF.FFGAN(img, z_noise)

def test_encoding_shape():
  for i in range(100, 200, 25):
    tf.reset_default_graph()
    img = tf.placeholder(tf.float32, (None, 128, 128, 3))
    z_noise = tf.placeholder(tf.float32, (None, i))
    model = FF.FFGAN(img, z_noise)
    assert int(model.encoding_real.shape[1]) == i
    assert len(model.encoding_real.shape) == 2
    assert int(model.encoding_fake.shape[1]) == i
    assert len(model.encoding_fake.shape) == 2

def test_k_update():
  img = tf.placeholder(tf.float32, (None, 128, 128, 3))
  z_noise = tf.placeholder(tf.float32, (None, 128))
  model = FF.FFGAN(img, z_noise)
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  sess.run(model.update_k, feed_dict={model.new_k: 1.0})
  assert sess.run(model.k) == 1.0

def test_new_k_calculation_increase():
  img = tf.placeholder(tf.float32, (None, 128, 128, 3))
  z_noise = tf.placeholder(tf.float32, (None, 128))
  model = FF.FFGAN(img, z_noise)
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  sess.run(model.update_k, feed_dict={
    model.descrim_loss: 1.0,
    model.gen_loss: 0.1})
  assert sess.run(model.k) > 0.0

def test_new_k_calculation_decrease():
  img = tf.placeholder(tf.float32, (None, 128, 128, 3))
  z_noise = tf.placeholder(tf.float32, (None, 128))
  model = FF.FFGAN(img, z_noise)
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  sess.run(model.update_k, feed_dict={
    model.descrim_loss: 0.1,
    model.gen_loss: 1.1})
  assert sess.run(model.k) < 0.0

def test_correct_vars_train_generator():
  img = tf.placeholder(tf.float32, (None, 128, 128, 3), name="Image")
  z_noise = tf.placeholder(tf.float32, (None, 128), name="z_noise")
  model = FF.FFGAN(img, z_noise)
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  before_encoder = sess.run(model.encoder_vars)
  before_decoder = sess.run(model.decoder_gen_vars)
  sess.run(model.train_generator, feed_dict={
    z_noise: np.ones((1, 128))
  })
  after_encoder = sess.run(model.encoder_vars)
  after_decoder = sess.run(model.decoder_gen_vars)
  for a,b in zip(after_encoder, before_encoder):
    assert (a == b).all()
  for a,b in zip(after_decoder, before_decoder):
    assert (a != b).any()

def test_correct_vars_train_descriminator():
  img = tf.placeholder(tf.float32, (None, 128, 128, 3))
  z_noise = tf.placeholder(tf.float32, (None, 128))
  model = FF.FFGAN(img, z_noise)
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  before_encoder = sess.run(model.encoder_vars)
  before_decoder = sess.run(model.decoder_gen_vars)
  sess.run(model.train_descrim, feed_dict={
    img: np.ones((1, 128, 128, 3)),
    z_noise: np.ones((1, 128))
  })
  after_encoder = sess.run(model.encoder_vars)
  after_decoder = sess.run(model.decoder_gen_vars)
  for a,b in zip(after_encoder, before_encoder):
    assert (a != b).any()
  for a,b in zip(after_decoder, before_decoder):
    assert (a == b).all()


def test_var_amounts():
  img = tf.placeholder(tf.float32, (None, 128, 128, 3))
  z_noise = tf.placeholder(tf.float32, (None, 128))
  model = FF.FFGAN(img, z_noise)
  assert len(model.encoder_vars) != 0
  assert len(model.decoder_gen_vars) != 0
  assert len(set(model.encoder_vars) & set(model.decoder_gen_vars)) == 0
  model_vars = set(model.encoder_vars) | set(model.decoder_gen_vars) 
  all_vars = set(tf.trainable_variables())
  assert all_vars == model_vars

def test_gan_train_op_sanity():
  img = tf.placeholder(tf.float32, (None, 128, 128, 3))
  z_noise = tf.placeholder(tf.float32, (None, 128))
  model = FF.FFGAN(img, z_noise)
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  sess.run(model.gan_train_op, feed_dict={
      z_noise: np.ones((1, 128)),
      img: np.ones((1, 128, 128, 3))
    })

def test_input_requirements():
  img = tf.placeholder(tf.float32, (None, 128, 128, 3), name="Image")
  z_noise = tf.placeholder(tf.float32, (None, 128), name="z_noise")
  model = FF.FFGAN(img, z_noise)
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  with pytest.raises(tf.errors.InvalidArgumentError):
    _ = sess.run(model.gen_out, feed_dict={
          img: np.ones((1, 128, 128, 3)),
        })

  with pytest.raises(tf.errors.InvalidArgumentError):
    _ = sess.run(model.encoding_fake, feed_dict={
          img: np.ones((1, 128, 128, 3)),
        })


  with pytest.raises(tf.errors.InvalidArgumentError):
    _ = sess.run(model.decoded_fake, feed_dict={
          img: np.ones((1, 128, 128, 3)),
        })

  with pytest.raises(tf.errors.InvalidArgumentError):
    _ = sess.run(model.encoding_real, feed_dict={
          z_noise: np.ones((1, 128)),
        })

  with pytest.raises(tf.errors.InvalidArgumentError):
    _ = sess.run(model.decoded_real, feed_dict={
          z_noise: np.ones((1, 128)),
        })