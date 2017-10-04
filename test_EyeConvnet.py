import EyeConvnet
import tensorflow as tf
import numpy as np
import pytest 

def setup_function(fnc):
  tf.reset_default_graph()

def test_sanity_build():
  face_tensor = tf.placeholder(tf.float32, (None, 128, 128, 3))
  left_eye_tensor = tf.placeholder(tf.float32, (None, 36, 60, 3))
  right_eye_tensor = tf.placeholder(tf.float32, (None, 36, 60, 3))
  face_pts_tensor = tf.placeholder(tf.float32, (None, 102))
  is_training = False
  EyeConvnet.EyeConvnet(False, face_tensor, left_eye_tensor, right_eye_tensor,
                        face_pts_tensor)

def test_variables_change():
  with tf.variable_scope("Convnet"):
    face_tensor = tf.placeholder(tf.float32, (None, 128, 128, 3))
    left_eye_tensor = tf.placeholder(tf.float32, (None, 36, 60, 3))
    right_eye_tensor = tf.placeholder(tf.float32, (None, 36, 60, 3))
    face_pts_tensor = tf.placeholder(tf.float32, (None, 102))
    is_training = False
    model = EyeConvnet.EyeConvnet(False, face_tensor, left_eye_tensor, right_eye_tensor,
                          face_pts_tensor)
    opt = tf.train.AdamOptimizer()
    train = opt.minimize(model.prediction)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    before = sess.run(tf.trainable_variables())
    _ = sess.run(train, feed_dict={
      face_tensor: np.ones((1, 128, 128, 3)),
      left_eye_tensor: np.ones((1, 36, 60, 3)),
      right_eye_tensor: np.ones((1, 36, 60, 3)),
      face_pts_tensor: np.ones((1, 102))
      })
    after = sess.run(tf.trainable_variables())
    for b, a in zip(before, after):
      assert (b != a).any()

def test_needs_inputs():
  with tf.variable_scope("Convnet"):
    face_tensor = tf.placeholder(tf.float32, (None, 128, 128, 3))
    left_eye_tensor = tf.placeholder(tf.float32, (None, 36, 60, 3))
    right_eye_tensor = tf.placeholder(tf.float32, (None, 36, 60, 3))
    face_pts_tensor = tf.placeholder(tf.float32, (None, 102))
    is_training = False
    model = EyeConvnet.EyeConvnet(False, face_tensor, left_eye_tensor, right_eye_tensor,
                          face_pts_tensor)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # No face_pts
    with pytest.raises(Exception):
      _ = sess.run(model.prediction, feed_dict={
        face_tensor: np.ones((1, 128, 128, 3)),
        left_eye_tensor: np.ones((1, 36, 60, 3)),
        right_eye_tensor: np.ones((1, 36, 60, 3)),
        })
    # No right eye
    with pytest.raises(Exception):
      _ = sess.run(model.prediction, feed_dict={
        face_tensor: np.ones((1, 128, 128, 3)),
        left_eye_tensor: np.ones((1, 36, 60, 3)),
        face_pts_tensor: np.ones((1, 102))
        })
    # No left eye
    with pytest.raises(Exception):
      _ = sess.run(train, feed_dict={
        face_tensor: np.ones((1, 128, 128, 3)),
        right_eye_tensor: np.ones((1, 36, 60, 3)),
        face_pts_tensor: np.ones((1, 102))
        })
    # No face
    with pytest.raises(Exception):
      _ = sess.run(train, feed_dict={
        left_eye_tensor: np.ones((1, 36, 60, 3)),
        right_eye_tensor: np.ones((1, 36, 60, 3)),
        face_pts_tensor: np.ones((1, 102))
        })