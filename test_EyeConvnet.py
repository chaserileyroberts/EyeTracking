import EyeConvnet
import tensorflow as tf


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
