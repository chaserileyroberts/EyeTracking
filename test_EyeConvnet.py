import EyeConvnet
import tensorflow as tf
import numpy as np
import pytest
import mltest

slim = tf.contrib.slim 


def setup_function(fnc):
    tf.reset_default_graph()

def setup():
    mltest.setup()

def test_suite():
    face_tensor = tf.placeholder(tf.float32, (None, 128, 128, 3))
    left_eye_tensor = tf.placeholder(tf.float32, (None, 36, 60, 3))
    right_eye_tensor = tf.placeholder(tf.float32, (None, 36, 60, 3))
    face_pts_tensor = tf.placeholder(tf.float32, (None, 102))
    model = EyeConvnet.EyeConvnet(
        True, face_tensor, left_eye_tensor, right_eye_tensor,
                          face_pts_tensor)
    opt = tf.train.AdamOptimizer()
    train = slim.learning.create_train_op(model.prediction, opt)
    mltest.test_suite(model.prediction, train, feed_dict={
      face_tensor: np.random.normal(size=(1, 128, 128, 3)) + 1000,
      left_eye_tensor: np.random.normal(size=(1, 36, 60, 3)),
      right_eye_tensor: np.random.normal(size=(1, 36, 60, 3)),
      face_pts_tensor: np.random.normal(size=(1, 102))
    })

