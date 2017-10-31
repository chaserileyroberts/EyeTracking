import pytest
import tensorflow as tf 
import EyeBEGAN

def test_EyeBEGAN_sanity():
  z = tf.placeholder(tf.float32, (None, 100))
  face_tensor = tf.placeholder(tf.float32, (None, 128, 128, 3))
  left_eye_tensor = tf.placeholder(tf.float32, (None, 36, 60, 3))
  right_eye_tensor = tf.placeholder(tf.float32, (None, 36, 60, 3))
  face_pts_tensor = tf.placeholder(tf.float32, (None, 102))
  model = EyeBEGAN.EyeBEGAN(z, face_tensor, left_eye_tensor, 
      right_eye_tensor, face_pts_tensor)

def test_correct_variables_change():
  pass
  
def test_began_step():
  pass