import tensorflow as tf
import numpy as np 
from misc import loadmat
from GAN.FeatureForcing import FFGAN
import random
import scipy.spatial
import math

if __name__ == '__main__':
  image_tensor = tf.placeholder(tf.float32, (None, 128, 128, 3))
  z_tensor = tf.placeholder(tf.float32, (None, 128))
  model = FFGAN(image_tensor, z_tensor)
  faces = loadmat('/media/roberc4/kang/final_dataset/kang/day01/Center/data.mat')['face']
  sess = tf.Session()
  sess.run(tf.global_variables_initalizer())
  vectors = []
  for i in np.random.choice(100, range(1990)):
    face = np.reshape(faces[random.randint(0, 1990)], (1, 128, 128, 3))
    flipped_face = np.fliplr(face)
    vec1 = sess.run(model.encoding_real, feed_dict={
          image_tensor: face
        })
    vec2 = sess.run(model.encoding_real, feed_dict={
          image_tensor: flipped_face
        })
    diff_vec = vec1 - vec2
    vectors.append(diff_vec)
  values = 
  for i,vec1 in enumerate(vectors):
    for j,vec2 in enumerate(vectors[i+1:]):
      values.append(math.abs(
          scipy.spatial.distance.cosine(vec1, vec2) - 1))
  print(sum(values)/len(values))