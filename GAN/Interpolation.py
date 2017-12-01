import tensorflow as tf
import numpy as np 
from misc import loadmat
from GAN.FeatureForcing import FFGAN
import random
import scipy.spatial
import math
import sklearn.decomposition


if __name__ == '__main__':
  image_tensor = tf.placeholder(tf.float32, (None, 128, 128, 3))
  z_tensor = tf.placeholder(tf.float32, (None, 128))
  model = FFGAN(image_tensor, z_tensor)
  faces = loadmat('/media/roberc4/kang/final_dataset/kang/day01/Center/data.mat')['face']
  faces2 = loadmat('/media/roberc4/kang/final_dataset/kang/day05/Center/data.mat')['face']
  faces = np.append(faces, faces2, axis=0)
  faces3 = loadmat('/media/roberc4/kang/final_dataset/kang/day08/Center/data.mat')['face']
  faces = np.append(faces, faces3, axis=0)
  saver = tf.train.Saver()
  path = tf.train.latest_checkpoint('/media/roberc4/kang/chase_models/kang_began')
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  saver.restore(sess, path)
  vectors = []
  diff_vectors = []
  print(faces.shape)
  for i in np.random.choice(faces.shape[0], 1000):
    face = np.reshape(faces[i], (1, 128, 128, 3))
    flipped_face = np.fliplr(face)
    vec1 = sess.run(model.encoding_real, feed_dict={
          image_tensor: face
        })
    vec2 = sess.run(model.encoding_real, feed_dict={
          image_tensor: flipped_face
        })
    vectors.append(vec1)
    diff_vectors.append(vec2)
  values = []
  pca = sklearn.decomposition.PCA(128)
  vectors = [v.flatten() for v in vectors]
  pca.fit(np.array(vectors))
  print("PCA results")
  print(pca.explained_variance_ratio_[0])
  diff_vectors = [v.flatten() for v in diff_vectors]
  pre_transform = vectors
  pre_diff = [a - b for a,b in zip(vectors, diff_vectors)]
  vectors = pca.transform(vectors)
  diff_vectors = pca.transform(np.array(diff_vectors))
  vectors = [v[0:] for v in vectors]
  diff_vectors = [v[0:] for v in diff_vectors]
  basis_vectors = [a - b for a,b in zip(vectors, diff_vectors)]
  for i,vec1 in enumerate(pre_diff):
    for j,vec2 in enumerate(pre_diff[i+1:]):
      values.append(math.fabs(
	  scipy.spatial.distance.cosine(vec1, vec2) - 1))
  print("Diff variance")
  print(sum(values)/len(values))
  for i,vec1 in enumerate(pre_transform):
    for j,vec2 in enumerate(pre_transform[i+1:]):
      values.append(math.fabs(
	  scipy.spatial.distance.cosine(vec1, vec2) - 1))
  print("Normal variance")
  print(sum(values)/len(values))
