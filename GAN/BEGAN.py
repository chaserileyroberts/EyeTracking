import tensorflow as tf
import numpy as np
from PIL import Image


def inverse_image(img):
    img = (img + 0.5) * 255.
    img[img > 255] = 255
    img[img < 0] = 0
    img = img[..., ::-1]  # bgr to rgb
    return img.astype('uint8')


def interpolate(a, b, alpha=0.5):
  return a * alpha + b * (1 - alpha)


def get_img_from_vector(vec, bottle_neck, output, index, sess):
  results = sess.run(output, feed_dict={bottle_neck: vec})
  image_array = results[0]
  image_array = inverse_image(image_array)
  img = Image.fromarray(image_array)
  img = img.resize((300, 300))
  return img


if __name__ == '__main__':
  sess = tf.Session()
  saver = tf.train.import_meta_graph(
      './BeganPretrained/began_gm0.4.model-429439.meta',
                                     clear_devices=True)
  saver.restore(sess, './BeganPretrained/began_gm0.4.model-429439')
  graph = tf.get_default_graph()
  x = graph.get_tensor_by_name('x:0')
  bottle_neck = graph.get_tensor_by_name('disc__2/enc_fc/BiasAdd:0')
  discrim_image = graph.get_tensor_by_name('disc__3/conv6_a/BiasAdd:0')
  generated = graph.get_tensor_by_name('gen_/conv6_a/BiasAdd:0')
  rand_vals = np.random.random((16, 128))
  rand_vals[2] = interpolate(rand_vals[0], rand_vals[1])
  get_img_from_vector(rand_vals,
