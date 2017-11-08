import tensorflow as tf
import Preprocess
import numpy as np

def setup():
  tf.reset_default_graph()

def test_random_image():
    # Makes tests deterministic.
    np.random.seed(0)
    random_image = np.random.randint(255, size=(36, 60, 3), dtype='uint8')
    tensor_input = tf.placeholder(tf.uint8, (36, 60, 3))
    float_image = Preprocess.image_preprocess(tensor_input)
    sess = tf.Session()
    result = sess.run(float_image, feed_dict={tensor_input: random_image})
    assert (result <= 1).all()
    assert (result >= -1).all()
