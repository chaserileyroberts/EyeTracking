import tensorflow as tf

slim = tf.contrib.slim

class FFGAN():
  def __init__(self, real_img, z_vector):
    """Feature Forcing Generative Adversarial Network
    Args:
      real_img: Tensor to input real images.
      z_vector: Tensor for the noise input vector.
    """
    decoder_gen_template = tf.make_template(
        "decoder_generator", self.make_decoder_generator)
    self.encoding = self.make_encoder(real_img, z_vector.shape[1])
    self 
  def make_encoder(self, img, encoding_size=100):
    """Convnet Feature Encoder
    Args:
      img: Input image tensor.
      encoding_size: Size of the output encoding. Default to 100.
    Returns:
      encoding: Encoding tensor of size (batch_size, encoding_size).
    """
    with tf.variable_scope("encoder"):
      # Try to normalize the input before convoluting
      net = slim.conv2d(img, 32, [11, 11], scope="conv1_11x11")
      net = slim.conv2d(net, 64, [5, 5], scope="conv2_5x5")
      net = slim.max_pool2d(net, [4, 4], stride=4, scope='pool1')
      net = slim.conv2d(net, 64, [5, 5], scope="conv3_5x5")
      net = slim.conv2d(net, 128, [3, 3], scope="conv4_3x3")
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.conv2d(net, 128, [3, 3], scope="conv5_3x3")
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.conv2d(net, 32, [1, 1], scope="conv6_1x1")
      net = slim.flatten(net)
      net = slim.fully_connected(net, int(encoding_size))
      return net

  def make_decoder_generator(self, encoding):
    """Deconvnet Generator/Decoder.
    Args:
      encoding: Encoding tensor. Either random or output from 
          encoder.
    Returns:
      output: Output image tensor.
    """
    # Defining this function for my ownsanity
    deconv = slim.conv2d_transpose

    net = slim.fully_connected(128, encoding)
    net = 

