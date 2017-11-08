import tensorflow as tf
import Preprocess

slim = tf.contrib.slim


class FFGAN():
  def __init__(self, real_img, z_vector, gamma=0.5):
    """Feature Forcing Generative Adversarial Network
    Args:
      real_img: Tensor to input real images.
      z_vector: Tensor for the noise input vector.
      gamma: Variation level for generator.
    """
    self.real_img = real_img
    self.z_vector = z_vector
    self.prop_gain = 0.01
    self.encoding_size = z_vector.shape[1]
    self.k = tf.Variable(10.0, name="k", trainable=False)
    encoder_template = tf.make_template("encoder", self.make_encoder)
    decoder_gen_template = tf.make_template(
        "decoder_generator", self.make_decoder_generator)
    # Make encoder/decoder for the real images.
    self.encoding_real = encoder_template(real_img, self.encoding_size)
    self.decoded_real = decoder_gen_template(self.encoding_real)

    # Make results for the generated images
    self.gen_out = decoder_gen_template(z_vector)
    self.encoding_fake = encoder_template(self.gen_out, self.encoding_size)
    self.decoded_fake = decoder_gen_template(self.encoding_fake)

    # Calculate the losses
    self.img_diff_real = tf.losses.mean_squared_error(
        real_img, self.decoded_real)
    self.img_diff_fake = tf.losses.mean_squared_error(
        self.gen_out, self.decoded_fake)
    self.descrim_loss = self.img_diff_real - self.k * self.img_diff_fake
    self.gen_loss = self.img_diff_fake
    # TODO(chase): Test this
    self.new_k = (self.k 
        + self.prop_gain * (gamma * self.descrim_loss - self.gen_loss))
    self.update_k = tf.assign(self.k, self.new_k)    

    # Build optimizers. Make sure to only train certain variables.
    optimizer = tf.train.AdamOptimizer()
    self.encoder_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES) #scope="encoder")
    self.decoder_gen_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="decoder_generator")
    self.train_descrim = slim.learning.create_train_op(
        self.descrim_loss, 
        optimizer,
        variables_to_train=self.encoder_vars)
    self.train_generator = slim.learning.create_train_op(
        self.gen_loss, 
        optimizer,
        variables_to_train=self.decoder_gen_vars)
    # Training step for GAN
    self.gan_train_op = tf.group(
        self.train_descrim, self.train_generator, self.update_k)

    self.convergence = (
        self.img_diff_real 
        + tf.abs(gamma * self.img_diff_real - self.img_diff_fake))
    # Some book keeping
    self.make_summaries()

  def make_summaries(self):
    """Creates summaries"""
    tf.summary.scalar("gen_loss", self.gen_loss)
    tf.summary.scalar("descrim_loss", self.descrim_loss)
    tf.summary.scalar("k", self.k)
    tf.summary.scalar("model_convergence", self.convergence)
    tf.summary.image("real_image", 
            Preprocess.image_correction(self.real_img))
    tf.summary.image("generated_image", 
            Preprocess.image_correction(self.gen_out))
    tf.summary.image("real_reconstruction", 
            Preprocess.image_correction(self.decoded_real))
    tf.summary.image("generated_reconstruction", 
            Preprocess.image_correction(self.decoded_fake))
    tf.summary.histogram("read_encodings", self.encoding_real)
    tf.summary.histogram("fake_encodings", self.encoding_fake)
    # Histogram for all of the variables.
    for var in tf.trainable_variables():
      tf.summary.histogram(var.name, var)

  def make_encoder(self, img, encoding_size):
    """Convnet Feature Encoder
    Args:
      img: Input image tensor.
      encoding_size: Size of the output encoding.
    Returns:
      encoding: Encoding tensor of size (batch_size, encoding_size).
    """
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
    net = slim.fully_connected(net, 256)
    net = slim.fully_connected(net, int(encoding_size), activation_fn=None)
    return net

  def make_decoder_generator(self, encoding):
    """Deconvnet Generator/Decoder.
    Args:
      encoding: Encoding tensor. Either random or output from 
          encoder.
    Returns:
      output: Output image tensor.
    """
    #TODO(Chase) Add skip connections.
    net = slim.fully_connected(encoding, 256)
    net = tf.reshape(net, [-1, 16, 16, 1])
    net = slim.conv2d(net, 32, [3, 3], scope="conv1_3x3")
    net = tf.image.resize_images(net, [int(i) * 2 for i in net.shape[1:3]])
    net = slim.conv2d(net, 128, [3, 3], scope="conv2_3x3")
    net = tf.image.resize_images(net, [int(i) * 2 for i in net.shape[1:3]])
    net = slim.conv2d(net, 128, [3, 3], scope="conv3_3x3")
    net = slim.conv2d(net, 64, [5, 5], scope="conv4_5x5")
    net = tf.image.resize_images(net, [int(i) * 2 for i in net.shape[1:3]])
    net = slim.conv2d(net, 64, [5, 5], scope="conv5_5x5")
    net = slim.conv2d(net, 3, [11, 11], scope="conv6_11x11")
    return net


