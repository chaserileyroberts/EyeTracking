import tensorflow as tf 
import EyeConvnet

slim = tf.contrib.slim

class EyeBEGAN(EyeConvnet.EyeConvnet):
  def __init__(self, 
               z_noise,
               face_tensor,
               left_eye_tensor,
               right_eye_tensor,
               face_pts):
    """Eye Boundary Equalibrium GAN.
    Experimental code used for unsupervised data augmentation.
    Args:
      z_noise: N-d noise vector. Used for the generator.
      face_tensor: Tensor used to input the entire face image.
      left_eye_tensor: Tensor used for the left eye image.
      right_eye_tensor: Tensor used for the right eye image.
      face_pts: Tensor for the face pts (possibly ignore this).
    """
    # DO NOT RUN THE super(...).__init__()
    # This network has a very differet structure than the 
    # EyeConvnet, but we still want some of its existing functionality.
    pass

  def make_encoder(self, z_size, face_img, left_img, right_img, face_pts):
    """ Create the encoder Convnet.
    Args:
      z_size: Size of the bottleneck
      face_img: Tensor used to input the entire face image.
      left_img: Tensor used for the left eye image.
      right_img: Tensor used for the right eye image.
      face_pts: Tensor for the face pts (possibly ignore this).
    Returns: 
      encoding: Tensor of size (batch_size, z_size) of the 
          bottleneck encoding.
    """
    pass

  def make_decoder_generator(self, z_size, 
      face_img, left_img, right_img, face_pts):
    """ Create the encoder Convnet.
    Args:
      z_size: Size of the bottleneck
      face_img: Tensor used to input the entire face image.
      left_img: Tensor used for the left eye image.
      right_img: Tensor used for the right eye image.
      face_pts: Tensor for the face pts (possibly ignore this).
    Returns: 
      face_out: Generated/Decoded entire face image tensor.
      left_out: Generated/Decoded left eye image tensor.
      right_out: Generated/Decoded right eye image tensor.
      face_pts_out: Generated/Decoded face_pts tensor
    """
    pass
