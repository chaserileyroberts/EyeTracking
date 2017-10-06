import tensorflow as tf


def image_preprocess(image):
    """Takes in a uint8 Tensor and transforms it to a (-1, 1) float Tensor.
    Args:
      image: Tensor uint8 array.
    Returns:
      float_image: Tensor float32 array with range (-1, 1)
    """
    return (tf.to_float(image) / 127.5) - 1.0


def gaze_images_preprocess(face, left, right, gaze, pts):
    """Preprocesses all of the gaze images.
    Args:
      face: Face tensor.
      left: Left eye tensor.
      right: Right eye tensor.
      gaze: Gaze direction tensor.
      pts: Face pts tensor.
    Returns:
      face: Preprocessed face image.
      left: Preprocessed left image.
      right: Preprocessed right image.
      gaze: Unchange gaze.
      pts: Unchanged pts.
    """
    return (image_preprocess(face),
            image_preprocess(left),
            image_preprocess(right),
            gaze,
            pts)
