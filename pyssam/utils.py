import numpy as np
from skimage import io
from skimage.color import rgb2gray

__all__ = ["euclidean_distance", "loadXR"]


def euclidean_distance(x, y):
  """Finds the euclidean distance between two arrays x, y.

  Calculated using pythagoras theorem
  """
  if x.size <= 3:
    return np.sqrt(np.sum((x - y) ** 2))
  else:
    return np.sqrt(np.sum((x - y) ** 2, axis=1))


def loadXR(file):
  """take input X-ray as png (or similar) and convert to grayscale np array.

  can add any necessary pre-processing steps in here, such as utils.he
  (histogram equalisation contrast enhancement)
  """
  g_im = rgb2gray(io.imread(file))
  g_im /= 255
  return g_im
