"""Utilities to aid modelling classes."""

from warnings import warn

import numpy as np
from skimage.io import imread

__all__ = ["euclidean_distance", "loadXR", "AppearanceFromXray"]


def euclidean_distance(x, y) -> np.ndarray:
  """Finds the euclidean distance between two arrays x, y.

  Parameters
  ----------
  x : array_like
      Coordinates as 1D or 2D array
  y : array_like
      Coordinates as 1D or 2D array

  Returns
  -------
  """
  if x.size <= 3:
    return np.sqrt(np.sum((x - y) ** 2))
  else:
    return np.sqrt(np.sum((x - y) ** 2, axis=1))


def loadXR(file) -> np.ndarray:
  """Take input X-ray as png (or similar) and convert to grayscale np array
  with range 0 to 1.

  Parameters
  ----------
  file : str
      Image file name of X-ray.

  Returns
  -------
  grayscale_image : array_like
      Pixel values as grayscale with range [0:1].
  """
  grayscale_image = imread(file, as_gray=True)
  return grayscale_image


class AppearanceFromXray:
  """Extract appearance information from X-ray.

  Parameters
  ----------
  imgs_all : array_like
      3D array for all images in dataset, where the first dimension
      is the number of samples. Second and third dimensions are the image pixels.
  img_origin : array_like
      2D array for spatial coordinates of origins for all images in dataset.
      The first dimension is the number of samples.
      If one image is used, only x and y are needed to expected size is 2.
  img_spacing : array_like
      2D array for pixel spacing of all images in dataset.
      The first dimension is the number of samples.
      If one image is used, only x and y are needed to expected size is 2.

  Examples
  ========
  >>> import numpy as np
  >>> import pyssam
  >>> num_samples = 5
  >>> num_pixels = 250
  >>> image_all = np.random.rand(num_samples, num_pixels, num_pixels)
  >>> origin_all = np.zeros((num_samples, 2))
  >>> spacing_all = np.ones((num_samples, 2))
  >>> appearance_helper = pyssam.utils.AppearanceFromXray(image_all, origin_all, spacing_all)
  using 2D coordinates for X-ray
  >>> print(appearance_helper.pixel_coordinates.shape)
  (5, 250, 2)
  >>> appearance_helper = pyssam.utils.AppearanceFromXray(np.random.rand(250,250), np.zeros(3), np.ones(3))
  using 3D coordinates for XR
  >>> print(appearance_helper.pixel_coordinates.shape)
  (1, 250, 3)
  """

  def __init__(
    self, imgs_all: np.ndarray, img_origin: np.ndarray, img_spacing: np.ndarray
  ):

    # initialise variables
    self.imgs_all = imgs_all
    # generate spatial coordinates for each pixel on X-ray
    assert (
      img_origin.shape == img_spacing.shape
    ), (
      "Image origin and spacing not with same number of spatial dimensions " 
      f"({img_origin.shape} != {img_spacing.shape})"
    )
    self.pixel_coordinates = self.radiograph_to_realworld_coordinates(
      self.imgs_all, img_origin, img_spacing
    )

  def radiograph_to_realworld_coordinates(
    self, img: np.ndarray, origin: np.ndarray, spacing: np.ndarray
  ) -> np.ndarray:
    """Get 2D or 3D set of coordinates (xy or xyz) for image pixels.

    Parameters
    ----------
    imgs : array_like
        3D array for all images in dataset, where the first dimension
        is the number of samples. Second and third dimensions are the image pixels.
    origin : array_like
        2D array for spatial coordinates of origins for all images in dataset.
        The first dimension is the number of samples.
        Second dimension is the number of spatial dimensions
        If one image is used, only x and y are needed to expected size is 2.
    spacing : array_like
        2D array for pixel spacing of all images in dataset.
        The first dimension is the number of samples.
        Second dimension is the number of spatial dimensions
        If one image is used, only x and y are needed to expected size is 2.

    Returns
    -------
    pixel_coordinates : array_like
        Array with spatial locations of each pixel.

    Raises
    ------
    AssertionError
        If final dimension of spacing has shape not equal to 2 or 3.
    AssertionError
        If size of final two dimensions of img not equal to each other (square image)
    """
    assert img.shape[-1] == img.shape[-2], f"image not square (shape {img.shape}"
    if spacing.shape[-1] == 2:
      print("using 2D coordinates for X-ray")
      return self.radiograph_to_realworld_coordinates_2D(img, origin, spacing)
    elif spacing.shape[-1] == 3:
      print("using 3D coordinates for XR")
      return self.radiograph_to_realworld_coordinates_3D(img, origin, spacing)
    else:
      raise AssertionError(
        f"Shape of spacing not recognised {spacing.shape}. "
        "Should represent 2D or 3D images (final value = 2 or 3)"
      )

  def radiograph_to_realworld_coordinates_2D(
    self, img: np.ndarray, origin: np.ndarray, spacing: np.ndarray
  ) -> np.ndarray:
    x_coords = np.zeros((origin.shape[0], img.shape[-2]))
    y_coords = np.zeros((origin.shape[0], img.shape[-1]))

    x_base = np.linspace(0, img.shape[-2], img.shape[-2])
    y_base = np.linspace(0, img.shape[-1], img.shape[-1])
    if origin.ndim == 2:
      x_coords = (
        origin[:, 0]
        + np.meshgrid(x_base, np.ones(spacing[:, 0].size))[0].T * spacing[:, 0]
      )
      y_coords = (
        origin[:, 1]
        + np.meshgrid(y_base, np.ones(spacing[:, 1].size))[0].T * spacing[:, 1]
      )
    elif origin.ndim == 1:
      x_coords = (
        origin[0]
        + np.meshgrid(x_base, np.ones(spacing[0].size))[0].T * spacing[0]
      )
      y_coords = (
        origin[1]
        + np.meshgrid(y_base, np.ones(spacing[1].size))[0].T * spacing[1]
      )
    else:
      raise AttributeError(
        "unexpected origin dimensions in radiograph_to_realworld_coordinates"
      )

    return np.dstack((np.swapaxes(x_coords, 0, 1), np.swapaxes(y_coords, 0, 1)))

  def radiograph_to_realworld_coordinates_3D(
    self, img: np.ndarray, origin: np.ndarray, spacing: np.ndarray
  ) -> np.ndarray:
    x_coords = np.zeros((origin.shape[0], img.shape[-2]))
    y_coords = np.zeros((origin.shape[0], img.shape[-2]))
    z_coords = np.zeros((origin.shape[0], img.shape[-1]))

    x_base = np.linspace(0, img.shape[-2], img.shape[-2])
    y_base = np.linspace(0, img.shape[-2], img.shape[-2])
    z_base = np.linspace(0, img.shape[-1], img.shape[-1])
    if origin.ndim == 2:
      x_coords = (
        origin[:, 0]
        + np.meshgrid(x_base, np.ones(spacing[:, 0].size))[0].T * spacing[:, 0]
      )
      y_coords = (
        origin[:, 1]
        + np.meshgrid(y_base, np.ones(spacing[:, 1].size))[0].T * spacing[:, 1]
      )
      z_coords = (
        origin[:, 2]
        + np.meshgrid(z_base, np.ones(spacing[:, 2].size))[0].T * spacing[:, 2]
      )
    elif origin.ndim == 1:
      x_coords = (
        origin[0]
        + np.meshgrid(x_base, np.ones(spacing[0].size))[0].T * spacing[0]
      )
      y_coords = (
        origin[1]
        + np.meshgrid(x_base, np.ones(spacing[1].size))[0].T * spacing[1]
      )
      z_coords = (
        origin[2]
        + np.meshgrid(z_base, np.ones(spacing[2].size))[0].T * spacing[2]
      )
    else:
      raise AttributeError(
        "Unexpected origin dimensions in radiograph_to_realworld_coordinates"
      )

    return np.dstack(
      (
        np.swapaxes(x_coords, 0, 1),
        np.swapaxes(y_coords, 0, 1),
        np.swapaxes(z_coords, 0, 1),
      )
    )

  def compute_landmark_density(
    self, landmarks: np.ndarray, img: np.ndarray, pixel_coordinates: np.ndarray
  ) -> np.ndarray:
    """Find the gray value at each landmark based on nearest neighbor
    interpolation to pixel coordinates on image. The grayvalues are all
    normalised to zero mean and unit variance.

    Parameters
    ----------
    landmarks: array_like
        Array of all landmarks for one sample. Expected shape is 2D, where
        first dimension has size equal to number of landmarks and
        second dimension has size of two.
    img: array_like
        Image to extract appearance from. Expected 2D array with equal size
        in first and second dimensions to represent pixels.
    pixel_coordinates:
        Array with spatial locations of each pixel.
        Note that pixel_coordinates axis=1 value assumes a square figure
        (same number of x and y pixels)

    Returns
    -------
    landmark_grayvalue: array_like
        1D array of grey value for each landmark.

    Raises
    ------
    AssertionError
        If shape of input arguments do not agree.
    """
    assert (
      landmarks.shape[1] == pixel_coordinates.shape[1]
    ), "landmarks must have same number of spatial dimensions as pixel_coordinates"
    assert img.ndim == 2, f"img has unexpected number of dimensions {img.shape}"
    assert pixel_coordinates.shape[0] == img.shape[0] == img.shape[1], (
      "Each axis of img should have same size as 0th axis of pixel_coordinates "
      f"{pixel_coordinates.shape[0]} == {img.shape[0]} == {img.shape[1]} "
      " (pixel_coordinates.shape[0] == img.shape[0] == img.shape[1]"
    )
    # use argmin to find nearest pixel neighboring a point
    nearest_pixel_xaxis = np.argmin(
      abs(landmarks[:, 0] - pixel_coordinates[:, 0].reshape(-1, 1)), axis=0
    )
    nearest_pixel_yaxis = np.argmin(
      abs(landmarks[:, 1] - pixel_coordinates[:, 1].reshape(-1, 1)), axis=0
    )

    landmark_grayvalue = img[
      len(img) - 1 - nearest_pixel_yaxis, nearest_pixel_xaxis
    ]
    normalised_density = landmark_grayvalue - landmark_grayvalue.mean()
    normalised_density /= normalised_density.std()

    return normalised_density

  def all_landmark_density(self, landmarks: np.ndarray) -> np.ndarray:
    """Returns density of all landmarks in a dataset based on comparing
    landmark coordinates to spatial coordinates of pixels in X-rays.

    Parameters
    ----------
    landmarks: array_like
        Landmark coordinates used to find appearance. Array has 3 dimensions
        with shape (num_samples, num_landmarks, 2)
    img: array_like
        Array of grey-values from X-rays, with 3 dimensions and shape is
        (num_samples, num_x_pixels, num_y_pixels). Images should be square
        (meaning same number of pixels in x and y axes).
    pixel_coordinates: array_like
        Spatial coordinates corresponding to each voxel, with shape
        with shape (num_samples, num_x_pixels, 2)

    Return
    ------
    density: array_like
        Normalised grey-value for each landmark location, with shape
        (num_samples, num_landmarks).
    """

    density = np.zeros(landmarks.shape[:-1])

    for i in range(landmarks.shape[0]):
      density[i] = self.compute_landmark_density(
        landmarks[i], self.imgs_all[i], self.pixel_coordinates[i]
      )

    return density
