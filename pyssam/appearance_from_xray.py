"""Extract appearance information from X-ray."""
import numpy as np


class AppearanceFromXray:
  def __init__(self, landmarks, imgs_all, img_origin, img_spacing):

    # initialise variables
    self.landmarks = landmarks
    self.imgs_all = imgs_all
    self.pixel_coordinates = self.radiograph_to_realworld_coordinates(
      self.imgs_all, img_origin, img_spacing
    )
    self.appearance_base = self.all_landmark_density(
      self.landmarks, self.imgs_all, self.pixel_coordinates
    )

  def radiograph_to_realworld_coordinates(self, img, origin, spacing):
    """Set bottom left coordinate to CT origin, and assign real world coord to
    DRR."""
    xCoords = np.zeros((origin.shape[0], img.shape[-2]))
    zCoords = np.zeros((origin.shape[0], img.shape[-1]))

    xBase = np.linspace(0, img.shape[-2], img.shape[-2])
    zBase = np.linspace(0, img.shape[-1], img.shape[-1])
    if origin.ndim == 2:
      xCoords = (
        origin[:, 0]
        + np.meshgrid(xBase, np.ones(spacing[:, 0].size))[0].T * spacing[:, 0]
      )
      zCoords = (
        origin[:, 2]
        + np.meshgrid(zBase, np.ones(spacing[:, 2].size))[0].T * spacing[:, 2]
      )
    elif origin.ndim == 1:
      xCoords = (
        origin[0]
        + np.meshgrid(xBase, np.ones(spacing[0].size))[0].T * spacing[0]
      )
      zCoords = (
        origin[2]
        + np.meshgrid(zBase, np.ones(spacing[2].size))[0].T * spacing[2]
      )
    else:
      Warning(
        "unexpected origin dimensions in SAM.radiograph_to_realworld_coordinates"
      )

    return np.dstack((np.swapaxes(xCoords, 0, 1), np.swapaxes(zCoords, 0, 1)))

  def compute_landmark_density(self, landmarks, img, pixel_coordinates):
    """Returns density of a landmark based on comparing landmark coordinates to
    pixel with nearest real world coordinate in x and z direction.

    Inputs:
        landmarks: (coords x 3) array of landmarks
        img: (drr x-dimension x drr y-dimension) array of grey values
        pixel_coordinates: (drr x-dimension x 2)
            Note that pixel_coordinates axis=1 value assumes a square figure
            (same number of x and y pixels)

    Return:
        density: (coords) array of grey value for each landmarks
    """
    # use argmin to find nearest pixel neighboring a point
    nearestX = np.argmin(
      abs(landmarks[:, 0] - pixel_coordinates[:, 0].reshape(-1, 1)), axis=0
    )
    nearestZ = np.argmin(
      abs(landmarks[:, 2] - pixel_coordinates[:, 1].reshape(-1, 1)), axis=0
    )

    return img[len(img) - 1 - nearestZ, nearestX]  # gives correct result

  def all_landmark_density(self, landmarks, img, pixel_coordinates):
    """Returns density of a landmark based on comparing landmark coordinates to
    pixel with nearest real world coordinate in x and z direction.

    Inputs:
        landmarks: (patients x coords x 3) array of landmarks
        img: (patients x drr x-dimension x drr y-dimension) array of grey values
        pixel_coordinates: (patients x drr x-dimension x 2)
            Note that pixel_coordinates axis=1 value assumes a square figure
            (same number of x and y pixels)

    Return:
        density: (patients x coords) array of grey value for each landmark
    """

    dshape = list(landmarks.shape[:-1])
    density = np.zeros(dshape)

    for p in range(landmarks.shape[0]):
      density[p] = self.compute_landmark_density(
        landmarks[p], img[p], pixel_coordinates[p]
      )

    return density
