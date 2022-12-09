"""Develops SSM of lung lobes based on landmarks provided by the user."""

import numpy as np

from . import StatisticalModelBase


class SSM(StatisticalModelBase):
  def __init__(self, landmarks):

    # align all samples to origin
    self.landmarks = landmarks - landmarks.mean(axis=1)[:, np.newaxis]
    self.landmarks_columns = self.landmarks.reshape(
      landmarks.shape[0], landmarks.shape[1] * landmarks.shape[2]
    )

    self.landmarks_columns_scale = (
      self.landmarks_columns / self.landmarks_columns.std(axis=1)[:, np.newaxis]
    )

  def compute_dataset_mean(self):
    """
    Args:
      landmark_files

    Returns:
      mean value of each landmark index to create the mean shape
    """
    return np.mean(self.landmarks_columns_scale, axis=0)
