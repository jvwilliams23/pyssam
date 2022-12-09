"""Use SSM and SAM to analyse joint correlation between shape and
appearance."""
import numpy as np

from . import SAM, SSM, StatisticalModelBase


class SSAM(StatisticalModelBase):
  def __init__(self, landmarks, appearance):

    # shape modelling classes
    self.ssm = SSM(landmarks)
    self.sam = SAM(appearance)

    # initialise input variables
    self.landmarks_columns_scale = self.ssm.landmarks_columns_scale
    self.appearance = self.sam.appearance_base
    self.appearance_columns_scale = self.sam.appearance_scale

    # align all samples to origin
    self.landmarks = landmarks - landmarks.mean(axis=1)[:, np.newaxis]
    # initialise shape model data
    self.landmarks_columns = self.landmarks.reshape(
      self.landmarks.shape[0], self.landmarks.shape[1] * self.landmarks.shape[2]
    )

    self.shape_appearance = np.dstack(
      (
        self.landmarks_columns_scale.reshape(landmarks.shape),
        self.appearance_columns_scale,
      )
    )
    self.shape_appearance_columns = self.shape_appearance.reshape(
      self.shape_appearance.shape[0],
      self.shape_appearance.shape[1] * self.shape_appearance.shape[2],
    )

  def compute_dataset_mean(self):
    """"""
    return np.mean(self.shape_appearance_columns, axis=0)
