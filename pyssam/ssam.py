"""Use SSM and SAM to analyse joint correlation between shape and
appearance."""

import argparse
from copy import copy
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

from . import SAM, SSM, StatisticalModelBase, euclidean_distance


class SSAM(StatisticalModelBase):
  def __init__(self, landmarks, appearance):

    # shape modelling classes
    self.ssm = SSM(landmarks)
    self.sam = SAM(appearance)

    # initialise input variables
    self.landmarks = landmarks
    self.appearance = self.sam.appearance_base

    # align all samples to origin
    self.landmarks = landmarks - landmarks.mean(axis=1)[:, np.newaxis]
    # initialise shape model data
    self.landmarks_columns = self.landmarks.reshape(
      self.landmarks.shape[0], self.landmarks.shape[1] * self.landmarks.shape[2]
    )
    (
      self.landmarks_columns_scale,
      self.appearance_columns_scale,
    ) = self.rescale_shape_and_appearance(
      self.landmarks_columns, self.appearance
    )

    self.shape_appearance = np.dstack(
      (
        self.landmarks_columns_scale.reshape(self.landmarks.shape),
        self.appearance_columns_scale,
      )
    )
    self.shape_appearance_columns = self.shape_appearance.reshape(
      self.shape_appearance.shape[0],
      self.shape_appearance.shape[1] * self.shape_appearance.shape[2],
    )

  def model_parameters(self):
    self.landmarks_columnvec_scale.shape[0]
    pass

  def compute_dataset_mean(self):
    """"""
    return np.mean(self.shape_appearance_columns, axis=0)

  def rescale_shape_and_appearance(self, landmarks_columns, appearance):
    """
    x: samples x 3landmarks matrix of coordinates
    g: samples x landmarks matrix of intensities (grey-value)
    """
    # normalise coordinates
    landmarks_columns_scale = landmarks_columns.copy()
    landmarks_columns_scale /= landmarks_columns_scale.std(axis=1)[
      :, np.newaxis
    ]
    landmarks_columns_scale = landmarks_columns_scale.reshape(
      self.landmarks.shape
    )
    # normalise gray values
    appearance_columns_scale = (
      appearance - appearance.mean(axis=1)[:, np.newaxis]
    )
    appearance_columns_scale = (
      appearance_columns_scale
      / appearance_columns_scale.std(axis=1)[:, np.newaxis]
    )

    return landmarks_columns_scale, appearance_columns_scale
