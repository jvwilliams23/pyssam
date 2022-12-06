"""Develops SAM based on landmarks and DRRs extracted from the same CT
dataset."""

from copy import copy

import numpy as np

from . import StatisticalModelBase


class SAM(StatisticalModelBase):
  def __init__(self, appearance):

    # initialise variables
    self.appearance_base = copy(appearance)
    appearance_aligned = (
      self.appearance_base - self.appearance_base.mean(axis=1)[:, np.newaxis]
    )
    self.appearance_scale = (
      appearance_aligned / appearance_aligned.std(axis=1)[:, np.newaxis]
    )

  def compute_dataset_mean(self):
    """"""
    return np.mean(self.appearance_scale, axis=0)
