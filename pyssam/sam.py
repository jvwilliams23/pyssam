"""Create statistical appearance model (SAM) for a set of samples."""

from copy import copy

import numpy as np

from . import StatisticalModelBase


class SAM(StatisticalModelBase):
  """Create statistical appearance model for a set of samples.

  Parameters
  ----------
  appearance : array_like
      Appearances for landmarks in dataset, with 2 dimensions.
      First dimension has size equal to the number of samples.
      Second dimension has size equal to the number of landmarks per sample.

  Examples
  ========
  >>> num_samples = 5
  >>> num_landmarks = 10
  >>> appearances = np.random.normal(size=(num_samples, num_landmarks, 3))
  >>> sam = pyssam.SAM(appearances)
  >>> print(sam.appearance_scale.shape)
  (5, 10)
  >>> print(sam.compute_dataset_mean().shape)
  (10,)
  """

  def __init__(self, appearance: np.ndarray):

    # initialise variables
    self.appearance_base = copy(appearance)
    _appearance_aligned = (
      self.appearance_base - self.appearance_base.mean(axis=1)[:, np.newaxis]
    )
    self.appearance_scale = (
      _appearance_aligned / _appearance_aligned.std(axis=1)[:, np.newaxis]
    )

  def compute_dataset_mean(self) -> np.array:
    """Average over all samples to produce a column-vector of the mean appearance.

    Returns
    -------
    mean_columnvector : array_like
        mean appearance of all samples in dataset
    """
    return np.mean(self.appearance_scale, axis=0)
