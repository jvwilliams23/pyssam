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
  >>> import numpy as np
  >>> import pyssam
  >>> num_samples = 5
  >>> num_landmarks = 10
  >>> appearances = np.random.normal(size=(num_samples, num_landmarks))
  >>> sam = pyssam.SAM(appearances)
  >>> print(sam.appearance_scale.shape)
  (5, 10, 1)
  >>> print(sam.compute_dataset_mean().shape)
  (10,)
  """

  def __init__(self, appearance: np.ndarray):

    # initialise variables
    self._num_landmarks = appearance.shape[1]
    self.appearance_base = copy(appearance)
    self.appearance_scale = self.scale_dataset(appearance)
    self.appearance_columns_scale = self.landmark_data_to_column(
      self.appearance_scale
    )

  def compute_dataset_mean(self) -> np.array:
    """Average over all samples to produce a column-vector of the mean
    appearance.

    Returns
    -------
    mean_columnvector : array_like
        mean appearance of all samples in dataset
    """
    return np.mean(self.appearance_columns_scale, axis=0)
