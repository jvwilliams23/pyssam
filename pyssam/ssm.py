"""Create statistical shape model (SSM) for a set of shapes."""
import numpy as np

from . import StatisticalModelBase


class SSM(StatisticalModelBase):
  """Create statistical shape model for a set of shapes.

  Parameters
  ----------
  landmarks : array_like
      Coordinates for landmarks in dataset, with 3 dimensions.
      First dimension has size equal to the number of samples.
      Second dimension has size equal to the number of landmarks per sample.
      Third dimension has size equal to the number of spatial
      dimensions occupied by the shapes (e.g. 3D or 2D).

  Examples
  ========
  >>> import numpy as np
  >>> import pyssam
  >>> num_samples = 5
  >>> num_landmarks = 10
  >>> landmarks = np.random.normal(size=(num_samples, num_landmarks, 3))
  >>> ssm = pyssam.SSM(landmarks)
  >>> print(ssm.landmarks_columns_scale.shape)
  (5, 30)
  >>> print(ssm.compute_dataset_mean().shape)
  (30,)
  """

  def __init__(self, landmarks: np.ndarray):

    # align all samples to origin
    self._num_landmarks = landmarks.shape[1]
    self.landmarks_scale = self.scale_dataset(landmarks)
    self.landmarks_columns_scale = self.landmark_data_to_column(
      self.landmarks_scale
    )

  def compute_dataset_mean(self) -> np.array:
    """Average over all samples to produce a column-vector of the mean shape.

    Returns
    -------
    mean_columnvector : array_like
        mean shape of all samples in dataset
    """
    return np.mean(self.landmarks_columns_scale, axis=0)
