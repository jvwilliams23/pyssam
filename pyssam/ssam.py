"""Create statistical shape and appearance model (SSAM) for a set of
samples."""
import numpy as np

from . import SAM, SSM, StatisticalModelBase


class SSAM(StatisticalModelBase):
  """Create statistical shape and appearance model for a set of samples.

  Parameters
  ----------
  landmarks : array_like
      Coordinates for landmarks in dataset, with 3 dimensions.
      First dimension has size equal to the number of samples.
      Second dimension has size equal to the number of landmarks per sample.
      Third dimension has size equal to the number of spatial
      dimensions occupied by the shapes (e.g. 3D or 2D).
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
  >>> landmarks = np.random.normal(size=(num_samples, num_landmarks, 3))
  >>> appearances = np.random.normal(size=(num_samples, num_landmarks))
  >>> ssam = pyssam.SSAM(landmarks, appearances)
  >>> print(ssam.shape_appearance_columns.shape)
  (5, 40)
  >>> print(ssam.compute_dataset_mean().shape)
  (40,)
  """

  def __init__(self, landmarks: np.ndarray, appearance: np.ndarray):

    # shape modelling classes
    self._num_landmarks = landmarks.shape[1]
    self.ssm = SSM(landmarks)
    self.sam = SAM(appearance)

    # import scaled variables from SSM and SAM classes
    self.landmarks_columns_scale = self.ssm.landmarks_columns_scale
    self.appearance_columns_scale = self.sam.appearance_scale

    # stack shape and appearance for each landmark
    self.shape_appearance = np.dstack(
      (
        self.ssm.landmarks_scale,
        self.sam.appearance_scale,
      )
    )
    # convert stacked shape + appearance into a single column for all landmarks
    self.shape_appearance_columns = self.landmark_data_to_column(
      self.shape_appearance
    )

  def compute_dataset_mean(self) -> np.ndarray:
    """Average over all samples to get column-vector of mean shape and
    appearance.

    Returns
    -------
    mean_columnvector : array_like
        mean shape and appearance of all samples in dataset
    """
    return np.mean(self.shape_appearance_columns, axis=0)
