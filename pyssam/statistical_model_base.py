from abc import ABC, abstractmethod, abstractproperty
from copy import copy
from typing import Any, Tuple

import numpy as np
import sklearn
from sklearn.decomposition import PCA


class StatisticalModelBase(ABC):
  """Abstract base class for statistical model."""

  def landmark_data_to_column(self, landmark_array):
    num_data_features = landmark_array.shape[-1]
    reduced_array = landmark_array.reshape(
      -1, self.num_landmarks * num_data_features
    )
    return np.squeeze(reduced_array)

  @property
  def num_landmarks(self):
    return self._num_landmarks

  @num_landmarks.setter
  def num_landmarks(self, _num_landmarks):
    raise AttributeError("Not allowed to overwrite num_landmarks")

  def scale_dataset(self, dataset):
    """Take 2D or 3D array representing landmarks for multiple samples in a
    population, and scale to have zero mean and unit std dev.

    Parameters
    ----------
    dataset : array_like
        Landmarks for multiple samples in a population, with 2 or 3 dimensions

    Returns
    -------
    scaled_dataset : array_like
        Landmarks for multiple samples, each centered on mean = 0 and std dev = 1

    Raises
    ------
    AssertionError
        If number of dimensions on input data are not equal to 2 or 3

    Examples
    ========
    >>> landmarks_in = np.random.uniform(0, 100, size=(5, 100, 3))
    >>> ssm = pyssam.SSM(landmarks_in)
    >>> scaled_landmarks = ssm.scale_dataset(landmarks_in)
    >>> print(scaled_landmarks.mean(), scaled_landmarks.std())
    0 1
    """
    dataset = self._check_data_shape(dataset)
    aligned_dataset = dataset - dataset.mean(axis=1)[:, np.newaxis]
    scaled_dataset = (
      aligned_dataset / aligned_dataset.std(axis=(1, 2))[:, None, None]
    )
    return scaled_dataset

  def _check_data_shape(self, dataset: np.ndarray) -> np.ndarray:
    """Check that data to be scaled for PCA model has ndim = 3. This is useful
    for appearance data, which is often a single-scalar, therefore ndim is usually 2.
    In this case, an additional dimension of size 1 is added.

    Parameters
    ----------
    dataset : array_like
        N-dimension array of data to model, where each row on the first axis is one sample
        and each column on the second axis is a landmark value

    Returns
    -------
    dataset_reshaped : array_like
        3D array of data, where 2D input arrays have a third dimension of size 1.
        Input 3D arrays are untouched

    Raises
    ------
    AssertionError
        If ndim not equal to 2 or 2
    """
    if dataset.ndim == 2:
      return np.expand_dims(dataset, axis=-1)
    elif dataset.ndim == 3:
      return copy(dataset)
    else:
      raise AssertionError(f"Unexpected shape {dataset.shape}")

  @abstractmethod
  def compute_dataset_mean(self) -> np.array:
    """Average over all samples to produce a column-vector of the mean shape,
    appearance, or other quantity included in model.

    Returns
    -------
    mean_columnvector : array_like
    """
    pass

  def create_pca_model(
    self, dataset: np.ndarray, desired_variance: float = 0.9
  ) -> None:
    """Perform principal component analysis to create statistical model and
    extract modelling quantites from the PCA class.

    Parameters
    ----------
    dataset : array_like
        2D array of data to model, where each row on the first axis is one sample
        and each column on the second axis is e.g. shape or appearance for a landmark
    desired_variance : float
        Fraction of total variance to be described by the reduced-dimension model

    Returns
    -------
    None
    """
    assert (
      0.0 < desired_variance <= 1.0
    ), f"desired_variance out of bounds {desired_variance}"
    # perform principal component analysis to train shape model
    self.pca_object, self.required_mode_number = self.do_pca(
      dataset, desired_variance
    )
    # get principal components (eigenvectors) and variances (eigenvalues)
    self.pca_model_components = self.pca_object.components_
    self.variance = self.pca_object.explained_variance_
    # get standard deviation of each component,
    # and initialise model parameters as zero
    self.std = np.sqrt(self.variance)
    self.model_parameters = np.zeros(len(self.std))
    # save desired_variance as global variable to help post-processing and debugging
    self.desired_variance = desired_variance

  def morph_model(
    self,
    mean_dataset_columnvector: np.array,
    pca_model_components: np.array,
    model_parameters: np.array,
    num_modes: int = 1000000,
  ) -> np.array:
    """Morph the mean dataset based on the PCA weights and variances, with some
    user-defined model parameters to create a new sample.

    Parameters
    ----------
    mean_dataset_columnvector : array_like
        mean shape of the training data in a 1D array
    pca_model_components : array_like
        eigenvectors of covariance matrix, obtain by PCA
    model_parameters : array_like
        model parameters used to perturb each principal component by some amount
        1D array, where values should all be within +/- 3
    num_modes : int
        Number of principal components (or `modes') to include in model
        to morph. By default this is set to a high number to set all modes
        as included.

    Returns
    -------
    morphed output : array_like
        A 1D array which has been perturbed from the mean shape based
        on the pca_model and model_parameters

    Raises
    ------
    Warning
        If model parameters are outwith +/- 3
    AssertionError
        If number of dimension in pca_model_components not equal to 2
    """
    if np.any(abs(model_parameters) > 3.0):
      Warning(
        f"Applying large model parameter ({abs(model_parameters).max()}) "
        "which may produce unrealistic output"
      )
    assert pca_model_components.ndim == 2, (
      f"pca model not of expected number of dimensions"
      f" (shape is {pca_model_components.shape})"
    )
    model_weight = model_parameters * self.std[:num_modes]
    return mean_dataset_columnvector + np.dot(
      pca_model_components[:num_modes].T, model_weight
    )

  def do_pca(
    self, dataset: np.ndarray, desired_variance: float = 0.9
  ) -> Tuple[Any, int]:
    """Fit principal component analysis to given dataset.

    Parameters
    ----------
    dataset : array_like
        2D array of data to model, where each row on the first axis is one sample
        and each column on the second axis is e.g. shape or appearance for a landmark
    desired_variance : float
        Fraction of total variance to be described by the reduced-dimension model

    Returns:
    pca : sklearn.decomposition._pca.PCA
        Object containing fitted PCA information e.g. components, explained variance
    required_mode_number : int
        Number of principal components needed to produce desired_variance
    """
    pca = PCA(svd_solver="full")
    pca.fit(dataset)
    required_mode_number = np.where(
      np.cumsum(pca.explained_variance_ratio_) > desired_variance
    )[0][0]
    print(
      f"Reduced to {required_mode_number} components "
      f"from {len(pca.explained_variance_ratio_)} "
      f"for {desired_variance*100}% variation"
    )

    return pca, required_mode_number
