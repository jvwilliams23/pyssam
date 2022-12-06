"""Develops SSM of lung lobes based on landmarks provided by the user."""
from abc import ABC, abstractmethod, abstractproperty

import numpy as np
from sklearn.decomposition import PCA


class StatisticalModelBase(ABC):
  @abstractmethod
  def compute_dataset_mean(self):
    pass

  def create_pca_model(self, dataset_columnvector, desired_variance=0.9):
    # perform principal component analysis to train shape model
    self.pca_object, self.k = self.do_pca(
      dataset_columnvector, desired_variance
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
    self, mean_dataset_columnvector, pca_model_components, model_parameters
  ):
    """Return point cloud that has been adjusted by a specified shape vector
    (b)

    Args:
      mean_dataset_columnvector (3n,) array: mean shape
      pca_model_components (sampleNum, 3n) array: PCA components
      b (sampleNum, ) array: shape vector to vary points by
    """
    model_weight = model_parameters * self.std
    return mean_dataset_columnvector + np.dot(
      pca_model_components.T, model_weight
    )

  def do_pca(self, dataset_columnvector, desired_variance=0.95):
    """
    Args:
      dataset_columnvector = 2D array of components

    Returns:
      pca, variation of each mode
      desired_variance, amount of modes to reach desired variance ratio
    """
    pca = PCA(svd_solver="full")
    pca.fit(dataset_columnvector)
    required_num_modes = np.where(
      np.cumsum(pca.explained_variance_ratio_) > desired_variance
    )[0][0]
    print(
      f"Reduced to {required_num_modes} components "
      f"from {len(pca.explained_variance_ratio_)} "
      f"for {desired_variance*100}% variation"
    )

    return pca, required_num_modes
