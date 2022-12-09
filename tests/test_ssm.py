import unittest

import numpy as np
import pyssam


class TestSSM(unittest.TestCase):
  def test_morph_model(self):
    num_repititions = 10
    for _ in range(0, num_repititions):
      num_samples = 50
      tree_class = pyssam.datasets.Tree(num_extra_ends=1)
      landmark_coordinates = np.array(
        [tree_class.make_tree_landmarks() for i in range(0, num_samples)]
      )

      ssm_obj = pyssam.SSM(landmark_coordinates)
      ssm_obj.create_pca_model(ssm_obj.landmarks_columns_scale)
      mean_shape_columnvector = ssm_obj.compute_dataset_mean()
      mean_shape = mean_shape_columnvector.reshape(-1, 3)
      pca_model_components = ssm_obj.pca_model_components

      test_shape_columnvec = (
        landmark_coordinates[0] - landmark_coordinates[0].mean(axis=0)
      ).reshape(-1)
      test_shape_columnvec /= test_shape_columnvec.std()
      test_params = (
        np.dot(
          (test_shape_columnvec - mean_shape_columnvector), pca_model_components.T
        )
        / ssm_obj.std
      )

      # we get some issue with large shape parameters of modes that contribute
      # very small variance. We remove these temporarily to avoid interfering
      # with our test
      relevant_parameters = test_params[
        ssm_obj.pca_object.explained_variance_ratio_ > 0.01
      ]
      # shape parameters should generally be < 3 times the mode standard deviation
      # check all parameters below 5 (5 instead of 3 to be conservative)
      assert np.all(abs(relevant_parameters) < 5), (
        f"shape parameters too high (max {abs(relevant_parameters).max()}), "
        f"may be scaling issue {abs(relevant_parameters)}"
      )

      morphed_test_shape = ssm_obj.morph_model(
        mean_shape_columnvector, pca_model_components, test_params
      )

      assert np.allclose(
        test_shape_columnvec, morphed_test_shape
      ), "reconstructed shape not close to landmarks"


if __name__ == "__main__":
  unittest.main()
