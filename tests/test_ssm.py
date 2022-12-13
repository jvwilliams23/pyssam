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

      test_shape_columnvec = (
        landmark_coordinates[0] - landmark_coordinates[0].mean(axis=0)
      ).reshape(-1)
      test_shape_columnvec /= test_shape_columnvec.std()
      test_params = (
        np.dot(
          (test_shape_columnvec - mean_shape_columnvector),
          ssm_obj.pca_model_components.T,
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
        mean_shape_columnvector, ssm_obj.pca_model_components, test_params
      )

      assert np.allclose(
        test_shape_columnvec, morphed_test_shape
      ), "reconstructed shape not close to landmarks"

  def test_morph_model_reduced_dimension(self):
    num_repititions = 10
    for _ in range(0, num_repititions):
      num_samples = 50
      tree_class = pyssam.datasets.Tree(num_extra_ends=1)
      landmark_coordinates = np.array(
        [tree_class.make_tree_landmarks() for i in range(0, num_samples)]
      )

      ssm_obj = pyssam.SSM(landmark_coordinates)
      ssm_obj.create_pca_model(ssm_obj.landmarks_columns_scale, desired_variance=0.7)
      num_modes = ssm_obj.required_mode_number
      mean_shape_columnvector = ssm_obj.compute_dataset_mean()
      mean_shape = mean_shape_columnvector.reshape(-1, 3)

      test_shape_columnvec = (
        landmark_coordinates[0] - landmark_coordinates[0].mean(axis=0)
      ).reshape(-1)
      test_shape_columnvec /= test_shape_columnvec.std()
      test_params = (
        np.dot(
          (test_shape_columnvec - mean_shape_columnvector),
          ssm_obj.pca_model_components.T,
        )
        / ssm_obj.std
      )

      # we get some issue with large shape parameters of modes that contribute
      # very small variance. We remove these temporarily to avoid interfering
      # with our test
      relevant_parameters = test_params[
        ssm_obj.pca_object.explained_variance_ratio_ > 0.01
      ]

      morphed_test_shape = ssm_obj.morph_model(
        mean_shape_columnvector, ssm_obj.pca_model_components, test_params[:num_modes], num_modes=num_modes
      )

      assert not np.allclose(
        test_shape_columnvec, morphed_test_shape
      ), (
        f"reconstructed shape is close to landmarks "
        f"(should not be for {ssm_obj.desired_variance} variance)"
      )

  def test_desired_variance_bounds(self):
    tree_class = pyssam.datasets.Tree(num_extra_ends=0)
    num_samples = 10
    landmark_coordinates = np.array(
      [tree_class.make_tree_landmarks() for i in range(0, num_samples)]
    )
    ssm_obj = pyssam.SSM(landmark_coordinates)
    desired_variance_list = [-0.0001, -50, 1.1, 1.00001, 50]
    for desired_variance in desired_variance_list:
      try:
        ssm_obj.create_pca_model(
          ssm_obj.landmarks_columns_scale, desired_variance=desired_variance
        )
      except AssertionError:
        pass
      else:
        AssertionError("AssertionError for desired variance not identified")

  def test_property_decorator(self):
    dataset_ssm = np.ones((10,10,3))
    ssm = pyssam.SSM(dataset_ssm)
    try:
      ssm.num_landmarks = 20
    except AttributeError:
      print("passed test")
    else:
      raise AssertionError(
        "Should not be able to overwrite ssm.num_landmarks"
      )

  def test__check_data_shape(self):
    dataset_ssm = np.ones((10,10,3))
    base_dataset_ssm = np.ones((10,10,3))
    assert pyssam.SSM(base_dataset_ssm)._check_data_shape(dataset_ssm).ndim == 3, "ndim not equal to 3"
    dataset_ssm = np.ones((10,10))
    assert pyssam.SSM(base_dataset_ssm)._check_data_shape(dataset_ssm).ndim == 3, "ndim not equal to 3"
    
    dataset_ssm = np.ones((10))
    try:
      pyssam.SSM(base_dataset_ssm)._check_data_shape(dataset_ssm).ndim
    except AssertionError:
      pass
    else:
      AssertionError(f"SSM._check_data_shape not recognising input with ndim = {dataset_ssm.ndim}")

    dataset_ssm = np.ones((10)*4)
    try:
      pyssam.SSM(base_dataset_ssm)._check_data_shape(dataset_ssm).ndim
    except AssertionError:
      pass
    else:
      AssertionError(f"SSM._check_data_shape not recognising input with ndim = {dataset_ssm.ndim}")

class TestSSAM(unittest.TestCase):
  def test_morph_model(self):
    num_repititions = 10
    for _ in range(0, num_repititions):
      num_samples = 100
      tree_class = pyssam.datasets.Tree(num_extra_ends=1)
      landmark_coordinates = np.array(
        [tree_class.make_tree_landmarks() for i in range(0, num_samples)]
      )
      appearance = np.random.normal(size=(num_samples, landmark_coordinates.shape[1]))
      appearance /= appearance.std(axis=0)

      ssam_obj = pyssam.SSAM(landmark_coordinates, appearance)
      ssam_obj.create_pca_model(ssam_obj.shape_appearance_columns)
      mean_shape_columnvector = ssam_obj.compute_dataset_mean()
      mean_shape = mean_shape_columnvector.reshape(-1, 4)
      
      test_shape_columnvec = (
        landmark_coordinates[0] - landmark_coordinates[0].mean(axis=0)
      )
      test_shape_columnvec /= test_shape_columnvec.std()
      test_shape_columnvec = np.hstack((test_shape_columnvec, appearance[0][:, None])).reshape(-1)
      test_params = (
        np.dot(
          (test_shape_columnvec - mean_shape_columnvector),
          ssam_obj.pca_model_components.T,
        )
        / ssam_obj.std
      )

      # we get some issue with large shape parameters of modes that contribute
      # very small variance. We remove these temporarily to avoid interfering
      # with our test
      relevant_parameters = test_params[
        ssam_obj.pca_object.explained_variance_ratio_ > 0.01
      ]
      # shape parameters should generally be < 3 times the mode standard deviation
      # check all parameters below 5 (5 instead of 3 to be conservative)
      assert np.all(abs(relevant_parameters) < 5), (
        f"shape parameters too high (max {abs(relevant_parameters).max()}), "
        f"may be scaling issue {abs(relevant_parameters)}"
      )

      morphed_test_shape = ssam_obj.morph_model(
        mean_shape_columnvector, ssam_obj.pca_model_components, test_params
      )

      assert np.allclose(
        test_shape_columnvec, morphed_test_shape
      ), (
        "reconstructed shape not close to landmarks "
        f"mean difference = {np.mean(test_shape_columnvec - morphed_test_shape)}"
      )

if __name__ == "__main__":
  unittest.main()
