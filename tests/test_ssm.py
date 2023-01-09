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
      test_params = ssm_obj.fit_model_parameters(test_shape_columnvec, ssm_obj.pca_model_components)

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
      test_params = ssm_obj.fit_model_parameters(test_shape_columnvec, ssm_obj.pca_model_components)

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

  def test_fit_model_parameters_all_modes(self):
    num_repititions = 10
    for _ in range(0, num_repititions):
      num_samples = 50
      tree_class = pyssam.datasets.Tree(num_extra_ends=1)
      landmark_coordinates = np.array(
        [tree_class.make_tree_landmarks() for i in range(0, num_samples)]
      )

      ssm_obj = pyssam.SSM(landmark_coordinates)
      ssm_obj.create_pca_model(ssm_obj.landmarks_columns_scale, desired_variance=0.7)
      target_shape = ssm_obj.landmarks_columns_scale[0]
      model_parameters = ssm_obj.fit_model_parameters(target_shape, ssm_obj.pca_model_components)
      model_parameters = np.where(model_parameters < 5, model_parameters, 3)
      model_parameters = np.where(model_parameters > -5, model_parameters, -3)
      
      dataset_mean = ssm_obj.compute_dataset_mean()
      morphed_shape = ssm_obj.morph_model(dataset_mean, ssm_obj.pca_model_components, model_parameters)
      error = abs(target_shape - morphed_shape)
      assert np.isclose(error.mean(), 0), f"error is non-zero ({error.mean()})"

  def test_fit_model_parameters_reduced_modes(self):
    num_repititions = 10
    for _ in range(0, num_repititions):
      num_samples = 50
      tree_class = pyssam.datasets.Tree(num_extra_ends=1)
      landmark_coordinates = np.array(
        [tree_class.make_tree_landmarks() for i in range(0, num_samples)]
      )

      ssm_obj = pyssam.SSM(landmark_coordinates)
      ssm_obj.create_pca_model(ssm_obj.landmarks_columns_scale, desired_variance=0.7)
      target_shape = ssm_obj.landmarks_columns_scale[0]
      model_parameters = ssm_obj.fit_model_parameters(target_shape, ssm_obj.pca_model_components, num_modes=2)
      dataset_mean = ssm_obj.compute_dataset_mean()
      morphed_shape = ssm_obj.morph_model(dataset_mean, ssm_obj.pca_model_components, model_parameters, num_modes=2)
      error = abs(target_shape - morphed_shape)
      assert not np.isclose(error.mean(), 0), f"error is zero ({error.mean()}), but should be non-zero"

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

if __name__ == "__main__":
  unittest.main()
