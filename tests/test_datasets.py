import unittest

import numpy as np
import pyssam


class TestDatasets(unittest.TestCase):
  def test_number_of_ends(self):
    for num_extra_ends in range(0, 4):
      num_extra_ends = 1
      expected_ends = 2 ** (3 + num_extra_ends)
      tree_class = pyssam.datasets.Tree(num_extra_ends=num_extra_ends)
      num_ends = tree_class.make_tree_landmarks().shape[0]
      assert (
        num_ends == expected_ends
      ), f"expected landmarks {expected_ends} != {num_ends}"

  def test_torus_modes_normalised(self):
    N_SAMPLES = 100

    torus = pyssam.datasets.Torus()
    data = torus.make_dataset(N_SAMPLES)

    landmark_coordinates = np.array(
      [sample_i.points()[::10] for sample_i in data]
    )

    # when landmarks are scaled, should only be one mode (inner/outer radius ratio)
    EXPECTED_NONZERO_MODES = 1
    ssm_obj = pyssam.SSM(landmark_coordinates)
    ssm_obj.create_pca_model(ssm_obj.landmarks_columns_scale)
    explained_variance = ssm_obj.pca_object.explained_variance_ratio_
    num_nonzero_modes = (explained_variance > 0.1).sum()
    assert (
      num_nonzero_modes == EXPECTED_NONZERO_MODES
    ), f"num non-zero modes {num_nonzero_modes} but expected {EXPECTED_NONZERO_MODES} for variances {explained_variance}"

  def test_torus_modes_not_normalised(self):
    N_SAMPLES = 100

    torus = pyssam.datasets.Torus()
    data = torus.make_dataset(N_SAMPLES)

    landmark_coordinates = np.array(
      [sample_i.points()[::10] for sample_i in data]
    )

    # when landmarks are not scaled, should be two modes (inner radius, outer radius)
    EXPECTED_NONZERO_MODES = 2
    ssm_obj = pyssam.SSM(landmark_coordinates)
    ssm_obj.create_pca_model(landmark_coordinates.reshape(N_SAMPLES, -1))
    explained_variance = ssm_obj.pca_object.explained_variance_ratio_
    num_nonzero_modes = (explained_variance > 0.1).sum()
    assert (
      num_nonzero_modes == EXPECTED_NONZERO_MODES
    ), f"num non-zero modes {num_nonzero_modes} but expected {EXPECTED_NONZERO_MODES} for variances {explained_variance}"

  def test_torus_to_points(self):
    N_SAMPLES = 2

    torus = pyssam.datasets.Torus()
    data = torus.make_dataset(N_SAMPLES)

    # landmark_coordinates = np.array(
    #   [sample_i.points()[::10] for sample_i in data]
    # )
    assert type(data[0].points()) == np.ndarray, f"torus.points() type np.ndarray != {type(data[0].points())}"


if __name__ == "__main__":
  unittest.main()
