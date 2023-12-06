import unittest
from warnings import warn

import numpy as np
import pyssam


class TestMorphMesh(unittest.TestCase):
  def test_morph_mesh_torus(self):
    """Create a torus dataset with two samples.
    Use the first sample as a template, and the second as a target.

    Compare the morphed mesh (volume and area) with the ground truth.
    """
    # dataset etc is stochastic, so may not pass on first attempt.
    NUM_ATTEMPTS = 10 
    cond_met = False

    while not cond_met:
      torus = pyssam.datasets.Torus()
      torus_data = torus.make_dataset(2)

      landmark_coordinates = np.array(
        [sample_i.points()[::10] for sample_i in torus_data]
      )

      landmark_target = landmark_coordinates[-1]
      mesh_target_actual = torus_data[-1]
      landmark_coordinates = landmark_coordinates[:-1]

      input_coordinates = landmark_coordinates[0]
      input_mesh = torus_data[0]

      mesh_target_computed = pyssam.morph_mesh.MorphTemplateMesh(
        landmark_target, input_coordinates, input_mesh
      ).mesh_target

      volume_error = (
        100.0
        * abs(mesh_target_computed.volume() - mesh_target_actual.volume())
        / mesh_target_actual.volume()
      )
      area_error = (
        100.0
        * abs(mesh_target_computed.area() - mesh_target_actual.area())
        / mesh_target_actual.area()
      )

      print(f"volume error {volume_error}")
      print(f"area error {area_error}")
      if volume_error > 10.0 or area_error > 10.0:
        cond_met = False
        error_message = (
          "Morphed mesh not consistent for torus, "
          f"volume error is {volume_error}, area error is {area_error}"
        )
        cond_met = False
        warn(error_message)
      else:
        cond_met = True
    if cond_met == False:
        raise AssertionError(error_message)


if __name__ == "__main__":
  unittest.main()
