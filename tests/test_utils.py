import unittest
import warnings

import numpy as np
import pyssam

NUM_SAMPLES = 5
NUM_PIXELS = 100
NUM_LANDMARKS = 50
IMAGE_DATASET = np.random.rand(NUM_SAMPLES, NUM_PIXELS, NUM_PIXELS)
IMAGE_SPACING = np.ones((NUM_SAMPLES, 2))
IMAGE_ORIGIN = np.zeros((NUM_SAMPLES, 2))

class TestAppearanceFromXray(unittest.TestCase):
  def test_compute_landmark_density(self):
    appearance_helper = pyssam.utils.AppearanceFromXray(
      IMAGE_DATASET, IMAGE_ORIGIN, IMAGE_SPACING
    )
    _test_landmarks = np.random.rand(NUM_LANDMARKS, 3)
    try: 
      appearance_helper.compute_landmark_density(
        _test_landmarks, IMAGE_DATASET[0], appearance_helper.pixel_coordinates[0]
      )
    except AssertionError:
      pass
    else:
      raise AssertionError(
        "Assertion to check shape of landmarks compared to pixel_coordinates not recognised"
      )

  def test_img_ndim(self):
    """Test to check that class should not work with img.ndim != 2"""
    appearance_helper = pyssam.utils.AppearanceFromXray(
      IMAGE_DATASET, IMAGE_ORIGIN, IMAGE_SPACING
    )
    _test_landmarks = np.random.rand(NUM_LANDMARKS, 2)
    try: 
      appearance_helper.compute_landmark_density(
        _test_landmarks, IMAGE_DATASET, appearance_helper.pixel_coordinates[0]
      )
    except AssertionError:
      pass
    else:
      raise AssertionError(
        "Assertion to check dimensions of image not recognised"
      )

  def test_work_with_one_input(self):
    """Test to check no crashes when initialising AppearanceFromXray on 1 sample (not whole dataset)"""
    appearance_helper = pyssam.utils.AppearanceFromXray(
      IMAGE_DATASET[0], IMAGE_ORIGIN[0], IMAGE_SPACING[0]
    )
    test_landmarks = np.random.rand(NUM_LANDMARKS, 2)
    appearance_helper.compute_landmark_density(
      test_landmarks, IMAGE_DATASET[0], appearance_helper.pixel_coordinates[0]
    )

  def test_img_shape(self):
    """Test to check that class should not work with non-square image"""
    _image_dataset = np.random.rand(NUM_SAMPLES, NUM_PIXELS//2, NUM_PIXELS)
    try: 
      appearance_helper = pyssam.utils.AppearanceFromXray(
        _image_dataset, IMAGE_ORIGIN, IMAGE_SPACING
      )
    except AssertionError:
      pass
    else:
      raise AssertionError(
        "Assertion to check shape of image not recognised"
      )

  def test_all_landmark_density(self):
    """Test to check output of all_landmark_density is fine with expected behaviour"""
    appearance_helper = pyssam.utils.AppearanceFromXray(
      IMAGE_DATASET, IMAGE_ORIGIN, IMAGE_SPACING
    )
    _test_landmarks = np.random.rand(NUM_SAMPLES, NUM_LANDMARKS, 2)
    appearance_helper.all_landmark_density(
      _test_landmarks
    )

  def test_all_landmark_density_landmark_assertions(self):
    """Test to check output of all_landmark_density is fine with expected behaviour"""
    appearance_helper = pyssam.utils.AppearanceFromXray(
      IMAGE_DATASET, IMAGE_ORIGIN, IMAGE_SPACING
    )
    _test_landmarks = np.random.rand(NUM_SAMPLES, NUM_LANDMARKS, 3)
    try:
      appearance_helper.all_landmark_density(
        _test_landmarks
      )
    except AssertionError:
      pass
    else:
      raise AssertionError(
        "Assertion failed to recognise shape of landmarks not compatible"
      )

  def test_origin_shape_assertions(self):
    """Test to check output of all_landmark_density is fine with expected behaviour"""
    _test_origin = np.zeros((NUM_SAMPLES, 3))
    try:
      appearance_helper = pyssam.utils.AppearanceFromXray(
        IMAGE_DATASET, _test_origin, IMAGE_SPACING
      )
    except AssertionError:
      pass
    else:
      raise AssertionError(
        "Assertion failed to recognise shape of origin and spacing not equal"
      )

  def test_spacing_shape_assertions(self):
    """Test to check output of all_landmark_density is fine with expected behaviour"""
    _test_spacing = np.zeros((NUM_SAMPLES, 3))
    try:
      appearance_helper = pyssam.utils.AppearanceFromXray(
        IMAGE_DATASET, IMAGE_ORIGIN, _test_spacing
      )
    except AssertionError:
      pass
    else:
      raise AssertionError(
        "Assertion failed to recognise shape of origin and spacing not equal"
      )
    
if __name__ == "__main__":
  unittest.main()
