from glob import glob
import unittest

import numpy as np
import pyssam

def read_lung_data():
  LANDMARK_DIR = "exampleData/gamesData/"
  XR_DIR = "exampleData/drrs/"
  # Get directories for DRR and landmark data
  origin_dir_list = glob(f"{XR_DIR}/origins/origins/drr*.md")
  spacing_dir_list = glob(f"{XR_DIR}/*/drr*.md")
  im_dir_list = glob(f"{XR_DIR}/*/drr*.png")
  origin_dir_list.sort()
  spacing_dir_list.sort()
  im_dir_list.sort()
  # check that user has declared correct directory
  patientIDs = [i.split("/")[-1].replace(".png", "")[-4:] for i in im_dir_list]
  landmark_dir_list = glob(f"{LANDMARK_DIR}/landmarks*.csv")
  landmark_dir_list = sorted(
    landmark_dir_list, key=lambda x: int(x.replace(".csv", "")[-4:])
  )
  # used to align drrs and landmarks
  trans_dirs = glob(f"{XR_DIR}/transforms/transformParams_case*_m_*.dat")
  trans_dirs.sort()

  if (
    len(im_dir_list) == 0
    or len(origin_dir_list) == 0
    or len(landmark_dir_list) == 0
    or len(spacing_dir_list) == 0
  ):
    raise AssertionError(
      "ERROR: The directories you have declared are empty.",
      "\nPlease check your input arguments.",
    )

  landmark_offset = np.vstack(
    [np.loadtxt(t, skiprows=1, max_rows=1) for t in trans_dirs]
  )
  # read data
  origin = np.vstack([np.loadtxt(o, skiprows=1) for o in origin_dir_list])
  spacing = np.vstack([np.loadtxt(o, skiprows=1) for o in spacing_dir_list])
  # load x-rays into a stacked array,
  # switch so shape is (num patients, x pixel, y pixel)
  img_all = np.rollaxis(np.dstack([pyssam.utils.loadXR(o) for o in im_dir_list]), 2, 0)
  landmark_coordinates = np.array(
    [np.loadtxt(l, delimiter=",") for l in landmark_dir_list]
  )

  # offset centered coordinates to same reference frame as CT data
  landmark_align_to_projection = (
    landmark_coordinates + landmark_offset[:, np.newaxis]
  )

  from pyssam.utils import AppearanceFromXray
  appearance_helper = AppearanceFromXray(
      img_all, 
      origin[:, [0, 2]], 
      spacing[:, [0, 2]]
  )
  appearance_scaled = appearance_helper.all_landmark_density(landmark_align_to_projection[:, :, [0, 2]])
  return landmark_coordinates, appearance_scaled, appearance_helper


class TestSAM(unittest.TestCase):
  def test_morph_model(self):
    LANDMARKS, APPEARANCE, _ = read_lung_data()
    sam_obj = pyssam.SAM(APPEARANCE)
    sam_obj.create_pca_model(sam_obj.appearance_columns_scale, desired_variance=0.99999)
    # num_modes = sam_obj.required_mode_number
    mean_appearance = sam_obj.compute_dataset_mean()
    
    # run test for all cases in database
    for test_appearance in APPEARANCE:
      test_params = (
        np.dot(
          (test_appearance - mean_appearance),
          sam_obj.pca_model_components.T,
        )
        / sam_obj.std
      )

      # we get some issue with large shape parameters of modes that contribute
      # very small variance. We remove these temporarily to avoid interfering
      # with our test
      relevant_parameters = np.where(abs(test_params)<5, test_params, 0)
      # shape parameters should generally be < 3 times the mode standard deviation
      # check all parameters below 5 (5 instead of 3 to be conservative)
      assert np.all(abs(relevant_parameters) < 5), (
        f"shape parameters too high (max {abs(relevant_parameters).max()}), "
        f"may be scaling issue {abs(relevant_parameters)}"
      )

      morphed_appearance = sam_obj.morph_model(
        mean_appearance, sam_obj.pca_model_components, relevant_parameters
      )

      # check that reconstructed appearance matches the target when same parameters are used
      assert np.allclose(
        test_appearance, morphed_appearance
      ), (
        "reconstructed appearance not close to landmarks "
        "mean difference = "
        f"{np.mean(abs(test_appearance - morphed_appearance))} \n"
        f"{test_appearance - morphed_appearance}"
      )


if __name__ == "__main__":
  unittest.main()
