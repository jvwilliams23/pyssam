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


if __name__ == "__main__":
  unittest.main()
