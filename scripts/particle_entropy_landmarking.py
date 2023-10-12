import argparse
from random import choices
from sys import exit

import networkx as nx
import numpy as np
from sklearn.utils import shuffle
import vedo as v
from pyssam.utils import euclidean_distance

def get_inputs():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--meshfile",
    "-m",
    default="scripts/test_mesh/meanLUL.stl",
    type=str,
    help="directory containing meshes to grow",
  )
  return parser.parse_args()

def read_mesh(meshfile, coarsen=500):
  surface_points = v.load(meshfile).points()[::coarsen]
  surface_points -= surface_points.mean(axis=0)
  return surface_points

class ParticleEntropyBasedLandmarking:
  """
  Implementation of
  Cates, J. et al., 2007. Shape modeling and analysis with entropy-based particle systems.

  This is a group-wise landmarking approach, which is supposed to optimise 
  correspondences across the entire dataset, with respect to information entropy.
  """
  def __init__(self):
    pass
  

if __name__ == "__main__":
  args = get_inputs()

  COARSEN_FACTOR = 100
  surface_points_orig = read_mesh(args.meshfile, coarsen=COARSEN_FACTOR)

  # Should allow accuracy criteria to be consistent in different cases
  particle_entropy_landmarking = ParticleEntropyBasedLandmarking()
