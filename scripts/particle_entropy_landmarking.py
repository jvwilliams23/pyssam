import argparse
from random import choices
from sys import exit

import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist
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
  def __init__(self, samples, number_of_particles):
    self.samples = samples
    self.number_of_particles = number_of_particles
    self.landmarks = self._initialise_landmarks(samples, number_of_particles=self.number_of_particles)

  def _initialise_landmarks(self, samples, number_of_particles):  
    # define initial random sample of points
    landmarks = []
    for sample_i in samples:
      random_surf_points = sample_i[np.random.randint(0, len(sample_i), number_of_particles)]
      landmarks.append(random_surf_points)
    return np.array(landmarks)

  def _estimate_density(self, sample_i):
    # get PDF by 'parzen window sampling', using gaussian kernel
    prefactor = 1.0/(self.number_of_particles*(self.number_of_particles-1))
    all_dists = cdist(sample_i, sample_i)
    density_list = np.zeros(len(sample_i))
    for j, dist_j in enumerate(all_dists):
      kernel_j = self._gaussian_kernel(dist_j, dist_j.std())
      density_list[j] = prefactor * kernel_j.sum()
    return density_list

  def _gaussian_kernel(self, dist, std_dev):
    return np.exp(-dist/std_dev**2.0)
  
  def run_particle_optimisation(self):
    for sample_i in self.samples:
      density_list = self._estimate_density(sample_i)
      print(density_list)
    # function for cost-function

def random_spherical_coord(radius):
  phi = np.deg2rad(np.random.uniform(0,360))
  theta = np.deg2rad(np.random.uniform(0,360))
  x = radius * np.cos(phi) * np.sin(theta)
  y = radius * np.sin(phi) * np.sin(theta)
  z = radius * np.cos(theta)
  return np.array([x, y, z])

def make_peanut_dataset(n_samples=10):
  base_pos = np.array([0,0,0])
  radius = 1.0
  out_data = []
  min_length = np.infty
  for _ in range(n_samples):
    sphere_0 = v.Sphere(pos=base_pos, r=radius)
    perturb_dist = np.random.uniform(radius*0.3, radius*0.5)
    sphere_1 = v.Sphere(pos=base_pos+random_spherical_coord(perturb_dist), r=np.random.uniform(radius*0.7, radius))
    sphere_out = sphere_0.boolean("plus", sphere_1)
    # vp = v.Plotter()
    # vp += sphere_0
    # vp += sphere_1
    # vp.show()
    out_data.append(sphere_out.points())
    if sphere_out.points().shape[0] < min_length:
      min_length = sphere_out.points().shape[0]
  for i in range(n_samples):
    out_data[i] = out_data[i][:min_length]
  return np.array(out_data)

if __name__ == "__main__":
  args = get_inputs()

  COARSEN_FACTOR = 100
  surface_points_orig = read_mesh(args.meshfile, coarsen=COARSEN_FACTOR)

  dataset = make_peanut_dataset()

  # Should allow accuracy criteria to be consistent in different cases
  particle_entropy_landmarking = ParticleEntropyBasedLandmarking(dataset, number_of_particles=20)
  particle_entropy_landmarking.run_particle_optimisation()
