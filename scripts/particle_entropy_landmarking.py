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
  def __init__(self, samples_points, samples_surfaces, number_of_particles):
    self.samples_points = samples_points
    self.samples_surfaces = samples_surfaces
    self.number_of_particles = number_of_particles
    self.landmarks = self._initialise_landmarks(samples_points, number_of_particles=self.number_of_particles)

    self.samples_surfaces_points = self._get_surface_points(samples_surfaces)
    self.samples_surfaces_normals = self._get_surface_normals(samples_surfaces)

  def _get_surface_normals(self, samples_surfaces):
    # get the normals at each **point** on the surface mesh
    normals = [surf_i.normals() for surf_i in samples_surfaces]
    return normals
  
  def _get_surface_points(self, samples_surfaces):
    # get the point on the surface mesh
    points = [surf_i.points() for surf_i in samples_surfaces]
    return points

  def _initialise_landmarks(self, samples_points, number_of_particles):  
    # define initial random sample of points
    landmarks = []
    for sample_points_i in samples_points:
      random_surf_points = sample_points_i[np.random.randint(0, len(sample_points_i), number_of_particles)]
      landmarks.append(random_surf_points)
    return np.array(landmarks)
  
  def _estimate_density(self, point_i, all_points):
    dist_j = euclidean_distance(all_points, point_i)
    kernel_width = dist_j.std()
    kernel_j = self._gaussian_kernel(dist_j, kernel_width)
    density = kernel_j.sum() / (self.number_of_particles*(self.number_of_particles-1))
    gradient = ((point_i - all_points) * kernel_j[:, None]).sum(axis=0) / kernel_j.sum() / kernel_width**2.0
    return density, gradient

  def _loop_particles_one_sample(self, sample_points_i, case_id):
    # get PDF by 'parzen window sampling', using gaussian kernel
    # all_dists = cdist(sample_points_i, sample_points_i) # maybe better to get euclidean distance in the loop for each particle
    density_list = np.zeros(len(sample_points_i))
    gradient_list = np.zeros(sample_points_i.shape)
    nearest_surface_gradient_list = np.zeros(sample_points_i.shape)
    for j, point_j in enumerate(sample_points_i):
      density_list[j], gradient_list[j] = self._estimate_density(point_j, sample_points_i)
      nearest_surface_gradient_list[j] = self._nearest_surface_normal(sample_points_i[j], self.samples_surfaces_points[case_id], self.samples_surfaces_normals[case_id])
    return density_list, gradient_list, nearest_surface_gradient_list

  def _nearest_surface_normal(self, point_i, surface_points, surface_normals):
    return surface_normals[euclidean_distance(surface_points, point_i).argmin()]

  def _gaussian_kernel(self, dist, std_dev):
    return np.exp(-dist/std_dev**2.0)

  def _advance_sampling(self, sample_points_i, gradients_i, normals_i, time_step):
    # move points based on cost function gradient, projected onto nearest normal vector of surface
    projection_onto_normal = np.sum(gradients_i*normals_i, axis=-1, keepdims=True) * normals_i
    direction_to_travel = gradients_i - projection_onto_normal
    return sample_points_i + time_step * direction_to_travel
  
  def run_initial_sampling_optimisation(self, orig_sample_points_i, orig_sample_mesh_i, case_id):
    sample_points_i = orig_sample_points_i.copy()
    NUM_STEPS = 100
    TIME_STEP = 0.01
    for time_i in range(100):
      _density_list, _gradient_list, _normal_list = self._loop_particles_one_sample(sample_points_i, case_id)
      sample_points_i = self._advance_sampling(sample_points_i, _gradient_list, _normal_list, time_step=TIME_STEP)
    vp = v.Plotter()
    vp += v.Points(orig_sample_points_i, r=5, c="black")
    vp += v.Points(sample_points_i, r=5, c="blue")
    vp.show()
    return sample_points_i

  def run_particle_optimisation(self):
    case_id = 0
    self.run_initial_sampling_optimisation(self.samples_points[case_id], self.samples_surfaces[case_id], case_id)
    # for sample_points_i in self.samples:
    #   density_list, gradient_list = self._estimate_density(sample_points_i)
    #   print(density_list, density_list.min(), density_list.max(), density_list.sum())
    #   # differential_entropy = -1.0/len(density_list) * np.log(density_list).sum()
    #   # cost_gradient = 
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
  out_surfaces = []
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
    out_surfaces.append(sphere_out)
    if sphere_out.points().shape[0] < min_length:
      min_length = sphere_out.points().shape[0]
  for i in range(n_samples):
    out_data[i] = out_data[i][:min_length]
  return np.array(out_data), out_surfaces

if __name__ == "__main__":
  args = get_inputs()

  COARSEN_FACTOR = 100
  surface_points_orig = read_mesh(args.meshfile, coarsen=COARSEN_FACTOR)

  dataset_points, dataset_surfaces = make_peanut_dataset()

  # Should allow accuracy criteria to be consistent in different cases
  particle_entropy_landmarking = ParticleEntropyBasedLandmarking(dataset_points, dataset_surfaces, number_of_particles=20)
  particle_entropy_landmarking.run_particle_optimisation()
