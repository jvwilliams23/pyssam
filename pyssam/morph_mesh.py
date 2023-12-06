"""Create a surface mesh by morphing a template mesh and landmarks 
to a new set of landmarks.
This is done using a radial basis function with a gaussian kernel.
Source: Grassi et al. (2011) Medical Engineering & Physics. 
"""
import numpy as np
import trimesh
import vedo as v
from . import utils 

__all__ = ["MorphTemplateMesh"]

class MorphTemplateMesh:
  """Create a mesh for a new set of landmarks based on a pre-existing 'template' mesh,
  for which we already have a set of landmarks.
  This is done using a radial basis function with a gaussian kernel.
  Source: Grassi et al. (2011) Medical Engineering & Physics.
  
  Note that this requires the meshes are already aligned. 
  In rare cases, the error may be high. In this case, I would suggest trying a template
  that is closer to the target if available.

  Examples
  ========
  >>> import pyssam
  >>> torus = pyssam.datasets.Torus()
  >>> torus_mesh_list = torus.make_dataset(2)
  >>> landmark_coordinates = [sample_i.points()[::10] for sample_i in torus_mesh_list]
  >>> mesh_target_actual = torus_mesh_list[-1]
  >>> mesh_target_computed = pyssam.morph_mesh.MorphTemplateMesh(
          landmark_target=landmark_coordinates[-1],
          landmark_template=landmark_coordinates[0],
          mesh_template=torus_mesh_list[0]
      ).mesh_target
  >>> volume_error = 100.0 * abs(mesh_target_computed.volume() - mesh_target_actual.volume()) / mesh_target_actual.volume()
  >>> print("volume error below 5%?", volume_error < 5.0) 
  True
  """
  def __init__(
    self,
    landmark_target,
    landmark_template,
    mesh_template,
    # kernel="gaussian",
    kernel_width=0.3,
    smooth=False,
  ):

    self.smooth = smooth
    kernel = "gaussian" # only one implemented currently
    # gaussian kernel width
    self.kernel_width = kernel_width
    # select functions based on chosen morphing kernel
    if kernel == "gaussian":
      self.kernel_function = self.gaussian_kernel
    else:
      raise NotImplementedError(
        f"The kernel type '{kernel}' has not been implemented."
      )
    self.mesh_template = mesh_template

    (
      self.landmark_target,
      self.landmark_template,
      self.coords_template,
      self.std_scale,
    ) = self.scale_and_align_coordinates(
      landmark_target.copy(), landmark_template.copy(), mesh_template.points()
    )
    self.do_mesh_morphing()

  def do_mesh_morphing(self):
    """Compute coordinates of vertices on new mesh (corresponding to landmark_target).
    The new coordinates are then used to create a new mesh, where face connectivity is the same
    as the template mesh. The new mesh is checked for watertightness, and some cleanup is done.
    """
    # initialise coordinates of new mesh as a copy of the template mesh coordinates
    coords_new = self.coords_template.copy()
    # morph template surface coordinates
    for i, coords_template_i in enumerate(self.coords_template):
      w = self.get_weights(
        self.landmark_target, self.landmark_template, coords_template_i, self.kernel_function
      )
      kernel_output = self.kernel_function(self.landmark_template, coords_template_i)[:, np.newaxis]
      coords_new[i] = coords_template_i + np.sum(kernel_output * w, axis=0)

    # rescale from standard deviation normalised values to real-world
    coords_new *= self.std_scale
    
    self.mesh_target = self.create_new_mesh(coords_new)
    # self.mesh_target = self.clean_new_mesh(self.mesh_target)
    return self.mesh_target

  def create_new_mesh(self, coords):
    """Create a mesh from a set of coordinates, and faces of the template mesh.
    
    Parameters
    ----------
    coords : array_like
      A set of coordinates corresponding to vertices on a new mesh. Ordering must be 
      consistent with self.mesh_template.points()

    Returns
    -------
    mesh : vedo.Mesh
      vedo object containing coordinates and face connectivity for new surface mesh
    """
    # create mesh object from morphed vertices
    return v.Mesh([coords, self.mesh_template.faces()])

  def clean_new_mesh(self, mesh_target):
    """Use trimesh functionality to check mesh is watertight and do some cleaning
    operations and smoothing if desired.
    
    Parameters
    ----------
    mesh_target : vedo.Mesh
      vedo object containing coordinates and face connectivity for new surface mesh

    Returns
    -------
    mesh_target : vedo.Mesh
      vedo object containing coordinates and face connectivity for new surface mesh
    """
    # smoothing and clean up
    mesh_targettri = mesh_target.to_trimesh()
    watertight = mesh_targettri.is_watertight
    if not watertight:
      print("Watertight mesh?", watertight)
      trimesh.repair.fill_holes(mesh_targettri)
      trimesh.repair.broken_faces(mesh_targettri)

    if self.smooth:
      trimesh.smoothing.filter_humphrey(mesh_targettri, alpha=0.1)
      trimesh.smoothing.filter_humphrey(mesh_targettri, alpha=0.1)
      watertight = mesh_targettri.is_watertight

      if not watertight:
        trimesh.repair.fill_holes(mesh_targettri)
        trimesh.repair.broken_faces(mesh_targettri)
    return v.trimesh2vedo(mesh_targettri)

  def scale_and_align_coordinates(
    self, landmark_target, landmark_template, coords_template
  ):
    """Scale the template landmarks and mesh coordinates to the same size as the 
    target landmarks.

    Parameters
    ----------
    landmark_target : array_like
      Landmarks of the new shape which we want to morph the mesh to, shape (N,3).

    landmark_template : array_like
      Landmarks of the template shape (which we already have a mesh for), shape (N,3).

    template_coords_i : array_like
      coordinates on the template surface mesh.

    Returns
    -------
    landmark_target : array_like
      Landmarks of the new shape which we want to morph the mesh to, scaled to 1 std-dev

    landmark_template : array_like
      Landmarks of the template shape, scaled to same scale as target landmarks.

    template_coords_i : array_like
      coordinates on the template surface mesh, scaled to same scale as target landmarks.
    """
    # scale and align template mesh and landmarks
    size_target = landmark_target.max(axis=0) - landmark_target.min(axis=0)
    size_template = landmark_template.max(axis=0) - landmark_template.min(
      axis=0
    )
    size_ratio = size_target / size_template
    coords_template *= size_ratio
    landmark_template *= size_ratio
    coords_new = coords_template.copy()

    # scale to unit standard deviation, such that the gaussian filter is consistent
    # for shapes of different sizes
    std_scale = coords_new.std(axis=0)
    coords_new /= std_scale
    coords_template /= std_scale
    landmark_template /= std_scale
    landmark_target /= std_scale

    return landmark_target, landmark_template, coords_template, std_scale

  def gaussian_kernel(self, landmark_template, template_coords_i):
    """Function to find distance between a coordinate and all surrounding landmarks.
    We use a Gaussian kernel to smooth how the surrounding landmarks act on the point.

    Parameters
    ----------
    landmark_template : array_like
      Landmarks of the template shape (which we already have a mesh for), shape (N,3).

    template_coords_i : array_like
      coordinates on the template surface mesh, shape (3).

    Returns
    -------
    distances : array_like
      Scalar value of distances between all landmarks and the template mesh coordinate.
    """
    return np.exp(
      -(utils.euclidean_distance(landmark_template, template_coords_i) ** 2.0)
      / (2.0 * self.kernel_width**2)
    )

  def get_weights(
    self, landmark_target, landmark_template, template_coords_i, kernel_function
  ):
    """Find weight coefficients that control how the template mesh is morphed to
    the new geometry, based on the distance between the 'template' and 'target' landmarks
    (which are in correspondence, so the ordering is consistent).

    For more information, see equations (1) and (2) in
    Grassi et al. (2011) (Medical Engineering & Physics).

    Parameters
    ----------
    landmark_target : array_like
      Landmarks of the new shape which we want to morph the mesh to, shape (N,3).

    landmark_template : array_like
      Landmarks of the template shape (which we already have a mesh for), shape (N,3).

    template_coords_i : array_like
      coordinates on the template surface mesh.

    Returns
    -------
    weights : array_like
      Weights are coefficients that control how strongly the kernel effects each point.
    """
    kernel_output = kernel_function(landmark_template, template_coords_i)
    weights = (landmark_target - landmark_template) / kernel_output.sum()
    return weights

