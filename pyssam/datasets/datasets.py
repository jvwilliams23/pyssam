import networkx as nx
import numpy as np
import pyssam
from numpy import cos, sin

__all__ = ["Tree"]


class Tree:
  """Create tree object based on a set of pre-defined parameters.

  Parameters
  ----------
  length : list
        Minimum and maximum values for tree first branch segment
  length_ratio : list
        Minimum and maximum values to sample for length ratio between
        parent and child branches
  angle : list
        Minimum and maximum values to sample for angle between child
        branch and parent branch vectors
  num_extra_ends : int
        Number of additional bifurcation levels to generate

  Examples
  ========
  >>> import pyssam
  >>> tree_class = pyssam.datasets.Tree()
  >>> print(tree_class.make_tree_landmarks().shape)
  (8, 3)
  >>> tree_class = Tree(num_extra_ends=2)
  >>> print(tree_class.make_tree_landmarks().shape)
  (32, 3)
  """

  def __init__(
    self,
    length: list = [3.8, 4.2],
    length_ratio: list = [0.2, 0.8],
    angle: list = [5, 60],
    num_extra_ends: int = 0,
  ):
    self._length = length
    self._length_ratio = length_ratio
    self._angle = angle
    self._check_morphology_inputs(self._length)
    self._check_morphology_inputs(self._length_ratio)
    self._check_morphology_inputs(self._angle)

    self._num_extra_ends = num_extra_ends
    self._root = 0
    self.graph_baseline = self._initialise_tree()
    self.edges = list(self.graph_baseline.edges)

  def _check_morphology_inputs(self, parameter_list: list) -> None:
    assert (
      len(parameter_list) == 2
    ), "List length should be 2 (corresponding to min and max values)"

  def _grow_graph_end_nodes(self, graph: nx.DiGraph) -> nx.DiGraph:
    """Add additional bifurcation level to each end point in current tree.

    Parameters
    ----------
    graph : nx.DiGraph

    Returns
    -------
    graph : nx.DiGraph
    """
    graph_0 = graph.copy()
    for current_edge in nx.dfs_edges(graph, 0):
      current_node = current_edge[1]
      if graph.degree(current_node) == 1:
        graph_0.add_edge(current_node, max(graph_0.nodes) + 1)
        graph_0.add_edge(current_node, max(graph_0.nodes) + 1)
    return graph_0

  def _parent_branch_length(self, graph, edge_parent) -> float:
    pos_distal = graph.nodes[edge_parent[0]]["position"]
    pos_proximal = graph.nodes[edge_parent[1]]["position"]
    return pyssam.utils.euclidean_distance(pos_distal, pos_proximal)

  def _initialise_tree(self):
    """Create baseline tree structure to adapt from for creating population.

    Parameters
    ----------
    None

    Returns
    -------
    graph : nx.DiGraph
        Baseline graph, then node positions will be randomly generated later
    """
    edge_list_orig = [[0, 1], [1, 2], [1, 3]]
    end_nodes = [2, 3]
    graph = nx.DiGraph()
    graph.add_edges_from(edge_list_orig)
    # use original tree structure to grow more branches
    for node_i in end_nodes:
      graph.add_edge(node_i, max(graph.nodes) + 1)
      graph.add_edge(node_i, max(graph.nodes) + 1)

    for _ in range(0, self._num_extra_ends):
      graph = self._grow_graph_end_nodes(graph)

    graph.nodes[self._root]["position"] = np.array([0, 0, 0])
    return graph

  def make_tree(self) -> nx.DiGraph:
    """Make a tree structure based on a baseline graph by randomly creating
    nodal coordinates (also determined by angle, length and length_ratio)

    Returns
    -------
    graph : nx.DiGraph
        Randomly created graph that can be used with shape model
    """
    graph = self.graph_baseline.copy()
    length_i = np.random.uniform(self._length[0], self._length[1])
    graph.nodes[1]["position"] = np.array([0, 0, length_i])

    for current_edge in nx.dfs_edges(graph, 0):
      # get all attributes from current edge and each node in edge
      parent_node = current_edge[0]
      node_i = current_edge[1]
      parent_position = graph.nodes[parent_node]["position"]
      current_position = graph.nodes[node_i]["position"]
      child_edges = list(graph.out_edges(node_i))
      angle_multiplier = [1, -1]
      for angle_multiplier_i, edge_i in zip(angle_multiplier, child_edges):
        child_i = edge_i[1]
        vector_parent = (current_position - parent_position) / np.linalg.norm(
          current_position - parent_position
        )
        angle_from_zero = np.rad2deg(
          np.arccos(
            np.dot(vector_parent, [1, 0, 0])
            / np.sqrt(np.sum(vector_parent ** 2))
            / np.sqrt(np.sum(np.array([1, 0, 0]) ** 2))
          )
        )

        length_i = self._parent_branch_length(
          graph, current_edge
        ) * np.random.uniform(
          self._length_ratio[0], self._length_ratio[1]
        )
        angle_i = np.random.uniform(self._angle[0], self._angle[1])
        angle_out_deg = angle_from_zero - angle_multiplier_i * angle_i
        angle_out = np.deg2rad(angle_out_deg)

        x_child_i = current_position[0] + length_i * cos(angle_out)
        z_child_i = current_position[2] + length_i * sin(angle_out)
        graph.nodes[child_i]["position"] = np.array([x_child_i, 0, z_child_i])

    return graph

  def graph_to_coords(self, graph: nx.DiGraph) -> np.array:
    """Convert "position" key from all nodes in graph to a numpy array of
    coordinates.

    Parameters
    ----------
    graph : nx.DiGraph
        Graph with "position" entry in nodal attributes
    """
    coords_out = []
    for node_i in graph:
      coords_out.append(graph.nodes[node_i]["position"])
    return np.array(coords_out)

  def make_tree_landmarks(self) -> np.array:
    """Make tree landmarks based on a baseline graph by randomly creating nodal
    coordinates.

    Returns
    -------
    landmarks : array_like
        Landmarks for nodal coordinates on randomly created graph
        that can be used with shape model
    """
    tree_graph = self.make_tree()
    return self.graph_to_coords(tree_graph)
