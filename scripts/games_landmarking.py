import argparse
from copy import copy
from glob import glob
from random import choices
from sys import exit
from time import time

import networkx as nx
import numpy as np
from sklearn.utils import shuffle
import vedo as v

def getInputs():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--meshfile",
    "-m",
    default="dataset/*/STL/ET/*_ET_SmoothOpen.stl",
    type=str,
    help="directory containing meshes to grow",
  )
  return parser.parse_args()

def euclideanDist(x, y):
  """
  Finds the euclidean distance between two arrays x, y.
  Calculated using pythagoras theorem
  """
  if x.size == 3:
    return np.sqrt(np.sum((x - y) ** 2))
  else:
    return np.sqrt(np.sum((x - y) ** 2, axis=1))

def mahalanobisDist(x, y):
  """
  Finds the mahalanobis distance between two arrays x, y
  Calculated based on the inverse covariance matrix of the two arrays
  and the difference of each array (delta)
  """
  delta = x - y
  if len(np.where(delta == delta[0])[0]) == delta.size:
    return 0
  X = np.vstack([x, y])

  V = np.cov(X.T)

  if V[0, 0] == 0 or V[1, 1] == 0 or V[2, 2] == 0:
    print("SINGULAR MATRIX, ADJUSTING")
    V += np.eye(3) * 1.0e-6
    # return 10000000
    # exit()

  VI = np.linalg.inv(V)

  if np.sum(np.dot(delta, VI) * delta) < 0:
    return 10000000

  return np.sqrt(np.sum(np.dot(delta, VI) * delta, axis=-1))


def summandDist(
  pointCloud, modelNodes, distance_metricFind, distance_metricSum=euclideanDist
):
  """
  Returns the cumulative euclidean distance between
  surface point and nearest node.
  """
  # -shorten variable names, consistent with syntax of the paper
  # -surface_points = pointCloud
  # -A = modelNodes
  sumList = [
    distance_metricSum(p_k, modelNodes[np.argmin(distance_metricFind(modelNodes, p_k))])
    for i, p_k in enumerate(pointCloud)
  ]

  return sum(sumList), np.std(sumList), sumList

def f_T(alpha=1.05, beta=3.33):
  return 1 - (1 / alpha) * (1 - np.exp(-(alpha * 5) / beta))


def f_s(t_counter, alpha_w=1.05, beta_w=3.33):
  # -Firing value associated with s
  return 1 - (1 / alpha_w) * (1 - np.exp(-(alpha_w * t_counter) / beta_w))


def f_n(t_counter, alpha_n=1.05, beta_n=14.3):
  # -Firing value associated with s's neighbors
  return 1 - (1 / alpha_n) * (1 - np.exp(-(alpha_n * t_counter) / beta_n))


def grow_landmark_network(
  surface_points_orig, activation_threshold=0.01
):
  """
  Args:
  surface_points_orig (-1x3 array): Point cloud of entire surface mesh
  (used for accuracy comparison)

  Performs growing phase of the GAMEs algorithm.
  This function performs unsupervised clustering of a set of surface coordinates,
  landmarks are selected based on their euclidean distance from other points.
  This is then repeated using the Mahalanobis distance to choose landmarks that represent
  the subset of surrounding surface points best.
  """
  graph = nx.DiGraph()
  graph.add_nodes_from([0, 1])

  # -Initialise - need to adjust A to remove matching nodes
  surface_points = shuffle(surface_points_orig)
  landmark_setA = choices(surface_points, k=2)  # set of nodes
  # landmark_setA = [surface_points[0], surface_points[1000]]
  landmark_setA = np.array(landmark_setA, dtype="float64")

  if np.allclose(landmark_setA[0], landmark_setA[1]):
    exit()
  graph.nodes[0]["position"] = landmark_setA[0]
  graph.nodes[1]["position"] = landmark_setA[1]
  threshold_std_dev_dist_to_surf = 0.01  # Threshold accuracy
  threshold_average_dist_to_surf = 5
  edge_age_threshold = 50  # edge age threshold

  distance_metric = euclideanDist

  print("Beginning loop")
  initalRemoveCheck = 0
  loopedPoints = 0
  euclidean_dist_std_dev = 1.0
  average_distance_lm_to_surf = 100.0

  # initialisation

  while not round(average_distance_lm_to_surf, 1) <= threshold_average_dist_to_surf:
    for surface_point_i in surface_points:
      # find closest node to surface_point_i
      graph_positions = _graph_nodes_to_positions(graph)
      distance_to_point_i = distance_metric(graph_positions, surface_point_i)
      nearest_point_index = np.argmin(distance_to_point_i)
      nearest_node = list(graph.nodes)[nearest_point_index]
      position_nearest_node = graph.nodes[nearest_node]["position"]
      distance_closest_node = distance_to_point_i[nearest_point_index]
      # find second closest node
      distance_to_point_i[np.argmin(distance_to_point_i)] = np.inf
      second_nearest_point_index = np.argmin(distance_to_point_i)
      second_nearest_node = list(graph.nodes)[second_nearest_point_index]

      # create edge between best and second best matching node, set edge age = 0
      edge_i = (nearest_node, second_nearest_node)
      # TODO: maybe need to check if edge exists
      graph.add_edge(*edge_i)
      graph.edges[edge_i]["age"] = 0
      
      # TODO
      firing_value_nearest = 1

      # evaluate activation function for best matching node
      activation_value = np.exp(-distance_closest_node)
      # check if activtion less than cutoff.
      if activation_value < activation_threshold: #TODO: and f_s < f_T
        # add new node and connect to graph (grow graph)
        new_node = max(graph.nodes)+1
        graph.add_node(new_node, position=(surface_point_i+position_nearest_node)/2)
        graph.add_edge(new_node, nearest_node, age=0)
        graph.add_edge(new_node, second_nearest_node, age=0)
        graph.remove_edge(nearest_node, second_nearest_node)
      else:
        # adapt nearest node and its neighbors
        morph_parameter_nearest = 0.1 # e_w in paper
        # morph_parameter_neighbors = morph_parameter_nearest * np.exp(-) # e_n in paper
        connected_edges = list(graph.edges(nearest_node))
        for edge_j in connected_edges:
          node_j = edge_j[1]
          firing_value_neighbor = 1 # TODO
          position_node_j = graph.nodes[node_j]["position"]
          distance_nearest_to_neighbor = distance_metric(position_nearest_node, position_node_j)
          SIGMA = 0.42 # empirical parameter from Ferrarini et al.
          morph_parameter_neighbor_j = morph_parameter_nearest * np.exp(-distance_nearest_to_neighbor / (2 * SIGMA**2.0))
          # move neighbor nodes position
          graph.nodes[node_j]["position"] += morph_parameter_neighbor_j * firing_value_neighbor * (surface_point_i - position_node_j)
        # move nearest node position
        graph.nodes[nearest_node]["position"] += morph_parameter_neighbor_j * firing_value_neighbor * (surface_point_i - position_node_j)
      # increase age of edges ENDING in best matching node (implies we need a DiGraph)
      neighbor_edges = graph.out_edges(nearest_node)
      for edge_i in neighbor_edges:
        graph.edges[edge_i]["age"] += 1

      # decrease firing value of best matching node and its neighbors

      # remove edges where age is above a cutoff
      graph = _check_remove_edge_age(graph)

      # remove isolated nodes
      graph = graph.remove_nodes_from(nx.isolates(graph))
      continue

  return graph

def _check_remove_edge_age(graph, age_cutoff=50):
  remove_edges = []
  for edge_i in graph.edges:
    age = graph.edges[edge_i]["age"]
    if age > age_cutoff:
      remove_edges.append(edge_i)

  graph.remove_edges_from(remove_edges)
  return graph

def _graph_nodes_to_positions(graph):
  return np.array(list(nx.get_node_attributes(graph, "position").values()))

###################################################################################
# 							MAIN RUN FOR ADAPT CODE
###################################################################################

if __name__ == "__main__":
  print(__doc__)
  # Dictionary of labels for reading surface files
  # -importing loadcase name and other user args
  args = getInputs()

  surface_points_orig = v.load(args.meshfile).points()[::10]
  # scale to min -1000 and max 1000. 
  # Should allow accuracy criteria to be consistent in different cases
  graph = grow_landmark_network(
    surface_points_orig, activation_threshold=0.01
  )
