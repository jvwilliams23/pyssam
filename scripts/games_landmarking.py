"""
	USAGE: 
	python games_master.py -inp landmarksBase.csv edgesBase.csv 
													-meshdir /home/josh/luna16_labels/

	HELP:
	python games_master.py --help
	python games_master.py -h

	Uses GAMEs (Growing and Adaptive Meshes) alignment for 
	automatic landmarking of SSM.
	
	Based on:
	L. Ferrarini et al, Medical Image Analysis 11 (2007) 302-314

	Programmed by:		Josh Williams
	Began:				15/03/2020

  two most time consuming are advance t counter and arrayRowIntersection

"""

import argparse
import re
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
  surface_points_wholeSurf, case_name, coarsenFactor=4, activation_threshold=0.01, organ=""
):
  """
  Args:
  surface_points_wholeSurf (-1x3 array): Point cloud of entire surface mesh
  (used for accuracy comparison)

  Returns:
  A (-1x3 array): list of nodes, representing landmarks of surface model
  edge_age (-1x6 array): Edges connecting pairs of landmarks

  Performs growing phase of the GAMEs algorithm.
  This function performs unsupervised clustering of a set of surface coordinates,
  landmarks are selected based on their euclidean distance from other points.
  This is then repeated using the Mahalanobis distance to choose landmarks that represent
  the subset of surrounding surface points best.
  """
  graph = nx.DiGraph()

  # -Initialise - need to adjust A to remove matching nodes
  surface_points = shuffle(surface_points_wholeSurf[::coarsenFactor])
  landmark_setA = choices(surface_points, k=2)  # set of nodes
  if args.debug:
    landmark_setA = [surface_points[0], surface_points[1000]]
  landmark_setA = np.array(landmark_setA, dtype="float64")

  if np.allclose(landmark_setA[0], landmark_setA[1]):
    exit()
  edge_age = np.zeros(
    7
  )  # [] # empty set of edges and their edges (edge edge age = edge_age)
  edge_age.shape = (1, 7)
  threshold_std_dev_dist_to_surf = 0.01  # Threshold accuracy
  threshold_average_dist_to_surf = 5
  # activation_threshold = 0.001  # 0.025 # activation threshold
  edge_age_threshold = 50  # edge age threshold
  t_counterList = np.column_stack((surface_points, np.zeros(len(surface_points))))

  distance_metric = euclideanDist

  print("Beginning loop")
  initalRemoveCheck = 0
  loopedPoints = 0
  euclidean_dist_std_dev = 1.0
  average_distance_lm_to_surf = 100.0

  # initialisation

  while not round(average_distance_lm_to_surf, 1) <= threshold_average_dist_to_surf:
    for surface_point_i in surface_points:
      pass
      # find closest node to surface_point_i

      # find best matching node and second best batching node

      # create edge between best and second best matching node
      # set edge age = 0

      # evaluate activation function for best matching node

      # check if activtion less than cutoff.

      # increase age of edges ENDING in best matching node (implies we need a DiGraph)

      # decrease diring value of best matching node and its neighbors

      # remove edges where age is above a cutoff

      # remove isolated nodes

  return graph


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
