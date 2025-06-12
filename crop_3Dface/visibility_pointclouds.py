import numpy as np
import math
from scipy.spatial import ConvexHull
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import torch
from crop_3Dface.obj_handler import read_obj_file

def HPR(p, C, radius=0):
  if isinstance(radius, torch.Tensor):
    radius = radius.item()
  if radius == 0:
    radius = calc_ideal_radius(p, init_param = 1, debug=False)

  dim = p.shape[1]
  num_pts = p.shape[0]
  # Move the points s.t. C is the origin
  p = p - np.tile(C, [num_pts, 1])
  # Calculate ||p||
  normp = np.sqrt(np.sum(p*p, axis=1))
  # Sphere radius
  R = np.tile(np.max(normp)*(10 ** radius), [num_pts, 1])
  # Spherical flipping
  normp = np.reshape(normp, (num_pts, 1))
  t = np.tile(R - normp, [1, dim])
  P = p + 2 * np.divide(np.multiply(t, p), np.tile(normp, [1, dim]))
  # Compute the convex hull on the flipped points and the origin (the view point)
  visiblePtInds = np.unique(ConvexHull(np.concatenate((P,np.zeros((1, 3))), axis=0)).vertices)
  # Remove the index for the origin before returning the indices of the visible points
  visiblePtInds = np.delete(visiblePtInds, np.where(visiblePtInds==num_pts))
  return visiblePtInds

def applyHPR(p, radius=0):
  C = camera(p, d=10)
  v = HPR(p, C, radius)
  points = []
  old_new_index_map = dict()
  for i in range(len(p)):
    if i in v:
      points.append(p[i])
  return np.asarray(points)

def gradient(x, y, index):
  return np.sign((y[index] - y[index-1]) / (x[index] - x[index-1]))

def center(p):
  return np.array([
      (np.min(p[:, 0]) + np.max(p[:, 0])) / 2,
      (np.min(p[:, 1]) + np.max(p[:, 1])) / 2,
      (np.min(p[:, 2]) + np.max(p[:, 2])) / 2])
def calc_ideal_radius(p, init_param = 1.0, debug=False):
  # A viewpoint opposite to the current viewpoint C on the line connecting the
  # original viewpoint to the object’s center of mass
  C = camera(p, d=10)
  C2 = 2*center(p) - C

  g = 1
  rate = 0.1
  disjoins = np.array([])
  params = np.array([init_param])
  v1 = HPR(p, C, params[-1])
  v2 = HPR(p, C2, params[-1])
  disjoins = np.append(disjoins, len(np.union1d(v1, v2)) - len(np.intersect1d(v1, v2, assume_unique=True)))
  while True:
    params = np.append(params, params[-1]+rate*g)
    v1 = HPR(p, C, params[-1])
    v2 = HPR(p, C2, params[-1])
    disjoins = np.append(disjoins, len(np.union1d(v1, v2)) - len(np.intersect1d(v1, v2, assume_unique=True)))
    delta = disjoins[-1] - disjoins[-2]
    if debug:
      print("param=", params[-1], ", rate=", rate, ", delta=", delta)
    # Stop if either the delta is small enough or the rate is too small
    if -5 < delta < 5 or rate < 1e-12:
      break
    # Reduce rate by half if the gradient changes direction
    g = gradient(params, disjoins, -1)
    if len(disjoins) > 2 and g != gradient(params, disjoins, -2):
      rate /= 2

  return params[-1]

def camera(p, d=1.0, h_angle=0.0, v_angle=0.0):
  min_x = np.min(p[:, 0])
  min_y = np.min(p[:, 1])
  min_z = np.min(p[:, 2])
  max_x = np.max(p[:, 0])
  max_y = np.max(p[:, 1])
  max_z = np.max(p[:, 2])
  mean_x = (min_x + max_x) / 2
  mean_y = (min_y + max_y) / 2
  mean_z = (min_z + max_z) / 2
  r = d * max((max_x-min_x), (max_y-min_y), (max_z-min_z))
  return np.array([mean_x+r*math.cos(math.radians(v_angle))*math.sin(math.radians(h_angle)),
                   mean_y+r*math.sin(math.radians(v_angle)),
                   mean_z+r*math.cos(math.radians(v_angle))*math.cos(math.radians(h_angle))])
