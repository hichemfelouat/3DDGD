import numpy as np
from PIL import Image, ImageDraw
import crop_3Dface.visibility_pointclouds as vpc
from crop_3Dface.obj_handler import read_obj_file, npy2obj
from scipy.stats import rankdata
from scipy.spatial import KDTree
import os


def array2img(point_cloud_base, img_size):
    point_cloud = point_cloud_base.copy()
    # convert coordinates from center (0,0) to top left (0,0)
    point_cloud[:,0] = point_cloud[:,0] + img_size/2
    point_cloud[:,1] = point_cloud[:,1] + img_size/2
    # flip the y axis
    point_cloud[:,1] = img_size - point_cloud[:,1]
    return point_cloud

def draw_point_cloud(point_cloud, img, color):
    # draw the point cloud on the image
    draw = ImageDraw.Draw(img)
    for i in range(len(point_cloud)):
        draw.point((point_cloud[i][0], point_cloud[i][1]), fill=color)

def generate_colors(point_cloud: np.array):
    #find percentile of all points
    percentile = rankdata(point_cloud[:, 2])*100/len(point_cloud[:, 2])
    percentile = percentile / 100

    #assign colors based off percentile - colors in values from 0-1
    colors = np.zeros((len(point_cloud), 3))
    for i in range(len(point_cloud)):
        colors[i] = [percentile[i], percentile[i], percentile[i]]

    if False: #alternative way with direct color value assignment
      lowest = np.min(point_cloud[:,2])
      highest = np.max(point_cloud[:,2])
      z_range = highest - lowest
      colors = np.zeros((len(point_cloud), 3))
      for i in range(len(point_cloud)):
        value = (point_cloud[i][2]-lowest)/z_range
        colors[i] = [value, value, value]

    return colors
    
def compute_tranformations(point_cloud_base: np.array, img_size:int, scaling_factor=2):
    point_cloud = point_cloud_base.copy()

    # get min and max values
    x_min = min(point_cloud[:,0])
    y_min = min(point_cloud[:,1])
    x_max = max(point_cloud[:,0])
    y_max = max(point_cloud[:,1])

    # calculate the middle value
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2

    #recompute the range
    x_min = x_min - x_mid
    x_max = x_max - x_mid
    y_min = y_min - y_mid
    y_max = y_max - y_mid

    # compute the scaling factor
    abs_max = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
    scale = img_size / (2 * abs_max)

    return (x_mid, y_mid),scale

def calcColorTreshold(pixel_x: int, pixel_y: int, closest_points: np.array, closest_points_color: np.array):
    # find the bounds
    min_x = np.min(closest_points[:, 0]).item()
    max_x = np.max(closest_points[:, 0]).item()
    min_y = np.min(closest_points[:, 1]).item()
    max_y = np.max(closest_points[:, 1]).item()

    if (min_x <= pixel_x and pixel_x <= max_x and min_y <= pixel_y and pixel_y <= max_y):
      #in bounds, return average color of 3 closest points
      return np.mean(closest_points_color[:3], axis = 0)
    else:
      #out of bounds, return black
      return np.zeros(3)
    
def process_img(img_size: int, point_cloud_2d:np.array , point_cloud_colors: np.array):
    # generate all coordinates of image pixels
    coords = np.stack(np.meshgrid(np.arange(img_size), np.arange(img_size)), axis=-1).reshape(-1, 2)
    coords = np.fliplr(coords)

    #generate closest neighbor points indices for all pixels
    kdtree = KDTree(point_cloud_2d)
    k = 15 #number of points to search
    dists, inds = kdtree.query(coords, k=k)

    #fill in the pixels
    pixels = np.zeros((img_size, img_size, 3))
    point_cloud_colors = np.asarray(point_cloud_colors)
    for x in range(img_size):
      for y in range(img_size):
        pixels[x][y] = calcColorTreshold(x, y, point_cloud_2d[inds[img_size*x + y]], point_cloud_colors[inds[img_size*x + y]])

    #flip and rotate image to make the point cloud
    pixels = np.rot90(pixels, k=3)
    pixels = np.fliplr(pixels)
    return pixels

def filter_point_cloud(point_cloud_og:np.array, visibility_radius = 0):
    point_cloud = point_cloud_og.copy()
    point_cloud = vpc.applyHPR(point_cloud, radius = visibility_radius)

    return point_cloud

def pc2d(point_cloud_base: np.array, img_size=512, visibility_radius = 0,no_npi=False):
    point_cloud = point_cloud_base.copy()

    # filter the point cloud
    if visibility_radius is not None:
      point_cloud = filter_point_cloud(point_cloud, visibility_radius=visibility_radius)
    
    #scale point cloud to image dimensions
    translation, scale = compute_tranformations(point_cloud, img_size)

    # apply transformation and scaling to the point cloud
    point_cloud[:,0] = point_cloud[:,0] - translation[0]
    point_cloud[:,1] = point_cloud[:,1] - translation[1]
    point_cloud[:,0] = point_cloud[:,0] * scale
    point_cloud[:,1] = point_cloud[:,1] * scale 
   
    # populate colors values
    point_cloud_colors = generate_colors(point_cloud.copy())

    # convert to image coordinates
    point_cloud = array2img(point_cloud.copy(), img_size)

    # process the image
    if no_npi:
        # colors a pixel for each point
        pixels = np.zeros((img_size, img_size, 3))
        for i in range(len(point_cloud)):
            x = int(point_cloud[i][0])
            if x >= img_size:
                x = img_size - 1
            y = int(point_cloud[i][1])
            if y >= img_size:
                y = img_size - 1
            pixels[x][y] = point_cloud_colors[i]
    else:
        pixels = process_img(img_size, point_cloud[:,:2].copy(), point_cloud_colors)
    return pixels, point_cloud


def get_landmarks_3d(landmarks_2d: np.array, point_cloud_base:np.array, img_size: int):
    point_cloud = point_cloud_base.copy()
    #transformations on point cloud to fit image dimensions
    translation,scale = compute_tranformations(point_cloud, img_size)
    point_cloud[:,0] = point_cloud[:,0] - translation[0]
    point_cloud[:,1] = point_cloud[:,1] - translation[1]
    point_cloud[:,0] = point_cloud[:,0] * scale
    point_cloud[:,1] = point_cloud[:,1] * scale 
    # convert to image coordinate
    point_cloud = array2img(point_cloud.copy(), img_size)

    # get the closest points 3d points of 2d landmarks
    kdtree = KDTree(point_cloud[:, :2])
    k = 1 #number of points to search
    dists, inds = kdtree.query(landmarks_2d, k=k)

    # get the coordinates of lm in the original point cloud
    closest_points_3d = np.zeros((len(inds),3))
    for i in range(len(inds)):
        closest_points_3d[i] = point_cloud_base[inds[i]]
    
    return closest_points_3d
