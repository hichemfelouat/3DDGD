from crop_3Dface.projection_2d import pc2d,get_landmarks_3d,filter_point_cloud
from crop_3Dface.obj_handler import read_obj_file, npy2obj
from crop_3Dface.landmark_detecter_2d import get_landmarks_mp
import numpy as np
from PIL import Image,ImageDraw

#------------------------------------------------------------------------------------------------
def compute_mask(landmarks:np.ndarray):
  #cropping point cloud using 7 rough landmarks
  lm = landmarks
  nose_bottom = np.array(lm[4])
  nose_bridge = (np.array(lm[1]) + np.array(lm[2]))/2
  face_centre = nose_bottom + 0.3 * (nose_bridge-nose_bottom)

  outer_eye_dist = np.linalg.norm(np.array(lm[0]) - np.array(lm[3]))
  nose_dist = np.linalg.norm(nose_bridge - nose_bottom)

  mask_radius = 1.4*(outer_eye_dist + nose_dist)/2
  return (face_centre, mask_radius)

#------------------------------------------------------------------------------------------------
def crop(landmarks:np.ndarray, pointcloud:np.ndarray):
  v = pointcloud;
  face_centre, mask_radius = compute_mask(landmarks)
  dist = np.linalg.norm(v-face_centre, axis=1)
  ids = np.where(dist <= mask_radius * 1.3)[0]
  return v[ids]

#------------------------------------------------------------------------------------------------
def compute_landmarks_pc(point_cloud, img_size=256, cropping = True, img_save_nocrop = None, img_save_crop = None, visibility_radius = 0, visibility_obj = None):

  if not cropping:
    rough_landmarks, _ = __compute_landmarks_pc_helper(point_cloud, img_size, img_save_nocrop, visibility_radius = visibility_radius, visibility_obj = visibility_obj)
    return rough_landmarks, None
    
  else:
    rough_landmarks, point_cloud_visibility_applied = __compute_landmarks_pc_helper(point_cloud, img_size, img_save = img_save_nocrop, visibility_radius = visibility_radius, visibility_obj = visibility_obj)
    
    if rough_landmarks is None:
      return rough_landmarks, None 
      
    mp_i_7 = [33, 133, 362, 263, 2, 76, 292]
    pc_cropped = crop(rough_landmarks[mp_i_7], point_cloud_visibility_applied)
    cropped_landmarks, _ = __compute_landmarks_pc_helper(pc_cropped, img_size, img_save = img_save_crop, visibility_radius = 0,visibility_obj=None)
    
    return rough_landmarks, cropped_landmarks

#------------------------------------------------------------------------------------------------     
def __compute_landmarks_pc_helper(point_cloud: np.array, img_size: int, img_save = None, visibility_radius = 0, visibility_obj = None):
    #filter point cloud
    if visibility_radius is not None:
      point_cloud_copy = filter_point_cloud(point_cloud.copy(), visibility_radius=visibility_radius)
      if visibility_obj is not None:
        npy2obj(point_cloud_copy, visibility_obj)
    else:
      point_cloud_copy = point_cloud.copy()
    
    # 2d projection
    pixels, point_cloud_img = pc2d(point_cloud_copy.copy(),img_size=img_size, visibility_radius = None)
    
    img = Image.fromarray((pixels * 255).astype(np.uint8))
    """
    if img_save is not None:
      print(f"Saving Depth Map image to {img_save}")
      img.save(img_save)
    """
    draw = ImageDraw.Draw(img)

    # obtain landmarks
    landmarks = get_landmarks_mp((pixels * 255).astype(np.uint8).copy())
    if landmarks is None : 
      return None, None
      
    if img_save is not None:
      for point in landmarks:
        dot_size = img_size/300 + 1
        draw.ellipse((point[0]-dot_size, point[1]-dot_size, point[0]+dot_size, point[1]+dot_size), fill=(0,0,255,255))
        
      #print("Saving Depth Map image with landmarks to" + img_save.split('.')[0] + " with lm.png")
      #img.save(img_save.split('.')[0] + " with lm.png") #save projection with labeled landmarks

    landmarks_3d = get_landmarks_3d(landmarks, point_cloud_copy, img_size=img_size)

    return landmarks_3d, point_cloud_copy
#------------------------------------------------------------------------------------------------



