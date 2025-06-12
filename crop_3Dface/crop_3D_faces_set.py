import numpy as np
import matplotlib.pyplot as plt
import trimesh
from trimesh import Trimesh
from glob import glob
#import mayavi.mlab as mlab

from crop_3D_face import*
from compute_3d_landmarks import compute_landmarks_pc
from obj_handler import read_obj_file

import argparse 
import time

start_time = time.time()
#import os
#os.system("cls")

#-------------------------------------------------------------------------------
def create_fake_example(obj_file, lst_eye_l_ind, lst_eye_r_ind, is_from_faceScape):
  mesh = np.asarray(read_obj_file(obj_file))
  # 3D facial landmarks extraction
  lm_nocrop, lm_crop = compute_landmarks_pc(
    mesh, 
    img_size          = 256, 
    cropping          = True, 
    img_save_nocrop   = None, 
    img_save_crop     = None, 
    visibility_radius = 0, 
    visibility_obj    = None)
  
  # List of landmarks
  try:
    if lm_crop == None:
      print("len lm_nocrop : ",len(lm_nocrop))
      lm_crop = lm_nocrop.copy()
  except: 
    #print("len lm_nocrop : ",len(lm_nocrop))
    print("len lm_crop   : ",len(lm_crop))

  # Get cropped face
  alpha   = 1.0
  mrg     = 0.0001
  if is_from_faceScape == 0:
    mrg = 0.0001
  else:
    mrg = 3.00 
  
  cropped_face = crop_face_from_obj(obj_file, lm_crop, lst_eye_l_ind, lst_eye_r_ind, mrg, alpha, is_from_faceScape)
  
  # Remove irrelevant parts
  cropped_face_irr_p = remove_irrelevant_parts(cropped_face)
  
  # Remove eyes outliers
  vertices_face   = cropped_face_irr_p.vertices
  faces_face      = cropped_face_irr_p.faces
  vertices, faces = remove_eyes_outlier(vertices_face.tolist(), faces_face.tolist(),  lm_crop, lst_eye_l_ind, lst_eye_r_ind, 0.001)
  
  vertices_arr     = np.array(vertices)
  faces_arr        = np.array(faces)
  cropped_face_new = Trimesh(vertices_arr, faces_arr)
  
  return cropped_face_new

#-------------------------------------------------------------------------------
def main(args):
  inputfolder     = args.inputfolder
  savefolder      = args.savefolder
  isfromfaceScape = args.isfromfaceScape
     
  lst_eye_l_ind  = [29,27,28,56,157,153,145,24,110,25,33]
  lst_eye_r_ind  = [286,258,257,388,390,254,253,252,256,463,414]
  
  path_meshs = glob(inputfolder+"/*.obj")

  save_folder = savefolder+"/"

  for i in range(len(path_meshs)):
    try:
      obj_file = path_meshs[i]
      name_obj = obj_file.split("/")[-1]
  
      # Get cropped face
      cropped_face = create_fake_example(obj_file, lst_eye_l_ind, lst_eye_r_ind, isfromfaceScape)
    
      # Export cropped face 
      cropped_face.export(file_obj=save_folder+name_obj, file_type="obj")

      print("Done ...")
      
    except:
      print("An exception occurred (path) : ",obj_file)
      continue
 
    
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="cropped 3d face.")
  
  parser.add_argument("-i", "--inputfolder",    default="data/examples", type=str, help="path to the test data.")
  parser.add_argument("-s", "--savefolder" ,    default="data/examples/results", type=str, help="path to the output directory.")
  parser.add_argument("-t","--isfromfaceScape", default=0, type=int, help="Is this input from the FaceScape dataset?" )
  main(parser.parse_args())


