import os
import sys
import argparse
import numpy as np

import matplotlib.pyplot as plt
import trimesh
from trimesh import Trimesh
from glob import glob
import time
from tqdm import tqdm
import trimesh
from trimesh import Trimesh

from crop_3Dface.crop_3D_face import*
from crop_3Dface.compute_3d_landmarks import compute_landmarks_pc
from crop_3Dface.obj_handler import read_obj_file

from features_3Dmesh.features_3d_mesh import*

import torch
from torch.utils.data import Dataset, DataLoader
from pickle import load
from sklearn.preprocessing import StandardScaler

from Tabtransformer.tabtransformer import *
from Mesh_MLP_MHA.meshmlp_mha_net import *

os.system("clear")

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def crop3Dface(obj_file, is_from_faceScape=0):

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
      #print("len lm_nocrop : ",len(lm_nocrop))
      lm_crop = lm_nocrop.copy()
  except: 
    #print("len lm_nocrop : ",len(lm_nocrop))
    #print("len lm_crop   : ",len(lm_crop))
    alpha   = 1.0
  
  lst_eye_l_ind  = [29,27,28,56,157,153,145,24,110,25,33]
  lst_eye_r_ind  = [286,258,257,388,390,254,253,252,256,463,414]
  
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

class CustomInferenceDataset(Dataset):
  def __init__(self, features, scaler_path=None):
  
    self.features = features
    
    # Load the saved scaler
    if scaler_path is not None:
        self.scaler = load(open(scaler_path, "rb"))
        # Apply the scaler to the features
        self.features = self.scaler.transform(np.array(features).reshape(1, -1))
  
  def __len__(self):
    return 1  # Since it's a single example
  
  def __getitem__(self, idx):
    return torch.tensor(self.features, dtype=torch.float32)

def error_message(error_type):
  print("****************************")
  if error_type == 1:
    print("Error: Failed to load the model. Please ensure the path is correct and the model file exists at the specified location.")
  elif error_type == 2:
    print("Error: An issue occurred during the 3D face cropping process.")
  elif error_type == 3:
    print("Error: An issue occurred during the feature extraction process.")
  else:
    print("Error: An issue occurred during the prediction process.") 
  print("****************************")


def load_model(model_name, feature_dim):
  # Load the model 
  print("Load the model ... (",model_name,")")

  if model_name == "Tabtransformer":
      try:
        if feature_dim == 624:
          model_path = "weights/Tabtransformer/Tabtransformer_G0.pth"
        if feature_dim == 2496:
          model_path = "weights/Tabtransformer/Tabtransformer_G1.pth"
        if feature_dim == 6006:
          model_path = "weights/Tabtransformer/Tabtransformer_G2.pth"

        d_model      = 128
        num_heads    = 4
        num_layers   = 3
        d_ff         = 256
        num_classes  = 1  
        dropout_rate = 0.1
        
        model_tab = TabTransformer(feature_dim, d_model, num_heads, num_layers, d_ff, num_classes, dropout_rate)  
        model_tab.load_state_dict(torch.load(model_path))
        model_tab.eval()

        return model_tab

      except:
        error_type = 1
        error_message(error_type)
        print("model_path : ",model_path)
        sys.exit(1)

  if model_name == "Mesh_MLP_MHA":
      try:
        if feature_dim == 624:
          model_path = "weights/Mesh_MLP_MHA/Mesh_MLP_MHA_G0.pth" 
        if feature_dim == 2496:
          model_path = "weights/Mesh_MLP_MHA/Mesh_MLP_MHA_G1.pth"
        if feature_dim == 6006:
          model_path = "weights/Mesh_MLP_MHA/Mesh_MLP_MHA_G2.pth"

        num_classes = 1
        drop_prob   = 0.1
        k_eig_list  = [2047, 128, 32]

        model_MHA = Net(C_in=feature_dim, C_out=num_classes, drop_path_rate=drop_prob, k_eig_list=k_eig_list)
        model_MHA.load_state_dict(torch.load(model_path))
        model_MHA.eval()

        return model_MHA

      except:
        error_type = 1
        error_message(error_type)
        print("model_path : ",model_path)
        sys.exit(1)

    
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def main(args):
    start_time = time.time()
    print("\nStart ...")
    
    input_data   = args.input_data
    feature_type = args.feature_type
    model_name   = args.model_name
    is_cropped   = args.is_cropped
    
    print("Input Data  : ", args.input_data)
    print("Feature Type: ", args.feature_type)
    print("Model Name  : ", args.model_name)
    print("Is Cropped  : ", args.is_cropped)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k      = 16
    in_dim = 624
    
    if feature_type == "G1":
      k = 64
      in_dim = 2496
    if feature_type == "G2":
      k = 154
      in_dim = 6006
    
    error_type = 0
    
    print("k      : ",k)
    print("in_dim : ",in_dim)
    model_loaded = load_model(model_name, in_dim)

    obj_paths = glob(input_data+"/*.obj")
    print("-" * 50)
    print("-" * 50)
    for i in tqdm(range(len(obj_paths))):     
      error_type    = 0
      face_obj_path = obj_paths[i]
      name_face_obj = face_obj_path.split("/")[-1]
      print(i," - "+name_face_obj+" : ")

      ts1 = time.time()
      try:
        if is_cropped == 0:
          cropped_face  = crop3Dface(face_obj_path)
          print(name_face_obj+" cropped is OK.")
          #cropped_face.export(file_obj=input_data+"/cropped_"+name_face_obj, file_type="obj")
        else:  
          cropped_face = trimesh.load(face_obj_path)
          print(name_face_obj+" loaded is OK.")
        
      except:
        error_type = 2
        error_message(error_type)
        #sys.exit(1)
        
      if error_type == 0:
        try:
          face_mesh = normalize_faces_from_mesh(cropped_face, 15000) 
          face_mesh = normalize_mesh(face_mesh)
          vertices  = face_mesh.vertices
          faces     = face_mesh.faces
          print(f"Vertices: {vertices.shape}, Faces: {faces.shape}")
          te1 = time.time()
          print("Data Preparation Time (s)            : ",te1-ts1)
          
          ts2 = time.time()
          print("Extrinsic Features ...")
          vertex_coordinates = get_vertex_coordinates(face_mesh)
          vertex_normals     = get_vertex_normals(face_mesh)
          dihedral_angles    = generate_dihedral_angles(face_mesh)
          te2 = time.time()
          print("Extrinsic Feature Extraction Time (s): ",te2-ts2)
          
          ts3 = time.time()
          print("Intrinsic Features ...")
          mesh_gaussian_curvature     = generate_gaussian_curvature(face_mesh)
          eigen_vectors, eigen_values = generate_cot_eigen_vectors(face_mesh, device, k)
          hks_features                = HKS(eigen_vectors, eigen_values, 25)

          vertex_coordinates = np.array(vertex_coordinates, copy=True)
          vertex_normals     = np.array(vertex_normals, copy=True)
          dihedral_angles    = np.array(dihedral_angles, copy=True)
          eigen_vectors      = np.array(eigen_vectors, copy=True)

          # The input features
          features = torch.cat([
              torch.from_numpy(vertex_coordinates).float(),
              torch.from_numpy(vertex_normals).float(),
              torch.from_numpy(dihedral_angles).float(),
              mesh_gaussian_curvature.float(),
              torch.from_numpy(eigen_vectors[:, 1:21]).float(),
              torch.from_numpy(hks_features).float()
          ], dim=1)
          
          print("features : ",features.shape)

          eigen_vectors     = eigen_vectors.astype(np.float32)  # Convert to float32 for compatibility
          features_G_tensor = torch.from_numpy(eigen_vectors)[:, :k].T.to(device) @ features.to(device)
          features_G        = torch.flatten(features_G_tensor, start_dim=0).tolist()

          print("Input features : ",len(features_G))
          te3 = time.time()
          print("Intrinsic Feature Extraction Time (s): ",te3-ts3)
        
        except:
          error_type = 3
          error_message(error_type)
          #sys.exit(1)
        
        if error_type == 0:
          try: 
            ts4 = time.time() 
            # Create inputs
            my_dataset = CustomInferenceDataset(features_G)
            # Create DataLoader
            inputloader = DataLoader(my_dataset, batch_size=1)
            
            # Use the loaded model for inference
            print("Inference ...")
            for batch in inputloader:
              with torch.no_grad():
                output    = model_loaded(batch)
                predicted = output.squeeze().item()
                if predicted >= 0.5 :
                  print("Model output : "+name_face_obj+" is Fake. (", predicted,")")
                else:
                  print("Model output : "+name_face_obj+" is Real. (", predicted,")")
  
            te4 = time.time()
            print("Inference Time (s)                   : ",te4-ts4)
            print("-" * 50)
          
          except:
            error_type = 4
            error_message(error_type)
            #sys.exit(1)
              
    end_time = time.time()
    print("Total Execution Time (s): ",end_time-start_time)
    
#-------------------------------------------------------------------------------
#------------------------------------------------------------------------------- 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data"  , type=str, help="Input data path")
    parser.add_argument("--feature_type", choices=["G0", "G1", "G2"], default="G0", help="Type of feature to use") 
    parser.add_argument("--model_name"  , choices=["Tabtransformer", "Mesh_MLP_MHA"], help="Model name", default="Tabtransformer")
    parser.add_argument('--is_cropped'  , type=int, help="If the 3D face is already cropped = 1 otherwise 0.", default=0)
    
    args = parser.parse_args()
    main(args)




