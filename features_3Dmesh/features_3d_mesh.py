import trimesh
from trimesh import Trimesh
import numpy as np
import torch
import open3d as o3d
import os
import sys
import igl
import robust_laplacian

from scipy.sparse.linalg import eigsh
from scipy import sparse

from glob import glob
import pandas as pd

import time
start_time = time.time()

def normalize_faces(mesh_path, target_faces_count=15000):
  # Load the mesh directly with trimesh for better performance
  trimesh_mesh = trimesh.load(mesh_path)
  
  # Convert to Open3D mesh only once
  mesh           = o3d.geometry.TriangleMesh()
  mesh.vertices  = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
  mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
  
  # Simplify the mesh
  simplified_mesh_open3d = mesh.simplify_quadric_decimation(target_faces_count)
  
  # Extract vertices and faces from Open3D mesh
  vertices = np.asarray(simplified_mesh_open3d.vertices)
  faces    = np.asarray(simplified_mesh_open3d.triangles)

  # Create Trimesh mesh
  simplified_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
  return simplified_mesh

def normalize_faces_from_mesh(trimesh_mesh, target_faces_count=15000):
  # Create an open3d mesh
  mesh = o3d.geometry.TriangleMesh()
  
  # Assign vertices and faces from trimesh to open3d mesh
  mesh.vertices  = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
  mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
  
  # Simplify the mesh
  simplified_mesh_open3d = mesh.simplify_quadric_decimation(target_faces_count)
  
  # Extract vertices and faces from Open3D mesh
  vertices = np.asarray(simplified_mesh_open3d.vertices)
  faces    = np.asarray(simplified_mesh_open3d.triangles)

  # Create Trimesh mesh
  simplified_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
  return simplified_mesh

def normalize_mesh(mesh):
  # Optimize by doing operations in-place when possible
  vertices = mesh.vertices
  min_vals = vertices.min(axis=0)
  vertices = vertices - min_vals
  max_val  = vertices.max()

  if max_val > 0:  # Avoid division by zero
    vertices = vertices / max_val
  mesh.vertices = vertices
  return mesh

def get_vertex_coordinates(mesh):
  # Access vertex coordinates directly
  return mesh.vertices.view(np.ndarray)
  
def get_vertex_normals(mesh):
  # Compute vertex normals only if needed
  if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
    mesh.vertex_normals = mesh.vertex_normals

  return mesh.vertex_normals

def generate_dihedral_angles(mesh):
  # Pre-allocate matrix with correct size
  vertex_count = mesh.vertices.shape[0]
  face_count   = mesh.faces.shape[0]
  vertex_faces_adjacency_matrix = np.zeros((vertex_count, face_count), dtype=np.float32)
  
  # Populate the adjacency matrix more efficiently
  for vertex, faces in enumerate(mesh.vertex_faces):
    valid_faces = faces[faces != -1]
    if len(valid_faces) > 0:
      vertex_faces_adjacency_matrix[vertex, valid_faces] = 1
  
  # Pre-allocate dihedral angles list with correct structure
  dihedral_angle = [[] for _ in range(face_count)]
  face_adjacency = mesh.face_adjacency
  face_normals   = mesh.face_normals
  
  # Compute dihedral angles using vectorized operations where possible
  for adj_faces in face_adjacency:
    angle = np.abs(np.dot(face_normals[adj_faces[0]], face_normals[adj_faces[1]]))
    dihedral_angle[adj_faces[0]].append(angle)
    dihedral_angle[adj_faces[1]].append(angle)
  
  # Process non-watertight mesh more efficiently
  for i, angles in enumerate(dihedral_angle):
    # Pad to ensure exactly 3 values
    while len(angles) < 3:
      angles.append(1)
  
  face_dihedral_angle = np.array(dihedral_angle, dtype=np.float32).reshape(-1, 3)
  V_dihedral_angles   = vertex_faces_adjacency_matrix.dot(face_dihedral_angle)
  
  return V_dihedral_angles

def generate_gaussian_curvature(mesh):
  # Compute Gaussian curvature
  mesh_gaussian_curvature = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, 0)
  
  if np.isnan(mesh_gaussian_curvature).sum() > 0 or np.isinf(mesh_gaussian_curvature).sum() > 0:
    print("gaussian_curvature errors, exit")
    sys.exit()
  
  # Calculate and normalize using vectorized operations
  gaussian_curvature = torch.exp(-(torch.tensor(mesh_gaussian_curvature, dtype=torch.float32)))
  
  if torch.isnan(gaussian_curvature).sum() > 0 or torch.isinf(gaussian_curvature).sum() > 0:
    print('gaussian_curvature errors, exit')
    sys.exit()
  
  min_val = gaussian_curvature.min()
  max_val = gaussian_curvature.max()
  gaussian_curvature = ((gaussian_curvature - min_val) / (max_val - min_val)).unsqueeze(1)
  
  return gaussian_curvature

def generate_cot_eigen_vectors(mesh, device="cuda", k=64):
  # Get the cotangent Laplacian matrix (returns a sparse matrix)
  cot = -igl.cotmatrix(mesh.vertices, mesh.faces)
  
  # igl.cotmatrix returns a sparse matrix, so we can directly check symmetry with .data
  if not sparse.issparse(cot):
    # Convert to sparse if not already
    cot = sparse.csr_matrix(cot)
  
  # Check if matrix is symmetric by comparing non-zero entries
  # in the difference between the matrix and its transpose
  diff         = cot - cot.transpose()
  is_symmetric = (abs(diff).sum() < 1e-10)
  
  if not is_symmetric:
    print("Using robust Laplacian due to non-symmetric cotangent matrix")
    L, M = robust_laplacian.mesh_laplacian(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    cot = L  # L is already a sparse matrix
  
  # Use scipy.sparse.linalg.eigsh for much faster partial eigendecomposition
  # Ensure we don't request more eigenvectors than possible
  k = min(k, cot.shape[0]-2)  # Keep at least 2 less than matrix dimension for safety
  
  try:
    eigen_values, eigen_vectors = eigsh(cot, k=k, which='SM')
  except Exception as e:
    print(f"Error in eigendecomposition: {e}")
    print("Falling back to dense eigendecomposition (slower)")
    # Fall back to dense eigendecomposition if sparse fails
    dense_cot = cot.toarray()
    eigen_values, eigen_vectors = np.linalg.eigh(dense_cot)
    # Take only the k smallest eigenvalues/vectors
    idx           = np.argsort(eigen_values)[:k]
    eigen_values  = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]
  
  # Sort by eigenvalues (smallest to largest)
  idx           = np.argsort(eigen_values)
  eigen_values  = eigen_values[idx]
  eigen_vectors = eigen_vectors[:, idx]
  
  return eigen_vectors, eigen_values

def HKS(eigen_vector, eigen_values, nbr_samples=64):
  # Ensure eigen_values is at least length 2 to avoid indexing errors
  if len(eigen_values) < 2:
    print("Not enough eigenvalues for HKS computation. Need at least 2.")
    sys.exit()
  
  # Use fewer time samples and focus on the most informative eigenvalues
  t_min = 4 * np.log(10) / eigen_values.max()
  t_max = 4 * np.log(10) / eigen_values[1]
  
  # Use float32 instead of float128 for better performance
  ts = np.linspace(t_min, t_max, num=nbr_samples, dtype=np.float32)
  
  # Pre-compute the exponential decay factors
  exp_factors = np.exp(-eigen_values[:, None] * ts[None, :])
  
  # Compute all HKS features in one shot
  hkss = (eigen_vector[:, :, None]**2) * exp_factors[None, :, :]
  hks = np.sum(hkss, axis=1)
  
  # Convert to tensor once
  hks_tensor = torch.tensor(hks, dtype=torch.float32)
  
  # Extract and normalize only the needed time samples
  selected_indices = [1, 2, 3, 4, 5, 8, 10, 15, 20]
  valid_indices    = [i for i in selected_indices if i < hks_tensor.shape[1]]
  
  if len(valid_indices) == 0:
    print("No valid HKS time samples, exit")
    sys.exit()
  
  hks_cat = None
  for idx in valid_indices:
    if idx >= hks_tensor.shape[1]:
      continue
    
    feature = hks_tensor[:, idx]
    min_val = feature.min()
    max_val = feature.max()
    
    # Avoid division by zero
    if max_val - min_val > 1e-10:
      normalized = ((feature - min_val) / (max_val - min_val)).unsqueeze(1)
    else:
      normalized = torch.zeros_like(feature).unsqueeze(1)
    
    if hks_cat is None:
      hks_cat = normalized
    else:
      hks_cat = torch.cat((hks_cat, normalized), dim=1)
  
  if torch.isnan(hks_cat).sum() > 0 or torch.isinf(hks_cat).sum() > 0:
    print("hks errors, exit")
    sys.exit()
  
  return hks_cat.numpy()

  

