import numpy as np
import matplotlib.pyplot as plt
import trimesh
from trimesh import Trimesh
from shapely.geometry import Polygon, Point
#import mayavi.mlab as mlab

from crop_3Dface.compute_3d_landmarks import compute_landmarks_pc
from crop_3Dface.obj_handler import read_obj_file


def plot_face_mesh(mesh_plot):
  vertices_face = mesh_plot.vertices
  faces_face    = mesh_plot.faces
  mesh = mlab.triangular_mesh(vertices_face[:, 0], vertices_face[:, 1], vertices_face[:, 2], faces_face)
  mlab.show()

def plot_face(vertices_face, faces_face):
  mesh = mlab.triangular_mesh(vertices_face[:, 0], vertices_face[:, 1], vertices_face[:, 2], faces_face)
  mlab.show()
  
def plot_mesh_graph(vertices, faces):
  # Create a 3D plot
  fig = plt.figure(figsize=(16, 12))
  ax  = fig.add_subplot(111, projection="3d")
  
  # Plot the nodes (vertices)
  ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=3, c='b')  # adjust marker size (s) and color (c)
  
  # Plot the edges (faces)
  for face in faces:
      # Extract vertex indices for each face
      f0, f1, f2 = face
  
      # Plot lines between vertices using separate plot calls
      ax.plot(
          [vertices[f0, 0], vertices[f1, 0], vertices[f2, 0]],
          [vertices[f0, 1], vertices[f1, 1], vertices[f2, 1]],
          zs=[vertices[f0, 2], vertices[f1, 2], vertices[f2, 2]],
          color='gray',
          linewidth=0.5
      )
  
  # Set labels and title
  #ax.set_xlabel("X")
  #ax.set_ylabel("Y")
  #ax.set_zlabel("Z")
  ax.grid(False)
  #plt.title("")
  # Hide all axes
  plt.gca().axis("off")
  plt.show() 

def plot_3D_landmarks(lst_landmarks, lst_eye_l_ind, lst_eye_r_ind):
  indices = range(0, len(lst_landmarks))  # Create indices as numbers 0 to length-1
  lst_landmarks_new = []
 
  lst_eye_l = []
  lst_eye_r = []
  for i in range(len(lst_landmarks)):
    if i not in lst_eye_l_ind and i not in lst_eye_r_ind:
      lst_landmarks_new.append(lst_landmarks[i])
    else:
      if i in lst_eye_l_ind:
        lst_eye_l.append((lst_landmarks[i][0],lst_landmarks[i][1],lst_landmarks[i][2]))
      else:
        if i in lst_eye_r_ind:
          lst_eye_r.append((lst_landmarks[i][0],lst_landmarks[i][1],lst_landmarks[i][2]))
  
  # Extract coordinates
  x, y, z = zip(*lst_landmarks_new)
  xe_l, ye_l, ze_l = zip(*lst_eye_l)
  xe_r, ye_r, ze_r = zip(*lst_eye_r)
  # Create Polygon
  vertices_r = []
  vertices_l = []
  for i in range(len(lst_eye_r_ind)):
    p3dr = lst_landmarks[lst_eye_r_ind[i]]
    p3dl = lst_landmarks[lst_eye_l_ind[i]]
    vertices_r.append((p3dr[0],p3dr[1]))
    vertices_l.append((p3dl[0],p3dl[1]))
  my_polygon_l = Polygon(vertices_l)
  my_polygon_r = Polygon(vertices_r)
  # Extract x and y coordinates from the polygon vertices
  xl, yl = zip(*my_polygon_l.exterior.coords)
  xr, yr = zip(*my_polygon_r.exterior.coords)
  # Create a Matplotlib Polygon patch
  polygon_patch_l = plt.Polygon(xy=list(zip(xl, yl)), closed=True)
  polygon_patch_r = plt.Polygon(xy=list(zip(xr, yr)), closed=True)
  # Create the 2D plot
  fig, ax = plt.subplots(figsize=(16, 12))
  ax.scatter(x, y, s=5, c="b")  # adjust marker size (s) and color (c)
  ax.scatter(xe_l, ye_l, s=5, c="r") 
  ax.scatter(xe_r, ye_r, s=5, c="r")
  
  ax.add_patch(polygon_patch_l)
  ax.add_patch(polygon_patch_r)

  for i, p in enumerate(lst_landmarks):
      ax.text(p[0], p[1], indices[i], size=6)

  ax.grid(False)
  plt.gca().axis("off")
  #plt.show()
  
  # Create the 3D plot
  fig = plt.figure(figsize=(16, 12))
  ax = fig.add_subplot(111, projection="3d")
  # Plot points
  ax.scatter(x, y, z, s=5, c="b")  # adjust marker size (s) and color (c)
  ax.scatter(xe_l, ye_l, ze_l, s=5, c="r") 
  ax.scatter(xe_r, ye_r, ze_r, s=5, c="r")
  
  # Add labels 
  for i, p in enumerate(lst_landmarks):
      ax.text(p[0], p[1], p[2], indices[i], size=6)
  # Set labels and title
  #ax.set_xlabel("X")
  #ax.set_ylabel("Y")
  #ax.set_zlabel("Z")
  ax.grid(False)
  #plt.title("3D Landmarks with Numbers (Labels):")
  # Hide all axes
  plt.gca().axis("off")
  plt.show()

def get_bounding_box(lst_landmarks, mrg=0.0001, alpha=0.9, is_from_faceScape=0):
  # Find the minimum and maximum bounding box coordinates of the landmarks
  min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
  max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')
  for landmark in lst_landmarks:
    min_x = min(min_x, landmark[0])
    min_y = min(min_y, landmark[1])
    min_z = min(min_z, landmark[2])
    max_x = max(max_x, landmark[0])
    max_y = max(max_y, landmark[1])
    max_z = max(max_z, landmark[2])

  min_x = min_x - mrg 
  min_y = min_y - mrg 
  
  if is_from_faceScape== 1:
   min_z = 0.001
  else:
    min_z = min_z - mrg 
    
  max_x = max_x + mrg
  max_y = max_y + mrg
  max_z = max_z + mrg + alpha
  return min_x, min_y, min_z, max_x, max_y, max_z
  
def is_point_in_polygon(point, polygon):
  return polygon.contains(point)
   
def crop_face_from_obj(obj_file, lst_landmarks, lst_eye_l_ind, lst_eye_r_ind, mrg, alpha, is_from_faceScape):
  # Load the mesh from the OBJ file
  try:
    mesh     = trimesh.load(obj_file)
    vertices = mesh.vertices
    faces    = mesh.faces
  except FileNotFoundError:
    raise ValueError(f"OBJ file not found: {obj_file}")

  if len(lst_landmarks)==0:
    raise ValueError("Landmark list (lm_crop) cannot be empty.")

  # Find the minimum and maximum bounding box coordinates of the landmarks
  min_x, min_y, min_z, max_x, max_y, max_z = get_bounding_box(lst_landmarks, mrg, alpha, is_from_faceScape)
  
  #print("min_x, min_y, min_z :",min_x, min_y, min_z)
  #print("max_x, max_y, max_z :",max_x, max_y, max_z)
  #print("vertices : ",vertices.shape)
  #print("faces    : ",faces.shape)
  
  #----------------------------------------------------------------------------- 
  # Create Polygons for eyes
  lst_eye_l = []
  lst_eye_r = []
  for i in range(len(lst_eye_r_ind)):
    p3dr = lst_landmarks[lst_eye_r_ind[i]]
    p3dl = lst_landmarks[lst_eye_l_ind[i]]
    lst_eye_r.append((p3dr[0],p3dr[1]))
    lst_eye_l.append((p3dl[0],p3dl[1]))
  
  my_polygon_r = Polygon(lst_eye_r)
  my_polygon_l = Polygon(lst_eye_l)
  
  # Filter vertices within the Polygon
  cropped_vertices          = []
  cropped_vertices_indx_old = []
  my_vertices               = vertices.tolist()

  for i in range(len(my_vertices)):
    vertex  = my_vertices[i]
    if vertex[0]> min_x and vertex[1] > min_y and vertex[2] > min_z and vertex[0] < max_x and vertex[1] < max_y and vertex[2] < max_z: 
      if(not is_point_in_polygon(Point(vertex[0], vertex[1]), my_polygon_r) and (not is_point_in_polygon(Point(vertex[0], vertex[1]), my_polygon_l) )):
        cropped_vertices.append(vertex)
        cropped_vertices_indx_old.append(i)

  # print("Cropped_vertices : ",type(cropped_vertices), len(cropped_vertices))
  
  # Filter faces that use the filtered vertices
  cropped_faces = []
  my_faces      = faces.tolist()
  for face in my_faces:
    if face[0] in cropped_vertices_indx_old and face[1] in cropped_vertices_indx_old and face[2] in cropped_vertices_indx_old:
      ind0 = cropped_vertices_indx_old.index(face[0])
      ind1 = cropped_vertices_indx_old.index(face[1])
      ind2 = cropped_vertices_indx_old.index(face[2])
      cropped_faces.append([ind0, ind1, ind2])
      
  # print("Cropped_faces : ",type(cropped_faces), len(cropped_faces))
  
  cropped_vertices = np.array(cropped_vertices)
  cropped_faces    = np.array(cropped_faces)
  # Create a Trimesh object from vertices and faces
  cropped_face = Trimesh(cropped_vertices, cropped_faces)
  # cropped_face.export(file_obj="my_imgs/cropped_face.obj", file_type="obj")
  # Plot the mesh
  # cropped_face.show()
  
  return cropped_face
  
def remove_irrelevant_parts(face_mesh):
  # Perform connected component analysis
  components = face_mesh.split(only_watertight=False)
  # Print the number of unconnected parts
  # print("Number of unconnected parts:", len(components))

  max_vertices_ind = -1
  max_vertices_nb  = -1
  for i in range(len(components)):
    component = components[i]
    component_vertices = np.array(component.vertices)
    if component_vertices.shape[0] > max_vertices_nb:
      max_vertices_nb  = component_vertices.shape[0]
      max_vertices_ind = i

  vertices = np.array(components[max_vertices_ind].vertices)
  faces    = np.array(components[max_vertices_ind].faces)
  # Create a Trimesh object from vertices and faces
  face_mesh_result = Trimesh(vertices, faces) 
  return face_mesh_result

def count_X_lst(lst, x):
  count = 0
  for sub_lst in lst:
    for elm in sub_lst:
      if (elm == x):
          count = count + 1
  return count

def remove_eyes_outlier(lst_vertices, lst_faces, lst_landmarks, lst_eye_l_ind, lst_eye_r_ind, mrg=0.001):
  lst_eye_l = []
  lst_eye_r = []
  for i in range(len(lst_eye_r_ind)):
    p3dr = lst_landmarks[lst_eye_r_ind[i]]
    p3dl = lst_landmarks[lst_eye_l_ind[i]]
    lst_eye_r.append((p3dr[0],p3dr[1],p3dr[2]))
    lst_eye_l.append((p3dl[0],p3dl[1],p3dl[2]))

  min_xel, min_yel, min_zel, max_xel, max_yel, max_zel = get_bounding_box(lst_eye_l, mrg, 0.9)
  min_xer, min_yer, min_zer, max_xer, max_yer, max_zer = get_bounding_box(lst_eye_r, mrg, 0.9)

  # Filter vertices and faces within the bounding box
  lst_vertices_in_box = []
  for i in range(len(lst_vertices)):
    vertex  = lst_vertices[i]
    if  (vertex[0]> min_xel and vertex[1] > min_yel and vertex[0] < max_xel and vertex[1] < max_yel) or (vertex[0]> min_xer and vertex[1] > min_yer and vertex[0] < max_xer and vertex[1] < max_yer)  :
      lst_vertices_in_box.append(i)

  lst_faces_in_box     = []
  lst_faces_in_box_ind = []
  for i in range(len(lst_faces)):
    face  = lst_faces[i]
    if face[0] in lst_vertices_in_box or face[1] in lst_vertices_in_box or face[2] in lst_vertices_in_box:
      lst_faces_in_box.append(face)
      lst_faces_in_box_ind.append(i)

  # print("----------------------------------------------------------------------")
  # print("(Eyes) vertices in bbox : ",len(lst_vertices_in_box))
  # print("(Eyes) faces in bbox    : ",len(lst_faces_in_box))

  # Delete outliers (vertices and faces)
  lst_deleted_vertices = []
  lst_deleted_faces    = []
  bol                  = True
  while len(lst_vertices_in_box) > 0 and len(lst_faces_in_box) > 0 and bol:
    bol           = False
    tmp_lst_v     = []
    for i in range(len(lst_vertices_in_box)):
      nbr_faces = count_X_lst(lst_faces_in_box, lst_vertices_in_box[i])
      if nbr_faces < 3:
        bol = True
        # Remove the vertex at index i from the vertices within the bounding box
        lst_deleted_vertices.append(lst_vertices_in_box[i])
        removed_vertex = lst_vertices_in_box[i]
        
        tmp_lst_f     = []
        tmp_lst_f_ind = []
        for j in range(len(lst_faces_in_box)):
          face  = lst_faces_in_box[j]
          if removed_vertex in face:
            lst_deleted_faces.append(lst_faces_in_box_ind[j])
          else:
            tmp_lst_f.append(lst_faces_in_box[j])
            tmp_lst_f_ind.append(lst_faces_in_box_ind[j])
        
        lst_faces_in_box     = tmp_lst_f.copy()
        lst_faces_in_box_ind = tmp_lst_f_ind.copy()

      else:
        tmp_lst_v.append(lst_vertices_in_box[i])

    lst_vertices_in_box = tmp_lst_v.copy()

  new_vertices          = []
  lst_vertices_indx_old = []
  for i in range(len(lst_vertices)):
    if i not in lst_deleted_vertices:
      new_vertices.append(lst_vertices[i])
      lst_vertices_indx_old.append(i)
  
  new_faces = []
  for i in range(len(lst_faces)):
    if i not in lst_deleted_faces:
      face = lst_faces[i]
      ind0 = lst_vertices_indx_old.index(face[0])
      ind1 = lst_vertices_indx_old.index(face[1])
      ind2 = lst_vertices_indx_old.index(face[2])
      new_faces.append([ind0, ind1, ind2])

  return new_vertices, new_faces

def remove_face_outlier(lst_vertices, lst_faces, lst_landmarks, sub_bbox_face, mrg, alpha): 
  
  min_xf1, min_yf1, min_zf1, max_xf1, max_yf1, max_zf1 = get_bounding_box(lst_landmarks, mrg, alpha)

  lst_landmarks_sub_face = []
  for i in sub_bbox_face:
    lst_landmarks_sub_face.append(lst_landmarks[i])

  min_xf2, min_yf2, min_zf2, max_xf2, max_yf2, max_zf2 = get_bounding_box(lst_landmarks_sub_face, 0.0, 0.0)

  # Filter vertices and faces within the bounding box
  lst_vertices_in_box = []
  for i in range(len(lst_vertices)):
    vertex  = lst_vertices[i]
    if (vertex[0]> min_xf1 and vertex[1] > min_yf1 and vertex[0] < max_xf1 and vertex[1] < max_yf1) and not (vertex[0]> min_xf2 and vertex[1] > min_yf2 and vertex[0] < max_xf2 and vertex[1] < max_yf2):
      lst_vertices_in_box.append(i)

  lst_faces_in_box     = []
  lst_faces_in_box_ind = []
  for i in range(len(lst_faces)):
    face  = lst_faces[i]
    if face[0] in lst_vertices_in_box or face[1] in lst_vertices_in_box or face[2] in lst_vertices_in_box:
      lst_faces_in_box.append(face)
      lst_faces_in_box_ind.append(i)

  # print("----------------------------------------------------------------------")
  # print("(Face) vertices in bbox : ",len(lst_vertices_in_box))
  # print("(Face) faces in bbox    : ",len(lst_faces_in_box))

  # Delete outliers (vertices and faces)
  lst_deleted_vertices = []
  lst_deleted_faces    = []
  bol                  = True
  while len(lst_vertices_in_box) > 0 and len(lst_faces_in_box) > 0 and bol:
    bol           = False
    tmp_lst_v     = []
    for i in range(len(lst_vertices_in_box)):
      nbr_faces = count_X_lst(lst_faces_in_box, lst_vertices_in_box[i])
      if nbr_faces < 3:
        bol = True
        # Remove the vertex at index i from the vertices within the bounding box
        lst_deleted_vertices.append(lst_vertices_in_box[i])
        removed_vertex = lst_vertices_in_box[i]
        
        tmp_lst_f     = []
        tmp_lst_f_ind = []
        for j in range(len(lst_faces_in_box)):
          face  = lst_faces_in_box[j]
          if removed_vertex in face:
            lst_deleted_faces.append(lst_faces_in_box_ind[j])
          else:
            tmp_lst_f.append(lst_faces_in_box[j])
            tmp_lst_f_ind.append(lst_faces_in_box_ind[j])
        
        lst_faces_in_box     = tmp_lst_f.copy()
        lst_faces_in_box_ind = tmp_lst_f_ind.copy()

      else:
        tmp_lst_v.append(lst_vertices_in_box[i])

    lst_vertices_in_box = tmp_lst_v.copy()

  new_vertices          = []
  lst_vertices_indx_old = []
  for i in range(len(lst_vertices)):
    if i not in lst_deleted_vertices:
      new_vertices.append(lst_vertices[i])
      lst_vertices_indx_old.append(i)
  
  new_faces = []
  for i in range(len(lst_faces)):
    if i not in lst_deleted_faces:
      face = lst_faces[i]
      ind0 = lst_vertices_indx_old.index(face[0])
      ind1 = lst_vertices_indx_old.index(face[1])
      ind2 = lst_vertices_indx_old.index(face[2])
      new_faces.append([ind0, ind1, ind2])

  return new_vertices, new_faces

def plot_bounding_box(ax, xmin, ymin, zmin, xmax, ymax, zmax):
  # Eight corners of the box
  corners = [(xmin, ymin, zmin), (xmax, ymin, zmin),
             (xmax, ymax, zmin), (xmin, ymax, zmin),
             (xmin, ymin, zmax), (xmax, ymin, zmax),
             (xmax, ymax, zmax), (xmin, ymax, zmax)]

  # Plot the faces as triangles with a wireframe style
  ax.plot_trisurf(
      [coord[0] for coord in corners],
      [coord[1] for coord in corners],
      [coord[2] for coord in corners],
      linewidth=1,
      alpha=0.1,
      edgecolor='red')  # Adjust color for the edges

def face_outlier_plot_bbox(lst_landmarks, sub_bbox_face, mrg, alpha, lst_eye_l_ind, lst_eye_r_ind): 
  
  min_xf1, min_yf1, min_zf1, max_xf1, max_yf1, max_zf1 = get_bounding_box(lst_landmarks, mrg, alpha)

  lst_landmarks_sub_face = []
  for i in sub_bbox_face:
    lst_landmarks_sub_face.append(lst_landmarks[i])

  min_xf2, min_yf2, min_zf2, max_xf2, max_yf2, max_zf2 = get_bounding_box(lst_landmarks_sub_face, 0.0, 0.0)


  lst_eye_l = []
  lst_eye_r = []
  for i in range(len(lst_eye_r_ind)):
    p3dr = lst_landmarks[lst_eye_r_ind[i]]
    p3dl = lst_landmarks[lst_eye_l_ind[i]]
    lst_eye_r.append((p3dr[0],p3dr[1],p3dr[2]))
    lst_eye_l.append((p3dl[0],p3dl[1],p3dl[2]))

  min_xel, min_yel, min_zel, max_xel, max_yel, max_zel = get_bounding_box(lst_eye_l, mrg, alpha)
  min_xer, min_yer, min_zer, max_xer, max_yer, max_zer = get_bounding_box(lst_eye_r, mrg, alpha)
  
  # Extract coordinates
  x, y, z = zip(*lst_landmarks)
  xe_l, ye_l, ze_l = zip(*lst_eye_l)
  xe_r, ye_r, ze_r = zip(*lst_eye_r)

  # Plot bbox
  # Create the 3D plot
  fig = plt.figure(figsize=(16, 12))
  ax = fig.add_subplot(111, projection="3d")
  # Plot points
  ax.scatter(x, y, z, s=5, c="b")  # adjust marker size (s) and color (c)
  ax.scatter(xe_l, ye_l, ze_l, s=5, c="r") 
  ax.scatter(xe_r, ye_r, ze_r, s=5, c="r")
  
  # Add labels 
  indices = range(0, len(lst_landmarks))
  for i, p in enumerate(lst_landmarks):
      ax.text(p[0], p[1], p[2], indices[i], size=6)

  # Plot bbox
  plot_bounding_box(ax, min_xf1, min_yf1, min_zf1, max_xf1, max_yf1, max_zf1)
  plot_bounding_box(ax, min_xf2, min_yf2, min_zf2, max_xf2, max_yf2, max_zf2)
  plot_bounding_box(ax, min_xel, min_yel, min_zel, max_xel, max_yel, max_zel)
  plot_bounding_box(ax, min_xer, min_yer, min_zer, max_xer, max_yer, max_zer)

  # Set labels and title
  #ax.set_xlabel("X")
  #ax.set_ylabel("Y")
  #ax.set_zlabel("Z")
  ax.grid(False)
  #plt.title("3D Landmarks with Numbers (Labels):")
  # Hide all axes
  plt.gca().axis("off")
  plt.show()
    


