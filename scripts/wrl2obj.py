from glob import glob
import subprocess as sp
import time

start_time = time.time()

def parse_wrl(wrl_file):
    vertices = []
    faces    = []
    in_coord = False
    in_coord_index = False

    with open(wrl_file, 'r') as file:
        for line in file:
            line = line.strip()

            # Look for the start of vertex coordinates (Coordinate node)
            if "coord Coordinate {" in line:
                in_coord = True

            # Look for the start of the point list (vertex list)
            if in_coord and "point [" in line:
                in_coord = True
                continue

            # Look for the end of the point list
            if in_coord and "]" in line:
                in_coord = False
                continue

            # Process vertex coordinates
            if in_coord:
                try:
                    # Extract vertex coordinates
                    vertex = [float(v) for v in line.replace(',', ' ').split() if v.strip()]
                    if len(vertex) == 3:
                        vertices.append(vertex)
                except ValueError:
                    # Ignore lines that cannot be converted to float
                    continue

            # Look for the start of the face indices (coordIndex)
            if "coordIndex [" in line:
                in_coord_index = True
                continue

            # Look for the end of the face indices list
            if in_coord_index and "]" in line:
                in_coord_index = False
                continue

            # Process face indices
            if in_coord_index:
                # Split on both commas and spaces, and ignore '-1'
                face = [int(f) for f in line.replace(',', ' ').split() if f.strip() and f != '-1']
                if face:
                    faces.append(face)

    return vertices, faces


def write_obj(obj_file, vertices, faces):
    with open(obj_file, 'w') as file:
        # Write vertices to OBJ format
        for vertex in vertices:
            file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

        # Write faces to OBJ format (OBJ uses 1-based indexing)
        for face in faces:
            face_indices = [f + 1 for f in face]
            file.write(f"f {' '.join(map(str, face_indices))}\n")


def convert_wrl_to_obj(wrl_file, obj_file):
    vertices, faces = parse_wrl(wrl_file)
    write_obj(obj_file, vertices, faces)
    #print(f"Conversion completed: {obj_file}")

#-------------------------------------------------------------------------------
path_ids_folders = glob("../Real_wrl/*")
save_folder = "../Real/"

for i in range(len(path_ids_folders)):
  path_id = path_ids_folders[i]
  name_id = path_id.split("/")[-1]
  print("---------------------------------------------------------------------")
  print("id = : ",name_id)
  cmd    = "mkdir "+save_folder+name_id
  output = sp.getoutput(cmd)
  
  path_objs = glob(path_id+"/*.wrl")
  for j in range(len(path_objs)):
    wrl_file_path = path_objs[j]
    name_obj = wrl_file_path.split("/")[-1].split(".")[0]

    obj_file_path = save_folder+name_id+"/"+name_obj+".obj"
    convert_wrl_to_obj(wrl_file_path, obj_file_path)

print("---------------------------------------------------------------------")
end_time = time.time()
print("The time of execution is (in s) : ",end_time-start_time)
print("End ...")

