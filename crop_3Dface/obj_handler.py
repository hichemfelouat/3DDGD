import numpy as np
import math
import sys

def read_obj_file(obj_file):
    vertices = []
    with open(obj_file, 'r') as file:
        for line in file:
            components = line.strip().split()
            if len(components) > 0 and components[0] == 'v':  
                vertex = list(map(float, components[1:4])) 
                vertices.append(vertex)
    return vertices

#input is 1 x 3n numpy array and output is in format of n by 3 text for obj file type
def npy2obj(npy, obj): 
    matrix = npy
    numPoints = int(matrix.shape[0])
    #print(numPoints)

    with open(obj, 'w') as f:
        for i in range(numPoints):
            f.write("v " + repr(matrix[i , 0]) + " " + repr(matrix[i , 1]) + " " + repr(matrix[i , 2]) + "\n")