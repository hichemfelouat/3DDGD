## Identifying Dense 3D Facial Landmarks by Utilizing 2D as an Intermediate

This repo contains the code that used in our paper: [Identifying Dense 3D Facial Landmarks by Utilizing 2D as an Intermediate](https://doi.org/10.31224/3320).

### Usage

```py
import numpy as np

from compute_3d_landmarks import compute_landmarks_pc
from obj_handler import read_obj_file

pointcloud_path = "123456.obj"
pointcloud= np.asarray(read_obj_file(pointcloud_path ))
lm_nocrop, lm_crop = compute_landmarks_pc(
    pointcloud, 
    img_size=256, 
    cropping = True, 
    img_save_nocrop = "nocropping.png", 
    img_save_crop = "cropping.png", 
    visibility_radius = 0, 
    visibility_obj = "123456 fr.obj"
    )
```

### Authors
**Elwin Li¹, Tahlia Wu², Paul Kronlund³**

¹ Mountain View High School, Mountain View, CA\
² Cupertino High School, California, CA\
³ Lycée Racine, Paris, France
