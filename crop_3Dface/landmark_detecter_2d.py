import mediapipe as mp
import numpy as np

def get_landmarks_mp(image: np.array):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    height, width, _ = image.shape
    result = face_mesh.process(image)
    lm_indices = [33, 133, 362, 263, 2, 76, 292, 4, 0, 17, 152]
    facial_landmarks = []
    if result.multi_face_landmarks is None:
      rot_image = np.flip(image, axis = 0)
      rot_image = np.ascontiguousarray(rot_image, dtype=np.uint8)
      height, width, _ = rot_image.shape
      result = face_mesh.process(rot_image)
      if result.multi_face_landmarks is None :
        return None

      facial_landmarks = result.multi_face_landmarks[0]
      used_landmarks = []
      for i in range(468):
          pt = facial_landmarks.landmark[i]
          x = int(pt.x * width)
          y = height - int(pt.y * height)
          used_landmarks.append((x, y))
    else: 
      facial_landmarks = result.multi_face_landmarks[0] 
      used_landmarks = []
      for i in range(468):
          pt = facial_landmarks.landmark[i]
          x = int(pt.x * width)
          y = int(pt.y * height)
          used_landmarks.append((x, y))    
    return used_landmarks
