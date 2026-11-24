import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Tuple, Union
import math
import numpy as np
import cv2
import dearpygui.dearpygui as dpg   

def clamp_number(minimum, maximum,int):
  return max(minimum, min(maximum, int))


def blank_image(height, width):
  frame=np.zeros((height, width, 4), dtype=np.uint8)
  frame[:, :, 3] = 1.0
  return frame


def is_valid_normalized_value(value: float) -> bool:
  """Checks if the float value is between 0 and 1."""
  return (value > 0) and (value < 1)


def normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  
  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


def init_dearpygui(window_name="temp",width=800, height=600):
  dpg.create_context()
  dpg.create_viewport(title=window_name, width=width, height=height)
  dpg.setup_dearpygui()


def init_camera(width,heigt):
  cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, heigt)
  return cap


def init_face_track():
  landmarker_base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
  landmarker_options = vision.FaceLandmarkerOptions(base_options=landmarker_base_options, output_face_blendshapes=True,output_facial_transformation_matrixes=True,num_faces=1,running_mode=vision.RunningMode.IMAGE)
  landmarker__detector=vision.FaceLandmarker.create_from_options(landmarker_options)
  return landmarker__detector


def detect_face(image,landmarker__detector):
  return landmarker__detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))


def convert_for_dpg(image):
  # Convert BGR to RGBA
  image_dpg=cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
  # Convert uint8 [0, 255] to float32 [0, 1]
  image_dpg = image_dpg.astype(np.float32) / 255.0
  image_dpg.flatten()
  return image_dpg
   

def annotate_image(image, detection_results, color=(255,255,255), thickness=1, radius=1):
  face_annotated=image.copy()
  height, width, _ = image.shape
  for landmarks in detection_results.face_landmarks:
      for landmark in landmarks:
        keypoint_px = normalized_to_pixel_coordinates(landmark.x, landmark.y, width, height)
        cv2.circle(face_annotated, keypoint_px, thickness, color, int(radius))
  return face_annotated






def update_blendshape_2d_sliders(slider_2ds,detection_result,alias_blendshapes_2D_slider):
  for blendshape in detection_result.face_blendshapes:
    for shape in blendshape:
      slider_2d_x=0.5
      slider_2d_y=0.5
      if shape.category_name in list(alias_blendshapes_2D_slider.keys()):
        blendshape_to_update=alias_blendshapes_2D_slider[shape.category_name]
        
        if blendshape_to_update == "eyeLookLeft_dir":
          
          slider_2d_y-=detection_result.face_blendshapes[0][11].score / 2
          slider_2d_y+=detection_result.face_blendshapes[0][17].score / 2
          slider_2d_x-=detection_result.face_blendshapes[0][13].score / 2
          slider_2d_x+=detection_result.face_blendshapes[0][15].score / 2

          
          slider_2ds[blendshape_to_update].update_2d_slider(slider_2d_x,slider_2d_y)
          continue
        if blendshape_to_update =="eyeLookRight_dir":
          slider_2d_y-=detection_result.face_blendshapes[0][12].score / 2
          slider_2d_y+=detection_result.face_blendshapes[0][18].score / 2
          slider_2d_x+=detection_result.face_blendshapes[0][14].score / 2
          slider_2d_x-=detection_result.face_blendshapes[0][16].score / 2

          slider_2ds[blendshape_to_update].update_2d_slider(slider_2d_x,slider_2d_y)
          continue
        if blendshape_to_update =="jaw_dir":
          slider_2d_y=1-detection_result.face_blendshapes[0][25].score
          slider_2d_x+=detection_result.face_blendshapes[0][24].score * 5
          slider_2d_x-=detection_result.face_blendshapes[0][26].score * 5



          slider_2ds[blendshape_to_update].update_2d_slider(slider_2d_x,slider_2d_y)
          continue
        



def update_blendshape_sliders(sliders, detection_result,alias_blendshapes_2D_slider):
  for blendshape in detection_result.face_blendshapes:
    for shape in blendshape:
      try:
        #if not shape.category_name in list(alias_blendshapes_2D_slider.keys()):
        dpg.set_value(shape.category_name,shape.score)
        sliders[shape.category_name]=shape.score
      except Exception as e:
        print(shape.category_name)
        print(sliders)
        
