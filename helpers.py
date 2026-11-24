from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import dearpygui.dearpygui as dpg
from typing import Tuple, Union
import mediapipe as mp
import numpy as np
import math
import cv2
   

def clamp_number(minimum, maximum,int):
  return max(minimum, min(maximum, int))


def is_valid_normalized_value(value: float) -> bool:
  """Checks if the float value is between 0 and 1."""
  return (value > 0) and (value < 1)


def normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  
  if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


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
   

def annotate_image(image, detection_results, color=(255,255,255), thickness=1, radius=1):
  face_annotated=image.copy()
  height, width, _ = image.shape
  for landmarks in detection_results.face_landmarks:
      for landmark in landmarks:
        keypoint_px = normalized_to_pixel_coordinates(landmark.x, landmark.y, width, height)
        cv2.circle(face_annotated, keypoint_px, thickness, color, int(radius))
  return face_annotated

