import math
import mediapipe as mp
import numpy as np
import cv2
import time
import helpers
import dearpygui.dearpygui as dpg 
from slider_2d import add_2d_slider
width=720
height=1280 
thickness=1
radius=1
color=(255, 255, 255,255)  # red


#MARGIN = 10  # pixels
#ROW_SIZE = 10  # pixels
#FONT_SIZE = 1
#FONT_THICKNESS = 1
#text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
#cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    





blendshapes_slider={
"eyeLookDownLeft":0,
"eyeLookUpLeft":0,
"eyeLookInLeft":0,
"eyeLookOutLeft":0,

"eyeLookDownRight":0,
"eyeLookUpRight":0,
"eyeLookInRight":0,
"eyeLookOutRight":0,

"jawOpen":0,
"jawLeft":0,
"jawRight":0,
"_neutral":0,
              "browDownLeft":0,
              "browDownRight":0,
              "browInnerUp":0,
              "browOuterUpLeft":0,
              "browOuterUpRight":0,
              "cheekPuff":0, 
              "cheekSquintLeft":0, 
              "cheekSquintRight":0, 
              "eyeBlinkLeft":0, 
              "eyeBlinkRight":0,
              "eyeSquintLeft":0, 
              "eyeSquintRight":0, 
              "eyeWideLeft":0, 
              "eyeWideRight":0, 
              "jawForward":0,
              "mouthClose":0, 
              "mouthDimpleLeft":0, 
              "mouthDimpleRight":0, 
              "mouthFrownLeft":0, 
              "mouthFrownRight":0, 
              "mouthFunnel":0, 
              "mouthLeft":0, 
              "mouthLowerDownLeft":0, 
              "mouthLowerDownRight":0, 
              "mouthPressLeft":0, 
              "mouthPressRight":0, 
              "mouthPucker":0, 
              "mouthRight":0, 
              "mouthRollLower":0, 
              "mouthRollUpper":0, 
              "mouthShrugLower":0, 
              "mouthShrugUpper":0, 
              "mouthSmileLeft":0, 
              "mouthSmileRight":0, 
              "mouthStretchLeft":0, 
              "mouthStretchRight":0, 
              "mouthUpperUpLeft":0, 
              "mouthUpperUpRight":0, 
              "noseSneerLeft":0, 
              "noseSneerRight":0}


#eyeLookLeft_dir (eyeLookDownLeft eyeLookInLeft eyeLookOutLeft eyeLookUpLeft)
#eyeLookRight_dir (eyeLookDownRight eyeLookInRight eyeLookOutRight eyeLookUpRight)
#jaw_dir (jawLeft jawOpen jawRight)
alias_blendshapes_2D_slider={"eyeLookDownLeft":"eyeLookLeft_dir",
                            "eyeLookUpLeft":"eyeLookLeft_dir",
                            "eyeLookInLeft":"eyeLookLeft_dir",
                            "eyeLookOutLeft":"eyeLookLeft_dir",
                             
                            "eyeLookDownRight":"eyeLookRight_dir",
                            "eyeLookUpRight":"eyeLookRight_dir",
                            "eyeLookInRight":"eyeLookRight_dir",
                            "eyeLookOutRight":"eyeLookRight_dir",
                             
                            "jawOpen":"jaw_dir",
                            "jawLeft":"jaw_dir",
                            "jawRight":"jaw_dir"}

blendshapes_2D_slider={"jaw_dir":None,"eyeLookLeft_dir":None,"eyeLookRight_dir":None}


cap=helpers.init_camera(width,height)
landmarker__detector=helpers.init_face_track()
helpers.init_dearpygui(window_name="test",width=800, height=600)

ret, image = cap.read()
if not ret:
  raise Exception("camera closed")
height,width,_=image.shape
detection_result=helpers.detect_face(image,landmarker__detector)
t=0
for blendshape in detection_result.face_blendshapes:
    for shape in blendshape:
      if shape.category_name in list(alias_blendshapes_2D_slider.keys()):
        print(shape.category_name)
        print(t)
      t+=1



initial_frame = np.zeros((height, width, 4), dtype=np.float32)
initial_frame[:, :, 3] = 1.0

with dpg.texture_registry():
    dpg.add_raw_texture(
        width=width,
        height=height,
        default_value=initial_frame.ravel(),
        format=dpg.mvFormat_Float_rgba,
        tag="camera_texture"
    )
    dpg.add_raw_texture(
            width=width,
            height=height,
            default_value=initial_frame.ravel(),
            format=dpg.mvFormat_Float_rgba,
            tag="mapped_face_points_texture"
        )







with dpg.window(label="Example Window",tag="Primary_Window"):
  with dpg.group(horizontal=True):
    dpg.add_image(texture_tag="camera_texture")
    dpg.add_image(texture_tag="mapped_face_points_texture")

  with dpg.group(horizontal=True):
    for blendshape in blendshapes_2D_slider.keys():
      slider_2d=add_2d_slider(100,100,label=blendshape,tag=blendshape,interactable=False,normalized_io=True)
      blendshapes_2D_slider[blendshape]=slider_2d 

  for blendshape in blendshapes_slider.keys():
    dpg.add_progress_bar(overlay=blendshape,tag=blendshape)

  

dpg.show_viewport()
#dpg.show_item_registry()
#dpg.show_metrics()
dpg.set_primary_window("Primary_Window", True)

try:
  while dpg.is_dearpygui_running():

    ret, image = cap.read()
    if not ret:
      raise Exception("camera closed")
    
    mapped_face_points=helpers.blank_image(height,width)

    detection_result=helpers.detect_face(image,landmarker__detector)

    # Update the blendshape sliders
    helpers.update_blendshape_sliders(blendshapes_slider, detection_result,alias_blendshapes_2D_slider)
    helpers.update_blendshape_2d_sliders(blendshapes_2D_slider,detection_result,alias_blendshapes_2D_slider)

    face_annotated=helpers.annotate_image(mapped_face_points, detection_result, color)

    
    image_dpg=helpers.convert_for_dpg(image)
    face_annotated_dpg=helpers.convert_for_dpg(face_annotated)

    
    # Update the texture
    dpg.set_value("camera_texture", image_dpg)
    
    dpg.set_value("mapped_face_points_texture", face_annotated_dpg)
    dpg.render_dearpygui_frame()

finally:
  cap.release()
  dpg.destroy_context()