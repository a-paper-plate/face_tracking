from slider_2d import add_2d_slider
import dearpygui.dearpygui as dpg 
import mediapipe as mp
import numpy as np
import helpers
import GUI
width=720
height=1280 
color=(255, 255, 255,255)
    

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

"""eyeLookLeft_dir (eyeLookDownLeft eyeLookInLeft eyeLookOutLeft eyeLookUpLeft)
eyeLookRight_dir (eyeLookDownRight eyeLookInRight eyeLookOutRight eyeLookUpRight)
jaw_dir (jawLeft jawOpen jawRight)"""

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
GUI.init_dearpygui(window_name="test",width=800, height=600)

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







GUI.init_texture_registry(height,width)

GUI.init_UI(blendshapes_slider,blendshapes_2D_slider)

  

dpg.show_viewport()
#dpg.show_item_registry()
#dpg.show_metrics()
dpg.set_primary_window("Primary_Window", True)

try:
  while dpg.is_dearpygui_running():

    ret, image = cap.read()
    if not ret:
      raise Exception("camera closed")
    
    mapped_face_points=GUI.blank_image(height,width)

    detection_result=helpers.detect_face(image,landmarker__detector)

    # Update the blendshape sliders
    GUI.update_blendshape_sliders(blendshapes_slider, detection_result,alias_blendshapes_2D_slider)
    GUI.update_blendshape_2d_sliders(blendshapes_2D_slider,detection_result,alias_blendshapes_2D_slider)

    face_annotated=helpers.annotate_image(mapped_face_points, detection_result, color)
    image_dpg=GUI.convert_for_dpg(image)
    face_annotated_dpg=GUI.convert_for_dpg(face_annotated)

    # Update the camera textures
    dpg.set_value("camera_texture", image_dpg)
    dpg.set_value("mapped_face_points_texture", face_annotated_dpg)

    dpg.render_dearpygui_frame()

finally:
  cap.release()
  dpg.destroy_context()