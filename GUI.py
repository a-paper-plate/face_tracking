from mediapipe.tasks.python import vision
from slider_2d import add_2d_slider
from mediapipe.tasks import python
import dearpygui.dearpygui as dpg
from typing import Tuple, Union
import mediapipe as mp
import numpy as np
import helpers
import math
import cv2


def init_dearpygui(window_name="temp",width=800, height=600):
  dpg.create_context()
  dpg.create_viewport(title=window_name, width=width, height=height)
  dpg.setup_dearpygui()


def  init_texture_registry(height,width):
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


def side_window_on():
    dpg.configure_item("data_window",show=True)
    dpg.configure_item("on_button",show=False)

def side_window_off():
    dpg.configure_item("data_window",show=False)
    dpg.configure_item("on_button",show=True)


def init_UI(blendshapes_slider,blendshapes_2D_slider):
    with dpg.window(label="Example Window",tag="Primary_Window"):
        with dpg.group(horizontal=True):
            with dpg.child_window(width=160, border=True,tag="data_window"):
                with dpg.table(header_row=False, borders_innerH=False, borders_outerH=False,borders_innerV=False, borders_outerV=False):
                    # Column 1: Shrinks to fit the exact width of the widget inside it
                    dpg.add_table_column(width_fixed=True)
                    # Column 2: Stretches to fill all open space on the left
                    dpg.add_table_column(width_stretch=True)
                    # Column 3: Shrinks to fit the exact width of the widget inside it
                    dpg.add_table_column(width_fixed=True)
                
                    with dpg.table_row():
                        dpg.add_text("DATA")
                        dpg.add_spacer()
                        dpg.add_button(label="X", callback=lambda:side_window_off())

                dpg.add_separator()
                for blendshape in blendshapes_2D_slider.keys():
                    slider_2d=add_2d_slider(100,100,label=blendshape,tag=blendshape,interactable=False,normalized_io=True)
                    blendshapes_2D_slider[blendshape]=slider_2d
                dpg.add_separator()
                for blendshape in blendshapes_slider.keys():
                    dpg.add_progress_bar(overlay=blendshape,tag=blendshape,width=-1)


            with dpg.child_window(width=-1, border=True,tag="image_window"):   
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Y", callback=lambda:side_window_on(), tag="on_button",show=False)
                    dpg.add_image(texture_tag="camera_texture")
                    dpg.add_image(texture_tag="mapped_face_points_texture")

        
        

        







def blank_image(height, width):
  frame=np.zeros((height, width, 4), dtype=np.uint8)
  frame[:, :, 3] = 1.0
  return frame


def convert_for_dpg(image):
  # Convert BGR to RGBA
  image_dpg=cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
  # Convert uint8 [0, 255] to float32 [0, 1]
  image_dpg = image_dpg.astype(np.float32) / 255.0
  image_dpg.flatten()
  return image_dpg


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


