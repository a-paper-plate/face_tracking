import dearpygui.dearpygui as dpg
from typing import Union,Any,Callable
import helpers

class add_2d_slider:
    def __init__(self,
                 width:int,
                 height:int,
                 label:str="",
                 user_data:Any="",
                 tag:Union[int, str]="",
                 parent:Union[int, str]=None,
                 before:Union[int, str]=None,
                 callback:Callable=None,
                 show:bool=True,
                 id:Union[int, str]=None,
                 coord_readout:bool=True,
                 knob_radius:int=6,
                 default_x:int=0,
                 default_y:int=0)->int|Union[int, str]:

        self.width=width
        self.height=height
        self.label=label
        self.user_data=user_data
        self.tag=tag
        self.parent=parent
        self.before=before
        self.callback=callback
        self.show=show
        self.id=id
        self.coord_readout=coord_readout
        self.knob_radius=knob_radius
        self.current_x=default_x
        self.current_y=default_y
        if id==None:
            self.uuid=dpg.generate_uuid()
        else:
            self.uuid=dpg.generate_uuid(id=self.id)
        
        # TODO allow for selecting the "parent" and "before" objects   for default parent dpg.last_item()
        with dpg.group(tag=self.tag,show=self.show):
            if self.coord_readout:
                dpg.add_text("X: 0.00, Y: 0.00", tag=f"{self.tag}_coord_readout")
                
            # create the 2D pad area
            with dpg.drawlist(width=self.width, height=self.height, tag=f"{self.tag}_slider_2d_canvas"):

                # Background & border
                dpg.draw_rectangle((0, 0), (self.width, self.height), color=(150, 150, 150, 255), fill=(35, 35, 40, 255))

                #crosshairs
                dpg.draw_line((0, self.height), (0, 0), color=(80, 80, 80, 255), tag=f"{self.tag}_vline")
                dpg.draw_line((self.width, 0), (self.width, self.height), color=(80, 80, 80, 255), tag=f"{self.tag}_hline")

                # draggable handle
                dpg.draw_circle((0, self.height), self.knob_radius, color=(255, 255, 255, 255), fill=(66, 150, 250, 255), tag=f"{self.tag}_knob")


        # set default pos
        self.set_value(default_x,default_y)

        # register global mouse handler
        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=self._update_2d_slider)
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=self._update_2d_slider)
        


    def update_2d_slider(self,x,y):
        new_x=helpers.clamp_number(0,self.width,x)
        new_y=helpers.clamp_number(0,self.height,y)
        self.current_x=new_x
        self.current_y=new_y
        # move the visual knob and crosshairs
        dpg.configure_item(f"{self.tag}_knob", center=(new_x, new_y))
        dpg.configure_item(f"{self.tag}_vline", p1=(new_x, 0), p2=(new_x, self.height))
        dpg.configure_item(f"{self.tag}_hline", p1=(0, new_y), p2=(self.width, new_y))
        if self.coord_readout==True:

            # map pixel coordinates (inverting Y so up is positive)
            norm_x = 0 + (new_x / self.width) * (1 - 0)
            norm_y = 1 - (new_y / self.height) * (1 - 0)
            dpg.set_value(f"{self.tag}_coord_readout", f"X: {norm_x:.2f}, Y: {norm_y:.2f}")

        if not self.callback==None:
            self.callback(norm_x,norm_y)


    def _update_2d_slider(self,x:int,y:int):

        # only update when the mouse is actively dragging over the canvas
        if dpg.is_item_hovered(f"{self.tag}_slider_2d_canvas") and dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            canvas_x,canvas_y=dpg.get_item_rect_min(f"{self.tag}_slider_2d_canvas")
            x,y=dpg.get_mouse_pos(local=False)
            self.update_2d_slider(x-canvas_x,y-canvas_y)
        

    def __call__(self,*args, **kwargs):
        return self.uuid


    def set_value(self,x,y):
        # invert Y axis because it starts at bottom but internally works from top
        self.update_2d_slider(x*self.width,-(y*self.height)+self.height)


    def get_value(self):
        return ( 0 + (self.current_x / self.width) * (1 - 0), 1 - (self.current_y / self.height) * (1 - 0))