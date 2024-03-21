#ref: https://www.ovito.org/docs/current/python/introduction/examples/overlays/data_plot.html
#ref2: https://www.ovito.org/docs/current/python/modules/ovito_modifiers.html
import os, math
#to use OpenGL renderer, do following, TachyonRenderer no need the following three lines
os.environ['OVITO_GUI_MODE'] = '1' # Request a session with OpenGL support
import PySide6.QtWidgets    # Make the program run in the context of the graphical desktop environment.
app = PySide6.QtWidgets.QApplication()

from ovito.vis import PythonViewportOverlay, Viewport, TachyonRenderer, OpenGLRenderer
from ovito.modifiers import CreateBondsModifier, ColorCodingModifier
from ovito.modifiers import SliceModifier
from ovito.io import import_file
from ovito.modifiers import CoordinationAnalysisModifier, HistogramModifier
from ovito.modifiers import SliceModifier

from ovito.vis import ViewportOverlayInterface
from ovito.data import DataCollection

import time
os.environ["QT_API"] = "pyqt5"



settings = {"figSize":(900,600),
            "cut_distance": ( 56,42,  28, 14),
            #"cut_distance": ( 128, 96,  64, 32 ),
            }


filepath = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\examples\am_path\pattern_repeat\3D_AM_speed_3_mpWidth_69_haz_114_thickness_14_7.dump" #size 56  #slice dim: 14, 28, 42 ,56
#filepath = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\examples\am_path\pattern_repeat\3D_AM_speed_3_mpWidth_10_haz_40_thickness_5_234.dump" #size 128



#cut along x-axis, get yz view - single test code
# dist_yz = 28
# img_yz = import_file(filepath)
# img_yz.add_to_scene()
# img_yz.modifiers.append(ColorCodingModifier(property='Particle Type'))
# img_yz.modifiers.append(SliceModifier(normal=(1, 0, 0), distance=dist_yz, operate_on={'particles'}, slab_width=5))
# vp_yz = Viewport(camera_dir=(1, 0, 0))  # camer_dir=(1,0,0) -> y<z^// (0,0,1)->x<y^ // (0,1,0)->x>z^
# # vp_slice1.type = Viewport.Type.Front  #Type options: Perspective, TOP, FRONT, ORTHO
# vp_yz.zoom_all(size=settings["figSize"])
# vp_yz.render_image(filename=f'YZ_{dist_yz}__TEST_dup.png', background=(0, 0, 0),
#                        size=settings["figSize"], renderer=TachyonRenderer())
# print("yz: ",dist_yz)
#

def get_distance_snapshot(filepath,slice_distance, direction, slab=5):

    if direction == "xy":
        normal_plane_vector = (0,0,1)
        camera_direction = (0,0,-1)
    elif direction == "yz":
        normal_plane_vector = (1,0,0)
        camera_direction = (1, 0, 0)
    elif direction == "xz":
        normal_plane_vector = (0,1,0)
        camera_direction = (0,1,0)

    img = import_file(filepath)
    img.add_to_scene()
    img.modifiers.append(ColorCodingModifier(property='Particle Type'))
    img.modifiers.append(SliceModifier(normal=normal_plane_vector, distance=slice_distance, operate_on={'particles'}, slab_width=slab))
    vp = Viewport(camera_dir=camera_direction)  # camer_dir=(1,0,0) -> y<z^// (0,0,1)->x<y^ // (0,1,0)->x>z^
    # vp_slice1.type = Viewport.Type.Front  #Type options: Perspective, TOP, FRONT, ORTHO
    vp.zoom_all(size=settings["figSize"])
    vp.render_image(filename=f'{direction}_{slice_distance}.png', background=(0, 0, 0),
                       size=settings["figSize"], renderer=OpenGLRenderer())  #renderer=TachyonRenderer()   #renderer=OpenGLRenderer()
    print(direction, ": ", slice_distance)
    img.remove_from_scene()


#Part 1: getting slices and output to png images ( we will get 3 (directions) * 3 (distances) = 9 output images)
def plot_distance_slices(filepath):
    for dist in settings["cut_distance"]:
        #get slice from xy direciton (cut in z-axis)
        get_distance_snapshot(filepath,dist, "xy", slab=5)

        #get slice from yz direction (cut in x-axis)
        get_distance_snapshot(filepath, dist, "yz", slab=5)

        #get slice from xz direction (cut in y-axis)
        get_distance_snapshot(filepath, dist, "xz", slab=5)

def plot_3D_view(filepath):
    #getting full view of simulation in 3D and output to png image. (get 1 image output of 3D view）
    pipeline = import_file(filepath)
    pipeline.modifiers.append(SliceModifier(normal=(0,28,0), distance=0, operate_on={'particles'}, slab_width=128))
    # Compute the effect of the slice modifier by evaluating the pipeline.
    output = pipeline.compute()
    print("Remaining particle count:", output.particles.count) # Multiplication of size (e.g: 56 * 56 * 56 = 175616)

    pipeline.add_to_scene()
    pipeline.modifiers.append(ColorCodingModifier(property='Particle Type'))
    viewport = Viewport(type = Viewport.Type.Perspective) #Type options: Perspective, TOP, FRONT, ORTHO

    ## View in 3D (Perspective) with manual setting direction
    viewport.camera_pos = (-100, -150, 150)
    viewport.camera_dir = (2, 3, -3)
    viewport.fov = math.radians(45.0)


    viewport.zoom_all(size=settings["figSize"])
    viewport.render_image(filename='full_view.png',background=(0,0,0), size=settings["figSize"], renderer=TachyonRenderer())  #rendere=OpenGLRenderer()
    pipeline.remove_from_scene()#make sure to add this to overcome duplicate layers that affects other vis in loop

    print("full view done")

def main():
    start_time = time.time()
    plot_3D_view(filepath) #function call: generate 3D view ->output single 3D image
    plot_distance_slices(filepath) #function call: generate multiple slices images according to distance

    end_time = time.time()
    running_time = end_time - start_time
    print(f"Program running time: {running_time} seconds")

if __name__ == "__main__":
    main()



