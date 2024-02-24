#ref: https://www.ovito.org/docs/current/python/introduction/examples/overlays/data_plot.html
from ovito.vis import PythonViewportOverlay, Viewport, TachyonRenderer
from ovito.modifiers import CreateBondsModifier, ColorCodingModifier

from ovito.io import import_file
from ovito.modifiers import CoordinationAnalysisModifier, HistogramModifier
from ovito.modifiers import SliceModifier

from ovito.vis import ViewportOverlayInterface
from ovito.data import DataCollection
import os, math



import time

os.environ["QT_API"] = "pyqt5"



settings = {"figSize":(900,600),
            "cut_distance": (14, 28, 42, 56)}

class CoordinationPlotOverlay(ViewportOverlayInterface):
    def render(self, canvas: ViewportOverlayInterface.Canvas, data: DataCollection, **kwargs):
        with canvas.mpl_figure(pos=(0.02,0.98), size=(0.5,0.5), anchor="north west", alpha=0.5, tight_layout=True) as fig:
            ax = fig.subplots()
            ax.set_title('Coordination number histogram')
            plot_data = data.tables['histogram[Coordination]'].xy()
            ax.bar(plot_data[:,0], plot_data[:,1], width=0.8)


start_time = time.time()

filepath = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\examples\am_path\pattern_repeat\3D_AM_speed_3_mpWidth_69_haz_114_thickness_14_7.dump" #size 56
# Pipeline setup:
#pipeline = import_file(r"C:\Users\zg0017\PycharmProjects\spparks-simulation\examples\am_path\pattern_repeat\generated_SPPARKS_scripts\speed_1_thickness_3_mpwidth_2_haz_4\3D_AM_speed_1_mpWidth_2_haz_4_thickness_399.dump")
#pipeline = import_file(r"C:\Users\zg0017\PycharmProjects\spparks-simulation\examples\am_path\pattern_repeat\3D_AM_speed_3_mpWidth_69_haz_114_thickness_14_7.dump") #size 56
#pipeline = import_file(r"C:\Users\zg0017\Documents\Spring 2024\Simulation Results Save\SPPARKS_scripts_generation_test_HPC_box256_18\speed_3_mpwidth_69_haz_25.0_thickness_14\3D_AM_speed_3_mpWidth_69_haz_25_thickness_14_7.dump") #size 56

#img_yz = pipeline
#img_yz.add_to_scene()
#img_yz.modifiers.append(ColorCodingModifier(property='Particle Type'))

#img_xz = pipeline
#img_xz.add_to_scene()
#img_xz.modifiers.append(ColorCodingModifier(property='Particle Type'))

#img_xy = pipeline
#img_xy.add_to_scene()
#img_xy.modifiers.append(ColorCodingModifier(property='Particle Type'))

#img.add_to_scene()
#img.modifiers.append(ColorCodingModifier(property='Particle Type'))
#loop cut_distance to get different dimension of cross-section samples.

for dist in settings["cut_distance"]:
    #cut along x-axis, get yz view
    img_yz = import_file(filepath)
    img_yz.add_to_scene()
    img_yz.modifiers.append(ColorCodingModifier(property='Particle Type'))
    img_yz.modifiers.append(SliceModifier(normal=(1, 0, 0), distance=dist))
    vp_yz = Viewport(camera_dir=(1, 0, 0))  # camer_dir=(1,0,0) -> y<z^// (0,0,1)->x<y^ // (0,1,0)->x>z^
    # vp_slice1.type = Viewport.Type.Front  #Type options: Perspective, TOP, FRONT, ORTHO
    vp_yz.zoom_all(size=settings["figSize"])
    vp_yz.render_image(filename=f'YZ_{dist}_dump_plot_56_s3mp69haz114t14_top.png', background=(0, 0, 0),
                           size=settings["figSize"], renderer=TachyonRenderer())
    print("yz: ",dist)
    #cut along y-axis, get xz view
    img_xz = import_file(filepath)
    img_xz.add_to_scene()
    img_xz.modifiers.append(ColorCodingModifier(property='Particle Type'))
    img_xz.modifiers.append(SliceModifier(normal=(0, 1, 0), distance=dist))
    vp_xz = Viewport(camera_dir=(0, 1, 0))  # camer_dir=(1,0,0) -> y<z^// (0,0,1)->x<y^ // (0,1,0)->x>z^
    # vp_slice1.type = Viewport.Type.Front  #Type options: Perspective, TOP, FRONT, ORTHO
    vp_xz.zoom_all(size=settings["figSize"])
    vp_xz.render_image(filename=f'XZ_{dist}_dump_plot_56_s3mp69haz114t14_top.png', background=(0, 0, 0),
                       size=settings["figSize"], renderer=TachyonRenderer())
    print("xz: ",dist)
    #cut along z-axis, get xy view

    img_xy = import_file(filepath)
    img_xy.add_to_scene()
    img_xy.modifiers.append(ColorCodingModifier(property='Particle Type'))
    img_xy.modifiers.append(SliceModifier(normal=(0, 0, -1), distance=dist))
    vp_xy = Viewport(camera_dir=(0, 0, -1))  # camer_dir=(1,0,0) -> y<z^// (0,0,1)->x<y^ // (0,1,0)->x>z^
    # vp_slice1.type = Viewport.Type.Front  #Type options: Perspective, TOP, FRONT, ORTHO
    vp_xy.zoom_all(size=settings["figSize"])
    vp_xy.render_image(filename=f'XY_{dist}_dump_plot_56_s3mp69haz114t14_top.png', background=(0, 0, 0),
                       size=settings["figSize"], renderer=TachyonRenderer())
    print("xy: ",dist)
'''
#get original input visualization (e.g.: top)
input_img = pipeline
input_img.add_to_scene()
input_img.modifiers.append(ColorCodingModifier(property='Particle Type'))
vp_input = Viewport()
vp_input.type = Viewport.Type.Top  #Type options: Perspective, TOP, FRONT, ORTHO
vp_input.zoom_all(size=settings["figSize"])
vp_input.render_image(filename='input_visualize_dump_plot_56_s3mp69haz114t14_top.png',background=(0,0,0), size=settings["figSize"], renderer=TachyonRenderer())


#get slice
slice1 = pipeline
# Insert a modifier that operates on the data:
slice1.add_to_scene()
slice1.modifiers.append(SliceModifier(normal=(1,0,0), distance=14))
vp_slice1 = Viewport(camera_dir = (1,0,0)) #camer_dir=(1,0,0) -> y<z^// (0,0,1)->x<y^ // (0,1,0)->x>z^
#vp_slice1.type = Viewport.Type.Front  #Type options: Perspective, TOP, FRONT, ORTHO
vp_slice1.zoom_all(size=settings["figSize"])
vp_slice1.render_image(filename='slice1_visualize_dump_plot_56_s3mp69haz114t14_top.png',background=(0,0,0), size=settings["figSize"], renderer=TachyonRenderer())

'''

'''
# Insert a modifier that operates on the data:
pipeline.modifiers.append(SliceModifier(normal=(0,28,0), distance=0))
# Compute the effect of the slice modifier by evaluating the pipeline.
output = pipeline.compute()
print("Remaining particle count:", output.particles.count)

pipeline.add_to_scene()
pipeline.modifiers.append(ColorCodingModifier(property='Particle Type'))
## pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=3.0))
## pipeline.modifiers.append(HistogramModifier(property='Coordination', bin_count=13, fix_xrange=True, xrange_start=0.5, xrange_end=13.5))



# Viewport setup and rendering:
#viewport = Viewport(type = Viewport.Type.Perspective)
vp = Viewport()
#vp.type = Viewport.Type.Perspective   #Type options: Perspective, TOP, FRONT, ORTHO
vp.type = Viewport.Type.Top  #Type options: Perspective, TOP, FRONT, ORTHO

## View in 3D (Perspective) with manual setting direction
#vp.camera_pos = (-100, -150, 150)
#vp.camera_dir = (2, 3, -3)
#vp.fov = math.radians(45.0)

#test values
#vp.camera_pos = (-108, -80, 117)
#vp.camera_dir = (2, 3, -3)
#vp.fov = math.radians(35.0)


vp.zoom_all(size=settings["figSize"])
#vp.render_image(filename='slice_visualize_dump_plot_256_s3mp69haz25t14_top.png',background=(0,0,0), size=settings["figSize"], renderer=TachyonRenderer())
vp.render_image(filename='slice_visualize_dump_plot_56_s3mp69haz114t14_top.png',background=(0,0,0), size=settings["figSize"], renderer=TachyonRenderer())

'''


end_time = time.time()
running_time = end_time - start_time
print(f"Program running time: {running_time} seconds")
