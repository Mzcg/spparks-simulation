#ref: https://www.ovito.org/docs/current/python/introduction/examples/overlays/data_plot.html
from ovito.vis import PythonViewportOverlay, Viewport, TachyonRenderer
from ovito.modifiers import CreateBondsModifier, ColorCodingModifier

from ovito.io import import_file
from ovito.modifiers import CoordinationAnalysisModifier, HistogramModifier

from ovito.vis import ViewportOverlayInterface
from ovito.data import DataCollection
import os, math

import time

os.environ["QT_API"] = "pyqt5"

settings = {"figSize":(900,600)}

class CoordinationPlotOverlay(ViewportOverlayInterface):
    def render(self, canvas: ViewportOverlayInterface.Canvas, data: DataCollection, **kwargs):
        with canvas.mpl_figure(pos=(0.02,0.98), size=(0.5,0.5), anchor="north west", alpha=0.5, tight_layout=True) as fig:
            ax = fig.subplots()
            ax.set_title('Coordination number histogram')
            plot_data = data.tables['histogram[Coordination]'].xy()
            ax.bar(plot_data[:,0], plot_data[:,1], width=0.8)


start_time = time.time()

# Pipeline setup:
#pipeline = import_file(r"C:\Users\zg0017\PycharmProjects\spparks-simulation\examples\am_path\pattern_repeat\generated_SPPARKS_scripts\speed_1_thickness_3_mpwidth_2_haz_4\3D_AM_speed_1_mpWidth_2_haz_4_thickness_399.dump")
#pipeline = import_file(r"C:\Users\zg0017\PycharmProjects\spparks-simulation\examples\am_path\pattern_repeat\3D_AM_speed_3_mpWidth_69_haz_114_thickness_14_7.dump") #size 56
pipeline = import_file(r"C:\Users\zg0017\Documents\Spring 2024\Simulation Results Save\SPPARKS_scripts_generation_test_HPC_box256_18\speed_3_mpwidth_69_haz_25.0_thickness_14\3D_AM_speed_3_mpWidth_69_haz_25_thickness_14_7.dump") #size 56

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
vp.render_image(filename='visualize_dump_plot_256_s3mp69haz25t14_top.png',background=(0,0,0), size=settings["figSize"], renderer=TachyonRenderer())

end_time = time.time()
running_time = end_time - start_time
print(f"Program running time: {running_time} seconds")
