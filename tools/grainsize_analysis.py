# Objective： this program is for analysis use. we will test it on single images for analysis purpose. this is not a general program to process file in large scale.

#from slice_image2grainsize import single_image_statistics_collection
import average_grain_size_encap
from PIL import Image
import numpy as np
import os
import csv

#testimage = r"D:\Zhaochen\simulation_SPPARKS_hpc\SPPARKS_scripts_generation_128_20240306\simulation_images_parameter_analysis\speed_3_mpwidth_25_haz_40_thickness_8\xy_64_cut75.png"
#testimage2 = r"D:\Zhaochen\simulation_SPPARKS_hpc\SPPARKS_scripts_generation_128_20240306\simulation_images_parameter_analysis\speed_72_mpwidth_25_haz_40_thickness_8\xy_64_cut75.png"


def calculate_sinlge_image_grainsize_with_resize(imagepath):
   #image_pil = Image.open(testimage2).convert('RGB')  # convert image mode to RGB (in case if not, e.g: some image may have 4 channels: RGBA)
   image_pil = Image.open(imagepath).convert('RGB').resize((128, 128), Image.Resampling.LANCZOS) # convert image mode to RGB (in case if not, e.g: some image may have 4 channels: RGBA)
   image_np = np.array(image_pil)                     # Convert the PIL image to a NumPy array #note: later our input image will be numpy array, so we don't need to do taht.

   # show_intermediate_segmentation = True  # if want to display the six images of intermediate steps of grain segmentation, set to True
   # draw_histogram = False                 # if want to display the visualization of grain size distribution histogram, set it true.
   # histogram_plot_default = True          # Choose want to use manual setting (bin_min,max) to plot or not. True: auto plot with default values False: use following manual setting to plot.

   hist, bins, bin_min, bin_max, avg_grain_size, num_regions, total_grain_size, labeled_image, regions_list = \
      average_grain_size_encap.grain_segmentation(image_np, show_intermediate_segmentation=True, histogram_plot_default=True, draw_histogram=False)

   return hist, bins, bin_min, bin_max, avg_grain_size, num_regions, total_grain_size, labeled_image, regions_list

#print("grain size: ", avg_grain_size)
#print("total grain size: ", total_grain_size) #sum of detected grain size, not necessary equals to pixel size
def main():

   analysis_folder_path = r"D:\Zhaochen\simulation_SPPARKS_hpc\SPPARKS_scripts_generation_128_20240306\simulation_images_parameter_analysis"
   csv_path = os.path.join(analysis_folder_path, "grain_size_analysis_summary.csv")

   with open (csv_path, "w", newline='') as csvfile:
      csv_header = ['simulation', 'slice_file', 'average_grain_size']
      writer = csv.writer(csvfile)
      writer.writerow(csv_header)
   #   print(csv_path)
      for simulation_folder in os.listdir(analysis_folder_path): #loop folder inside parent folder
         simulation_folder_path = os.path.join(analysis_folder_path, simulation_folder)
         if os.path.isdir(simulation_folder_path):   #Avoid reading non-direcotory file -- we will generate a csv file inside the folder, to avoid accidently read this as a simulation folder, we add this condition
            for image_file in os.listdir(simulation_folder_path):   #loop image file inside
               image_file_path = os.path.join(simulation_folder_path, image_file)
               if image_file.startswith("xy_64") or image_file.startswith("xz_64"):
                  hist, bins, bin_min, bin_max, avg_grain_size, num_regions, sum_grain_size, labeled_image, regions_list = calculate_sinlge_image_grainsize_with_resize(image_file_path)

                  writer.writerow([simulation_folder, image_file, round(avg_grain_size, 2)])
                  #print(simulation_folder, image_file, avg_grain_size)
   print("analysis grain size save to grain_size_analysis_summary.csv")


if __name__ == "__main__":
    #main()

    ##single test
    #testimage = r"D:\Zhaochen\simulation_SPPARKS_hpc\SPPARKS_scripts_generation_128_20240306\simulation_images_parameter_analysis\speed_42_mpwidth_30_haz_65_thickness_8\xy_64_cut75.png"
    testimage = r"D:\Zhaochen\simulation_SPPARKS_hpc\SPPARKS_scripts_generation_128_20240306\simulation_images_generation_cut75_JET\speed_13_mpwidth_20_haz_82_thickness_8\xy_96_cut75.png"
    hist, bins, bin_min, bin_max, avg_grain_size, num_regions, sum_grain_size, labeled_image, regions_list = calculate_sinlge_image_grainsize_with_resize(testimage)
    print("grain size: ", avg_grain_size)