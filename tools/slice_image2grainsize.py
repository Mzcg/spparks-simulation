import average_grain_size_encap
from PIL import Image, ImageFilter
import numpy as np
import os
import glob
import json

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Some errors require this to fix (oneDNN custom operations are on)
import tensorflow as tf

def single_image_statistics_collection(image_path):
   image_pil = Image.open(image_path).convert('RGB')  # convert image mode to RGB (in case if not, e.g: some image may have 4 channels: RGBA)
   image_np = np.array(image_pil)                     # Convert the PIL image to a NumPy array #note: later our input image will be numpy array, so we don't need to do taht.

   # show_intermediate_segmentation = True  # if want to display the six images of intermediate steps of grain segmentation, set to True
   # draw_histogram = False                 # if want to display the visualization of grain size distribution histogram, set it true.
   # histogram_plot_default = True          # Choose want to use manual setting (bin_min,max) to plot or not. True: auto plot with default values False: use following manual setting to plot.

   hist, bins, bin_min, bin_max, avg_grain_size, num_regions, total_grain_size, labeled_image, regions_list = \
       average_grain_size_encap.grain_segmentation(image_np, show_intermediate_segmentation=False, histogram_plot_default=True, draw_histogram=False)
   return hist, bins, bin_min, bin_max, avg_grain_size, num_regions, total_grain_size, labeled_image, regions_list




simulation_folder_path = os.getcwd()
#xy_dict= {"xy":{}} #set up xy_dict
big_dict = {}
xy_slice_dict = {} #set up subset dictionary for slice in different directions and distances.
xz_slice_dict = {}
yz_slice_dict = {}

xy_4slice_grainsize_min = []
xy_4slice_grainsize_max = []
xy_4slice_grainsize_mean = []

xz_4slice_grainsize_min = []
xz_4slice_grainsize_max = []
xz_4slice_grainsize_mean = []

yz_4slice_grainsize_min = []
yz_4slice_grainsize_max = []
yz_4slice_grainsize_mean = []
for png_file in glob.glob(os.path.join(simulation_folder_path, '*.png')): #png_file -> contains all path
   img_filename = os.path.basename(png_file)
   if img_filename.startswith("xy"):   #check the image file start with xy. (make sure to store them start with xy as beginning char)

      #print(img_filename)   #xy_14.png
      slice_name = img_filename.split('.')[0]
      print(slice_name) #xy_14

      xy_hist, xy_bins, xy_bin_min, xy_bin_max, xy_avg_grain_size, xy_num_regions, xy_sum_grain_size, xy_labeled_image, xy_regions_list = single_image_statistics_collection(png_file)

      xy_min_grain_size = min(xy_regions_list)
      xy_max_grain_size = max(xy_regions_list)

      xy_4slice_grainsize_min.append(xy_min_grain_size)
      xy_4slice_grainsize_max.append(xy_max_grain_size)
      xy_4slice_grainsize_mean.append(xy_avg_grain_size)
      xy_slice_dict["xy_allslice_min"] = min(xy_4slice_grainsize_min)
      xy_slice_dict["xy_allslice_max"] = max(xy_4slice_grainsize_max)
      xy_slice_dict["xy_allslice_mean"] = sum(xy_4slice_grainsize_mean) / len(xy_4slice_grainsize_mean)

      xy_slice_dict[slice_name] = {slice_name+"_average_grain_size": xy_avg_grain_size,
                                   slice_name+"_number_of_regions": xy_num_regions,
                                   slice_name+"_min_grain_size": xy_min_grain_size,
                                   slice_name+"_max_grain_size": xy_max_grain_size,
                                   slice_name+"_total_grain_size": xy_sum_grain_size }



      #xy_dict["xy"] = xy_slice_dict
      big_dict["xy"] = xy_slice_dict
      #xy_dict["xy"][slice_name] = xy_slice_dict
      #xy_dict["xy"].update(xy_slice_dict)

   if img_filename.startswith("xz"):

      slice_name = img_filename.split('.')[0]
      print(slice_name)  # xy_14

      xz_hist, xz_bins, xz_bin_min, xz_bin_max, xz_avg_grain_size, xz_num_regions, xz_sum_grain_size, xz_labeled_image, xz_regions_list = single_image_statistics_collection(
         png_file)
      xz_min_grain_size = min(xz_regions_list)
      xz_max_grain_size = max(xz_regions_list)

      xz_4slice_grainsize_min.append(xz_min_grain_size)
      xz_4slice_grainsize_max.append(xz_max_grain_size)
      xz_4slice_grainsize_mean.append(xz_avg_grain_size)
      xz_slice_dict["xz_allslice_min"] = min(xz_4slice_grainsize_min)
      xz_slice_dict["xz_allslice_max"] = max(xz_4slice_grainsize_max)
      xz_slice_dict["xz_allslice_mean"] = sum(xz_4slice_grainsize_mean) / len(xz_4slice_grainsize_mean)

      xz_slice_dict[slice_name] = {slice_name + "_average_grain_size": xz_avg_grain_size,
                                   slice_name + "_number_of_regions": xz_num_regions,
                                   slice_name + "_min_grain_size": xz_min_grain_size,
                                   slice_name + "_max_grain_size": xz_max_grain_size,
                                   slice_name + "_total_grain_size": xz_sum_grain_size
                                   }

      big_dict["xz"] = xz_slice_dict

   if img_filename.startswith("yz"):

      slice_name = img_filename.split('.')[0] # xy_14

      yz_hist, yz_bins, yz_bin_min, yz_bin_max, yz_avg_grain_size, yz_num_regions, yz_sum_grain_size, yz_labeled_image, yz_regions_list = single_image_statistics_collection(
         png_file)

      yz_min_grain_size = min(yz_regions_list)
      yz_max_grain_size = max(yz_regions_list)

      yz_4slice_grainsize_min.append(yz_min_grain_size)
      yz_4slice_grainsize_max.append(yz_max_grain_size)
      yz_4slice_grainsize_mean.append(yz_avg_grain_size)
      yz_slice_dict["yz_allslice_min"] = min(yz_4slice_grainsize_min)
      yz_slice_dict["yz_allslice_max"] = max(yz_4slice_grainsize_max)
      yz_slice_dict["yz_allslice_mean"] = sum(yz_4slice_grainsize_mean) / len(yz_4slice_grainsize_mean)

      yz_slice_dict[slice_name] = {slice_name + "_average_grain_size": yz_avg_grain_size,
                                   slice_name + "_number_of_regions": yz_num_regions,
                                   slice_name + "_min_grain_size": yz_min_grain_size,
                                   slice_name + "_max_grain_size": yz_max_grain_size,
                                   slice_name + "_total_grain_size": yz_sum_grain_size
                                   }
      #xz_dict["xz"] = xz_slice_dict
      big_dict["yz"] = yz_slice_dict
merged_grainsize_min = min(xy_4slice_grainsize_min + xz_4slice_grainsize_min + yz_4slice_grainsize_min)
merged_grainsize_max = max(xy_4slice_grainsize_max + xz_4slice_grainsize_max + yz_4slice_grainsize_max)
merged_grainsize_mean = sum(xy_4slice_grainsize_mean + xz_4slice_grainsize_mean + yz_4slice_grainsize_mean) /len(xy_4slice_grainsize_mean + xz_4slice_grainsize_mean + yz_4slice_grainsize_mean)
#print(merged_grainsize_min)
big_dict["all_grainsize_min"] = merged_grainsize_min
big_dict["all_grainsize_max"] = merged_grainsize_max
big_dict["all_grainsize_mean"] = merged_grainsize_mean

#print(xy_dict)
#print(xz_dict)
print("this is big dict: ", big_dict)

with open("sample_1.json", "w") as outfile:
   json.dump(big_dict, outfile, indent=4)




# ### Single Image Test
# single_img_path = r'C:\Users\zg0017\PycharmProjects\spparks-simulation\tools\xy_14.png'
#
#
# #image_pil = Image.open(os.path.join(image_folder_path,image_file)).convert('RGB') #convert image mode to RGB (in case if not, e.g: some image may have 4 channels: RGBA)
# image_pil = Image.open(single_img_path).convert('RGB') #convert image mode to RGB (in case if not, e.g: some image may have 4 channels: RGBA)
# image_np = np.array(image_pil) # Convert the PIL image to a NumPy array #note: later our input image will be numpy array, so we don't need to do taht.
#
# #variable descritpion: setting up variables for the code in average_grain_size_encap
# # show_intermediate_segmentation = True  # if want to display the six images of intermediate steps of grain segmentation, set to True
# # draw_histogram = False                 # if want to display the visualization of grain size distribution histogram, set it true.
# # histogram_plot_default = True          # Choose want to use manual setting (bin_min,max) to plot or not. True: auto plot with default values False: use following manual setting to plot.
#
# hist, bins, bin_min, bin_max, avg_grain_size, num_regions, total_grain_size, labeled_image, regions_list = \
#    average_grain_size_encap.grain_segmentation(image_np, show_intermediate_segmentation=False, histogram_plot_default=True, draw_histogram=False)
#
# print("average grain size: ",avg_grain_size)
# print("num of regions: ", num_regions)
# print("total grain size: ", total_grain_size)
# print("labeled_image: ", labeled_image)
# print("regions list: ", regions_list)
# print("minimum grain size: ", min(regions_list))
# print("maximum grain size: ", max(regions_list))





'''
import json

# Sample data
data = {
    "title": "Your Title Here",
    "xy": {"xy_1": "value1", "xy_2": "value2", "xy_3": "value3", "xy_4": "value4"},
    "yz": {"yz1": "value5", "yz2": "value6", "yz3": "value7", "yz4": "value8"}
}

# File path where you want to store the JSON data
file_path = "data.json"

# Open the file in write mode
with open(file_path, "w") as json_file:
    # Write the data to the file
    json.dump(data, json_file, indent=4)

print("Data has been stored to", file_path)

#JSON format
{
    "title": "Your Title Here",
    "mean_grain_size_all“： ”val"
    "xy": {
        "xy_1": ',
        "xy_2": "value2",
        "xy_3": "value3",
        "xy_4": "value4"
        "xy_min...
        "xy_max...
        "xy_mean...
    },
    "yz": {
        "yz1": "value5",
        "yz2": "value6",
        "yz3": "value7",
        "yz4": "value8"
    }
    
    
}

'''