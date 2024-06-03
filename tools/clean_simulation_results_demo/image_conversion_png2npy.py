# Objective: This program is to convert images with .png postfix to .npy files for the upcoming ML processes.

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def convert_png_to_npy(parent_folder_path, image_data_foldername):
    """
        This function take input folder with image files end with .png convert to .npy format and save to a new folder
        Input:
            - parent_folder_path: folder that stores all the simulation files
            - image_data_foldername: pick the folder that stores the processed .png data
        Output:
            - new folder (npy_folder_path): that has same name with image_data_foldername but adding postfix _npy at the end of filename. all converted .npy files store here.
    """
    folder_path = os.path.join(parent_folder_path, image_data_foldername)
    all_files = os.listdir(folder_path)
    png_file_list = [f for f in all_files if f.lower().endswith('.png')] #get files in a list

    #create new folder for npy file output
    npy_folder_name = folder_path.split('\\')[-1] + "_npy" #example: simulation_images_generation_cut75_JET_multiDirection_multiDistance_resize_npy
    npy_folder_path =os.path.join(parent_folder_path, npy_folder_name)
    os.makedirs(npy_folder_path, exist_ok = True) #create folder if it's not exist.

    for png_file in png_file_list:
        png_file_path = os.path.join(folder_path,png_file)
        image = Image.open(png_file_path)
        image_array = np.array(image)
        
        npy_filename = png_file.replace('.png', '.npy')
        npy_file_path = os.path.join(npy_folder_path, npy_filename)

        #save numpy array as .npy file with same parameter name as png
        np.save(npy_file_path, image_array)



def main():
    conversion_command = True
    single_npy_visualization_and_save = False

    if conversion_command:
        parent_folder= r'D:\Zhaochen\simulation_SPPARKS_hpc\SPPARKS_scripts_generation_128_20240306'
        #data_folder_input = "simulation_images_generation_cut75_JET_xy64only_resize" #JET xy_64 only folder
        #data_folder_input = "simulation_images_generation_cut75_JET_xy64_xz64_resize" #JET xy_64 and xz_64 folder
        data_folder_input = "simulation_images_generation_cut75_JET_multiDirection_multiDistance_resize" #JET multi directions (xy, xz, yz) and distances (32,64,96) folder

        convert_png_to_npy(parent_folder, data_folder_input) #generate new folder and convert png files to npy and save to new folder


    ## Visualize single npy file (to png) --> just want to check if the image works.
    if single_npy_visualization_and_save == True:
        single_npy_path = r"D:\Zhaochen\simulation_SPPARKS_hpc\SPPARKS_scripts_generation_128_20240306\simulation_images_generation_cut75_JET_xy64only_resize_npy\speed_3_mpwidth_10_haz_82_thickness_11.npy"
        image_npy = np.load(single_npy_path)

        plt.imshow(image_npy, cmap='gray')  # You can specify the colormap here if needed
        plt.axis('off')  # Turn off axis
        plt.show()

        temp_save_path = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\tools\test_npy_vis.png"  #create a temp path to store new image (name created in the path) for testing purpose.
        plt.imsave(temp_save_path, image_npy, cmap='gray') #save the image


if __name__ == '__main__':
    main()