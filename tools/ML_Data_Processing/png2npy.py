# reference: https://github.com/kvmani/machineLearning/blob/35a0537554ffb4180d146c390f85df8ddac776e5/src/grainStructure/png2npy.py
# src/grainStructure/png2npy.py

import os
import numpy as np
from PIL import Image

def convertPngToNpy(png_folder, npy_save_folder):
    # Create the save folder if it doesn't exist
    os.makedirs(npy_save_folder, exist_ok=True)

    # List all PNG files in the folder
    png_files = [file for file in os.listdir(png_folder) if file.endswith('.png')]

    for png_file in png_files:
        png_path = os.path.join(png_folder, png_file)

        # Open the image and convert it to a NumPy array
        with Image.open(png_path) as img:
            img_array = np.array(img)

        # Save the NumPy array as an NPY file
        npy_file_path = os.path.join(npy_save_folder, f"{os.path.splitext(png_file)[0]}.npy")
        np.save(npy_file_path, img_array)
        print(f"Converted and saved {png_file} to {npy_file_path}")

    print("PNG to NPY conversion completed successfully!")

# Example usage
data_parent_folder = r'E:\Data\data_augmentation_all'

experiment_set = "data_augmentation_xy_0_32_64"
experiment_set_path = os.path.join(data_parent_folder, experiment_set)
for effect_folder in os.listdir(experiment_set_path):
    effect_folder_path = os.path.join(experiment_set_path, effect_folder)
    png_folder = effect_folder_path

    npy_save_foldername = "NPY_"+effect_folder
    npy_save_folderpath = os.path.join(experiment_set_path, npy_save_foldername)
    if not os.path.exists(npy_save_folderpath):
        os.makedirs(npy_save_folderpath)
        print(f"{npy_save_foldername} created.")

    convertPngToNpy(png_folder, npy_save_folderpath)


#png_folder = r'D:\Aishwarya\grainStructure\dataV3\simulation_images_generation_cut75_multiDirection_multiDistance_resize'
#npy_save_folder = r'D:\Aishwarya\grainStructure\dataV3\NPY_simulation_images_generation_cut75_multiDirection_multiDistance_resize'

#convertPngToNpy(png_folder, npy_save_folder)