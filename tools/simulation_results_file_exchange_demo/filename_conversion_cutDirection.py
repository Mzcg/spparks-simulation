# Objective: This program is to change the file name of processed data, specifically we want to change the direction (xy, xz and yz ) to integer value
# xy - 1
# yz - 2
# xz - 3

import os

#processed_data_folder_path = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\tools\simulation_images_generation_cut75_multiDirection_multiDistance_resize" #demo test
processed_data_folder_path = r"E:\Data\simulation_images_generation_cut75_multiDirection_multiDistance_resize" # modify filename in flash drive

# Iterate over the files in the folder
for filename in os.listdir(processed_data_folder_path):
    # Split the filename into parts based on underscores
    parts = filename.split('_')

    # Iterate over the parts to find and replace 'xy', 'xz', and 'yz'
    for i, part in enumerate(parts):
        if part == 'xy':
            parts[i] = '1'
        elif part == 'xz':
            parts[i] = '3'
        elif part == 'yz':
            parts[i] = '2'

    # Join the modified parts back into a filename
    new_filename = '_'.join(parts)

    # Rename the file
    os.rename(os.path.join(processed_data_folder_path, filename), os.path.join(processed_data_folder_path, new_filename))