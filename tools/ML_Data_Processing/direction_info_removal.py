# NOT A FUNCTIONAL PROGRAM FOR GENERAL DATA PROCESSING PURPOSE
# This program aim to remove the direction information from xy_only data since if we only consider direction xy without other direction, this info is redundant.
# later if we generate the color code is unnecessary. so we want to remove this part from the file name before generating ML required data
# apply to folder "data_augmentation_xy_0_32_64" only

import os

work_parent_folder = r"E:\Data\data_augmentation_all\data_augmentation_xy_0_32_64"
string_to_remove = 'direction_1_'

for subfolder in os.listdir(work_parent_folder):
    subfolder_path = os.path.join(work_parent_folder, subfolder)
    for image_file_name in os.listdir(subfolder_path):
        new_image_name = image_file_name.replace(string_to_remove,'')
        os.rename(os.path.join(subfolder_path, image_file_name), os.path.join(subfolder_path, new_image_name))

        print(f"Change {image_file_name} to {new_image_name}")

print("File Renaming Complete! ")
