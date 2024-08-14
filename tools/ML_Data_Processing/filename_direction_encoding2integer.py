import os

#encoding rule:
# xy -> 1
# yz -> 2
# xz -> 3

def filename_direction_conversion(image_path):
    #image_name = image_path.split("\\")[-1] #e.g: speed_3_mpwidth_10_haz_40_thickness_5_direction_xz_distance_64.png
    image_directory, image_name = os.path.split(image_path)


    direction = image_file_name.split("_")[9] #get the direction code (xy, xz, yz)
    if direction == "xy":
        new_image_name = image_name.replace("xy", "1")

    if direction == "yz":
        new_image_name = image_name.replace("yz", "2")
    if direction == "xz":
        new_image_name = image_name.replace("xz", "3")

    new_image_path = os.path.join(image_directory, new_image_name) #change the image name to the new one

    return new_image_path

data_augmentation_all_folder_path = r"E:\Data\data_augmentation_all"
#print(data_augmentation_all_folder_path)

for dataset_case_name in os.listdir(data_augmentation_all_folder_path): #loop through experiment cases (xy_0,32,64, xy_xz,0,32,64, xy_xz_yz_0.32.64)
    dataset_case_path = os.path.join(data_augmentation_all_folder_path, dataset_case_name)

    for effect_type in os.listdir(dataset_case_path): #loop through effects folder (gray_contrast, jet, median_filter, flip, original)
        effect_type_path = os.path.join(dataset_case_path, effect_type)
        #print(effect_type_path)
        for image_file_name in os.listdir(effect_type_path):
            image_file_path = os.path.join(effect_type_path, image_file_name)
            #change file name here (just file name)
            #direction_option = image_file_name.split("_")[9] #get the direction code (xy, xz, yz)
            #if direction_option == "xy":
            new_imagename_path = filename_direction_conversion(image_file_path)

            os.rename(image_file_path, new_imagename_path)  #change the generated file name in the folder (no way back)

#if have a folder contains the image file directly, can use the following sectional code only (just copy from partial code above)
# effect_type_path = r"D:\Zhaochen\ML_training_data_without_augmentation\original_xy_xz_64"
#
# for image_file_name in os.listdir(effect_type_path):
#     image_file_path = os.path.join(effect_type_path, image_file_name)
#     #change file name here (just file name)
#     #direction_option = image_file_name.split("_")[9] #get the direction code (xy, xz, yz)
#     #if direction_option == "xy":
#     new_imagename_path = filename_direction_conversion(image_file_path)
#
#     os.rename(image_file_path, new_imagename_path)  #change the generated file name in the folder (no way back)
