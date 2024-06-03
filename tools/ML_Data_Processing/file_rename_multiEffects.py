#This program aim to rename the files in separate effects folder (Processed_jet, gray_contrast, median_blur_filter, original, horizontal flip)

import os
import shutil

parent_folder = r"E:\Data\data_augmentation_all\data_augmentation_xy_0_32_64"

merged_folder = os.path.join(parent_folder, "merged_data_xy_0_32_46")  # output folder
os.makedirs(merged_folder, exist_ok=True)

merged_folder_3_effects = os.path.join(parent_folder,"merged_data_3effects_xy_0_32_46")  # output folder: no color changes in effects
os.makedirs(merged_folder_3_effects, exist_ok=True)

def merge_all_effects(input_parent_folder, output_merged_folder):
    # Loop through each item in the parent folder
    for folder in os.listdir(input_parent_folder):
        effect_folder_path = os.path.join(input_parent_folder, folder)
        # Check if the item is a directory and starts with "Processed" #different effects stored in the folder start with "Processed_"
        if os.path.isdir(effect_folder_path) and folder.startswith('Processed'):
            #print(effect_folder_path)
            #print(folder)
            effect_name = "_".join(folder.split("_")[1:]) #getting effect name: original, jet, gray_contrast, median_blur_filter, horizontal_flip
            #print(effect_name)
            for subfolder in os.listdir(effect_folder_path):
                #print(subfolder)
                if subfolder == "stitchedImages":
                    subfolder_path = os.path.join(effect_folder_path, subfolder) #getting path for stitched Images
                    for image_data in os.listdir(subfolder_path):

                        filename, extension = os.path.splitext(image_data) #extension: .png
                        new_image_name = f"{filename}_{effect_name}{extension}" #create new file name -> ex: speed_72_mpwidth_40_haz_91_thickness_8_direction_1_distance_64_horizontal_flip.png
                        new_image_save_path = os.path.join(output_merged_folder, new_image_name)

                        image_data_path = os.path.join(subfolder_path, image_data)
                        shutil.copy(image_data_path, new_image_save_path,) #copy the file to the new folder with new name

def merge_selected_effects(input_parent_folder, output_merged_folder, selected_effects_list):
    processed_effects_foldername = []
    for effect_name in selected_effects_list:
        effect_folder_name = "Processed_" + effect_name
        effect_folder_name = "Processed_" + effect_name
        processed_effects_foldername.append(effect_folder_name)
    #print(processed_effects_foldername) #e.g: ['Processed_original', 'Processed_median_blur_filter', 'Processed_horizontal_flip']

    # Loop through each item in the parent folder
    for folder in os.listdir(input_parent_folder):
        effect_folder_path = os.path.join(input_parent_folder, folder)
        # Check if the item is a directory and starts with "Processed" #different effects stored in the folder start with "Processed_"
        #if os.path.isdir(effect_folder_path) and folder.startswith('Processed'):
        if os.path.isdir(effect_folder_path) and (folder in processed_effects_foldername): #targeting only the selected effects Processed folder
            # print(effect_folder_path)
            # print(folder)

            effect_name = "_".join(folder.split("_")[1:])  # obtain individual effect name (original, flip, blur...)
            # print(effect_name)
            for subfolder in os.listdir(effect_folder_path):
                #print(subfolder)
                if subfolder == "stitchedImages":
                    subfolder_path = os.path.join(effect_folder_path, subfolder)  # getting path for stitched Images
                    for image_data in os.listdir(subfolder_path):
                        filename, extension = os.path.splitext(image_data)  # extension: .png
                        new_image_name = f"{filename}_{effect_name}{extension}"  # create new file name -> ex: speed_72_mpwidth_40_haz_91_thickness_8_direction_1_distance_64_horizontal_flip.png
                        new_image_save_path = os.path.join(output_merged_folder, new_image_name)

                        image_data_path = os.path.join(subfolder_path, image_data)
                        shutil.copy(image_data_path, new_image_save_path)  # copy the file to the new folder with new name
def main():



    #if merge all effects:
    merge_all_effects(parent_folder, merged_folder)

    #if merge only selected effects: currently we focus on the effects without color mapping changes (exclude jet colormap and gray_contrast for now)
    #selected_effects = ["original","median_blur_filter","horizontal_flip"]
    #merge_selected_effects(parent_folder, merged_folder_3_effects, selected_effects)


if __name__ == "__main__":
    main()