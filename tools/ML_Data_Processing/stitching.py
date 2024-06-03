#ref: https://github.com/kvmani/machineLearning/blob/35a0537554ffb4180d146c390f85df8ddac776e5/src/grainStructure/stitching.py
import os
import numpy as np
from PIL import Image

def stitchImages(im0, im2, mode='horizontal'):
    """
    Stitching the two numpy array images and returning a combined image.
    """
    if im0.shape == im2.shape:
        if len(im0.shape) == 2:
            XPixcels, YPixcels = im0.shape
            dtype = im0.dtype
            if "horizontal" in mode:
                combined = np.zeros((XPixcels, YPixcels * 2), dtype=dtype)
                combined[:, 0:YPixcels] = im0[:, :]
                combined[:, YPixcels:2 * YPixcels] = im2[:, :]
            elif "vertical" in mode:
                combined = np.zeros((XPixcels * 2, YPixcels), dtype=dtype)
                combined[0:XPixcels, :] = im0[:, :]
                combined[XPixcels:2 * XPixcels, :] = im2[:, :]
            else:
                raise ValueError("Unknown stitching mode: Only horizontal and vertical are supported")
        else:
            XPixcels, YPixcels, channels = im0.shape
            dtype = im0.dtype
            if "horizontal" in mode:
                combined = np.zeros((XPixcels, YPixcels * 2, 3), dtype=dtype)
                combined[:, 0:YPixcels, :] = im0[:, :, :]
                combined[:, YPixcels:2 * YPixcels, :] = im2[:, :, :]
            elif "vertical" in mode:
                combined = np.zeros((XPixcels * 2, YPixcels, 3), dtype=dtype)
                combined[0:XPixcels, :, :] = im0[:, :, :]
                combined[XPixcels:2 * XPixcels, :, :] = im2[:, :, :]
            else:
                raise ValueError("Unknown stitching mode: Only horizontal and vertical are supported")
        return combined
    else:
        raise ValueError(f"The shapes of the images {im0.shape} {im2.shape} are not matching !!")


# Define folders containing the PNG images
#folder1_path = r"D:\Aishwarya\grainStructure\dataV2_for_code_cleaning\fodlertoHELPCode Fainalization\random_data\source_png"
#folder2_path = r"D:\Aishwarya\grainStructure\dataV2_for_code_cleaning\simulation_images_generation_cut75_xy64only_resize"

parent_folder = r"E:\Data\data_augmentation_all\data_augmentation_xy_0_32_64"

effect_list = ["original", "jet", "gray_contrast", "median_blur_filter", "horizontal_flip"]
#effect_name = "original" #select the effects to work on
#effect_name = "jet" #select the effects to work on
#effect_name = "gray_contrast" #select the effects to work on
#effect_name = "median_blur_filter" #select the effects to work on
#effect_name = "horizontal_flip" #select the effects to work on

for effect_name in effect_list:

    processed_folder_name = "Processed_"+effect_name
    processed_folder_path = os.path.join(parent_folder, processed_folder_name)
    #special folder creation: create processed folder here (after first run, we already have left side color coded source_png, we want to created folder to save results for other effects)
    os.makedirs(processed_folder_path, exist_ok=True)

    sourcePng_folder_name = "random_data\\source_png" #in last step sourceGeneration, we generate color code left side using "original" effect, so we can use the static address for accessing source data
    locate_original_folder_path = os.path.join(parent_folder,"Processed_original") #note, first run we generate using "original" effects data
    sourcePng_folder_path = os.path.join(locate_original_folder_path, sourcePng_folder_name) #NOTE: USE AFTER 1ST RUN: use this after first effects (original) already generated

    ######### USE ONLY FOR 1ST RUN ####################
    #sourcePng_folder_path = os.path.join(processed_folder_path, sourcePng_folder_name) #USE THIS FOR 1ST effect RUN, comment out in other effects since we have source_png in fixed location

    ########################################################################

    ######### USE THE FOLLOWING ADDRESS AFTER FIRST RUN ####################
    #COMMENT OUT THE FOLLOWING TWO LINES IN FIRST RUN, USE IT AFTERWARDS

    ########################################################################

    folder1_path = sourcePng_folder_path #get parameter color image as left side for pairing
    folder2_path = os.path.join(parent_folder, effect_name) #get image (target) as right side for pairing


    # Define output folder
    #output_folder = "D:\Aishwarya\grainStructure\dataV2_for_code_cleaning\stitchedImages"
    output_folder = os.path.join(processed_folder_path, "stitchedImages")

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over files in both folders and stitch corresponding images
    for filename1 in os.listdir(folder1_path):
        if filename1.endswith('.png'):
            # Check if there's a corresponding file in folder2
            corresponding_file_path = os.path.join(folder2_path, filename1)
            if os.path.exists(corresponding_file_path):
                # Load images from both folders
                img1 = np.array(Image.open(os.path.join(folder1_path, filename1)))
                img2 = np.array(Image.open(corresponding_file_path))

                # Stitch the images horizontally
                combined_image = stitchImages(img1, img2, mode='horizontal')

                # Save the combined image
                combined_image_path = os.path.join(output_folder, filename1)
                Image.fromarray(combined_image.astype(np.uint8)).save(combined_image_path)
                print(f"Combined image saved: {combined_image_path}")

print("Stitching completed!")
