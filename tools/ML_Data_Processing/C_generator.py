# Reference: Aishwary's Github: https://github.com/kvmani/machineLearning/blob/1224fec16cc39e6a74b45d7644e0484d41729ac6/src/grainStructure/C_generator.py

import os
import glob
import numpy as np
import re
parent_folder = r"E:\Data\data_augmentation_all\data_augmentation_xy_0_32_64"


effect_name = "original" #select the effects to work on
npy_folder_name = "NPY_"+effect_name
#npy_folder = r"E:\Data\data_augmentation_all\data_augmentation_xy_0_32_64\NPY_original"
npy_folder = os.path.join(parent_folder,npy_folder_name) #e.g: E:\Data\data_augmentation_all\data_augmentation_xy_0_32_64\NPY_original

folderA = "A"
folderB = "B"
folderC = "C"

processed_data_folder_name = "Processed_" + effect_name #e.g: Processed_original -> store all the processed data
processed_data_folder_path = os.path.join(parent_folder, processed_data_folder_name)

#output_directory_A = r"D:\Aishwarya\grainStructure\dataV2\A"
output_directory_A = os.path.join(processed_data_folder_path, folderA)
output_directory_B = os.path.join(processed_data_folder_path, folderB)
output_directory_C = os.path.join(processed_data_folder_path, folderC)


debug = False



normalization_params_all_materials = {

    "grainStructureXY":
                {
                'speed': 72.0,
                'meltpool_width': 40.0,
                'haz': 91.0,
                'layerThickness': 11.0,
            },
    "grainStructureXY_0_32_64":
                {
                'speed': 72.0,
                'meltpool_width': 40.0,
                'haz': 91.0,
                'layerThickness': 11.0,
                'distance': 64,
            },
    "grainStructure_multi_direction_multi_distance":
                {
                'speed': 72.0,
                'meltpool_width': 40.0,
                'haz': 91.0,
                'layerThickness': 11.0,
                'direction': 3,
                'distance': 64,
            },


}

normalization_params = normalization_params_all_materials["grainStructureXY_0_32_64"]

imDatasetMathType = "logarithmic" # choices = "nonLogarithmic" or "logarithmic"

# Iterate over .npy files in the folder
npy_files = glob.glob(os.path.join(npy_folder, "*.npy"))

pattern = r'[\d.]+'

# Create output directories if they don't exist
for output_directory in [output_directory_A, output_directory_B, output_directory_C]:
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

for npy_file in npy_files:
    matches = re.findall(pattern, os.path.basename(npy_file))
    matches = [match.rstrip('.') for match in matches]
    if len(matches) != len(normalization_params):
        print(f"Unable to extract all values from file name: {npy_file}")
        continue

    extracted_values = {key: float(value) for key, value in zip(normalization_params.keys(), matches)}

    tempArray = np.load(npy_file)


    def stitchImages(im0, im2, mode='horizontal'):
        """
        stiching the two numpy array images and returning a combined image (for Pix2Pix etc).
        """
        if im0.shape == im2.shape:
            if len(im0.shape) == 2:
                XPixcels, YPixcels = im0.shape
                dtype = im0.dtype
                if "horizontal" in mode:
                    combined = np.zeros((XPixcels, YPixcels * 2), dtype=dtype)
                    combined[:, 0:YPixcels, ] = im0[:, :, ]
                    combined[:, YPixcels:2 * YPixcels, ] = im2[:, :, ]
                elif "vertical" in mode:
                    combined = np.zeros((XPixcels * 2, YPixcels, 3), dtype=dtype)
                    combined[0:XPixcels, :, ] = im0[:, :, ]
                    combined[XPixcels:2 * XPixcels, :, ] = im2[:, :, ]
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
            raise ValueError(f"The shapes of the images {im0.shape} {im2.shape} are not matcing !!")


    def generateSourceArraypix2pix(extracted_values, normalization_params):
        """
        Generate a 256X256X3 nparray which, when plotted, looks like an image with three rows,
        each for one parameter except for temperature.

        :parameters: extracted_values (dictionary with 'power', 'velocity', 'timestamp'),
                     normalization_params (dictionary containing normalizing values)
        :return: 256X256X3 nparray
        """
        final_image = np.zeros((256, 256))

        for i, (param, value) in enumerate(extracted_values.items()):
            normalized_value = (float(value) / normalization_params[param]) - 0.5

            top = i * (256 // len(extracted_values))
            bottom = (i + 1) * (256 // len(extracted_values))
            final_image[top:bottom, :] = normalized_value

        stackedArray = np.dstack([final_image] * 3)

        return stackedArray


    def generateTargetArray(tempArray):


        return tempArray


    def makeTrainExampleNum2Pix(extracted_values, tempArraySlice, normalization_params, imDatasetMathType):
        """
        Create a training example for Num2Pix.

        :parameters : extracted_values (dictionary with 'power', 'velocity', 'timestamp'),
                      tempArraySlice (a slice of the temperature array),
                      normalization_params (dictionary containing normalizing values)
        :return: A tuple of three 256X512X3 normalized numpy arrays: A, B, C
        """
        A = generateSourceArraypix2pix(extracted_values, normalization_params)
        B = generateTargetArray(tempArraySlice)
        C = stitchImages(A, B)

        return A, B, C


    A, B, C = makeTrainExampleNum2Pix(extracted_values, tempArray, normalization_params, imDatasetMathType)

    output_file_A = os.path.join(output_directory_A, os.path.basename(npy_file))
    output_file_B = os.path.join(output_directory_B, os.path.basename(npy_file))
    output_file_C = os.path.join(output_directory_C, os.path.basename(npy_file))

    np.save(output_file_A, A)
    np.save(output_file_B, B)
    np.save(output_file_C, C)