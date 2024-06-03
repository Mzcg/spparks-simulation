# objective: this program is just to convert the iamges in simulation images parameter analysis to 256 * 256 (consistant) for comparison in same scale

import os
import cv2

# Parent directory containing subfolders
parent_dir = r"D:\Zhaochen\simulation_SPPARKS_hpc\SPPARKS_scripts_generation_128_20240306\simulation_images_parameter_analysis_parameterClassification_resize"

# Loop through subfolders
for subdir in os.listdir(parent_dir):
    subdir_path = os.path.join(parent_dir, subdir)

    # Check if it's a directory
    if os.path.isdir(subdir_path):
        print(f"Processing images in {subdir_path}")

        # Loop through images in the subfolder
        for filename in os.listdir(subdir_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Read image
                img_path = os.path.join(subdir_path, filename)
                img = cv2.imread(img_path)

                # Resize image to 256x256
                resized_img = cv2.resize(img, (256, 256))

                # Save resized image back to the folder
                cv2.imwrite(os.path.join(subdir_path, filename), resized_img)
                print(f"Resized {filename} to 256x256")

print("All images resized successfully.")