import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# Define the main directory path


parent_folder = r"E:\Data\data_augmentation_all\data_augmentation_xy_0_32_64"
effect_name = "original" #select the effects to work on
processed_folder_name = "Processed_"+effect_name
processed_folder_path = os.path.join(parent_folder, processed_folder_name)

main_dir = processed_folder_path
C_folder_path = os.path.join(processed_folder_path,"C")
#main_dir = r'D:\Aishwarya\grainStructure\dataV2\fodlertoHELPCode Fainalization'
#C_folder_path = r'D:\Aishwarya\grainStructure\dataV2\fodlertoHELPCode Fainalization\C'



# Create the 'random_data' directory
random_data_dir = os.path.join(main_dir, 'random_data')
os.makedirs(random_data_dir, exist_ok=True)

# Create necessary folders inside 'random_data'
folders = ['random_data/source', 'random_data/target', 'random_data/source_png']
for folder in folders:
    folder_path = os.path.join(main_dir, folder)
    os.makedirs(folder_path, exist_ok=True)

print("Folder structure created successfully!")

def destitchImages(combined, mode='horizontal'):
    if len(combined.shape) == 2:
        XPixcels, YPixcels = combined.shape
        if "horizontal" in mode:
            im0 = combined[:, 0:YPixcels // 2]
            im2 = combined[:, YPixcels // 2:]
        elif "vertical" in mode:
            im0 = combined[0:XPixcels // 2, :]
            im2 = combined[XPixcels // 2:, :]
        else:
            raise ValueError("Unknown destitching mode: Only horizontal and vertical are supported")
    else:
        XPixcels, YPixcels, _ = combined.shape
        if "horizontal" in mode:
            im0 = combined[:, 0:YPixcels // 2, :]
            im2 = combined[:, YPixcels // 2:, :]
        elif "vertical" in mode:
            im0 = combined[0:XPixcels // 2, :, :]
            im2 = combined[XPixcels // 2:, :, :]
        else:
            raise ValueError("Unknown destitching mode: Only horizontal and vertical are supported")

    return im0, im2

def convertNpyToPng(data_folder, png_save_path):
    os.makedirs(png_save_path, exist_ok=True)
    png_files = []

    files = os.listdir(data_folder)

    for file in files:
        if file.endswith('.npy'):
            file_path = os.path.join(data_folder, file)
            data = np.load(file_path)[:, :, 0]

            plt.imshow(data, vmin=-0.5, vmax=0.5, cmap='jet')
            plt.axis('off')

            # Save the resized image with the original filename
            resized_file_path = os.path.join(png_save_path, f"{os.path.splitext(file)[0]}.png")
            plt.savefig(resized_file_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            img = Image.open(resized_file_path)
            img_rgb = img.convert('RGB')
            img_resized = img_rgb.resize((256, 256))
            img_resized.save(resized_file_path)
            png_files.append(resized_file_path)
            print(f"Resized and added to png_files: {resized_file_path}")

    print("PNG Conversion completed successfully!")

    return png_files

def processNpyFiles(folder_path):
    source_folder = os.path.join(random_data_dir, 'source')
    for directory in [source_folder]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    npy_files = [file for file in os.listdir(folder_path) if file.endswith('.npy')]

    for npy_file in npy_files:
        combined = np.load(os.path.join(folder_path, npy_file))

        # Destitch the combined image into two separate images
        im0, _ = destitchImages(combined)  # We only need the source image

        # Save the first image to the source folder
        np.save(os.path.join(source_folder, npy_file), im0)

        print(f"Processed and saved source image from {npy_file}")

    source_png_files = convertNpyToPng(
        os.path.join(random_data_dir, 'source'),
        os.path.join(random_data_dir, 'source_png')
    )

    print(f"source_png_files: {source_png_files}")

processNpyFiles(C_folder_path)



