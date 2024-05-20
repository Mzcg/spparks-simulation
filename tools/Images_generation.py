# Objective: This program aim to generate 5*3 slice images + 1 full view image for each simulation will store it to separate folder and files for easily access further.
import get_last_dump  #import another code (this code has function to obtain the path of last dump file from each simulation folder)
import dump2image_slice #import another program(this prog has function to generate slices images and 3d view from dump file)
import os
import csv

"""
    Input:
        - Individual simluation folders (multiple) within the cleaned parent simulation result folder (SPPARKS_scripts_generation_128_20240306)
    Output:
        - Last_dump_file_list.csv: this program will get access to the last dump file from each individual simulation folder. we will output the list while we access them for reference. the folder will be save as path.
        - simulation_images_generation folder: we create a folder to store the sliced images separately. within this folder, we will generate simulation folders (same file name as we got simulation) and we store sliced iamges and 1 3d images (3 directions * 5 + 1 full 3d view)
"""

#data_folder_path = r'C:\Users\zg0017\PycharmProjects\spparks-simulation\tools\clean_simulation_results_demo'  #demo test
data_folder_path = r'D:\Zhaochen\simulation_SPPARKS_hpc\SPPARKS_scripts_generation_128_20240306'
dump_file_path_list = get_last_dump.get_last_dump_file(data_folder_path)

#image_generation_storage_folder_name = "simulation_images_generation"
image_generation_storage_folder_name = "simulation_images_generation_JET"
image_generation_storage_folder_path = os.path.join(data_folder_path, image_generation_storage_folder_name ) #set up path for creating a new folder to store images
os.makedirs(image_generation_storage_folder_path, exist_ok= True)#creating a new folder to store images (will have subfolder inside later) #exist_oK: if folder exists, no action will be taken.
#print(f"Folder '{image_generation_storage_folder_name}' has been created at '{data_folder_path}'")

print(len(dump_file_path_list))

######################################################
#save the dump_file_list to a csv file for reference
csv_path = os.path.join(data_folder_path, 'Last_dump_file_list.csv')
with open(csv_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file) #create writer object
    for dump_path in dump_file_path_list: #write each string from the list as a separate row in the CSV file
        writer.writerow([dump_path])
print(f'dump file list saved to {data_folder_path}')
#####################################################

for single_dump_file_path in dump_file_path_list:

    #step1: set up new folders to store the generated image data later
    simu_folder_name = single_dump_file_path.split("\\")[-2]  #get the folder name from the path and used it to create a new folder in simulation_images_generation folder
    image_simu_folder_path = os.path.join(image_generation_storage_folder_path, simu_folder_name) #set the new folder path
    os.makedirs(image_simu_folder_path, exist_ok=True) #folder generation: create folder for each simulations inside separate images folders

    print(image_simu_folder_path)
    #step2: generate images within created folders
    #step 2a: generate full view 3D images
    ## full_view_path = os.path.join(image_simu_folder_path,"full_view.png")
    ## dump2image_slice.plot_3D_view(single_dump_file_path, output_filepath=full_view_path)

    #step 2b: generate xy, yz, xz
    output_slices_folder_path = os.getcwd()
    dump2image_slice.plot_distance_slices(single_dump_file_path, image_simu_folder_path)  # function call: generate multiple slices images according to distance






