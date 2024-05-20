# Objective: The goal for this program is to getting the training samples.
# The requirement depends as follows:
# Stage 1: We will extract once slice of data from only one direction, specifically xy_64 only, (e.g. xy_64) from each simulation and save all of them to one folder and rename is using their simulation parameters (e.g: speed_3_mpwidth_25_haz_40_thickness_8.png)
# Stage 2: we will extract slices from two direction from one simulation (e.g. xy_64, xz_64) and save all to one separate folder using the simulation parameters as their names (e.g: speed_3_mpwidth_25_haz40_thickness_8_xy_64.png)
# Stage 3: We may considre more directions and more slices, like xy_64 and xy_32 and xy_96 and also xz_64.... the filename is based on these settings.
# The output of this program is a folders with extracted data inside.
import os
import shutil
import cv2

def getting_images_xy64only():
    """
        This function will extract original images (xy_64 only) that already being processed from original simulation and resize and write it to new folder.
        Algorithm:
        1. Getting original image folder path
        2. Set up new folder name for saving extracted images, then create the new folder if it doesn't exist.
        3. Iterate simulation folders and iterate images. find the wanted one (this function we only want image of xy_64).
        4. Read in image and resize to 256 * 256 pixels
        5. create new name for the image which used for saving in the new folder (new folder require image to have parameters in the filename)
        6. Write image with its new name to the new folder

        """
    #parent_folder = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\tools" #demo path for testing the code
    #simulation_folder_path = os.path.join(parent_folder, "simulation_cut75_demo") #demo path for testing the code

    parent_folder = r"D:\Zhaochen\simulation_SPPARKS_hpc\SPPARKS_scripts_generation_128_20240306"
    simulation_folder_path = os.path.join(parent_folder, "simulation_images_generation_cut75_JET")

    target_folder_xy64only = os.path.join(parent_folder,"simulation_images_generation_cut75_JET_xy64only_resize") #create new folder name and path

    if not os.path.exists(target_folder_xy64only):
        os.makedirs(target_folder_xy64only)
    for simu_folder in os.listdir(simulation_folder_path):
        simulation_name = simu_folder #getting the simulation name string
        simu_folder_path = os.path.join(simulation_folder_path, simu_folder)
        for img in os.listdir(simu_folder_path):
            if img.startswith("xy_64"):
                #resize image to 256 * 256
                img_path = os.path.join(simu_folder_path, img)  #getting wanted image (xy_64)
                org_img = cv2.imread(img_path)
                resized_img = cv2.resize(org_img, (256, 256))

                #rename the image to new name for storing in different folder
                img_rename = simulation_name+".png" #use simulation name as the image name
                destination_img_path = os.path.join(target_folder_xy64only, img_rename)

                #save resize image to new folder
                cv2.imwrite(destination_img_path, resized_img)
                #ref: create new name for images in target folder
                #dir_distance = "_".join(img.split("_")[:2])  #get prefix (e.g xy_64)
                #img_rename = simu_folder + "_"+dir_distance # eample output: speed_3_mpwidth_10_haz_48_thickness_11_xy_64.png
                #print(destination_img_path) #e.g: C:\Users\zg0017\PycharmProjects\spparks-simulation\tools\simulation_images_generation_cut75_xy64only\speed_3_mpwidth_10_haz_48_thickness_11.png
                # print(f"Copied {img_path} to {destination_img_path}")

                #shutil.copy(img_path, destination_img_path) #copy file from existing folder to new folder
                #print(f"Copied xy_64 image to {destination_img_path}")

def getting_images_xy64_xz64():
    """
    This function will extract original images (xy_64 and xz_64 ) that already being processed from original simulation and resize and write it to new folder.
    Algorithm:
    1. Getting original image folder path
    2. Set up new folder name for saving extracted images and then create the new folder if it doesn't exist.
    3. Iterate simulation folders and iterate images. find the wanted one (this function we want xy_64 and xz_64)
    4. Read in image and resize to 256 * 256 pixels
    5. create new name for the image which used for saving in the new folder (new folder require image to have parameters in the filename)
    6. Write image with its new name to the new folder

    """
    #parent_folder = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\tools" #demo path for testing the code
    #simulation_folder_path = os.path.join(parent_folder, "simulation_cut75_demo") #demo path for testing the code

    parent_folder = r"D:\Zhaochen\simulation_SPPARKS_hpc\SPPARKS_scripts_generation_128_20240306"
    simulation_folder_path = os.path.join(parent_folder, "simulation_images_generation_cut75_JET")

    target_folder_xy64_xz64 = os.path.join(parent_folder,"simulation_images_generation_cut75_JET_xy64_xz64_resize") #create new folder name and path

    if not os.path.exists(target_folder_xy64_xz64):
        os.makedirs(target_folder_xy64_xz64)
    for simu_folder in os.listdir(simulation_folder_path):
        simulation_name = simu_folder #getting the simulation name string
        simu_folder_path = os.path.join(simulation_folder_path, simu_folder)
        for img in os.listdir(simu_folder_path):
            if img.startswith("xy_64") or img.startswith("xz_64"):
                #resize image to 256 * 256
                img_path = os.path.join(simu_folder_path, img)  #getting wanted image (xy_64)
                org_img = cv2.imread(img_path)
                resized_img = cv2.resize(org_img, (256, 256)) #resize image to 256 * 256

                #rename the image to new name for storing in different folder
                image_direction = img.split("_")[0] #getting direction string（xy or xz)
                image_distance = img.split("_")[1]  #getting teh distance (64, 32 or 96 dep)
                img_rename = simulation_name+"_direction_"+image_direction+"_distance_"+image_distance+".png" #e.g: speed_3_mpwidth_10_haz_48_thickness_11_direction_xy_distance_64.png
                destination_img_path = os.path.join(target_folder_xy64_xz64, img_rename) #create the whole path for saving

                #save resize image to new folder
                cv2.imwrite(destination_img_path, resized_img)

def getting_images_multiDirection_multiDistance():
    """
        This function will extract images with multiple directions (xy, xz, yz) and multiple distances (32, 64, 96) that
                                already being processed from original simulation and resize and write it to new folder.
        Algorithm:
        1. Getting original image folder path
        2. Set up new folder name for saving extracted images and then create the new folder if it doesn't exist.
        3. Iterate simulation folders and iterate images. find the wanted ones
            (xy_32, xy_64, xy_96, xz_32, xz_64, xz_96, yz_32, yz_64, yz_96)
        4. Read in image and resize to 256 * 256 pixels
        5. create new name for the image which used for saving in the new folder (new folder require image to have parameters in the filename)
        6. Write image with its new name to the new folder

        """
    # Demo Simulation Data Folder (for code testing)
    #parent_folder = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\tools" #demo path for testing the code
    #simulation_folder_path = os.path.join(parent_folder, "simulation_cut75_demo") #demo path for testing the code

    # Real Simulation Data Folder
    parent_folder = r"D:\Zhaochen\simulation_SPPARKS_hpc\SPPARKS_scripts_generation_128_20240306"
    simulation_folder_path = os.path.join(parent_folder, "simulation_images_generation_cut75_JET")

    target_folder_multiDir_multiDist = os.path.join(parent_folder,"simulation_images_generation_cut75_JET_multiDirection_multiDistance_resize") #create new folder name and path

    if not os.path.exists(target_folder_multiDir_multiDist):
        os.makedirs(target_folder_multiDir_multiDist)
    for simu_folder in os.listdir(simulation_folder_path):
        simulation_name = simu_folder #getting the simulation name string
        simu_folder_path = os.path.join(simulation_folder_path, simu_folder)
        for img in os.listdir(simu_folder_path):
            image_prefix_list = ["xy_32", "xy_64", "xy_96", "xz_32", "xz_64", "xz_96", "yz_32", "yz_64", "yz_96"]
            #if img.startswith("xy_64") or img.startswith("xz_64"):
            if any(img.startswith(pf) for pf in image_prefix_list):
                #resize image to 256 * 256
                img_path = os.path.join(simu_folder_path, img)  #getting wanted image (xy_64)
                org_img = cv2.imread(img_path)
                resized_img = cv2.resize(org_img, (256, 256)) #resize image to 256 * 256

                #rename the image to new name for storing in different folder
                image_direction = img.split("_")[0] #getting direction string（xy or xz)
                image_distance = img.split("_")[1]  #getting teh distance (64, 32 or 96 dep)
                img_rename = simulation_name+"_direction_"+image_direction+"_distance_"+image_distance+".png" #e.g: speed_3_mpwidth_10_haz_48_thickness_11_direction_xy_distance_64.png
                destination_img_path = os.path.join(target_folder_multiDir_multiDist, img_rename) #create the whole path for saving

                #save resize image to new folder
                cv2.imwrite(destination_img_path, resized_img)

def main():
    stage1 = False  #getting xy_64 only data from all simualation (1 direction, 1 slice distance-64)
    stage2 = False #getting xy_64 and xz_64 from all simulations (2 or more directions (currently focus on xy and xz, 1 slice_distance-64)
    stage3 = True #getting multiple directions and multiple distances

    if stage1 == True:
        getting_images_xy64only()
    if stage2 == True:
        getting_images_xy64_xz64()
    if stage3 == True:
        getting_images_multiDirection_multiDistance()

if __name__ == "__main__":
    main()