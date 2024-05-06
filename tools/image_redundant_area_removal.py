import os
import cv2
import numpy as np
import shutil

'''
#Task 1: try on single demo image
Task 1a: remove the black area surrounding the simulation box area for single direction image
Task 1b: cut and main only 75/128 (based on size ratio) from the bottom.
'''

data_folder_path = r'C:\Users\zg0017\PycharmProjects\spparks-simulation\tools\clean_simulation_results_demo'  #demo test
#image_path = 'demo_remove_yz128.png' #image with black area outside of box area.
output_image_path_noblack = 'noblack.png'
def remove_black_area(image_path):

    image = cv2.imread(image_path) # Load the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
    _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY) # Threshold the grayscale image to create a binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours in the binary mask

    if contours: # Assuming the colored area is the largest contour, find it

        largest_contour = max(contours, key=cv2.contourArea) # Find the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour) # Get the bounding rectangle of the largest contour
        blackremoved_image = image[y:y + h, x:x + w] # Crop the original image to the bounding rectangle

        # Save the black-removed image
        #cv2.imwrite('noblack.png', blackremoved_image)
        cv2.imwrite(image_path, blackremoved_image)

        #print("black removed image saved as 'noblack.png")
    #else:
    #   print("No colored area found in the image.")


#TASK 1b: cut based on ratio
noblack_image_path = os.path.join(os.getcwd(), 'demo_remove_yz128.png')

def crop_75_from_bottom(noblack_image_path, output_cropped_image_folder_path):
    print(noblack_image_path)
    filename = noblack_image_path.split("\\")[-1]
    print(filename)
    prefix = filename.split(".")[0]
    new_string = prefix+ "_cut75.png"
    print(new_string)
    '''
    
    #before_crop = cv2.imread("noblack.png")
    before_crop = cv2.imread(noblack_image_path)
    height, width, channels = before_crop.shape
    #print("Before crop: Height-", height, " width-", width, )

    keep_height = int(height * 0.75) # Calculate the height to keep (75% of the total height)
    start_row = height - keep_height # Calculate the starting row for cropping (top part to remove)
    cut_image = before_crop[start_row:, :] #crop the image to keep the bottom 75% and remove the top 25%
    #cut_image_path = 'noblack_cutimage_75ratio.png'

    orig_filename = noblack_image_path.split("/")
    print(noblack_image_path)
    cv2.imwrite(output_cropped_image_folder_path, cut_image)
    #print(f"Cropped image saved as '{cut_image_path}'")

    # Display the size of the cropped image
    cut_height, cut_width, cut_channels = cut_image.shape
    print(f"Size of the cropped image: Width = {cut_width}, Height = {cut_height}, Channels = {cut_channels}")
    '''
crop_75_from_bottom(noblack_image_path, "tttest.png")

#no do resize now, will do it later when preprocessin for ML.
#resized_image = cv2.resize(cut_image, (128, 128), interpolation=cv2.INTER_CUBIC)
#cv2.imwrite("noblack_cutimage_resize128.png", resized_image)

#resizeback = cv2.resize(resized_image, (531,397)) #just test
#cv2.imwrite("noblack_resizeback.png", resizeback)

def main():
    REMOVE_BLACK = True
    CUT_75 = True


    #step 1: remove black area among the images (store to new folder: "simulation_images_generation_removeBlack")
    blackremoval_folder_name = "simulation_images_generation_removeBlack"
    blackremoval_folder_path = os.path.join(data_folder_path, blackremoval_folder_name)  # set up path for creating a new folder to store images
    src_folder = os.path.join(data_folder_path,"simulation_images_generation")
    '''
    if REMOVE_BLACK == True:
        shutil.copytree(src_folder, blackremoval_folder_path, dirs_exist_ok=True)  #copy the complete folder to new folder (will create the new folder at same time)
    
        for simu_folder in os.listdir(blackremoval_folder_path):
            simu_folder_path = os.path.join(blackremoval_folder_path, simu_folder)
            #print("folder name: " , simu_folder_path)
            for img in os.listdir(simu_folder_path):
                img_path = os.path.join(simu_folder_path, img)
                #print(img_path)
                remove_black_area(img_path)
    '''
    if CUT_75 == True:
        #step 2: cut 75/128 from bottom of images (store images to new folder) - keep extra files for references
        #2a: create an empty folder named "simulation_images_generation_cut75"
        cut75_folder_name = "simulation_images_generation_cut75"
        cut75_folder_path = os.path.join(data_folder_path, cut75_folder_name) # new folder to store 75 cut images.
        print(cut75_folder_path)
        os.makedirs(cut75_folder_path, exist_ok=True)  # folder generation: create folder for each simulations inside separate images folders
        #shutil.copytree(blackremoval_folder_path, cut75_folder_path)

        for simu_folder_noblack in os.listdir(blackremoval_folder_path): #get the noblack image data
            simu_folder_noblack_path_org = os.path.join(blackremoval_folder_path, simu_folder_noblack) #path for obtain the simulation folder from balck removal folder
            simu_folder_noblack_path_new = os.path.join(cut75_folder_path, simu_folder_noblack) #create path for each simulation folder in new folder
            os.makedirs(simu_folder_noblack_path_new, exist_ok=True) #create subfolder
            for img_noblack in os.listdir(simu_folder_noblack_path_org):
                if img_noblack == "full_view.png": #exclude the full_view (3d file) to process
                    continue
                img_noblack_path_org = os.path.join(simu_folder_noblack_path_org, img_noblack) #get the image as input to the function.
                print(img_noblack_path_org)

                # to do:
                #1. fix the fucntion of crop 75 from bottom
                #2. call function and put image in
                #3. store image to output file (simu_folder_noblack_path_new)
                #output will see in the new folder, we have cutted image of simulation witn 75/128 size from bottom)


if __name__ == "__main__":
    main()


