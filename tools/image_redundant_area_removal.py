import os
import cv2
import numpy as np

'''
#Task 1: try on single demo image
Task 1a: remove the black area surrounding the simulation box area for single direction image
Task 1b: cut and main only 75/128 (based on size ratio) from the bottom.
'''
import cv2
import numpy as np

# Load the image
image = cv2.imread('demo_remove_yz128.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the grayscale image to create a binary mask
_, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# Find contours in the binary mask
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assuming the colored area is the largest contour, find it
if contours:
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the original image to the bounding rectangle
    cropped_image = image[y:y + h, x:x + w]

    # Save the cropped image
    cv2.imwrite('noblack.png', cropped_image)

    print("Cropped image saved as 'noblack.png")
else:
    print("No colored area found in the image.")


#TASK 1b: cut based on ratio


before_crop = cv2.imread("noblack.png")
height, width, channels = before_crop.shape
print("Before crop: Height-", height, " width-", width, )



keep_height = int(height * 0.75) # Calculate the height to keep (75% of the total height)
start_row = height - keep_height # Calculate the starting row for cropping (top part to remove)
cut_image = before_crop[start_row:, :] #crop the image to keep the bottom 75% and remove the top 25%
cut_image_path = 'noblack_cutimage_75ratio.png'
cv2.imwrite(cut_image_path, cut_image)
print(f"Cropped image saved as '{cut_image_path}'")

# Display the size of the cropped image
cut_height, cut_width, cut_channels = cut_image.shape
print(f"Size of the cropped image: Width = {cut_width}, Height = {cut_height}, Channels = {cut_channels}")



#no do resize now, will do it later when preprocessin for ML.
#resized_image = cv2.resize(cut_image, (128, 128), interpolation=cv2.INTER_CUBIC)
#cv2.imwrite("noblack_cutimage_resize128.png", resized_image)

#resizeback = cv2.resize(resized_image, (531,397)) #just test
#cv2.imwrite("noblack_resizeback.png", resizeback)



