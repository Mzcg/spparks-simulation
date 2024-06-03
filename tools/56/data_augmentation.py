# objective: this program aim to generate more data by doing data augmentation
# It consist two part: 1) testing on different effects. 2) processing large images dataset generation

import os
import numpy as np
import imgaug.augmenters as iaa
import imageio
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageOps
import cv2
import random
import torch
from torchvision.transforms import v2
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def equalized(image_path):
    # Load the image
    #image = imageio.imread(r'C:\Users\zg0017\PycharmProjects\spparks-simulation\tools\speed_63_mpwidth_40_haz_74_thickness_11_direction_xy_distance_32.png')
    #image = imageio.imread(r'C:\Users\zg0017\PycharmProjects\spparks-simulation\tools\speed_23_mpwidth_25_haz_40_thickness_8_xy_64_cut75.png')
    image = imageio.imread(image_path)

    # Define the augmentation
    seq = iaa.Sequential([
        iaa.AllChannelsHistogramEqualization()
    ])

    # Apply the augmentation
    images_aug = seq(images=[image])

    # Display the original and equalized images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    plt.title('Equalized Image')
    plt.imshow(images_aug[0])

    plt.show()

def brightness(image_path):
    image = Image.open(image_path)
    # Adjust brightness
    enhancer = ImageEnhance.Brightness(image)
   #bright_image = enhancer.enhance(1.5)  # Increase brightness by 50%
    bright_image = enhancer.enhance(0.85)  # Increase brightness by 50%

    # Display the original and brightened images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    plt.title('Brightened Image')
    plt.imshow(bright_image)

    plt.show()

def brightness_contrast(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image is loaded successfully
    if image is None:
        print("Error: Unable to load image.")
        return

    # Adjust brightness
    adjusted_brightness = cv2.convertScaleAbs(image, alpha=0.6, beta=0)

    # Enhance contrast
    adjusted_image = cv2.convertScaleAbs(adjusted_brightness, alpha=1.3, beta=0)

    # Save the processed image
    #cv2.imwrite(output_path, adjusted_image)

    # Display the original and processed images (optional)
    cv2.imshow('Original Image', image)
    cv2.imshow('Processed Image', adjusted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def auto_contrast(image_path, output_path=None):
    """
    Apply auto-contrast to an image and save the result.

    Parameters:
    - image_path: str, path to the input image.
    - output_path: str, path to save the output image. If None, the image won't be saved.

    Returns:
    - Image object with auto-contrast applied.
    """

    image = Image.open(image_path) # Load the image
    auto_contrasted_image = ImageOps.autocontrast(image) # Apply auto-contrast
    if output_path:  # Save the result if an output path is provided
        auto_contrasted_image.save(output_path)

    #return auto_contrasted_image
    #auto_contrasted_image.show()

    # Display the original and brightened images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    plt.title('Autocontrast Image')
    plt.imshow(auto_contrasted_image)

    plt.show()

def enhanced_auto_contrast(image_path, output_path=None):
    """
    Apply enhanced auto-contrast to an image and save the result.

    Parameters:
    - image_path: str, path to the input image.
    - output_path: str, path to save the output image. If None, the image won't be saved.

    Returns:
    - Image object with enhanced auto-contrast applied.
    """
    image = cv2.imread(image_path) # Load the image
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb) # Convert the image to YCrCb color space
    y, cr, cb = cv2.split(ycrcb) # Split into channels
    y_eq = cv2.equalizeHist(y)  # Apply histogram equalization to the Y channel
    ycrcb_eq = cv2.merge((y_eq, cr, cb)) # Merge the equalized Y channel back with Cr and Cb channels
    result = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR) # Convert back to BGR color space

    # Display the original and brightened images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    plt.title('enhanced-autocontrast Image')
    plt.imshow(result)
    plt.show()

    # Save the result if an output path is provided
    #if output_path:
    #    cv2.imwrite(output_path, result)

    #return result



def edge_enhance(image_path, blur_kernel_size=(5, 5), threshold=100):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(image, blur_kernel_size, 0)

    # Apply thresholding to increase contrast
    _, thresholded = cv2.threshold(blurred_image, threshold, 255, cv2.THRESH_BINARY)

    # Apply Sobel operator to calculate gradients
    sobelx = cv2.Sobel(thresholded, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(thresholded, cv2.CV_64F, 0, 1, ksize=3)

    # Combine horizontal and vertical gradients to get edge magnitude
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Enhance edges by adding the original image to the scaled edge magnitude
    enhanced_image = np.clip(image + magnitude, 0, 255).astype(np.uint8)

    # Plot the original, blurred, thresholded, and enhanced images
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 4, 2)
    plt.title('Blurred Image')
    plt.imshow(blurred_image, cmap='gray')

    plt.subplot(1, 4, 3)
    plt.title('Thresholded Image')
    plt.imshow(thresholded, cmap='gray')

    plt.subplot(1, 4, 4)
    plt.title('Enhanced Image')
    plt.imshow(enhanced_image, cmap='gray')

    plt.show()

    return enhanced_image

def apply_histogram_equalization(image_path):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(image)

    # Plot the original and equalized images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title('Equalized Image')
    plt.imshow(equalized_image, cmap='gray')

    plt.show()


def create_digital_negative(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Invert the pixel values
    negative_image = 255 - image

    # Plot the original and negative images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title('Digital Negative')
    plt.imshow(cv2.cvtColor(negative_image, cv2.COLOR_BGR2RGB))

    plt.show()

    return negative_image

def sharpen_image(image_path, kernel_size=(5, 5), sigma=0.8, strength=1):
    """
    Sharpen the input image using a sharpening kernel.

    Args:
    - image_path: Path to the input image file.
    - kernel_size: Size of the Gaussian kernel (tuple of integers).
    - sigma: Standard deviation of the Gaussian kernel (float).
    - strength: Strength of the sharpening effect (float).

    Returns:
    - Sharpened image (numpy array).
    """
    # Read the input image
    input_image = cv2.imread(image_path)

    # Apply Gaussian blur to the image
    blurred = cv2.GaussianBlur(input_image, kernel_size, sigma)

    # Calculate the sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])

    # Apply the sharpening kernel to the blurred image
    sharpened = cv2.filter2D(blurred, -1, kernel * strength)

    # Display the original and sharpened images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    plt.title('Sharpened Image')
    plt.axis('off')

    plt.show()

    return sharpened


def canny_edges(image_path, low_threshold=5, high_threshold=10):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)

    # Display the original image and the edge detection map
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Detection Using Canny")

    plt.show()


def enhance_structure(image_path):

    image = cv2.imread(image_path) # Read the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Apply Gaussian blur to reduce noise
    edges = cv2.Canny(blurred, 8, 15) # Perform Canny edge detection
    edges = cv2.dilate(edges, None, iterations=1)  # Dilate the edges to make them more prominent (optional)

    # Perform morphological closing to connect nearby edges
    kernel = np.ones((3,3 ), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Overlay the closed edges onto the original image
    output = np.copy(image)
    output[edges_closed != 0] = [0, 0, 0]  # Highlight edges in black
    #output[edges_closed != 0] = [0, 0, 255]  # Highlight edges in red

    # Display the original image and the enhanced structure
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title("Enhanced Edge")

    plt.show()

def flip_horizontal(image_path, output_path):

    image = cv2.imread(image_path) # Read the image
    flipped_image = cv2.flip(image, 1) # Flip the image horizontally

    cv2.imwrite(output_path, flipped_image)
    # plt.figure(figsize=(10, 5))  # Display the original and flipped images
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title("Original Image")
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
    # plt.title("Horizontally Flipped Image")
    #
    # plt.show()

def apply_median_filter(image_path, output_path, ksize=5):
    image = cv2.imread(image_path)  # Read the image
    median_filterd_image = cv2.medianBlur(image, ksize)

    cv2.imwrite(output_path, median_filterd_image)
    # plt.figure(figsize=(10, 5))  # Display the original and flipped images
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title("Original Image")
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(median_filterd_image, cv2.COLOR_BGR2RGB))
    # plt.title("Median Filtered Image")
    #
    # plt.show()

def apply_non_local_means(image_path, h=5, templateWindowSize=5, searchWindowSize=10):
    image = cv2.imread(image_path)
    non_local_mean_image = cv2.fastNlMeansDenoisingColored(image, None, h, h, templateWindowSize, searchWindowSize)

    plt.figure(figsize=(10, 5))  # Display the original and flipped images

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(non_local_mean_image, cv2.COLOR_BGR2RGB))
    plt.title("Non Local Means Image")

    plt.show()

def apply_bilateral_filter(image_path, d=5, sigmaColor=50, sigmaSpace=75):
    image = cv2.imread(image_path)
    bilateral_image = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

    plt.figure(figsize=(10, 5))  # Display the original and flipped images

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(bilateral_image, cv2.COLOR_BGR2RGB))
    plt.title("Apply Bilateral Filter Image")

    plt.show()


def saturation(image_path, output_path= None, saturation_scale=1.8):
    """
    Load an image, adjust its saturation, save the result, and plot both original and adjusted images.

    Parameters:
    - image_path (str): Path to the input image.
    - output_path (str): Path to save the processed image.
    - saturation_scale (float): Factor to scale the saturation by (default is 1.5).
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Scale the saturation channel
    hsv_image[:, :, 1] = cv2.multiply(hsv_image[:, :, 1], saturation_scale)

    # Clip the saturation values to be in the valid range [0, 255]
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0, 255)

    # Convert the image back to BGR color space
    adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Save the adjusted image
    #cv2.imwrite(output_path, adjusted_image)

    # Plot the original and adjusted images
    plt.figure(figsize=(10, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Saturation adjusted image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB))
    plt.title('Saturation Adjusted Image')
    plt.axis('off')

    # Show the plot
    plt.show()

def gray_contrast(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(gray_image)

    # Convert the equalized grayscale image back to BGR (3 channels)
    equalized_image_3ch = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)

    # Apply bilateral filter to reduce noise while preserving edges
    #denoised_image = cv2.bilateralFilter(equalized_image_3ch, d=9, sigmaColor=75, sigmaSpace=75)

    # Save the processed image
    #cv2.imwrite(output_path, denoised_image)
    cv2.imwrite(output_path, cv2.cvtColor(equalized_image_3ch, cv2.COLOR_BGR2RGB))

    # Plot the original and processed images
    # plt.figure(figsize=(10, 5))

    # # Original image
    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title('Original Image')
    # plt.axis('off')
    #
    # # Processed image (Equalized and Denoised)
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(equalized_image_3ch, cv2.COLOR_BGR2RGB))
    # #plt.imshow(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB))
    # plt.title('Equalized and Denoised Image')
    # plt.axis('off')
    #
    # # Show the plot
    # plt.show()


def color_jitter_and_sharpen_and_smooth(image_path, output_path, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1,
                                        smooth_kernel_size=5):
    """
    Apply color jittering, sharpening, and smoothing to an image, and visualize the result.

    Parameters:
    - image_path (str): Path to the input image.
    - brightness (float): Maximum delta for brightness adjustment.
    - contrast (float): Lower and upper bounds for contrast adjustment.
    - saturation (float): Lower and upper bounds for saturation adjustment.
    - hue (float): Maximum delta for hue adjustment.
    - smooth_kernel_size (int): Size of the smoothing kernel (default is 5).
    """

    def color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        image = tf.image.random_brightness(image, max_delta=brightness)
        image = tf.image.random_contrast(image, lower=1 - contrast, upper=1 + contrast)
        image = tf.image.random_saturation(image, lower=1 - saturation, upper=1 + saturation)
        image = tf.image.random_hue(image, max_delta=hue)
        return image

    def sharpen_image(image):
        image = image.numpy()
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return tf.convert_to_tensor(sharpened, dtype=tf.float32)

    def smooth_image(image, kernel_size=5):
        image = image.numpy()
        kernel_size = int(kernel_size)  # Ensure kernel size is an integer
        smoothed = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return tf.convert_to_tensor(smoothed, dtype=tf.float32)

    # Read and decode the image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32) / 255.0

    # Save original image for comparison
    original_image = image

    # Apply color jitter
    image = color_jitter(image, brightness, contrast, saturation, hue)

    # Sharpen the image
    image = tf.py_function(sharpen_image, [image], tf.float32)

    # Smooth the image
    image = tf.py_function(smooth_image, [image, smooth_kernel_size], tf.float32)

    # Convert TensorFlow tensor to NumPy array
    #image = image.numpy()

    # Ensure pixel values are in the range 0-255 (uint8)
    #image = np.clip(image, 0, 255).astype(np.uint8)

    # Convert RGB to BGR format for OpenCV
    #image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Save the image
    #cv2.imwrite(output_path, image_bgr)

    # Visualize the original and augmented images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image.numpy())
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Augmented Image")
    plt.imshow(image.numpy())
    plt.axis('off')

    plt.show()


# Example usage
#auto_contrasted_image = auto_contrast('path_to_image.jpg', 'auto_contrasted_image.jpg')
#auto_contrasted_image.show()
#auto_contrasted_image.show()

path = r'C:\Users\zg0017\PycharmProjects\spparks-simulation\tools\speed_23_mpwidth_25_haz_40_thickness_8_xy_64_cut75.png'
#path = r'C:\Users\zg0017\PycharmProjects\spparks-simulation\tools\speed_3_mpwidth_25_haz_48_thickness_11_direction_xy_distance_64.png'

#path = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\tools\speed_23_mpwidth_25_haz_40_thickness_8_direction_xy_distance_64.png" #jet
#path = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\tools\Despeckle_GaussianBlur1_speed_3_mpwidth_10_haz_48_thickness_11_direction_1_distance_64.png"

#equalized(path)
#brightness(path)
#brightness_contrast(path)
#auto_contrast(path)
#enhanced_auto_contrast(path)
#edge_enhance(path)


#apply_histogram_equalization(path)
#create_digital_negative(path)
#sharpen_image(path)
#canny_edges(path)
#enhance_structure(path)
#flip_horizontal(path)
#apply_median_filter(path)
#apply_non_local_means(path) #not very good. adjusting parameter is too solid.
#apply_bilateral_filter(path) #not very good, adjusting parameter is too solid
#saturation(path)
#gray_contrast(path)
#color_jitter_and_sharpen_and_smooth(path, output_path = None)

def main():
    #note: we are going to have total 5 types of images: original, median filter, horizontal flip, color_jitter_sharpen_smooth, and jet (this we did already that extract images directly using ovito from dump)

    #demo folder test
    #parent_folder = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\tools"
    #data_folder = os.path.join(parent_folder, "data_aug_demo_xy0_32_64")

    # Real Folder and data generation
    parent_folder = r"D:\Zhaochen\simulation_SPPARKS_hpc\SPPARKS_scripts_generation_128_20240306"
    #data_folder = os.path.join(parent_folder, "simulation_images_generation_cut75_xy0_xy32_xy64_resize") #case 1: xy_0, xy_32, xy_64
    #data_folder = os.path.join(parent_folder, "simulation_images_generation_cut75_xy0_32_64_xz0_32_64_resize") #case 2: xy_0, xy_32, xy_64, xz_0, xz_32, xz_64
    data_folder = os.path.join(parent_folder, "simulation_images_generation_cut75_multiDirection_multiDistance_0_32_64_resize")  #case 3: xy_0, xy_32, xy_64, xz_0, xz_32, xz_64, yz_0, yz_32, yz_64


    #create folders for different effects (data augmentation)
    #data_augmentation_foldername = "data_augmentation_xy_0_32_64"
    #data_augmentation_foldername = "data_augmentation_xy_xz_0_32_64"
    data_augmentation_foldername = "data_augmentation_xy_xz_yz_0_32_64"

    data_augmentation_path = os.path.join(parent_folder, data_augmentation_foldername)

    # Effect 1 folder creation: median filter (adding blur)
    effect1_folder_path = os.path.join(data_augmentation_path, "median_blur_filter") #create folder for saving median blur filtered images
    if not os.path.exists(effect1_folder_path):
        os.makedirs(effect1_folder_path)
        print(f"Subfolder 'median_blur_filter' created in '{data_augmentation_foldername}'.")
    else:
        print(f"Subfolder 'median_blur_filter' already exists in '{data_augmentation_foldername}'.")

    # effect 2 folder creation: horizontal flip
    effect2_folder_path = os.path.join(data_augmentation_path,"horizontal_flip")  # create folder for saving horizontal flipped images
    if not os.path.exists(effect2_folder_path):
        os.makedirs(effect2_folder_path)
        print(f"Subfolder 'horizontal_flip' created in '{data_augmentation_foldername}'.")
    else:
        print(f"Subfolder 'horizontal_flip' already exists in '{data_augmentation_foldername}'.")

    # effect3 folder creation: color_jitter_sharpen_smooth
    effect3_folder_path = os.path.join(data_augmentation_path, "gray_contrast")  # create folder for saving gray contrast images
    if not os.path.exists(effect3_folder_path):
        os.makedirs(effect3_folder_path)
        print(f"Subfolder 'gray_contrast' created in '{data_augmentation_foldername}'.")
    else:
        print(f"Subfolder 'gray_contrast' already exists in '{data_augmentation_foldername}'.")


    for image_file in os.listdir(data_folder):
        image_input_path = os.path.join(data_folder, image_file)
        #print(image_path)

        #apply effects 1: median filter:
        image_output_path1 = os.path.join(effect1_folder_path, image_file)
        apply_median_filter(image_input_path, image_output_path1)

        # apply effects 2: horizontal flip:
        image_output_path2 = os.path.join(effect2_folder_path, image_file)
        flip_horizontal(image_input_path, image_output_path2)

        # apply effects 3: color_jitter_sharpen_smooth
        image_output_path3 = os.path.join(effect3_folder_path, image_file)
        gray_contrast(image_input_path, image_output_path3)

    return 0


if __name__ == "__main__":
    main()
