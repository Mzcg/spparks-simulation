from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, measure
from scipy.ndimage import binary_erosion, binary_dilation
import random
import os

def visualize_segment_process(image1, image2, image3, image4, image5, image6):
    """
            This funpction will display 6 images to show the original image and intermediate results of image processing and final segmentation results.

            Parameters:
            - image1: original image
            - image2: Gaussian filtered image
            - image3: Grayscale Image after Gaussian filter
            - image4: Sobel Edge detection
            - image5: Binary edges with added dilation
            - image6: final segmented results with color map

            Returns: (None)
            - Just plot images in a window (2*3)

    """
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 3, 1)
    plt.imshow(image1)
    plt.title('Original Image')
    plt.axis('off')

    # Gaussian Filtered Image
    plt.subplot(2, 3, 2)
    plt.imshow(image2)
    plt.title('Gaussian Filtered Image')
    plt.axis('off')

    # Grayscale Image
    plt.subplot(2, 3, 3)
    plt.imshow(image3, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')

    # Sobel Edges
    plt.subplot(2, 3, 4)
    plt.imshow(image4, cmap='gray')
    plt.title('Sobel Edges')
    plt.axis('off')

    # Binary Edges
    plt.subplot(2, 3, 5)
    plt.imshow(image5, cmap='binary')
    # plt.imshow(binary_edges, cmap='binary')
    plt.title('Binary Edges')
    plt.axis('off')

    # Labeled Image
    plt.subplot(2, 3, 6)
    plt.imshow(image6, cmap='jet')  # Use a colormap for better visualization
    plt.title('Labeled Image')
    plt.axis('off')

    plt.show()

def grain_segmentation(image, gaussian_radius=1, sobel_threshold=0.01, dilation_iterations=1,bin_min=0, bin_max=300, bins=15, plot_numbers_on_grains=True):

    """
        This function takes an image in numpy array format to process to segment the microstructure (grain) into different regions. and then calcualte the (weighted) average grain size for each image.

        Parameters:
        - image (numpy array): input image in RGB mode ( Nrow * Ncolumn * 3) - make sure it is 3 channels when input.
        - gaussian_radius (int): radius used when applying GaussianBlur filter
        - sobel_threshold (float): define the threshold to binary_edges after apply sobel edge detection.
        - dilation_iterations (int): iterations used in binary_dilation function (range from 1 to 2)
        - bin_min (int): lower limit set to used for histogram plot
        - bin_max (int): upper limit set to used for histogram plot
        - bins (int): number of bins

        Returns:
        - histogram values (int): 
            - bin_min: lower limit of the grain size to be displayed in histogram (min value can be 0)
            - bin_max: upper limit of the grain size to be displayed in histogram (max values can be the calculated maximum grain size, or self-defined values)
            - bins: number of bins that used for plotting histogram
        - weighted average grain size (float): calculated weighted average grain size for each image
        - number of regions (int): number of grain detected using the algorithms
        - total grain size (float): The sum of all detected grains' size
        - labeled image (array): segmented images saved in 2D-array
        - region list (array): save grain size for all the detected grain regions in a list.
    """
    total_size = 0      #total size of all detected grains (sum of each grain size)
    weighted_sum = 0    #used for calculate weighted average grain size of an image
    regions_list = []   #save all the grain sizes in the list


    # Apply Gaussian filter
    gaussian_image = image_pil.filter(ImageFilter.GaussianBlur(radius=gaussian_radius))
    gaussian_np = np.array(gaussian_image) # Convert the PIL image to a NumPy array after Gaussian filtering

    # Edge Detection with Sobel
    gray_image = gaussian_image.convert('L') # Convert to grayscale for Sobel edge detection
    sobel_edges = filters.sobel(np.array(gray_image)) # Apply Sobel edge detector
    binary_edges = sobel_edges < sobel_threshold  # set Sobel threshold the edge image to binary
    dilated_edges = binary_dilation(binary_edges, iterations=dilation_iterations)

    # Convert binary image to labeled image
    labeled_image, num_labels = measure.label(dilated_edges, connectivity=1, return_num=True)

    # coloring the segmented results with random set colors
    random.seed(42)  # Set a seed for reproducibility
    label_colors = {label: tuple(np.random.randint(0, 256, size=3)) for label in range(1, num_labels + 1)}
    randomized_labels = list(range(1, num_labels + 1))
    random.shuffle(randomized_labels)

    randomized_colored_image = np.zeros_like(image_np)
    for label, color in zip(randomized_labels, label_colors.values()):
        randomized_colored_image[labeled_image == label] = color

    # option: Display marked region number on segmented grains
    if plot_numbers_on_grains:
        plt.figure(figsize=(20, 15))
        plt.imshow(randomized_colored_image, cmap='jet')

    #calculate
    for region in measure.regionprops(labeled_image):

        #option (cont): Display marked region number on segmented grains
        # Draw text annotations at the centroid of each region
        if plot_numbers_on_grains:
            centroid = region.centroid
            plt.text(centroid[1], centroid[0], str(region.label), color='white', fontsize=8, ha='center',
                     va='center')  # Annotate the labeled image with region values

        # get each grain region statistics
        print(f"Region {region.label}: Size = {region.area} pixels")  # print results in loop
        regions_list = np.append(regions_list, region.area)

        # calculate weighted sum and total_size of all grains
        weighted_sum += region.area * region.area  # accumulate the numerator (denominator is total_size)
        total_size += region.area  # get total size of all grain area

    try:
        weighted_avg_grain_size = weighted_sum / total_size     # Calculate weighted average grain size
        #print("weighted average grain size is", weighted_avg_grain_size, "pixels")

    except ZeroDivisionError:
        print("Error: Division by zero. Number of detected region is zero.")
        weighted_avg_grain_size = None  # or set a default value or handle it in another way


    # -- Visualization of Intermediate Processes)
    #visualize_segment_process(image, gaussian_np, gray_image, sobel_edges, binary_edges,randomized_colored_image) #binary edges without dilation.
    visualize_segment_process(image, gaussian_np, gray_image, sobel_edges, dilated_edges,randomized_colored_image)


    return bins, bin_min, bin_max, weighted_avg_grain_size, num_labels, total_size, labeled_image, regions_list



def histogram_plot(grain_region_size_list, bin_num, bin_min, bin_max):
    """
            This is a docstring for the example_function.

            Parameters:
            - grain_region_size_list: a list of 1D array that stored all the region size
            - bin_num: number of bins want to set up
            - bin_min: lower level boundary used for setting range of histogram plot.
            - bin_max: upper level boundary for bin plot.

            Return: None
            - Visualization of the plot
    """
    n, bins, cts = plt.hist(grain_region_size_list, bins=bin_num, range=(bin_min, bin_max), ec='black')
    plt.bar_label(cts)  # Add numbers to the top of each bar
    plt.xlabel("Grain Size (pixels)")
    plt.ylabel("Number of Grain Regions")
    plt.title("Grain Size Distribution of " + file_name)
    plt.savefig(r"../../../data/Results_grain_size_distribution/Grain Size Distribution of " + file_name+ ".png")
    plt.show()

if __name__ == "__main__":

    histogram_plot_default = True  # True: plot automatically without manual setting for bin_min & _max. False: use following manual setting to plot.

    # Manually set up the values below:
    gaussian_radius = 1
    sobel_threshold = 0.01
    dilation_iterations = 1
    bin_min = 0
    bin_max = 400
    bins = 15
    plot_numbers_on_grains = False

    #image_path = r'../../../data/test_data/sample6.png'
    #file_name = os.path.basename(image_path).split(".")[0]  # Get the image name without postfix
    #image_pil = Image.open(image_path).convert('RGB')  # convert image mode to RGB (in case if not, e.g: some image may have 4 channels: RGBA)
    #image_np = np.array(image_pil)  # Convert the PIL image to a NumPy array

    image_folder_path = r'../../../data/test_data'
    file_list = os.listdir(image_folder_path)
    for image_file in file_list:
        file_name = image_file.split(".")[0]
        print(file_name)
        #Read Images
        image_pil = Image.open(image_file).convert('RGB') #convert image mode to RGB (in case if not, e.g: some image may have 4 channels: RGBA)
        image_np = np.array(image_pil) # Convert the PIL image to a NumPy array



        #run with manual setup parameters
        bin_number, bin_lower, bin_upper, avg_grain_size, num_regions, total_grain_size, labeled_image, region_list = grain_segmentation(
            image_np, gaussian_radius, sobel_threshold, dilation_iterations,bin_min, bin_max, bins, plot_numbers_on_grains)

        #run with default value
        #bin_number,  bin_lower, bin_upper, avg_grain_size, num_regions, total_grain_size, labeled_image, region_list = grain_segmentation(image_np)

        print(f"bin_number = {bin_number}, bin_min = {bin_lower}, bin_max = {bin_upper}, average grain size = {avg_grain_size:.3f} pixels, # of regions = {num_regions}, total grain size = {total_grain_size}")
        print("Printing the \"labeled_image\" and \"region_list\" upon request.")
        # print(labeled_image)

        # Plotting Histogram of grain size distribution
        if histogram_plot_default:
            histogram_plot(region_list, bin_number, np.min(region_list), np.max(region_list))
        else: #if use manual setting for bin range.
            histogram_plot(region_list, bin_number, bin_lower, bin_upper)