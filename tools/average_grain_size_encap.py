import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import binary_dilation
from skimage import filters, measure

import tensorflow as tf
from tensorflow.keras.preprocessing import image


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

def grain_segmentation(image, gaussian_radius=1, sobel_threshold=0.01, dilation_iterations=1,bin_min=0, bin_max=300, n_bins=15, plot_numbers_on_grains=True, show_intermediate_segmentation=False, histogram_plot_default=True, draw_histogram=False):

    """
        This function takes an image in numpy array format to process to segment the microstructure (grain) into different regions. and then calcualte the (weighted) average grain size for each image.

        Parameters:
        - image (numpy array): input image in RGB mode ( Nrow * Ncolumn * 3) - make sure it is 3 channels when input.
        - gaussian_radius (int): radius used when applying GaussianBlur filter
        - sobel_threshold (float): define the threshold to binary_edges after apply sobel edge detection.
        - dilation_iterations (int): iterations used in binary_dilation function (range from 1 to 2)
        - bin_min (int): lower limit set to used for histogram plot
        - bin_max (int): upper limit set to used for histogram plot
        - n_bins (int): number of bins


        Returns:
        - histogram values:
            - bin_min (int): lower limit of the grain size to be displayed in histogram (min value can be 0)
            - bin_max (int): upper limit of the grain size to be displayed in histogram (max values can be the calculated maximum grain size, or self-defined values)
            - bins (array): a list of values that shows the number of each bin edge (e.g: [0, 100, 200,... 10000]). length of array is bin_max/n_bins
            - hist_n (array of int/float): a list of values that represent the count of each bar (e.g: if we have 15 bins, then there are 15 values in the list. each represent the number of counts (int), or in normazlied value (float) of each bar.
        - weighted average grain size (float): calculated weighted average grain size for each image
        - number of regions (int): number of grain detected using the algorithms
        - total grain size (float): The sum of all detected grains' size
        - labeled image (array): segmented images saved in 2D-array
        - region list (array): save grain size for all the detected grain regions in a list.
    """


    hist_n, bins = [], []
    weighted_avg_grain_size, num_labels, total_size = 0, 0, 0
    labeled_image, regions_list = np.array([]), []


    #total_size = 0      #total size of all detected grains (sum of each grain size)
    weighted_sum = 0    #used for calculate weighted average grain size of an image
    #regions_list = []   #save all the grain sizes in the list

    #check the image input format:
    if isinstance(image, tf.Tensor): #check if the input image is in Tensor format
        #test: from tensor to numpy to pil
        image_np = image.numpy()
        pil_image = tf.keras.preprocessing.image.array_to_img(image_np)
    elif isinstance(image, np.ndarray): #check if image input is in numpy array format.
        # Create a PIL Image from the NumPy array (if the input image is in numpy array format)
        image_np = image
        pil_image = Image.fromarray(image_np)
    else:
        print("Warning: double check the image input format to make sure it is either np.array or Tensor format!")

    # Apply Gaussian filter
    gaussian_image = pil_image.filter(ImageFilter.GaussianBlur(radius=gaussian_radius))
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
        #print(f"Region {region.label}: Size = {region.area} pixels")  # print results in loop for the image
        #regions_list = np.append(regions_list, region.area)
        regions_list.append(region.area)

        # calculate weighted sum and total_size of all grains
        weighted_sum += region.area * region.area  # accumulate the numerator (denominator is total_size)
        total_size += region.area  # get total size of all grain area

    try:
        weighted_avg_grain_size = weighted_sum / total_size     # Calculate weighted average grain size
        #print("weighted average grain size is", weighted_avg_grain_size, "pixels")

    except ZeroDivisionError:
        print("Error: Division by zero. Number of detected region is zero.")
        weighted_avg_grain_size = None  # or set a default value or handle it in another way


    # -- Visualization of Segmentation Intermediate Steps
    if show_intermediate_segmentation:
        visualize_segment_process(image, gaussian_np, gray_image, sobel_edges, binary_edges,randomized_colored_image) #binary edges without dilation.
        #visualize_segment_process(image, gaussian_np, gray_image, sobel_edges, dilated_edges,randomized_colored_image) #recommended


    #get histogram and visualize if needed.
    hist_n, bins = get_histogram(regions_list, n_bins, bin_min, bin_max, histogram_plot_default, draw_histogram)


    #return hist_n, bins, bin_min, bin_max, weighted_avg_grain_size, num_labels, total_size, labeled_image, regions_list
    return hist_n, bins, bin_min, bin_max, weighted_avg_grain_size, num_labels, total_size, labeled_image, regions_list



def get_histogram(grain_region_size_list, bin_num, bin_min, bin_max, histogram_plot_default, draw_histogram):
    """
            Grain size distribution plot

            Parameters:
            - grain_region_size_list: a list of 1D array that stored all the region size
            - bin_num: number of bins want to set up
            - bin_min: lower level boundary used for setting range of histogram plot.
            - bin_max: upper level boundary for bin plot.

            Return:
            - n: a list  of values on each bar (if it is normalized with setting density=True, or original values of each bar)
            - bins: a list of values that has width of each bar. if number of bins =100, bin_max = 10000, then the list values of difference of 100 -> [0, 100, 200, .. 100000]
    """



    if histogram_plot_default:  #using min and max values from calculation directly to plot.

        ### using np.histogram to plot the histogram
        #n, bins, = np.histogram(grain_region_size_list, bins=bin_num, range=(np.min(grain_region_size_list), np.max(grain_region_size_list)),density=True) #density: use normalized y-value if true.
        n, bin_edges, = np.histogram(grain_region_size_list, bins=bin_num, density=True) #density: use normalized y-value if true.

        if draw_histogram == True:
            plt.bar(bin_edges[:-1], n, width=np.diff(bin_edges), color='blue', edgecolor='black', alpha=0.7)  # Plot the histogram using plt.bar

            ### Option: if want to label the bar, uncomment below
            #bars = plt.bar(bin_edges[:-1], n, width=np.diff(bins), color='blue', edgecolor='black', alpha=0.7) # Plot the histogram using plt.bar
            #plt.bar_label(bars, fmt='{:.2e}', label_type='edge') # Add labels to the bars using plt.bar_label

            ### Option: plot lines connected bars, uncomment below
            #plt.plot(bin_edges[:-1],n, color = 'orange') #plot the lines above the bar

            plt.xlabel("Grain Size (pixels)")
            plt.ylabel("Number of Grain Regions")
            plt.title("Grain Size Distribution of " + file_name)
            # plt.savefig(r"../../../tmp/Grain Size Distribution of " + file_name+ ".png")
            plt.show()
    else: #if histogram_plot_default = False, plot with MANUAL setting values (bin_min, bin_max)
        n, bin_edges, = np.histogram(grain_region_size_list, bins=bin_num,
                                range=(bin_min, bin_max),
                                density=True)  # density: use normalized y-value if true.

        if draw_histogram == True:
            plt.bar(bin_edges[:-1], n, width=np.diff(bin_edges), color='blue', edgecolor='black', alpha=0.7)  # Plot the histogram using plt.bar
            plt.xlabel("Grain Size (pixels)")
            plt.ylabel("Number of Grain Regions")
            plt.title("Grain Size Distribution of " + file_name)
            # plt.savefig(r"../../../tmp/Grain Size Distribution of " + file_name+ ".png")
            plt.show()



    return n, bin_edges #array of histogram bar values (without normalization, it's the count of each bar; with normalization, it's the normalized values)

def compare_images(image_np1, image_np2, gaussian_radius=1, sobel_threshold=0.01, dilation_iterations=1,bin_min=0, bin_max=300, n_bins=100,plotOn=False):
    """
        This function compare the two images grain size distribution (histogram), and output the similarity index (similarity score: intersect; distance score: bhattacharyya)
        We take two images (numpy array), and first call grain_segmentation to calculate the histogram, then we compare them using cv2 histogram comparison function.

        Parameters:
            - image_np1 (numpy array): image 1 as numpy array
            - image_np2 (numpy array): image 2 input as numpy array
            - gaussin_radius, sobel_threshold,, dilation_iterations, bin_min, bin_max, n_bins --> used for call grain_segmentation.

        Return:
            - similarity_score (float): compute use cv2.HISTCMP_INTERSECT
            - distance_score (float): compute use cv2.HISTCMP_BHATTACHARYYA
        """


    hist1, bins1, bin_min1, bin_max1, avg_grain_size1, num_regions1, total_grain_size1, labeled_image1, regions_list1 = grain_segmentation(
         image_np1, gaussian_radius, sobel_threshold, dilation_iterations, bin_min, bin_max, n_bins, plot_numbers_on_grains)


    hist2, bins2, bin_min2, bin_max2, avg_grain_size2, num_regions2, total_grain_size2, labeled_image2, regions_list2 = grain_segmentation(
        image_np2, gaussian_radius, sobel_threshold, dilation_iterations, bin_min, bin_max, n_bins, plot_numbers_on_grains)


    hist1 = np.array(hist1).astype(np.float32)
    hist2 = np.array(hist2).astype(np.float32)

    # d = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL) #correlation (similarity)
    # d_chi = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR) #chi-squre (distance)
    d_intersect = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)  # intersect (similarity)
    d_bhat = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)  # bhattacharyya (distance)

    similarity_score = d_intersect * 10 ** 2
    distance_score = d_bhat

    #print(f"Histogram Comparison: similarity score (intersect) = {similarity_score:.4f}e-02, distance (bhattacharyya) = {distance_score}") #test inside function
    # ref: print(f" h1 vs h2: similarity index: correlation: {d:.4f}, chi-squred: {d_chi:.4f}, intersect: {d_intersect * 10 ** 2:.4f}e+02, bhattacharyya: {d_bhat:.4f}")

    # plot log
    kernel_size = 3
    kernel = np.ones(kernel_size) / kernel_size

    smoothed_hist1 = np.convolve(hist1, kernel, mode='same')
    smoothed_hist2 = np.convolve(hist2, kernel, mode='same')

    if plotOn:
        plt.semilogx(bins1[:-1], smoothed_hist1, 'r')
        plt.semilogx(bins2[:-1], smoothed_hist2, 'g')

        # plt.semilogx(bins1[:-1], hist1, 'b')
        # plt.semilogx(bins2[:-1], hist2, 'y')

        plt.xlim((1e2, 1e4))
        plt.ylim((0, 2e-3))
        plt.legend(['image1', 'image2'])
        plt.title("")
        plt.show()

    return similarity_score, distance_score, avg_grain_size1, avg_grain_size2


if __name__ == "__main__":

    show_intermediate_segmentation = False  #if want to display the six images of intermediate steps of grain segmentation, set to True
    draw_histogram = False  #if want to display the visualization of grain size distribution histogram, set it true.
    histogram_plot_default = True # Choose want to use manual setting (bin_min,max) to plot or not. True: auto plot with default values False: use following manual setting to plot.



    # Manually set up the values below:
    gaussian_radius = 1
    sobel_threshold = 0.01
    dilation_iterations = 1
    bin_min = 0
    bin_max = 10000
    n_bins = 100
    plot_numbers_on_grains = False


    image_folder_path = r'../data/test_data'

    file_list = os.listdir(image_folder_path)

    image_dict = {} #key: image name; values: numpy array of np
    image_tensor_dict = {}

    for image_file in file_list:
        file_name = image_file.split(".")[0]  #file name: name without .png (e.g image_file = 'sample.png', file_name = 'sample')
        image_path = os.path.join(image_folder_path, image_file)

        #### Get Image Data as Numpy Array format

        ## Read Images (with PIL) to numpy array
        #image_pil = Image.open(os.path.join(image_folder_path,image_file)).convert('RGB') #convert image mode to RGB (in case if not, e.g: some image may have 4 channels: RGBA)
        #image_np = np.array(image_pil) # Convert the PIL image to a NumPy array #note: later our input image will be numpy array, so we don't need to do taht.
        ## save images into dict
        #image_dict[image_file] = image_np


        #Tensor Load: Load image as tensor format
        image_raw = tf.io.read_file(image_path) #step1: read the image
        image_tensor = tf.image.decode_image(image_raw, channels=3) #step2: decode the image to a tensor, and only get 3 channels (my image had 4 so to convert)
        #print(image_tensor)
        #save tensor image into dict
        image_tensor_dict[image_file] = image_tensor


    #print(image_tensor_dict)


    ##Get Image in Numpy Array format.
    #print(image_dict)
    #imagename_list  = list(image_dict.keys())
    ## get image nparray to pass to the fucntion
    #img1 = image_dict[imagename_list[0]]  # another way pass image (numpy array): list(image_dict.values())[0]
    #img2 = image_dict[imagename_list[1]]

    #Get Image in Tensor format
    imagename_list = list(image_tensor_dict.keys())
    img1 = image_tensor_dict[imagename_list[0]]
    img2 = image_tensor_dict[imagename_list[1]]
    #print("TYPE OF IMAGE 1:",type(img1))


    #Get file name for result printing in screen (optional)
    img1_name_str = imagename_list[0]
    img2_name_str = imagename_list[1]


    #Run comapre images with default setting.
    #s_score, d_score = compare_images(img1, img2)
    #print(f"Default Image comparison between\'{list(image_dict.keys())[0]}\' and \'{list(image_dict.keys())[1]}\': similarity_score = {s_score}, distance_score = {d_score}")

    #if use manual setting, use the following code
    s_score, d_score, compute_grainsize_1, compute_grainsize_2 = compare_images(img1,img2, gaussian_radius, sobel_threshold,dilation_iterations, bin_min, bin_max, n_bins)
    print(f"Weighted average grain size for \'{img1_name_str}\' is {compute_grainsize_1:.3f} pixels, and for \'{img2_name_str}\' is {compute_grainsize_2:.3f} pixels.")
    print(f"Image comparison between\'{img1_name_str}\' and \'{img2_name_str}\': similarity_score (intersect) = {s_score:.4f}e-02, distance_score (bhattacharyya) = {d_score:.4f}")








