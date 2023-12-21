import cv2
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

def grain_segmentation(image_np, gaussian_radius=1, sobel_threshold=0.01, dilation_iterations=1,bin_min=0, bin_max=300, n_bins=15, plot_numbers_on_grains=True):

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

    # Create a PIL Image from the NumPy array
    pil_image = Image.fromarray(image_np)

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
    #visualize_segment_process(image, gaussian_np, gray_image, sobel_edges, binary_edges,randomized_colored_image) #binary edges without dilation.
    #visualize_segment_process(image, gaussian_np, gray_image, sobel_edges, dilated_edges,randomized_colored_image) #recommended


    #get histogram and visualize if needed.
    hist_n, bins = get_histogram(regions_list, n_bins, bin_min, bin_max )


    #return hist_n, bins, bin_min, bin_max, weighted_avg_grain_size, num_labels, total_size, labeled_image, regions_list
    return hist_n, bins, bin_min, bin_max, weighted_avg_grain_size, num_labels, total_size, labeled_image, regions_list



def get_histogram(grain_region_size_list, bin_num, bin_min, bin_max):
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
            plt.bar(bin_edges[:-1], n, width=np.diff(bins), color='blue', edgecolor='black', alpha=0.7)  # Plot the histogram using plt.bar

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
            plt.bar(bin_edges[:-1], n, width=np.diff(bins), color='blue', edgecolor='black', alpha=0.7)  # Plot the histogram using plt.bar
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

    return similarity_score, distance_score





def compare_histogram(image1_name, image2_name , dict_distribution):
    """
                This function compare the two images grain size distribution (histogram), and output the similarity index (similarity score: intersect; distance score: bhattacharyya)
                We first calculate the distribution of each image, and then compare them using cv2 histogram comparison function.

                Parameters:
                - image1 (np array)
                - image2 (np array):
                - dict_distribution (dictionary): key - image name, values: array of histogram values (each bar)

                Return:
                - similarity_score (float): compute use cv2.HISTCMP_INTERSECT
                - distance_score (float): compute use cv2.HISTCMP_BHATTACHARYYA
    """

    hist1 = np.array(dict_distribution[image1_name]).astype(np.float32)
    hist2 = np.array(dict_distribution[image2_name]).astype(np.float32)

    #d = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL) #correlation (similarity)
    #d_chi = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR) #chi-squre (distance)
    d_intersect = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT) #intersect (similarity)
    d_bhat = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA) #bhattacharyya (distance)

    similarity_score = d_intersect * 10 ** 2
    distance_score = d_bhat

    print(f"Histogram Comparison between '{image1_name}' and '{image2_name}': similarity score (intersect) = {similarity_score:.4f}e-02, distance (bhattacharyya) = {distance_score}")
    #ref: print(f" h1 vs h2: similarity index: correlation: {d:.4f}, chi-squred: {d_chi:.4f}, intersect: {d_intersect * 10 ** 2:.4f}e+02, bhattacharyya: {d_bhat:.4f}")

    # plot log
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size

    plt.semilogx(bins[:-1], hist1, 'r')
    plt.semilogx(bins[:-1], hist2, 'g')


    plt.xlim((1e2, 1e4))
    plt.ylim((0, 2e-3))
    plt.legend([image1_name, image2_name])
    plt.show()

    return similarity_score, distance_score


if __name__ == "__main__":


    draw_histogram = False  #if want to display the visualization of grain size distribution histogram, set it true.
    histogram_plot_default = False  # Choose want to use manual setting (bin_min,max) to plot or not. True: auto plot with default values False: use following manual setting to plot.



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
    dict_distribution = {} #save the map of key: image name, value: array of histogram values
    for image_file in file_list:
        file_name = image_file.split(".")[0]
        #print(file_name)
        #Read Images
        image_pil = Image.open(os.path.join(image_folder_path,image_file)).convert('RGB') #convert image mode to RGB (in case if not, e.g: some image may have 4 channels: RGBA)
        image_np = np.array(image_pil) # Convert the PIL image to a NumPy array #note: later our input image will be numpy array, so we don't need to do taht.

        #run with manual setup parameters
        #hist_n, bins, bin_min, bin_max, avg_grain_size, num_regions, total_grain_size, labeled_image, output_regions_list = grain_segmentation(
        #    image_np, gaussian_radius, sobel_threshold, dilation_iterations,bin_min, bin_max, n_bins, plot_numbers_on_grains)
        #run with default value
        #bins,  bin_min, bin_max, avg_grain_size, num_regions, total_grain_size, labeled_image, regions_list = grain_segmentation(image_np)

        #print(f"bin_num = {n_bins}, bin_min = {bin_min}, bin_max = {bin_max}, average grain size = {avg_grain_size:.3f} pixels, # of regions = {num_regions}, total grain size = {total_grain_size}")
        # print("Printing the \"bins\",\"labeled_image\" and \"regions_list\" upon request (list).")
        # print(labeled_image)


        #store grain size distribution histogram results.
        #dict_distribution[image_file] = hist_n #save histogram bar values to dictionary. key: image name, value: array of bar values (normlized if density=True when plot)

        #save images into dict
        image_dict[image_file] = image_np



    #print(image_dict[])
    imagename_list  = list(image_dict.keys())

    #get image nparray to pass to the fucntion
    img1 = image_dict[imagename_list[0]]  # another way pass image (numpy array): list(image_dict.values())[0]
    img2 = image_dict[imagename_list[1]]


    #Run comapre images with default setting.
    #s_score, d_score = compare_images(img1, img2)
    #print(f"Default Image comparison between\'{list(image_dict.keys())[0]}\' and \'{list(image_dict.keys())[1]}\': similarity_score = {s_score}, distance_score = {d_score}")

    #if use manual setting, use the following code
    s_score, d_score = compare_images(img1,img2, gaussian_radius, sobel_threshold,dilation_iterations, bin_min, bin_max, n_bins)
    print(f"Image comparison between\'{list(image_dict.keys())[0]}\' and \'{list(image_dict.keys())[1]}\': similarity_score (intersect) = {s_score:.4f}e-02, distance_score (bhattacharyya) = {d_score:.4f}")


    #compare two distribution histogram

    ### mehtod1: define the image name by hand
    #image1 = 'sample5.png'
    #image2 = 'sample6.png'
    #image3 = 'speed_45_hatch_15_cropped.png'
    #image4 = 'speed_90_hatch_15_cropped.png'
    #sim_score, distance_score = compare_histogram(image1, image2, dict_distribution)
    #print(f"Histogram Comparison between '{image1}' and '{image2}': similarity score (intersect) = {sim_score:.4f}e+02, distance (bhattacharyya) = {distance_score}")

    #method 2: note if we only have 2 images (original, predicted image), instead of using the name, we will loop the dictionary
    #imagename_list = list(dict_distribution.keys())
    #print(imagename_list) #currently is ['sample5.png', 'sample6.png', 'speed_45_hatch_15_cropped.png', 'speed_90_hatch_15_CROP.png']. supposed only have two images (original, predict)
    #similarity_score, distance_score = compare_histogram(imagename_list[0], imagename_list[1], dict_distribution)







    """
    #plot all 4 sample images and their histogram for comparison 
    
    #using file name to compare
    hist1 = np.array(dict_distribution['sample5.png']).astype(np.float32)
    hist2 = np.array(dict_distribution['sample6.png']).astype(np.float32)
    hist3 = np.array(dict_distribution['speed_45_hatch_15_cropped.png']).astype(np.float32)
    hist4 = np.array(dict_distribution['speed_90_hatch_15_CROP.png']).astype(np.float32)

    d = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    d_chi = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    d_intersect = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
    d_bhat = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    print(f" h1 vs h2: similarity index: correlation: {d:.4f}, chi-squred: {d_chi:.4f}, intersect: {d_intersect*10**2:.4f}e+02, bhattacharyya: {d_bhat:.4f}")



    d = cv2.compareHist(hist2, hist4, cv2.HISTCMP_CORREL)
    d_chi = cv2.compareHist(hist2, hist4, cv2.HISTCMP_CHISQR)
    d_intersect = cv2.compareHist(hist2, hist4,cv2.HISTCMP_INTERSECT)
    d_bhat = cv2.compareHist(hist2, hist4, cv2.HISTCMP_BHATTACHARYYA)
    print(f" h2 vs h4: similarity index: correlation: {d:.4f}, chi-squred: {d_chi:.4f}, intersect: {d_intersect*10**2:.4f}e+02, bhattacharyya: {d_bhat:.4f}")


    d = cv2.compareHist(hist2, hist2, cv2.HISTCMP_CORREL)
    d_chi = cv2.compareHist(hist2, hist2, cv2.HISTCMP_CHISQR)
    d_intersect = cv2.compareHist(hist2, hist2, cv2.HISTCMP_INTERSECT)
    d_bhat = cv2.compareHist(hist2, hist2, cv2.HISTCMP_BHATTACHARYYA)
    print(f" h2 vs h2: similarity index: correlation: {d:.4f}, chi-squred: {d_chi:.4f}, intersect: {d_intersect*10**2:.4f}e+02, bhattacharyya: {d_bhat:.4f}")

    d = cv2.compareHist(hist3, hist4, cv2.HISTCMP_CORREL)
    d_chi = cv2.compareHist(hist3, hist4, cv2.HISTCMP_CHISQR)
    d_intersect = cv2.compareHist(hist3, hist4, cv2.HISTCMP_INTERSECT)
    d_bhat = cv2.compareHist(hist3, hist4,cv2.HISTCMP_BHATTACHARYYA)
    print( f" h3 vs h4 similarity index: correlation: {d:.4f}, chi-squred: {d_chi:.4f}, intersect: {d_intersect*10**2:.4f}e+02, bhattacharyya: {d_bhat:.4f}")

    d = cv2.compareHist(hist1, hist3, cv2.HISTCMP_CORREL)
    d_chi = cv2.compareHist(hist1, hist3, cv2.HISTCMP_CHISQR)
    d_intersect = cv2.compareHist(hist1, hist3, cv2.HISTCMP_INTERSECT)
    d_bhat = cv2.compareHist(hist1, hist3, cv2.HISTCMP_BHATTACHARYYA)
    print(f" h1 vs h3 similarity index: correlation: {d:.4f}, chi-squred: {d_chi:.4f}, intersect: {d_intersect*10**2:.4f}e+02, bhattacharyya: {d_bhat:.4f}")

    d = cv2.compareHist(hist2, hist3, cv2.HISTCMP_CORREL)
    d_chi = cv2.compareHist(hist2, hist3, cv2.HISTCMP_CHISQR)
    d_intersect = cv2.compareHist(hist2, hist3, cv2.HISTCMP_INTERSECT)
    d_bhat = cv2.compareHist(hist2, hist3, cv2.HISTCMP_BHATTACHARYYA)

    print(f" h2 vs h3 similarity index: correlation: {d:.4f}, chi-squred: {d_chi:.4f}, intersect: {d_intersect*10**2:.4f}e+02, bhattacharyya: {d_bhat:.4f}")

    d = cv2.compareHist(hist1, hist4, cv2.HISTCMP_CORREL)
    d_chi = cv2.compareHist(hist1, hist4, cv2.HISTCMP_CHISQR)
    d_intersect = cv2.compareHist(hist1, hist4, cv2.HISTCMP_INTERSECT)
    d_bhat = cv2.compareHist(hist1, hist4, cv2.HISTCMP_BHATTACHARYYA)

    print(
        f" h1 vs h4 similarity index: correlation: {d:.4f}, chi-squred: {d_chi:.4f}, intersect: {d_intersect * 10 ** 2:.4f}e+02, bhattacharyya: {d_bhat:.4f}")
   

    #plot log
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size

    plt.semilogx(bins[:-1],hist1,'r')
    plt.semilogx(bins[:-1],hist2,'g')
    plt.semilogx(bins[:-1],hist3,'b')
    plt.semilogx(bins[:-1],hist4,'k')

    plt.xlim((1e2,1e4))
    plt.ylim((0,2e-3))
    plt.legend(['sample5','sample6','speed_45_hatch_15','speed_45_hatch_15'])
    plt.show()
    
    """





