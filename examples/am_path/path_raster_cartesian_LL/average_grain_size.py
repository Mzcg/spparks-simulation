from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, measure
from scipy.ndimage import binary_erosion, binary_dilation
import random

plot_numbers_on_grains = True

# Read the image using PIL
#image_path = 'sample5-1.png'
#image_path = 'comp2.png'
image_path = 'speed_45_hatch_15_cropped.png'
#image_path='speed_90_hatch_15_CROP_8bitcolor.png'

image_pil = Image.open(image_path)

image_pil = image_pil.convert('RGB') #convert image mode to RGB (in case if not, e.g: some image may have 4 channels: RGBA)

# Convert the PIL image to a NumPy array
image_np = np.array(image_pil)

# Apply Gaussian filter
gaussian_image = image_pil.filter(ImageFilter.GaussianBlur(radius=1))

# Convert the PIL image to a NumPy array after Gaussian filtering
gaussian_np = np.array(gaussian_image)

# Convert to grayscale for Sobel edge detection
gray_image = gaussian_image.convert('L')

# Apply Sobel edge detector
sobel_edges = filters.sobel(np.array(gray_image))

# Threshold the edge image to binary
binary_edges = sobel_edges < 0.01  # Adjust threshold as needed
dilated_edges = binary_dilation(binary_edges,iterations=1)


# Convert binary image to labeled image
labeled_image, num_labels = measure.label(dilated_edges, connectivity=1, return_num=True)

random.seed(42)  # Set a seed for reproducibility
label_colors = {label: tuple(np.random.randint(0, 256, size=3)) for label in range(1, num_labels + 1)}
randomized_labels = list(range(1, num_labels + 1))
random.shuffle(randomized_labels)

randomized_colored_image = np.zeros_like(image_np)
for label, color in zip(randomized_labels, label_colors.values()):
    randomized_colored_image[labeled_image == label] = color


if plot_numbers_on_grains:
    plt.figure(figsize=(20,15))
    plt.imshow(randomized_colored_image, cmap='jet')


total_size = 0
weighted_sum = 0
regions_list = []
for region in measure.regionprops(labeled_image):
    # Draw text annotations at the centroid of each region
    if plot_numbers_on_grains:
        centroid = region.centroid
        plt.text(centroid[1], centroid[0], str(region.label), color='white', fontsize=8, ha='center', va='center') # Annotate the labeled image with region values

    #get each grain region statistics
    print(f"Region {region.label}: Size = {region.area} pixels")  # print results in loop
    regions_list = np.append(regions_list, region.area)

    #calculate weighted sum and total_size of all grains
    weighted_sum += region.area * region.area  # accumulate the numerator (denominator is total_size)
    total_size += region.area # get total size of all grain area

print("total size: ", total_size)
print("Number of grain regions: ", num_labels)
#print(regions_list)
try:
    #calculate average grain size (no weigthed added)
    #avg_grain_size = total_size / num_labels
    #print("average grain size is ", avg_grain_size, "pixels")

    #calculate weighted average grain size
    weighted_avg_grain_size = weighted_sum / total_size
    print("weighted average grain size is", weighted_avg_grain_size, "pixels")

except ZeroDivisionError:
    print("Error: Division by zero. Number of detected region is zero.")
    avg_grain_size = None  # or set a default value or handle it in another way




# Display the results (in figure)
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(2, 3, 1)
plt.imshow(image_np)
plt.title('Original Image')
plt.axis('off')

# Gaussian Filtered Image
plt.subplot(2, 3, 2)
plt.imshow(gaussian_np)
plt.title('Gaussian Filtered Image')
plt.axis('off')

# Grayscale Image
plt.subplot(2, 3, 3)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

# Sobel Edges
plt.subplot(2, 3, 4)
plt.imshow(sobel_edges, cmap='gray')
plt.title('Sobel Edges')
plt.axis('off')

# Binary Edges
plt.subplot(2, 3, 5)
plt.imshow(dilated_edges, cmap='binary')
#plt.imshow(binary_edges, cmap='binary')
plt.title('Binary Edges')
plt.axis('off')

# Labeled Image
plt.subplot(2, 3, 6)
plt.imshow(randomized_colored_image, cmap='jet')  # Use a colormap for better visualization
plt.title('Labeled Image')
plt.axis('off')

plt.tight_layout()
plt.show()


#plot histogram
#plt.hist(regions_list, bins=15, ec='black')
n, bins, cts = plt.hist(regions_list, bins=15,  range=(0,100),  ec='black')
plt.bar_label(cts) # Add numbers to the top of each bar
plt.xlabel("Grain Size")
plt.ylabel("Number of Grain Regions")
plt.show()