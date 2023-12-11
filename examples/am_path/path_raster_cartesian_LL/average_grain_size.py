import os
import cv2

from scipy import ndimage as ndi
import matplotlib.pyplot as plt

from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte

from skimage import io, color, measure, segmentation

image_path = r"C:\Users\zg0017\Documents\Fall 2023\AM project - grain\grain_size_detection\c2.png"
file_name = os.path.basename(image_path).split(".")[0]

print("File Name:", file_name)
image_input = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)



#image = gray

# Apply Gaussian blur to the image
#blurred = cv2.GaussianBlur(gray, (19,19), 0) #for sample 4 use this setting (19,19)
#note: for pixel change samples (in same region, pixel varies), use the GaussianBlur, for images with clear color, like sample2,5, do not use GaussianBlur will get best results.

# denoise image
image = gray
denoised = rank.median(image, disk(2))

#denoised = rank.median(blurred, disk(2))

# find continuous region (low gradient -
# where less than 10 for this image) --> markers
# disk(5) is used here to get a more smooth image
markers = rank.gradient(denoised, disk(5)) < 10
markers = ndi.label(markers)[0]

# local gradient (disk(2) is used to keep edges thin)
gradient = rank.gradient(denoised, disk(2))

# process the watershed
#labels = watershed(gradient, markers) #original
labels = watershed(gradient, markers, connectivity=2)


# display results
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8),
                         sharex=True, sharey=True)
ax = axes.ravel()


#ax[0].imshow(blurred)
#ax[0].set_title("Blurred")

ax[0].imshow(gray, cmap=plt.cm.gray)
ax[0].set_title("Original")

ax[1].imshow(gradient, cmap=plt.cm.nipy_spectral)
ax[1].set_title("Local Gradient")

ax[2].imshow(markers, cmap=plt.cm.nipy_spectral)
ax[2].set_title("Markers")

ax[3].imshow(gray, cmap=plt.cm.gray)
ax[3].imshow(labels, cmap=plt.cm.nipy_spectral, alpha=.5)
ax[3].set_title("Segmented")

for a in ax:
    a.axis('off')

fig.tight_layout()
plt.savefig("segmented_Results_"+file_name+".png") #save all the displayed results to 1 png file.

#save only the segmented results
# Create a new figure for the fourth subplot and save it
fig_segmented = plt.figure(figsize=(4, 4))
ax_segmented = fig_segmented.add_subplot(111)
ax_segmented.imshow(gray, cmap=plt.cm.gray)
ax_segmented.imshow(labels, cmap=plt.cm.nipy_spectral, alpha=.5)
ax_segmented.set_title("Segmented")

# Save only the fourth subplot
segmented_image_name = "segmented_" + file_name+".png"
fig_segmented.savefig(segmented_image_name)

#plt.show() #showing the watershed segmentation (4 images together: original, gradient, markers, and segmented)



# Calculate the size of each segmented area
region_sizes = measure.regionprops(labels)


total_size = 0
#counter = 0
weighted_sum = 0


# Print the size of each segmented area
for region in region_sizes:
    minr, minc, maxr, maxc = region.bbox
    rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
    ax[3].add_patch(rect) #showing red box along each separated region
    ax[3].text(minc, minr, f"Size: {region.area} pixels", color='red', fontsize=8) #show size value on image along box

        # Print the size of each region
    print(f"Region {region.label}: Size = {region.area} pixels") #pring results

    weighted_sum += region.area * region.area #accumulate the numerator (denominator is total_size)
    #print("weighted sum is ", weighted_sum)
    #get total size of all grain area
    total_size += region.area #sum all the region values = should be same as the pixel values of all images
    #counter += 1

plt.show() #showing the separate region with red box #if want to show this, comment the above plt.show()

print("total size: ", total_size)
#print("Number of grain regions: ", counter)
try:
    #avg_grain_size = total_size / counter
    weighted_avg_grain_size = weighted_sum / total_size
    #print("average grain size is ", avg_grain_size, "pixels")
    print("weighted average grain size is ", weighted_avg_grain_size)
except ZeroDivisionError:
    print("Error: Division by zero. Number of detected region is zero.")
    avg_grain_size = None  # or set a default value or handle it in another way

