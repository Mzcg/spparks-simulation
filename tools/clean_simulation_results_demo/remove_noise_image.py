#Objective: simulated images may have grid pattern noises in the image, trying to reduce the noise to make better quality images.
# we did not use this, decided to use just original image is fine.
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    return image

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=1.0):
    return cv2.GaussianBlur(image, kernel_size, sigma)

def apply_bilateral_filter(image, diameter=15, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

def visualize_images(original, processed):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Processed Image')
    plt.imshow(processed)
    plt.axis('off')

    plt.show()

def main():
    image_path = r'C:\Users\zg0017\PycharmProjects\spparks-simulation\tools\speed_63_mpwidth_40_haz_74_thickness_11_direction_xy_distance_32.png'  # Replace with your image path

    # Load the image
    image = load_image(image_path)

    # Apply Gaussian Blur
    blurred_image = apply_gaussian_blur(image, kernel_size=(5, 5), sigma=1.0)

    # Apply Bilateral Filter
    filtered_image = apply_bilateral_filter(blurred_image, diameter=15, sigma_color=75, sigma_space=75)

    # Visualize the original and processed images
    visualize_images(image, filtered_image)

    saved_path = os.path.join(r"C:\Users\zg0017\PycharmProjects\spparks-simulation\tools", "pythonfilter_speed_63_mpwidth_40_haz_74_thickness_11_direction_xy_distance_32.png")
    print(saved_path)
    cv2.imwrite(saved_path, filtered_image )


if __name__ == '__main__':
    main()