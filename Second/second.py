import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale image
img = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("❌ Could not load image. Check the path.")
    exit()

# Define a simple 3x3 kernel (e.g., edge detection)
kernel = np.array([[ -1, -1, -1],
                   [ -1,  8, -1],
                   [ -1, -1, -1]])

# Correlation (as done by cv2.filter2D)
correlation = cv2.filter2D(img, -1, kernel)

# Convolution = correlation with flipped kernel
kernel_flipped = np.flipud(np.fliplr(kernel))  # Flip both horizontally and vertically
convolution = cv2.filter2D(img, -1, kernel_flipped)

# Plotting results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Correlation")
plt.imshow(correlation, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Convolution")
plt.imshow(convolution, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
