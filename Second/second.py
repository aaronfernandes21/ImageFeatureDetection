import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale image
img = cv2.imread('image3.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Could not load image. Check the path.")
    exit()

# Define a 3x3 edge detection kernel
kernel = np.array([[ -1, -1, -1],
                   [ -1,  8, -1],
                   [ -1, -1, -1]])

# Correlation using filter2D
correlation = cv2.filter2D(img, cv2.CV_64F, kernel)
correlation_abs = cv2.convertScaleAbs(correlation)  # Scale and take abs for better visibility

# Convolution (flip kernel and apply filter2D)
kernel_flipped = np.flipud(np.fliplr(kernel))
convolution = cv2.filter2D(img, cv2.CV_64F, kernel_flipped)
convolution_abs = cv2.convertScaleAbs(convolution)

# Optional: Contrast enhance using histogram equalization
correlation_eq = cv2.equalizeHist(correlation_abs)
convolution_eq = cv2.equalizeHist(convolution_abs)

# Plotting results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Correlation (Enhanced)")
plt.imshow(correlation_eq, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Convolution (Enhanced)")
plt.imshow(convolution_eq, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
