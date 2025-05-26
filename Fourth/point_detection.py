# point_detection.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
img = cv2.imread('image5.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Image not found.")
    exit()

# Apply Laplacian operator
laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
laplacian_abs = cv2.convertScaleAbs(laplacian)

# Increase contrast of laplacian (manually stretch intensity)
laplacian_contrast = cv2.equalizeHist(laplacian_abs)

# Threshold to detect stronger points only (tweak threshold for clarity)
_, point_mask = cv2.threshold(laplacian_contrast, 60, 255, cv2.THRESH_BINARY)

# Overlay detected points (in black) on the original image
overlay = img.copy()
overlay[point_mask == 255] = 0  # Make detected points dark/black

# Display
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Laplacian Contrast")
plt.imshow(laplacian_contrast, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Detected Points")
plt.imshow(overlay, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
