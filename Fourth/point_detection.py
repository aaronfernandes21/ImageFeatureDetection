import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
img = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("❌ Could not load image.")
    exit()

# Apply Laplacian filter for point detection
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# Convert to uint8 for display
laplacian_result = cv2.convertScaleAbs(laplacian)

plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Point Detection")
plt.imshow(laplacian_result, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
