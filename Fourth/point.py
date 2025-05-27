import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
img = cv2.imread('image5.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Could not load image. Check the path or file name.")
    exit()

# Convert to float32 for Harris
gray = np.float32(img)

# Harris corner detection
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
dst = cv2.dilate(dst, None)

# Create a copy to mark detected points
point_img = img.copy()
point_img[dst > 0.01 * dst.max()] = 255  # white points where corners are detected

# Display side-by-side
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Harris Corners")
plt.imshow(point_img, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
