import cv2
import numpy as np
from scipy import stats
from skimage.measure import shannon_entropy  # For entropy
import matplotlib.pyplot as plt

# Load grayscale image
img = cv2.imread('image4.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print(" Failed to load image. Check the file path.")
    exit()

# Flatten image to 1D for statistical calculations
flat = img.flatten()

# Statistical Measures
mean = np.mean(flat)
median = np.median(flat)
mode = stats.mode(flat, keepdims=True).mode[0]
std_dev = np.std(flat)
entropy = shannon_entropy(img)

# Print Results
print("Image Statistical Analysis")
print(f"Mean: {mean:.2f}")
print(f"Median: {median}")
print(f"Mode: {mode}")
print(f"Standard Deviation: {std_dev:.2f}")
print(f"Entropy: {entropy:.4f}")

# Optional: Show the image
plt.imshow(img, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()
