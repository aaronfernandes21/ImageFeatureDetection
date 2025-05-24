import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load two images (same size for arithmetic operations)
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')

# Resize images to the same size if necessary
image1 = cv2.resize(image1, (400, 400))
image2 = cv2.resize(image2, (400, 400))

# Arithmetic Operations
add = cv2.add(image1, image2)
subtract = cv2.subtract(image1, image2)
multiply = cv2.multiply(image1, image2)
# To avoid divide-by-zero, convert to float32 and add small epsilon
divide = cv2.divide(image1.astype(np.float32), image2.astype(np.float32) + 1e-5)
divide = np.clip(divide, 0, 255).astype(np.uint8)

# Plotting using matplotlib
titles = ['Image 1', 'Image 2', 'Addition', 'Subtraction', 'Multiplication', 'Division']
images = [image1, image2, add, subtract, multiply, divide]

plt.figure(figsize=(12, 8))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
