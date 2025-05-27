import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image3.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Could not load image. Check the path or file name.")
    exit()

edges = cv2.Canny(img, 50, 150)

lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

line_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Hough Lines")
plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))   f
plt.axis('off')

plt.tight_layout()
plt.show()
