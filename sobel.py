import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("pic-good.jpg", cv.IMREAD_GRAYSCALE)
# Output dtype = cv.CV_64F. Then take its absolute and convert to cv.CV_8U
sobelx64f = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)


plt.figure(figsize=(24, 12))

plt.subplot(1, 2, 1), plt.imshow(img, cmap="gray")
plt.title("Original"), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(sobel_8u, cmap="gray")
plt.title("Sobel abs(CV_64F)"), plt.xticks([]), plt.yticks([])

plt.show()
