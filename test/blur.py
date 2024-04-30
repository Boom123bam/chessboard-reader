import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
 
img = cv.imread('pic-good.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
 
blur = cv.blur(img,(30,30))

plt.figure(figsize=(24, 12))
 
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()