import numpy as np
import cv2
from matplotlib import pyplot as plt

original = cv2.imread("board.png")

# rescale, grayscale and blur
img = cv2.resize(original, (500, 500), interpolation=cv2.INTER_AREA)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5, 5), 0)

# # Display the original and processed images
# cv2.imshow("Original Image", original)
# cv2.imshow("Resized, Grayscale, and Blurred Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def getSaddle(gray_img):
    img = gray_img.astype(np.float64)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    gxx = cv2.Sobel(gx, cv2.CV_64F, 1, 0)
    gyy = cv2.Sobel(gy, cv2.CV_64F, 0, 1)
    gxy = cv2.Sobel(gx, cv2.CV_64F, 0, 1)

    S = gxx * gyy - gxy**2
    return S


def pruneSaddle(s):
    thresh = 128
    score = (s > 0).sum()
    while score > 10000:
        thresh = thresh * 2
        s[s < thresh] = 0
        score = (s > 0).sum()


saddle = getSaddle(img)
saddle = -saddle
saddle[saddle < 0] = 0

pruneSaddle(saddle)


plt.figure(figsize=(10, 10))
plt.imshow(saddle, cmap="Greys_r")


# # Split the image into tiles
# w, h = image.shape
# tiles = []

# for x in range(8):
#     for y in range(8):
#         x1 = round(w * (x/8))
#         x2 = round(w * ((x + 1)/8))
#         y1 = round(h * (y/8))
#         y2 = round(h * ((y + 1)/8))
#         tiles.append(image[x1:x2,y1:y2])


# for tile in tiles:
#     cv.imshow('Block', tile)
#     cv.waitKey(0)
# cv.destroyAllWindows()
