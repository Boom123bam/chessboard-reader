import numpy as np
import cv2 as cv
from matplotlib import pyplot as pl
 
image = cv.imread('board.png', cv.IMREAD_GRAYSCALE)

# TODO transform and crop the image


# Split the image into tiles
w, h = image.shape
tiles = []

for x in range(8):
    for y in range(8):
        x1 = round(w * (x/8))
        x2 = round(w * ((x + 1)/8))
        y1 = round(h * (y/8))
        y2 = round(h * ((y + 1)/8))
        tiles.append(image[x1:x2,y1:y2])
        


for tile in tiles:
    cv.imshow('Block', tile)
    cv.waitKey(0)
cv.destroyAllWindows()