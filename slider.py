from __future__ import print_function
import cv2 as cv
import numpy as np


# src = cv.imread("pic-good.jpg")
src = cv.imread("board.png")
width = src.shape[0]

window_name = "Threshold Demo"
sliders = [
    ["blur", 1, int(width / 100)],
    ["thresh", 11, int(width)],
    ["thresh2", 2, 10],
    ["close", 2, 3],
]


def update(val):
    vals = [cv.getTrackbarPos(slider[0], window_name) for slider in sliders]
    # blur = cv.medianBlur(src_gray, values[0])
    # blur = cv.blur(src_gray, (values[0], values[0]))
    # _, dst = cv.threshold(blur, values[1], 255, values[2])
    kernel = np.ones((vals[0], vals[0]), np.uint8)
    blur = cv.erode(src_gray, kernel)
    th2 = cv.adaptiveThreshold(
        blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, vals[1], vals[2]
    )
    if vals[3]:
        th2 = cv.morphologyEx(th2, cv.MORPH_CLOSE, kernel, iterations=vals[3])
    cv.imshow(window_name, th2)


src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

cv.namedWindow(window_name)

for slider in sliders:
    cv.createTrackbar(slider[0], window_name, slider[1], slider[2], update)
update(0)
# Wait until user finishes program
cv.waitKey()
