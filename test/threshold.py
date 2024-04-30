from __future__ import print_function
import cv2 as cv
import argparse

max_value = 255
max_type = 4
max_binary_value = 255
trackbar_blur = "blur"
trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted"
trackbar_value = "Value"
window_name = "Threshold Demo"


def Threshold_Demo(val):
    blur_val = cv.getTrackbarPos(trackbar_blur, window_name)
    blur = cv.blur(src_gray, (blur_val, blur_val))
    threshold_type = cv.getTrackbarPos(trackbar_type, window_name)
    threshold_value = cv.getTrackbarPos(trackbar_value, window_name)
    _, dst = cv.threshold(blur, threshold_value, max_binary_value, threshold_type)
    cv.imshow(window_name, dst)
    cv.imshow(window_name, dst)


parser = argparse.ArgumentParser(
    description="Code for Basic Thresholding Operations tutorial."
)
src = cv.imread("pic-good.jpg")

src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.namedWindow(window_name)
cv.createTrackbar(trackbar_blur, window_name, 1, 30, Threshold_Demo)
cv.createTrackbar(trackbar_type, window_name, 3, max_type, Threshold_Demo)
# Create Trackbar to choose Threshold value
cv.createTrackbar(trackbar_value, window_name, 0, max_value, Threshold_Demo)
# Call the function to initialize
Threshold_Demo(0)
# Wait until user finishes program
cv.waitKey()
