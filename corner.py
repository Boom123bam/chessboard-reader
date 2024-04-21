import numpy as np
import cv2 as cv

# filename = 'board.png'
filename = 'pic-good.jpg'

img = cv.imread(filename)

img = cv.blur(img,(30,30))

def process(img, val):
    result = img.copy()
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray,2,3,val/100)
    
    #result is dilated for marking the corners, not important
    dst = cv.dilate(dst,None)
    
    # Threshold for an optimal value, it may vary depending on the image.
    result[dst>0.01*dst.max()]=[0,0,255]
    return result

window = "result"
cv.namedWindow(window)

def handleSliderChange(val): 
    res = process(img, val)
    cv.imshow(window,res)


cv.createTrackbar("slider ",window, 1, 5, handleSliderChange)
 
if cv.waitKey(0) & 0xff == 27:
 cv.destroyAllWindows()