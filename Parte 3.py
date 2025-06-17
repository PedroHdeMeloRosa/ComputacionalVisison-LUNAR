import numpy as np
import cv2

img1 = cv2.imread("C:\\Users\Pedro\Downloads\\3D-Matplotlib.png")
img2 = cv2.imread("C:\\Users\Pedro\Downloads\mainsvmimage.png")

add = img1 + img2

cv2.imshow("add",add)
cv2.waitKey(0)
cv2.destroyAllWindows()
