import numpy as np
import cv2

img1 = cv2.imread("Teste.jpg")
img2 = cv2.imread("mainsvmimage.png")

add = img1 + img2

cv2.imshow("img1",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
