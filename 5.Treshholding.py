import numpy as np
import cv2

img = cv2.imread("Teste.jpg")

ImagemCinza = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ImagemAdpatvel = cv2.adaptiveThreshold(ImagemCinza,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,115,5)

cv2.imshow("Imagem com Thresh",ImagemAdpatvel)

cv2.waitKey(0)
cv2.destroyAllWindows()

