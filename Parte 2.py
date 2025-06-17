import numpy as np
import cv2

img = cv2.imread("C:\\Users\Pedro\Desktop\\UNESP-DOCS\DOCS. PESSSUAIS\Instagram\IMG_20231029_233121_873.jpg", cv2.IMREAD_COLOR)

cv2.line(img, (0,0) , (150,150), (255,255,255) , 10)
            #Ponto inicial e final em x e y; cor do rabisco; espessura da linha

cv2.rectangle(img, (0,0), (500,500), (255,0,255), (2) )

#cv2.circle, funciona do mesmo jeito

cv2.imshow("Imagem",img)
cv2.waitKey(0)
cv2.destroyAllWindows()