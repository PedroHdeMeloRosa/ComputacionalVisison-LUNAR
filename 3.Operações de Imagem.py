import numpy as np
import cv2

# Este script carrega a imagem 'watch.jpg' e realiza manipulações diretas nos pixels:
# - Acessa um pixel específico (55,55)
# - Define uma região da imagem como branca (255,255,255)
# - Recorta uma parte da imagem ('watch face') e a cola em outro lugar
# - Exibe a imagem final

img = cv2.imread('watch.jpg', cv2.IMREAD_COLOR)               # Lê a imagem colorida

print(img[55, 55])                                            # Exibe o valor BGR do pixel na posição (55, 55)

img[55, 55] = [255, 255, 255]                                  # Altera o pixel (55,55) para branco (B=255, G=255, R=255)

img[100:150, 100:150] = [255, 0, 255]                        # Define um bloco de 50x50 pixels como branco ou  qualquer outra cor (de linha 100 a 149, coluna 100 a 149)

watch_face = img[37:111, 107:194]                              # Recorta a "face do relógio" (região de 74x87 pixels)
img[0:74, 0:87] = watch_face                                   # Cola essa região no canto superior esquerdo da imagem

cv2.imshow('image', img)                                      # Exibe a imagem resultante
cv2.waitKey(0)                                                # Espera até que uma tecla seja pressionada
cv2.destroyAllWindows()                                       # Fecha todas as janelas
