import cv2
import numpy as np

# Este script lê a imagem "Teste.jpg" e desenha sobre ela várias formas geométricas e um texto.
# São aplicadas: linha, retângulo, círculo, polígono e escrita, cada uma com cor, espessura e posição definidas.

img1 = cv2.imread("Teste.jpg", cv2.IMREAD_COLOR)

#cv2.line(img1, (0, 0), (600, 700), (255, 255, 255), 15)                 # Linha branca de (0,0) até (600,700) com espessura 15

#cv2.rectangle(img1, (15, 25), (200, 150), (0, 255, 0), 5)               # Retângulo verde entre (15,25) e (200,150) com espessura 5

cv2.circle(img1, (1000, 4000), 55, (0, 0, 255), -1)                        # Círculo vermelho preenchido (espessura -1) com centro em (100,63) e raio 55

pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)       # Coordenadas dos vértices do polígono
pts = pts.reshape((-1, 1, 2))                                           # Ajusta a estrutura dos pontos para uso no OpenCV

cv2.polylines(img1, [pts], True, (0, 255, 255), 5)                      # Polígono amarelo fechado com espessura 5

font = cv2.FONT_HERSHEY_SIMPLEX                                         # Define a fonte do texto
cv2.putText(img1, 'OpenCV Tuts!', (0, 130), font, 1, (200, 255, 255), 5, cv2.LINE_AA)
# Texto "OpenCV Tuts!" na posição (0,130), com escala 1, cor azul clara, espessura 5 e suavização de bordas

cv2.imshow('image', img1)                                              # Exibe a imagem com os desenhos aplicados
cv2.waitKey(0)                                                         # Espera uma tecla ser pressionada
cv2.destroyAllWindows()                                               # Fecha todas as janelas abertas