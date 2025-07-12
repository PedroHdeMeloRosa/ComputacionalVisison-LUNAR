import cv2
import numpy

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()                          # Captura um frame da câmera (ret indica sucesso, frame é a imagem).
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # Converte o frame de BGR (colorido) para escala de cinza.
    
    cv2.imshow('frame', frame)                       # Exibe o frame original em uma janela chamada 'frame'.
    cv2.imshow('gray', gray)                         # Exibe o frame em tons de cinza em uma janela chamada 'gray'.

    if cv2.waitKey(1) & 0xFF == ord('p'):            # Espera 1 ms por uma tecla; sai do loop se a tecla for 'p'.
        break

cap.release()                                        # Libera o recurso da câmera.
cv2.waitKey(0)                                       # Espera indefinidamente por uma tecla (aqui pode ser opcional).
cv2.destroyAllWindows()                              # Fecha todas as janelas abertas do OpenCV.