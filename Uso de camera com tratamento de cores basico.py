import cv2
import numpy
import matplotlib.pyplot as plt


'''path = "C:/Users\Pedro\Pictures\Screenshots\Captura de tela 2025-03-26 161323.png"
img = cv2.imread(path , cv2.IMREAD_GRAYSCALE)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyWindow()

plt.imshow(img, cmap='gray', interpolation= 'bicubic')
plt.plot([50,500],[800,48], 'c' , linewidth=2)
plt.show()'''

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',frame)
    cv2.imshow('gray',gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
