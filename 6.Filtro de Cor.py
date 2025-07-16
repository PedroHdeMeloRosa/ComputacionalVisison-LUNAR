import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0,0,0])
    upper_red= np.array([255,255,255])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    result = cv2.bitwise_and(frame, frame, mask = mask)

    edgs = cv2.Canny(frame, 150, 200)

    #cv2.imshow("Imagem 1 ", hsv)
    #cv2.imshow("mask", mask)
    #cv2.imshow("Result",result)
    cv2.imshow("edgs",edgs)

    if cv2.waitKey(5) & 0XFF == ord('p'):
        break


cv2.destroyAllWindows()
cap.release()
