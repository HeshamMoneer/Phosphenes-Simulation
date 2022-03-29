import cv2
from phosphenesSim import pSim

cap = cv2.VideoCapture('./videos/vid2.mp4')
while cap.isOpened():
    ret,frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = pSim(frame)
    cv2.imshow('window-name', frame)
    if cv2.waitKey(30) & 0xFF == ord('0'):
        break

cap.release()
cv2.destroyAllWindows() # destroy all opened windows