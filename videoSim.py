import cv2
from phosphenesSim import pSim
import time

cap = cv2.VideoCapture('./videos/vid2.mp4')
noFrames = 0
start = time.time()
while cap.isOpened():
    ret,frame = cap.read()
    if(not ret): break
    noFrames += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = pSim(frame)
    cv2.imshow('Phosphenated', frame)
    if cv2.waitKey(1) & 0xFF == ord('0'):
        break
end = time.time()

print('FPS = '+ str(noFrames/(end-start)))
cap.release()
cv2.destroyAllWindows() # destroy all opened windows