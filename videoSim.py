import cv2
import time

import simConfig as sc
from phosphenesSim import (pSim)
from bboxes import (updateBBoxes, applyBBoxes)

def switch_face(event, x, y, flags, *params):
    if event == cv2.EVENT_LBUTTONUP:
        sc.faceIndex += 1

def vSim(cap):
    fps = cap.get(cv2.CAP_PROP_FPS) # get the original video FPS
    print("Original FPS: "+str(fps))

    cv2.namedWindow(sc.windowName)
    cv2.setMouseCallback(sc.windowName, switch_face)

    while True:
        ret,frame = cap.read()
        if(not ret): break

        # startTime = time.time()
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        updateBBoxes(frame)
        frame = applyBBoxes(frame)
        frame = pSim(frame)
        cv2.imshow(sc.windowName, frame)
        
        # endTime = time.time()
        # print('FPS: '+ str(int(1/(endTime-startTime))), end='\r')

        if cv2.waitKey(1) & 0xFF == ord('0'): break
    cap.release()
    cv2.destroyAllWindows()

def main():
    sc.init()
    print("Enter Video number: ", end ="")
    vidNumber = eval(input())
    cap = cv2.VideoCapture('./videos/vid' + str(vidNumber) + '.mp4' if vidNumber > 0 else 0) # vidNumber <= 0 opens the webcam
    vSim(cap)

if __name__ == '__main__':
    main()