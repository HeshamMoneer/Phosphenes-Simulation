import cv2
import time

import simConfig as sc
from phosphenesSim import (pSim)
from bboxes import (updateBBoxes, applyBBoxes)

def switch_face(event, x, y, flags, *params):
    if event == cv2.EVENT_LBUTTONUP:
        sc.faceIndex += 1
    if event == cv2.EVENT_LBUTTONDBLCLK:
        sc.skip_enhancements_flag = not sc.skip_enhancements_flag
        sc.counter = 0

def vSim(cap):
    fps = cap.get(cv2.CAP_PROP_FPS) # get the original video FPS
    print("Original FPS: "+str(fps))
    original_frame_ms = int((1/fps) * 1000)

    cv2.namedWindow(sc.windowName,  cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(sc.windowName, switch_face)
    cv2.setWindowProperty(sc.windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    while True:
        ret,frame = cap.read()
        if(not ret): break

        startTime = time.time()
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not sc.skip_enhancements_flag:
            updateBBoxes(frame)
            frame = applyBBoxes(frame)
        frame = pSim(frame)
        cv2.imshow(sc.windowName, frame)
        
        endTime = time.time()
        # print('FPS: '+ str(int(1/(endTime-startTime))), end='\r')
        elapsed_ms = int((endTime - startTime) * 1000)
        waiting_time = original_frame_ms - elapsed_ms
        if waiting_time <= 0 : waiting_time = 1
        if cv2.waitKey(waiting_time) & 0xFF == ord('0'): break
    cap.release()
    cv2.destroyAllWindows()

def main():
    sc.init()
    print("Enter Experiment type: ", end ="")
    experiment_type = input()
    path = './experiment/tests/'
    if experiment_type.isnumeric():
        path += 'G' + experiment_type + '/Identity test.mp4'
    else:
        path += experiment_type + '.mp4'
    cap = cv2.VideoCapture(path) # vidNumber <= 0 opens the webcam
    vSim(cap)

if __name__ == '__main__':
    main()