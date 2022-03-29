import cv2
from phosphenesSim import pSim
import time

def vSim(cap, width = 32, height = 32, noColors = 16):
    # get the original video FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Original FPS: "+str(fps))
    while True:
        ret,frame = cap.read()
        if(not ret): break
        cv2.imshow('Original', frame)
        startTime = time.time()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = pSim(frame, width, height, noColors)
        endTime = time.time()
        print('Actual FPS: '+ str(int(1/(endTime-startTime))), end='\r')
        cv2.imshow('Phosphenated', frame)
        if cv2.waitKey(1) & 0xFF == ord('0'): break

    cap.release()
    cv2.destroyAllWindows() # destroy all opened windows

def main():
    vidNumber = eval(input("Enter Video number: "))
    # vidNumber <= 0 opens the webcam
    cap = cv2.VideoCapture('./videos/vid' + str(vidNumber) + '.mp4' if vidNumber > 0 else 0)
    vSim(cap)

if __name__ == '__main__':
    main()