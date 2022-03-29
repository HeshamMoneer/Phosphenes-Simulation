import cv2
from phosphenesSim import pSim
import time

def vSim(cap, width = 32, height = 32, noColors = 16):
    # get the original video FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Original FPS: "+str(fps))
    noFrames = 0
    elapsedTime = 0
    while True:
        startTime = time.time()
        ret,frame = cap.read()
        if(not ret): break
        noFrames += 1
        cv2.imshow('Original', frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = pSim(frame, width, height, noColors)
        cv2.imshow('Phosphenated', frame)
        elapsedTime += time.time() - startTime
        if cv2.waitKey(1) & 0xFF == ord('0'): break

    print('Actual FPS: '+ str(noFrames/elapsedTime))
    cap.release()
    cv2.destroyAllWindows() # destroy all opened windows

def main():
    vidNumber = eval(input("Enter Video number: "))
    # vidNumber <= 0 opens the webcam
    cap = cv2.VideoCapture('./videos/vid' + str(vidNumber) + '.mp4' if vidNumber > 0 else 0)
    vSim(cap)

if __name__ == '__main__':
    main()