import time
import cv2

def printFPS(funct):
  startTime = time.time()
  returned = funct()
  endTime = time.time()
  print('FPS: '+ str(int(1/(endTime-startTime))), end='\r')
  return returned

def printOrginialFPS(cap):
  fps = cap.get(cv2.CAP_PROP_FPS) # get the original video FPS
  print("Original FPS: "+str(fps))
