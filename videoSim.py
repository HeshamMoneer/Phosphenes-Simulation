import cv2
from phosphenesSim import (pSim, Simode)
from facesDetector import (detectAllFaces, scaleToFirstFace, brightenFirstFace)
from fps import (printFPS, printOrginialFPS)
from gaussArray import gaussArr
import enum

class Modes(enum.Enum): # Face Detection mode
    NOTHING = 0
    DETECT_ALL_FACES = 1
    SCALE_TO_FIRST_FACE = 2
    BRIGHTEN_FIRST_FACE = 3

def vSim(cap, dim = 32, dimWin = 640, mLevels = 16, simode = Simode.BCM, facesMode = Modes.NOTHING):
    # Computer the gauss array in case needed
    gArr = None
    if simode == Simode.ACM or simode == Simode.ASM:
        squareSide = dimWin//dim
        radius = int(squareSide * 0.7)
        gArr = gaussArr(radius)
    
    printOrginialFPS(cap)
    classifier = cv2.CascadeClassifier('cc.xml')
    while True:
        ret,frame = cap.read()
        if(not ret): break
        # cv2.imshow('Original', frame)
        frame = printFPS(lambda: fSim(frame, classifier, dim, dimWin, mLevels, gArr, simode, facesMode))
        cv2.imshow('Phosphenated ' + simode.name, frame)
        if cv2.waitKey(1) & 0xFF == ord('0'): break
    cap.release()
    cv2.destroyAllWindows()

def fSim(frame, classifier, dim, dimWin, mLevels, gArr, simode, facesMode):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if facesMode == Modes.DETECT_ALL_FACES: 
        frame = detectAllFaces(frame, classifier)
    elif facesMode == Modes.SCALE_TO_FIRST_FACE: 
        frame = scaleToFirstFace(frame, classifier)
    elif facesMode == Modes.BRIGHTEN_FIRST_FACE: 
        frame = brightenFirstFace(frame, classifier)
    return pSim(img = frame, dim = dim, dimWin = dimWin, mLevels = mLevels, simode = simode, gArr = gArr)


def main():
    vidNumber = eval(input("Enter Video number: "))
    cap = cv2.VideoCapture('./videos/vid' + str(vidNumber) + '.mp4' if vidNumber > 0 else 0) # vidNumber <= 0 opens the webcam
    vSim(cap, simode = Simode.BCM, facesMode = Modes.SCALE_TO_FIRST_FACE)

if __name__ == '__main__':
    main()