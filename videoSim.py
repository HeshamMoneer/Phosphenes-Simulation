import cv2
import dlib
import time

from enums import (Modes, Simode)
from phosphenesSim import (pSim, Simode)
from bboxes import (updateBBoxes, applyBBoxes)
from gaussArray import gaussArr
from cropper import squareCrop


def vSim(cap, dim = 32, dimWin = 640, mLevels = 16, simode = Simode.BSM, facesMode = Modes.NOTHING, cache = {}):
    # Computer the gauss array in case needed
    gArr = None
    if simode == Simode.ACM or simode == Simode.ASM:
        squareSide = dimWin//dim
        radius = int(squareSide * 0.7)
        gArr = gaussArr(radius)
    
    fps = cap.get(cv2.CAP_PROP_FPS) # get the original video FPS
    print("Original FPS: "+str(fps))

    faces_classifier = cv2.CascadeClassifier('classifiers/cc.xml')
    eyes_classifier = cv2.CascadeClassifier('classifiers/ecc.xml')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('classifiers/shape_predictor_68_face_landmarks.dat')
    classifiers = [faces_classifier, eyes_classifier, detector, predictor]
    bboxes = []
    counter = 0
    while True:
        ret,frame = cap.read()
        if(not ret): break

        startTime = time.time()
        
        frame = squareCrop(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bboxes, counter = updateBBoxes(frame, bboxes, counter, classifiers, facesMode)
        frame = applyBBoxes(frame, bboxes, facesMode)
        frame, cache = pSim(img = frame, dim = dim, dimWin = dimWin, mLevels = mLevels, simode = simode, gArr = gArr, cache = cache)
        cv2.imshow('Phosphenated ' + simode.name, frame)
        
        endTime = time.time()
        print('FPS: '+ str(int(1/(endTime-startTime))), end='\r')

        if cv2.waitKey(1) & 0xFF == ord('0'): break
    cap.release()
    cv2.destroyAllWindows()

def main():
    vidNumber = eval(input("Enter Video number: "))
    cap = cv2.VideoCapture('./videos/vid' + str(vidNumber) + '.mp4' if vidNumber > 0 else 0) # vidNumber <= 0 opens the webcam
    vSim(cap, facesMode = Modes.SFR_ROI_M)

if __name__ == '__main__':
    main()