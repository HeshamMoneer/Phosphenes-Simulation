import cv2
from enums import (Modes, Simode)
from phosphenesSim import (pSim, Simode)
from bboxes import (updateBBoxes, applyBBoxes)
from fps import (printFPS, printOrginialFPS)
from gaussArray import gaussArr
from cropper import squareCrop
import dlib


def vSim(cap, dim = 32, dimWin = 640, mLevels = 16, simode = Simode.BSM, facesMode = Modes.NOTHING):
    # Computer the gauss array in case needed
    gArr = None
    if simode == Simode.ACM or simode == Simode.ASM:
        squareSide = dimWin//dim
        radius = int(squareSide * 0.7)
        gArr = gaussArr(radius)
    
    printOrginialFPS(cap)
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
        frame = squareCrop(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bboxes, counter = updateBBoxes(frame, bboxes, counter, classifiers, facesMode)
        frame = applyBBoxes(frame, bboxes, facesMode)
        frame = pSim(img = frame, dim = dim, dimWin = dimWin, mLevels = mLevels, simode = simode, gArr = gArr)
        cv2.imshow('Phosphenated ' + simode.name, frame)
        if cv2.waitKey(1) & 0xFF == ord('0'): break
    cap.release()
    cv2.destroyAllWindows()

def main():
    vidNumber = eval(input("Enter Video number: "))
    cap = cv2.VideoCapture('./videos/vid' + str(vidNumber) + '.mp4' if vidNumber > 0 else 0) # vidNumber <= 0 opens the webcam
    vSim(cap, facesMode = Modes.SCALE_TO_FIRST_FACE)

if __name__ == '__main__':
    main()