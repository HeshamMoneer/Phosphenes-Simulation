import cv2
import numpy as np
from dlib import rectangle

import simConfig as sc
from preprocessing import contrastBrightness
from enums import Modes

def updateBBoxes(frame):
    if sc.counter == 0:
        #scaleFactor & minNeighbors can make detection faster but compromise accuracy
        bboxes = sc.classifiers[0].detectMultiScale(frame)
        if sc.facesMode == Modes.DETECT_FACES_WITH_EYES:
            allEyes = []
            for x,y,w,h in bboxes:
                subImg = frame[y:y+h, x:x+w]
                eyes = sc.classifiers[1].detectMultiScale(subImg)
                for x2, y2, w2, h2 in eyes: allEyes.append([x+x2, y+y2, w2, h2])
            for box in bboxes: allEyes.append(box.tolist())
            bboxes = allEyes
        sc.bboxes = bboxes
    sc.counter = (sc.counter + 1) % sc.ur if sc.ur > 1 else 0 # update bboxes every ur frames

def applyBBoxes(frame):
    sc.faceIndex = 0 if sc.faceIndex >= len(sc.bboxes) else sc.faceIndex

    if sc.facesMode == Modes.DETECT_ALL_FACES or sc.facesMode == Modes.DETECT_FACES_WITH_EYES: 
        for x,y,w,h in sc.bboxes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 1)

    elif sc.facesMode == Modes.VJFR_ROI_M or sc.facesMode == Modes.SFR_ROI_M:
        if len(sc.bboxes) > 0:
            x, y, w, h = sc.bboxes[sc.faceIndex]
            if sc.facesMode == Modes.SFR_ROI_M:
                x, y, w, h = VJFR_to_SFR(x, y, w, h, frame)
            frame = frame[y:y+h, x:x+w]

    elif sc.facesMode == Modes.DETECT_FACE_FEATURES:
        if len(sc.bboxes) > 0:
            x, y, w, h = sc.bboxes[sc.faceIndex]
            rect = rectangle(x, y, x+w, y+h)
            predictor = sc.classifiers[2]
            shape = predictor(frame, rect)
            for i in range(0,68):
                x = shape.part(i).x
                y = shape.part(i).y
                cv2.circle(frame, (x, y), 1, 255, 1)
    
    elif sc.facesMode == Modes.BRIGHTEN_FIRST_FACE: 
        if len(sc.bboxes) > 0:
            x, y, w, h = sc.bboxes[sc.faceIndex] 
            frame[y:y+h, x:x+w] = contrastBrightness(frame[y:y+h, x:x+w], 1, 30)
    
    return frame

def VJFR_to_SFR(xOld, yOld, wOld, hOld, frame):
    predictor = sc.classifiers[2]
    rect = rectangle(xOld,yOld,xOld+wOld, yOld+hOld)
    nose = predictor(frame, rect).part(33)
    xNose = nose.x
    yNose = nose.y

    ERW = 1.34
    ERHU = 1.802
    ERHD = 1.2
    
    # xNose - xNew = ERW * (xNose - xOld)
    xNew = xNose - ERW * (xNose - xOld)
    xNew = max(0, int(xNew))
    
    # yNose - yNew = ERHU * (yNose - yOld)
    yNew = yNose - ERHU * (yNose - yOld)
    yNew = max(0, int(yNew))

    # wNew = wOld * ERW
    wNew = int(ERW * wOld)
    if xNew + wNew > frame.shape[1]: wNew = frame.shape[1] - xNew

    # hNew + yNew - yNose = ERHD * (hOld + yOld - yNose)
    hNew = int(ERHD * (hOld + yOld - yNose) - yNew + yNose)
    if yNew + hNew > frame.shape[0]: hNew = frame.shape[0] - yNew

    return (xNew,yNew,wNew,hNew)