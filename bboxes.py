import cv2
import numpy as np
from preprocessing import contrastBrightness
from enums import Modes

def updateBBoxes(frame, bboxes, counter, classifiers, facesMode):
    if counter == 0:
        bboxes = classifiers[0].detectMultiScale(frame)
        if facesMode == Modes.DETECT_FACES_WITH_EYES:
            allEyes = []
            for x,y,w,h in bboxes:
                subImg = frame[y:y+h, x:x+w]
                eyes = classifiers[1].detectMultiScale(subImg)
                allEyes.extend(eyes)
            bboxes.extend(allEyes)
    counter = (counter + 1) % 10 # update bboxes every 10 frames
    return bboxes, counter

def applyBBoxes(frame, bboxes, facesMode):
    if facesMode == Modes.DETECT_ALL_FACES or facesMode == Modes.DETECT_FACES_WITH_EYES: 
        for x,y,w,h in bboxes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 1)
    elif facesMode == Modes.SCALE_TO_FIRST_FACE: 
        x, y, w, h = bboxes[0] if len(bboxes) > 0 else (0,0,frame.shape[1],frame.shape[0])
        frame = frame[y:y+h, x:x+w]
    elif facesMode == Modes.BRIGHTEN_FIRST_FACE: 
        x, y, w, h = bboxes[0] if len(bboxes) > 0 else (0,0,0,0)
        frame = np.vectorize(contrastBrightness)(frame[y:y+h, x:x+w], 1, 30)
    return frame