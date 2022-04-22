import cv2
import numpy as np
from preprocessing import contrastBrightness
from enums import Modes
import dlib

def updateBBoxes(frame, bboxes, counter, classifiers, facesMode, ue = 5):
    if counter == 0:
        if facesMode == Modes.DETECT_FACE_FEATURES:
            detector = classifiers[2]
            predictor = classifiers[3]
            rects = detector(frame, 1)
            if len(rects) > 0:
                rect = rects[0]
                shape = predictor(frame, rect)
                for i in range(0,68):
                    x = shape.part(i).x
                    y = shape.part(i).y
                    cv2.circle(frame, (x, y), 2, 255, 2)

        else:
            bboxes = classifiers[0].detectMultiScale(frame, scaleFactor = 1.3)
            if len(bboxes) > 0 and facesMode == Modes.SFR_ROI_M:
                xOld,yOld,wOld,hOld = bboxes[0]

                predictor = classifiers[3]
                rect = dlib.rectangle(xOld,yOld,xOld+wOld, yOld+hOld)
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

                bboxes[0] = (xNew,yNew,wNew,hNew)
            elif facesMode == Modes.DETECT_FACES_WITH_EYES:
                allEyes = []
                for x,y,w,h in bboxes:
                    subImg = frame[y:y+h, x:x+w]
                    eyes = classifiers[1].detectMultiScale(subImg)
                    for x2, y2, w2, h2 in eyes: allEyes.append([x+x2, y+y2, w2, h2])
                for box in bboxes: allEyes.append(box.tolist())
                bboxes = allEyes
    counter = (counter + 1) % ue if ue > 1 else 0 # update bboxes every ue frames
    return bboxes, counter

def applyBBoxes(frame, bboxes, facesMode):
    if facesMode == Modes.DETECT_ALL_FACES or facesMode == Modes.DETECT_FACES_WITH_EYES: 
        for x,y,w,h in bboxes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 1)
    elif facesMode == Modes.VJFR_ROI_M or facesMode == Modes.SFR_ROI_M: 
        x, y, w, h = bboxes[0] if len(bboxes) > 0 else (0,0,frame.shape[1],frame.shape[0])
        frame = frame[y:y+h, x:x+w]
    elif facesMode == Modes.BRIGHTEN_FIRST_FACE: 
        if len(bboxes) > 0:
            x, y, w, h = bboxes[0] 
            frame[y:y+h, x:x+w] = np.vectorize(contrastBrightness)(frame[y:y+h, x:x+w], 1, 30)
    return frame