import cv2
import numpy as np
from dlib import rectangle
import skimage

import simConfig as sc
from preprocessing import contrastBrightness
from enums import Modes
from caricaturing.__init__ import caric
from talking_detection.__init__ import talking_probability
from emotion_recognition.__init__ import detectEmo

def updateBBoxes(frame):
    if sc.facesMode == Modes.SFR_ROI_M_TD or sc.facesMode == Modes.VJFR_ROI_M_TD:
        bboxes = sc.classifiers[0].detectMultiScale(frame, scaleFactor = 1.3)
        sc.bboxes = bboxes
        if len(sc.talkingAcc) == 0 or len(sc.talkingAcc) != len(bboxes): 
            sc.talkingAcc = [[] for _ in range(len(bboxes))]
        for i in range(len(bboxes)):
            x, y, w, h = bboxes[i]
            sc.talkingAcc[i].append(frame[y:y+h, x:x+w])
        if len(sc.talkingAcc) > 0 and len(sc.talkingAcc[0]) == 25:
            predictor = sc.classifiers[2]
            for i in range(len(sc.talkingAcc)):
                sc.talkingAcc[i] = talking_probability(sc.talkingAcc[i], predictor, sc.talkingModel, sc.talkingScaler)
            sc.faceIndex = np.argmax(sc.talkingAcc)
            print(sc.talkingAcc)
            print(sc.faceIndex)
            sc.talkingAcc = []
        return
    if sc.counter == 0:
        #scaleFactor & minNeighbors can make detection faster but compromise accuracy
        bboxes = sc.classifiers[0].detectMultiScale(frame, scaleFactor = 1.3)
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
    sc.faceIndex = len(sc.bboxes) - 1 if sc.faceIndex >= len(sc.bboxes) else sc.faceIndex

    if sc.facesMode in [Modes.DETECT_ALL_FACES, Modes.DETECT_FACES_WITH_EYES]: 
        for x,y,w,h in sc.bboxes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 1)

    elif sc.facesMode in [Modes.VJFR_ROI_M, Modes.SFR_ROI_M, Modes.VJFR_ROI_C, Modes.SFR_ROI_HE, Modes.SFR_ROI_M_TD, Modes.SFR_ROI_M_ER, Modes.VJFR_ROI_M_TD, Modes.VJFR_ROI_M_ER, Modes.VJFR_ROI_HE]:
        if len(sc.bboxes) > 0:
            x, y, w, h = sc.bboxes[sc.faceIndex]
            if sc.skip_enhancements_flag:
                x, y, w, h = zoomout(frame, x, y, w, h)
            elif sc.zoom_counter < 10:
                x, y, w, h = zoomin(frame, x, y, w, h)
            if sc.facesMode in [Modes.SFR_ROI_M, Modes.SFR_ROI_M_TD]:
                x, y, w, h = VJFR_to_SFR(x, y, w, h, frame)
                frame = frame[y:y+h, x:x+w]
            elif sc.facesMode == Modes.SFR_ROI_HE or sc.facesMode == Modes.VJFR_ROI_HE:
                subFrame = frame[y:y+h, x:x+w] #VJFR
                subFrame = heq(subFrame) #equalization
                if sc.facesMode == Modes.SFR_ROI_HE:
                    x, y, w, h = VJFR_to_SFR(x, y, w, h, frame)
                frame = frame[y:y+h, x:x+w]
            elif sc.facesMode in [Modes.SFR_ROI_M_ER, Modes.VJFR_ROI_M_ER]:
                subframe = frame[y:y+h, x:x+w]
                if sc.counter == 1:
                    sc.emotionIndex = detectEmo(subframe, sc.emotionsModel)
                if sc.facesMode == Modes.SFR_ROI_M_ER: x, y, w, h = VJFR_to_SFR(x, y, w, h, frame)
                frame = frame[y:y+h, x:x+w]
            elif sc.facesMode == Modes.VJFR_ROI_C:
                frame = frame[y:y+h, x:x+w]
                frame = caric(frame)
            elif sc.facesMode in [Modes.VJFR_ROI_M, Modes.VJFR_ROI_M_TD]:
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

def heq(frame):
    rect = rectangle(0, 0, frame.shape[1], frame.shape[0])
    predictor = sc.classifiers[2]
    shape = predictor(frame, rect)
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    outline = landmarks[[*range(17), *range(26,16,-1)]]
    Y, X = skimage.draw.polygon(outline[:,1], outline[:,0], frame.shape)
    cropped_face = np.zeros(frame.shape, dtype=np.uint8)
    cropped_face[Y, X] = frame[Y, X]
    cropped_face = cv2.equalizeHist(cropped_face)
    frame[Y, X] = cropped_face[Y, X]
    return frame

def zoomout(frame, x, y, w, h):
    sc.zoom_counter += 1
    enlarge_factor = sc.zoom_counter * frame.shape[0]//20
    x = max(x-enlarge_factor, 0)
    y = max(y-enlarge_factor, 0)
    x2 = min(x + w + enlarge_factor*2, frame.shape[1] - 1)
    y2 = min(y + h + enlarge_factor*2, frame.shape[0] - 1)
    w = x2 - x
    h = y2 - y
    return (x,y,w,h)

def zoomin(frame, x, y, w, h):
    sc.zoom_counter += 1
    shrink_factor_y = sc.zoom_counter * frame.shape[0]//10
    shrink_factor_x = sc.zoom_counter * frame.shape[1]//10
    x1 = min(shrink_factor_x, x)
    y1 = min(shrink_factor_y, y)
    x2 = max(frame.shape[1] - shrink_factor_x, x + w)
    y2 = max(frame.shape[0] - shrink_factor_y, y + h)
    w = x2 - x1
    h = y2 - y1
    return (x1,y1,w,h)