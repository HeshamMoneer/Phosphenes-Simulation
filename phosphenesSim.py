import cv2
import numpy as np
import time

import simConfig as sc
from gaussianBlur import blur
from enums import (Simode, Modes)
from preprocessing import prep
from bboxes import updateBBoxes, applyBBoxes

def drawPhosphene(phosphenes, tlf, color):
    color = int(color)

    if not color in sc.cache:
        newCircle = np.zeros((sc.squareSide, sc.squareSide, 1), dtype=np.uint8)
        center = (sc.squareSide//2, sc.squareSide//2)

        if sc.simode == Simode.BCM:
            cv2.circle(newCircle, center, sc.radius, color, -1)

        elif sc.simode == Simode.BSM:
            curRadius = (sc.radius*color)//255
            cv2.circle(newCircle, center, curRadius, 255, -1)

        elif sc.simode == Simode.ACM:
            for i in range(sc.radius):
                cv2.circle(newCircle, center, i, color * sc.gArr[i], 1)

        elif sc.simode == Simode.ASM:
            curRadius = (sc.radius*color)//255
            step = len(sc.gArr)/curRadius if curRadius > 0 else 0
            for i in range(curRadius):
                index = int(i * step)
                cv2.circle(newCircle, center, i, 255 * sc.gArr[index], 1)
        
        newCircle = blur(newCircle, sc.blurKernel)
        sc.cache[color] = newCircle

    x, y = tlf
    phosphenes[y:y+sc.squareSide, x:x+sc.squareSide] = sc.cache[color]


def pSim(img):
    img = prep(img) # image preprocessing
    phosphenes = np.zeros((sc.dimWin, sc.dimWin), dtype=np.uint8) # pixel grid that displays phosphenes

    if sc.facesMode == Modes.SFR_ROI_M_ER or sc.facesMode == Modes.VJFR_ROI_M_ER:
        print(str(sc.emotionIndex) + " " +sc.emotion_dict[sc.emotionIndex], end='\r')
        binary = format(sc.emotionIndex, '03b')
        for i in range(len(binary)):
            img[0, i] = int(binary[i]) * 255

    getTLF = lambda var : int(var * sc.squareSide) # get the top left corner of phosphene square
    it = np.nditer(img, flags=['multi_index'])
    while not it.finished:
        y, x = it.multi_index
        color = it[0]
        it.iternext()
        tlf = (getTLF(x), getTLF(y)) # corresponding phosphene square top left corner
        drawPhosphene(phosphenes, tlf, color)

    # phosphenes = blur(phosphenes, sc.blurKernel)

    return phosphenes

def main():
    sc.init()

    imgNumber = eval(input("Enter Image number: "))
    img = cv2.imread('./images/img' + str(imgNumber) + '.jpg',0) # read desired image in grey scale

    updateBBoxes(img)
    tmpImg = applyBBoxes(img)
    tmpImg = pSim(tmpImg)

    cv2.imshow("ACM", tmpImg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

