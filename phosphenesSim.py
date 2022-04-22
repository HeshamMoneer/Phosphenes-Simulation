import cv2
import numpy as np
from gaussianBlur import blur
from enums import (Simode)
from preprocessing import prep
from gaussArray import gaussArr
import time

def drawPhosphene(phosphenes, tlf, radius, color, gArr, simode = Simode.BCM, cache = {}, squareSide = 0):
    color = int(color)

    if not color in cache:
        newCircle = np.zeros((squareSide, squareSide, 1), dtype=np.uint8)
        center = (squareSide//2, squareSide//2)

        if simode == Simode.BCM:
            cv2.circle(newCircle, center, radius, color, -1)  # -1 means solid circle

        elif simode == Simode.BSM:
            curRadius = (radius*color)//255
            cv2.circle(newCircle, center, curRadius, 255, -1)  # -1 means solid circle

        elif simode == Simode.ACM:
            for i in range(radius):
                cv2.circle(newCircle, center, i, color * gArr[i], 1)

        elif simode == Simode.ASM:
            curRadius = (radius*color)//255
            step = len(gArr)/curRadius if curRadius > 0 else 0
            for i in range(curRadius):
                index = int(i * step)
                cv2.circle(newCircle, center, i, 255 * gArr[index], 1)
        
        cache[color] = newCircle

    x, y = tlf
    phosphenes[y:y+squareSide, x:x+squareSide] = cache[color]

'''
Inputs:
img: source image to be phosphenated
dim: number of phosphenes -> 32 * 32 resulting phosphene image for example
dimWin: phosphene image dimensions in pixels
mLevels: number of modulation levels
gArr: gauss array; needed if simode is ACM or ASM
simode: simulation mode
    could be gaussian blur (B) based or gaussian array (A) based
    could be color moduled (CM) or size modulated (SM)
'''
def pSim(img, dim = 32, dimWin = 640, mLevels = 16, gArr = None, simode = Simode.BCM, cache = {}):
    img = prep(img, dim, mLevels) # image preprocessing
    phosphenes = np.zeros((dimWin, dimWin, 1), dtype=np.uint8) # pixel grid that displays phosphenes
    squareSide = dimWin//dim # square pixels that will contain a phosphene
    radius = 0

    if simode == Simode.ACM or simode == Simode.ASM:
        radius = int(squareSide * 0.7)
        if gArr == None: gArr = gaussArr(radius)

    elif simode == Simode.BCM:
        radius = int(squareSide * 0.25)

    elif simode == Simode.BSM:
        radius = int(squareSide * 0.3)
    
    getTLF = lambda var : int(var * squareSide) # get the top left corner of phosphene square
    it = np.nditer(img, flags=['multi_index'])
    while not it.finished:
        y, x = it.multi_index
        color = it[0]
        it.iternext()
        tlf = (getTLF(x), getTLF(y)) # corresponding phosphene square top left corner
        drawPhosphene(phosphenes, tlf, radius, color, gArr, simode, cache, squareSide)
    
    blurKernel = 0
    if simode == Simode.ACM or simode == Simode.ASM:
        blurKernel = 3

    elif simode == Simode.BCM or simode == Simode.BSM:
        blurKernel = radius * 2

    phosphenes = blur(phosphenes, blurKernel)

    return phosphenes, cache

def main():
    imgNumber = eval(input("Enter Image number: "))
    img = cv2.imread('./images/img' + str(imgNumber) + '.jpg',0) # read desired image in grey scale

    start = time.time()
    counter = 0
    cache = {}
    while time.time() - start < 1:
        counter += 1
        tmpImg, cache = pSim(img, simode = Simode.BSM, cache = cache)

    print(counter)

    cv2.imshow("BSM", tmpImg)

    # cv2.imshow("ACM", pSim(img, simode = Simode.ACM))

    # cv2.imshow("ASM", pSim(img, simode = Simode.ASM))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

