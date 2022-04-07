import cv2
import numpy as np
from modulator import modulate
from gaussianBlur import blur
from cropper import squareCrop
import enum
from gaussArray import gaussArr

class Simode(enum.Enum): # Simulation mode
    BCM = 0 # Blur color modulated
    BSM = 1 # Blur size modulated
    ACM = 2 # Array color modulated
    ASM = 3 # Array size modulated

def drawPhosphene(phosphenes, center, radius, color, gArr, simode = Simode.BCM):
    color = int(color)

    if simode == Simode.BCM:
        cv2.circle(phosphenes, center, radius, color, -1)  # -1 means solid circle

    elif simode == Simode.BSM:
        curRadius = (radius*color)//255
        cv2.circle(phosphenes, center, curRadius, 255, -1)  # -1 means solid circle

    elif simode == Simode.ACM:
        for i in range(radius):
            cv2.circle(phosphenes, center, i, color * gArr[i], 1)

    elif simode == Simode.ASM:
        curRadius = (radius*color)//255
        step = len(gArr)/curRadius if curRadius > 0 else 0
        for i in range(curRadius):
            index = int(i * step)
            cv2.circle(phosphenes, center, i, 255 * gArr[index], 1)

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
def pSim(img, dim = 32, dimWin = 640, mLevels = 16, gArr = None, simode = Simode.BCM):
    img = squareCrop(img) # crop image to be a square
    img = cv2.resize(img, (dim, dim)) # resize image to desired resolution; bilinear interpolation
    img = np.vectorize(modulate)(img, 255, mLevels - 1) # modulate colors to given levels; SIMD applied
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
    
    getCenter = lambda var : int(var * squareSide + squareSide/2) # get the center of phosphene square
    it = np.nditer(img, flags=['multi_index'])
    while not it.finished:
        y, x = it.multi_index
        color = it[0]
        it.iternext()
        center = (getCenter(x), getCenter(y)) # corresponding phosphene square center
        drawPhosphene(phosphenes, center, radius, color, gArr, simode)
    
    blurKernel = 0
    if simode == Simode.ACM or simode == Simode.ASM:
        blurKernel = 3

    elif simode == Simode.BCM or simode == Simode.BSM:
        blurKernel = radius * 2

    phosphenes = blur(phosphenes, blurKernel)
    return phosphenes

def main():
    imgNumber = eval(input("Enter Image number: "))
    img = cv2.imread('./images/img' + str(imgNumber) + '.jpg',0) # read desired image in grey scale

    cv2.imshow("BCM", pSim(img, simode = Simode.BCM))

    cv2.imshow("BSM", pSim(img, simode = Simode.BSM))

    cv2.imshow("ACM", pSim(img, simode = Simode.ACM))

    cv2.imshow("ASM", pSim(img, simode = Simode.ASM))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

