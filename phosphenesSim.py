import cv2
import numpy as np
import colorSampler as cS
from gaussianBlur import blur
from cropper import crop

def pSim(imgNumber, width = 32, height = 32, noColors = 16):
    # create color samples array
    colorSampler = cS.colorSampler(noColors)
    # read desired image in grey scale
    img = cv2.imread('./images/img' + str(imgNumber) + '.jpg',0)
    # crop image to be square
    img = crop(img)
    # resize image to desired resolution
    img = cv2.resize(img, (width, height))
    # resulting image width and height
    widthWin = heightWin = 640
    # pixel grid that displays phosphenes
    phosphenes = np.zeros((widthWin, heightWin, 1), dtype=np.uint8)
    # square pixels that will contains a phosphene
    squareSide = widthWin//width
    # phosphene radius
    radius = int(squareSide * 0.25) 
    for x in range(0, width):
        for y in range(0, height):
            color = img[y,x]
            # find the center of the corresponding phosphene cell
            xI = int(x * squareSide + squareSide/2)
            yI = int(y * squareSide + squareSide/2)
            # sample the picked color and type cast it from float
            color = int(colorSampler.sample(color))
            # -1 means solid circle
            cv2.circle(phosphenes, (xI, yI), radius, color, -1)
    phosphenes = blur(phosphenes, radius)
    return phosphenes

imgNumber = eval(input("Enter Image number: "))
# width = eval(input("Enter Image width in phosphenes: "))
# height = eval(input("Enter Image height in phosphenes: "))
# noColors = eval(input("Enter number of colors: "))

phosphenes = pSim(imgNumber)

cv2.imshow("Phosphenes",phosphenes)
cv2.waitKey(0)
cv2.destroyAllWindows()

