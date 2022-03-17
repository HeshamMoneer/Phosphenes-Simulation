import cv2
import numpy as np
import colorSampler as cS

imgNumber = eval(input("Enter Image number: "))
width = eval(input("Enter Image width: "))
height = eval(input("Enter Image height: "))
radius = eval(input("Enter phosphene radius: "))
distance = eval(input("Enter distance between phosphenes: "))
noColors = eval(input("Enter number of colors: "))
colorSampler = cS.colorSampler(noColors)


# read desired image in grey scale
img = cv2.imread('./images/img' + str(imgNumber) + '.jpg',0)

# resize image to desired resolution
img = cv2.resize(img, (width, height))

# square pixels that will contains a phosphene
squareSide = 2 * radius + distance 

# pixel grid that displays phosphenes
phosphenes = np.zeros((squareSide * height, squareSide * width, 1), dtype=np.uint8)

for x in range(0, width):
    for y in range(0, height):
        color = img[y,x]
        # find the center of the corresponding phosphene cell
        xI = int(x * squareSide + squareSide/2)
        yI = int(y * squareSide + squareSide/2)
        color = int(colorSampler.sample(color))
        cv2.circle(phosphenes, (xI, yI), int(radius * 0.85), int(color), -1)

blurFactor = squareSide // 2
blurFactor += 1 if blurFactor % 2 == 0 else 0
phosphenes = cv2.GaussianBlur(phosphenes,(blurFactor, blurFactor), cv2.BORDER_DEFAULT)
phosphenes = cv2.resize(phosphenes, (width * 10, height * 10))
cv2.imshow("Phosphenes",phosphenes)
cv2.waitKey(0)
cv2.destroyAllWindows()

