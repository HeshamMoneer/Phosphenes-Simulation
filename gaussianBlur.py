import cv2

def blur(img, radius):
  kernelSize = int(radius * 2)
  kernelSize = kernelSize if kernelSize % 2 == 1 else kernelSize - 1
  sigma = radius
  return cv2.GaussianBlur(img,(kernelSize, kernelSize), sigma, cv2.BORDER_DEFAULT)