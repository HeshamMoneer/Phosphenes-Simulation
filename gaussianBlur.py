import cv2

def blur(img, factor):
  kernelSize = int(factor * 1.9)
  kernelSize = kernelSize if kernelSize % 2 == 1 else kernelSize - 1
  sigma = factor
  return cv2.GaussianBlur(img,(kernelSize, kernelSize), sigma, cv2.BORDER_DEFAULT)