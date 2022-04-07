import cv2

def blur(img, kernelSize = 3, lastValue = 3):
  kernelSize = int(kernelSize)
  kernelSize = kernelSize if kernelSize % 2 == 1 else kernelSize - 1
  sigma = kernelSize//lastValue
  return cv2.GaussianBlur(img,(kernelSize, kernelSize), sigma, cv2.BORDER_REPLICATE)