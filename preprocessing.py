import cv2
import numpy as np

import simConfig as sc
from modulator import modulate

def prep(img):
  # img = squareCrop(img) # crop image to be a square; if not already a square
  img = squareFit(img)
  img = cv2.resize(img, (sc.dim, sc.dim)) # resize image to desired resolution; bilinear interpolation
  img = cv2.equalizeHist(img)
  img = modulate(img, 255, sc.mLevels - 1) # modulate colors to given levels; SIMD applied
  return img

def contrastBrightness(val, a, b): # a -> contrast, b -> brightness
  return np.clip(a * np.uint16(val) + b, 0, 255)

def squareCrop(img):
  height, width = img.shape[0], img.shape[1]
  if height == width: return img
  dim = min([height, width])
  sh = int(height/2 - dim /2)
  sw = int(width/2 - dim /2)
  return img[sh:sh+dim, sw:sw+dim]

def squareFit(img):
  height, width = img.shape[0], img.shape[1]
  if height == width: return img
  dim = max([height, width])
  res = np.zeros((dim, dim), dtype=np.uint8)
  sh = int((dim - height)/2)
  sw = int((dim - width)/2)
  res[sh:height+sh, sw:width+sw] = img
  return res