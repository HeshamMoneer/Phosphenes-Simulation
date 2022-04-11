import numpy as np
from cropper import squareCrop
from modulator import modulate
import cv2

def prep(img, dim, mLevels):
  img = squareCrop(img) # crop image to be a square; if not already a square
  img = cv2.resize(img, (dim, dim)) # resize image to desired resolution; bilinear interpolation
  img = modulate(img, 255, mLevels - 1) # modulate colors to given levels; SIMD applied
  img = contrastBrightness(img, 2, 20)
  return img

def contrastBrightness(val, a, b): # a -> contrast, b -> brightness
  return np.clip(a * val + b, 0, 255)