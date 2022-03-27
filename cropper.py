import math

def crop(img):
  height, width = len(img), len(img[0])
  dim = min([height, width])
  return img[0:dim, 0:dim]
