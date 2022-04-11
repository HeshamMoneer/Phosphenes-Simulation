def squareCrop(img):
  height, width = img.shape[0], img.shape[1]
  if height == width: return img
  dim = min([height, width])
  return img[0:dim, 0:dim]
