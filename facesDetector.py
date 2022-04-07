import cv2

def detectAllFaces(img, classifier):
  bboxes = classifier.detectMultiScale(img)
  for box in bboxes:
      x, y, width, height = box
      cv2.rectangle(img, (x, y), (x + width, y + height), 255, 1)
  return img

def scaleToFirstFace(img, classifier):
  bboxes = classifier.detectMultiScale(img)
  x, y, width, height = bboxes[0] if len(bboxes) > 0 else (0,0,img.shape[1],img.shape[0])
  return img[y:y+height, x:x+width]

def brightenFirstFace(img, classifier):
  bboxes = classifier.detectMultiScale(img)
  if len(bboxes) > 0:
    x, y, width, height = bboxes[0]
    for xI in range(x, x+width):
      for yI in range(y, y+height):
        img[yI, xI] += 30
        if img[yI, xI] > 255: img[yI, xI] = 255
  return img

  