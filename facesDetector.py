import cv2
import numpy as np

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
    it = np.nditer(img[y:y+height, x:x+width], op_flags=['readwrite'])
    while not it.finished:
      it[0] = it[0] + 30 if it[0] < 225 else 255
      it.iternext()
  return img

  