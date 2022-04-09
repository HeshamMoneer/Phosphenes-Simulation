import cv2
import numpy as np

def detectAllFaces(img, classifiers):
  bboxes = classifiers[0].detectMultiScale(img)
  for box in bboxes:
      x, y, width, height = box
      cv2.rectangle(img, (x, y), (x + width, y + height), 255, 1)
  return img

def scaleToFirstFace(img, classifiers):
  bboxes = classifiers[0].detectMultiScale(img)
  x, y, width, height = bboxes[0] if len(bboxes) > 0 else (0,0,img.shape[1],img.shape[0])
  return img[y:y+height, x:x+width]

def brightenFirstFace(img, classifiers):
  bboxes = classifiers[0].detectMultiScale(img)
  if len(bboxes) > 0:
    x, y, width, height = bboxes[0]
    it = np.nditer(img[y:y+height, x:x+width], op_flags=['readwrite'])
    while not it.finished:
      it[0] = it[0] + 30 if it[0] < 225 else 255
      it.iternext()
  return img

def detectFacesWithEyes(img, classifiers):
  face_detector = classifiers[0]
  eyes_detector = classifiers[1]

  faces = face_detector.detectMultiScale(img)

  for x,y, width, height in faces:
    cv2.rectangle(img, (x, y), (x + width, y + height), 255, 1)
    subImg = img[y:y+height, x:x+width]
    eyes = eyes_detector.detectMultiScale(subImg)
    for x2,y2, width2, height2 in eyes:
      cv2.rectangle(img, (x + x2, y + y2), (x + x2 + width2, y + y2 +height2), 255, 1)
  
  return img

  