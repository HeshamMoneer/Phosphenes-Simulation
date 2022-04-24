import cv2
import math
import numpy as np
import dlib
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

def dist(p1, p2):
  p1_x = p1[0]
  p2_x = p2[0]
  p1_y = p1[1]
  p2_y = p2[1]
  dist = math.sqrt((p2_x - p1_x) ** 2 + (p2_y - p1_y) ** 2)
  return dist

# processes 25 facial frames at a time
def talking_probability(frames, predictor, model, scaler):
  avg_gaps = []

  for frame in frames:
    faceRect = dlib.rectangle(0,0,frame.shape[1], frame.shape[0])
    shape = predictor(frame, faceRect)

    part_50 = (shape.part(50).x, shape.part(50).y)
    part_58 = (shape.part(58).x, shape.part(58).y)
    part_51 = (shape.part(51).x, shape.part(51).y)
    part_57 = (shape.part(57).x, shape.part(57).y)
    part_52 = (shape.part(52).x, shape.part(52).y)
    part_56 = (shape.part(56).x, shape.part(56).y)

    A = dist(part_50, part_58)
    B = dist(part_51, part_57)
    C = dist(part_52, part_56)

    avg_gap = (A + B + C) / 3.0

    avg_gaps.append([avg_gap])

  avg_gaps = scaler.fit_transform(avg_gaps)

  X_data = np.array([avg_gaps])
  y_pred = model.predict_on_batch(X_data)
  y_max = y_pred[0].argmax() # 0 -> silent, 1 -> speaking
  return y_pred[0][1]
  

def detect_face(img, classifier):
  if len(img.shape) > 2 :
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = classifier.detectMultiScale(img)
  if (len(faces) == 0):
    return img, None
  (x, y, w, h) = faces[0]
  return img[y:y+w, x:x+h], faces[0]

def main():
  model = load_model('model.h5')
  classifier = cv2.CascadeClassifier('../classifiers/cc.xml')
  scaler = MinMaxScaler()
  predictor = dlib.shape_predictor('../classifiers/shape_predictor_68_face_landmarks.dat')

  cap = cv2.VideoCapture(0)
  frames = []
  while True:
    ret,frame = cap.read()
    if(not ret): break
    frame, rect = detect_face(frame, classifier)
    if rect is not None: frames.append(frame)
    if len(frames) == 25:
      print(talking_probability(frames, predictor, model, scaler))
      frames = []
    frame = cv2.resize(frame, (300,300))
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('0'): break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
