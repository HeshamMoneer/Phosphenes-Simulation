import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

def createModel():
  model = Sequential()

  model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
  model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(1024, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(7, activation='softmax'))

  return model

def detectEmo(faceGray, model):
  cropped_img = np.expand_dims(np.expand_dims(cv2.resize(faceGray, (48, 48)), -1), 0)
  prediction = model.predict(cropped_img)[0]
  max1, max2 = 0, 0
  for i in range(len(prediction)):
    if(prediction[i] > prediction[max1]):
      max2 = max1
      max1 = i
  
  if max1 != 4: return max1
  if max2 > 0: return max2
  return max1


def main():
  model = createModel()
  facecasc = cv2.CascadeClassifier('../classifiers/cc.xml')
  model.load_weights('model.h5')
  emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
  cap = cv2.VideoCapture(0)
  while True:
      ret, frame = cap.read()
      if not ret: break
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      faces = facecasc.detectMultiScale(gray)
      for (x, y, w, h) in faces:
          cv2.rectangle(frame, (x, y), (x+w, y+h), 200, 1)
          faceGray = gray[y:y + h, x:x + w]
          index = detectEmo(faceGray, model)
          cv2.putText(frame, emotion_dict[index], (x, y-5), cv2.FONT_HERSHEY_PLAIN, 1.5, 255, 2)

      cv2.imshow('Video', frame)
      if cv2.waitKey(1) & 0xFF == ord('0'):
          break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()

