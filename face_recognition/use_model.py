import sys
import cv2
from train_model import detect_face

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), 255, 2)
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, 255, 2)

def predict(test_img, face_recognizer, subjects = ['', 'chris', 'luke', 'Hesham', 'Amy']):
    img = test_img.copy()
    face, rect = detect_face(img)
    if face is not None:
      label= face_recognizer.predict(face)
      print(label)
      label = label[0] if label[1] < 100 else 0
      label_text = subjects[label]
      draw_rectangle(img, rect)
      draw_text(img, label_text, rect[0], rect[1]-5)
    return img

def main():
  face_recognizer = cv2.face.LBPHFaceRecognizer_create()
  face_recognizer.read('models/model.yml')
  # img = cv2.imread('test_images/'+sys.argv[1]+'.jpg')

  # img = predict(img, face_recognizer)
  cap = cv2.VideoCapture(0)
  while True:
    ret, frame = cap.read()
    if(not ret): break
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.equalizeHist(frame)
    frame = predict(frame, face_recognizer)
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('0'): break
  
  # cv2.waitKey(0)
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()