import cv2
import os
import numpy as np

def detect_face(img):
    if len(img.shape) > 2 :
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    face_cascade = cv2.CascadeClassifier('../classifiers/cc.xml')
    faces = face_cascade.detectMultiScale(gray)
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path='training_images'):
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
    
    #let's go through each directory and read images within it
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue
        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            # cv2.imshow("Training on image...", image)
            # cv2.waitKey(100)
            face, rect = detect_face(image)
            # cv2.imshow(image_name, face)
            # cv2.waitKey(500)
            # cv2.destroyAllWindows()
            if face is not None:
                faces.append(face)
                labels.append(label)
    
    return faces, labels

  
def main():
  faces, labels = prepare_training_data()
  face_recognizer = cv2.face.LBPHFaceRecognizer_create()
  face_recognizer.train(faces, np.array(labels))
  face_recognizer.write('models/model.yml')

if __name__ == '__main__':
  main()