import dlib
import numpy as np
import cv2

class NoFaceFound(Exception):
   """Raised when there is no face found"""
   pass

def generate_face_correspondences(faceImg, avgS, predictor):
    faceImg = cv2.resize(faceImg, (avgS[1], avgS[0]))
    faceP = generate_face_land_marks(faceImg, predictor)
    return [faceImg,faceP]


def generate_face_land_marks(faceImg, predictor):
    result = []
    faceRect = dlib.rectangle(0,0,faceImg.shape[1], faceImg.shape[0])
    shape = predictor(faceImg, faceRect)
    for i in range(0,68):
        x = shape.part(i).x
        y = shape.part(i).y
        result.append((x, y))
    
    result.append((1,1))
    result.append((faceImg.shape[1]-1,1))
    result.append(((faceImg.shape[1]-1)//2,1))
    result.append((1,faceImg.shape[0]-1))
    result.append((1,(faceImg.shape[0]-1)//2))
    result.append(((faceImg.shape[1]-1)//2,faceImg.shape[0]-1))
    result.append((faceImg.shape[1]-1,faceImg.shape[0]-1))
    result.append(((faceImg.shape[1]-1),(faceImg.shape[0]-1)//2))
    return np.array(result)