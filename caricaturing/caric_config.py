import cv2
import dlib

from .delaunay_triangulation import make_delaunay
from .face_landmark_detection import generate_face_land_marks

'''
Caricaturing configuration values that could be manipulated

avg(Path): to choose which path is to the average face
alpha: caricaturing factor; typically 0 <= alpha <= 1
'''

def init(fc = None, p = None):
  global face_cascade, predictor, avg, avgP, avgS, tri, alpha
  if fc: face_cascade = fc
  else: face_cascade = cv2.CascadeClassifier('caricaturing/utils/cc.xml')

  if p: predictor = p
  else: predictor = dlib.shape_predictor('caricaturing/utils/shape_predictor_68_face_landmarks.dat')

  avg = cv2.imread('caricaturing/images/aligned_images/avg.png')
  avg, _ = detect_face(avg)
  avg = cv2.resize(avg, (160, 160))

  avgP = generate_face_land_marks(avg, predictor) # average landmarks
  avgS = avg.shape # average shape

  tri = None #make_delaunay(avgS, avgP)
  alpha = 0.6 
  '''
  We selected 60% as the highest caricature level because Irons et al. (2014) found caricature
  advantages up to this level; note that higher caricature strengths
  tend to produce morphing artifacts in the image, and eventually
  will make the faces distorted beyond the range perceived as nor-
  mal for real faces
  '''

def detect_face(img):
  if len(img.shape) > 2 :
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(img, minNeighbors = 8)
  if (len(faces) == 0):
    return img, None
  (x, y, w, h) = faces[0]
  return img[y:y+w, x:x+h], faces[0]
