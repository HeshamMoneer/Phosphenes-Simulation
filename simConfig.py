import cv2
import dlib
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

from gaussArray import gaussArr
from enums import (Simode, Modes)
import caricaturing.caric_config as cc

'''
Simulation configuration values that could be manipulated

dim: number of phosphenes = dim x dim
dimWin: the window has a size = dimWin x dimWin
mLevels: number of modulation levels (no. colors)
simode: simulation mode
faceMode: enhancement mode
ur: update rate
'''

def init():
  global faceIndex, dim, dimWin, mLevels, simode, facesMode, cache
  faceIndex = 0
  dim = 32
  dimWin = 640
  mLevels = 16
  simode = Simode.BSM
  facesMode = Modes.SFR_ROI_M
  cache = {}

  global squareSide, radius, blurKernel, gArr
  squareSide = dimWin//dim
  radius = 0
  if simode == Simode.ACM or simode == Simode.ASM:
    radius = int(squareSide * 0.7)

  elif simode == Simode.BCM:
    radius = int(squareSide * 0.25)

  elif simode == Simode.BSM:
    radius = int(squareSide * 0.3)

  blurKernel = 0
  if simode == Simode.ACM or simode == Simode.ASM:
      blurKernel = 3

  elif simode == Simode.BCM or simode == Simode.BSM:
      blurKernel = radius * 2

  
  if simode == Simode.ACM or simode == Simode.ASM: gArr = gaussArr(radius)
  else: gArr = None

  global ur, classifiers
  ur = 5

  faces_classifier = cv2.CascadeClassifier('classifiers/cc.xml')
  eyes_classifier, predictor = None, None
  if facesMode == Modes.DETECT_FACES_WITH_EYES:
    eyes_classifier = cv2.CascadeClassifier('classifiers/ecc.xml')
  elif facesMode in [Modes.DETECT_FACE_FEATURES, Modes.SFR_ROI_M, Modes.SFR_ROI_HE, Modes.SFR_ROI_M_TD]:
    predictor = dlib.shape_predictor('classifiers/shape_predictor_68_face_landmarks.dat')
  elif facesMode == Modes.VJFR_ROI_C:
    cc.init(faces_classifier, predictor)
    ur = 1

  classifiers = [faces_classifier, eyes_classifier, predictor]

  global bboxes, counter, windowName
  bboxes = []
  counter = 0
  windowName = 'Phosphenated ' + simode.name + ' & ' + facesMode.name

  global talkingAcc, talkingModel, talkingScaler
  talkingAcc = None
  talkingModel = None
  talkingScaler = None
  if facesMode == Modes.SFR_ROI_M_TD:
    talkingAcc = []
    talkingModel = load_model('./talking_detection/model.h5')
    talkingScaler = MinMaxScaler()
