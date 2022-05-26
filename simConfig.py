import cv2
import dlib
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

from gaussArray import gaussArr
from enums import (Simode, Modes)
import caricaturing.caric_config as cc
from emotion_recognition.__init__ import createModel

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
  simode = Simode.ASM
  facesMode = Modes.VJFR_ROI_C
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

  global ur, counter, skip_enhancements_flag, zoom_counter, classifiers
  counter = 0
  ur = 5
  skip_enhancements_flag = False
  zoom_counter = 10

  faces_classifier = cv2.CascadeClassifier('classifiers/cc.xml')
  eyes_classifier, predictor = None, None
  if facesMode == Modes.DETECT_FACES_WITH_EYES:
    eyes_classifier = cv2.CascadeClassifier('classifiers/ecc.xml')
  elif facesMode in [Modes.DETECT_FACE_FEATURES, Modes.SFR_ROI_M, Modes.SFR_ROI_HE, Modes.SFR_ROI_M_TD, Modes.SFR_ROI_M_ER, Modes.VJFR_ROI_M_TD, Modes.VJFR_ROI_M_ER, Modes.VJFR_ROI_HE]:
    predictor = dlib.shape_predictor('classifiers/shape_predictor_68_face_landmarks.dat')
  elif facesMode == Modes.VJFR_ROI_C:
    cc.init(faces_classifier, predictor)
    ur = 1

  classifiers = [faces_classifier, eyes_classifier, predictor]

  global bboxes, windowName
  bboxes = []
  windowName = 'Phosphenated ' + simode.name + ' & ' + facesMode.name

  global talkingAcc, talkingModel, talkingScaler
  talkingAcc = None
  talkingModel = None
  talkingScaler = None
  if facesMode == Modes.SFR_ROI_M_TD or facesMode == Modes.VJFR_ROI_M_TD:
    talkingAcc = []
    talkingModel = load_model('./talking_detection/model.h5')
    talkingScaler = MinMaxScaler()

  global emotionsModel, emotion_dict, emotionIndex
  emotionsModel = None
  emotion_dict = None
  emotionIndex = 4
  if facesMode == Modes.SFR_ROI_M_ER or facesMode == Modes.VJFR_ROI_M_ER:
    emotionsModel = createModel()
    emotionsModel.load_weights('./emotion_recognition/model.h5')
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
