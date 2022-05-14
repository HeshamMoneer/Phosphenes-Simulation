import cv2

from . import caric_config as cc
from .face_landmark_detection import generate_face_correspondences
from .face_morph import generate_morph_frame

def doMorphing(faceImg):
	[faceImg, faceP] = generate_face_correspondences(faceImg, cc.avgS, cc.predictor)
	cc.tri = make_delaunay(faceImg.shape, faceP)
	return generate_morph_frame(faceImg, faceP, cc.avgP, cc.tri, cc.alpha)

def caric(frame, faceOnly = True):
	if not faceOnly: 
		frame, rect = cc.detect_face(frame)
	else:	
		rect = (0,0,frame.shape[1], frame.shape[0])

	if rect is not None:
		caricature = doMorphing(frame)
		h, w = frame.shape
		caricature = cv2.resize(caricature, (w,h))
		caricature[caricature==0] = frame[caricature==0]
		frame = caricature
	return frame

def main():
	# change paths in init to run main properly
	cc.init()
	cap = cv2.VideoCapture(0)
	while True:
		ret, frame = cap.read()
		if(not ret): break
		frame = caric(frame, False)
		cv2.imshow('caricature', frame)
		if cv2.waitKey(1) & 0xFF == ord('0'): break
	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()