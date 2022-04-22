from face_landmark_detection import (generate_face_correspondences, generate_face_land_marks)
from delaunay_triangulation import make_delaunay
from face_morph import generate_morph_frame

import cv2
import dlib

def detect_face(img, face_cascade):
    if len(img.shape) > 2 :
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, minNeighbors = 8)
    if (len(faces) == 0):
      return img, None
    (x, y, w, h) = faces[0]
    return img[y:y+w, x:x+h], faces[0]

def doMorphing(faceImg, avgP, avgS, predictor, tri, alpha = 1):
	[faceImg, faceP] = generate_face_correspondences(faceImg, avgS, predictor)
	# for point in faceP:
	# 	cv2.circle(faceImg, point, 1, 255,1)
	# cv2.rectangle(faceImg, (2,2), (faceImg.shape[1] -2, faceImg.shape[0] -2), 255, 2)
	# return faceImg
	# for t in tri:
	# 	cv2.line(faceImg, faceP[t[0]],faceP[t[1]], 255, 2)
	# 	cv2.line(faceImg, faceP[t[2]],faceP[t[1]], 255, 2)
	# 	cv2.line(faceImg, faceP[t[0]],faceP[t[2]], 255, 2)
	return generate_morph_frame(faceImg, faceP, avgP, tri, alpha)

def main():
	face_cascade = cv2.CascadeClassifier('utils/cc.xml')
	predictor = dlib.shape_predictor('utils/shape_predictor_68_face_landmarks.dat')
	avg = cv2.imread('images/aligned_images/avgM.png')
	avg, _ = detect_face(avg, face_cascade)
	avg = cv2.resize(avg, (160, 160))
	avgP = generate_face_land_marks(avg, predictor) # average landmarks
	avgS = avg.shape # average shape
	tri = make_delaunay(avgS, avgP)

	cap = cv2.VideoCapture(0)

	# result = cv2.VideoWriter('caricaturing.avi', 
  #                        cv2.VideoWriter_fourcc(*'MJPG'),
  #                        20, (int(cap.get(3)), int(cap.get(4))), 0)
	while True:
		ret, frame = cap.read()
		if(not ret): break
		face, rect = detect_face(frame, face_cascade)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		if rect is not None:
			face = doMorphing(face, avgP, avgS, predictor, tri)
			# face = cv2.equalizeHist(face)
			# frame = cv2.equalizeHist(frame)
			(x,y,w,h) = rect
			face = cv2.resize(face, (w,h))
			subframe = frame[y:y+h, x:x+w]
			face[face==0] = subframe[face==0]
			frame[y:y+h, x:x+w] = face
			# frame = face
		# result.write(frame)
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('0'): break

	cap.release()
	# result.release()
	cv2.destroyAllWindows()

	# parser = argparse.ArgumentParser()
	# parser.add_argument("--img1", required=True, help="The First Image")
	# parser.add_argument("--img2", required=True, help="The Second Image")
	# args = parser.parse_args()

	# image1 = cv2.imread(args.img1)
	# image2 = cv2.imread(args.img2)

	# image1 = cv2.resize(image1, (300,300))
	# image2 = cv2.resize(image2, (300,300))

	# image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
	# image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

	# for i in range(11):
	# 	img = doMorphing(image1, image2, i * 0.1)
	# 	cv2.imshow('', img)
	# 	if cv2.waitKey(1) & 0xFF == ord('0'):
	# 		break
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

if __name__ == "__main__":
	main()