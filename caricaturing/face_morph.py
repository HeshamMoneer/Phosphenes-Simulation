import numpy as np
import cv2

## Uncomment to show avg face on top of your face
# face_cascade = cv2.CascadeClassifier('utils/cc.xml')
# avg = cv2.imread('images/aligned_images/avgM.png')
# avg = cv2.cvtColor(avg, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(avg, minNeighbors = 8)
# (x, y, w, h) = faces[0]
# avg = avg[y:y+w, x:x+h]
# avg = cv2.resize(avg, (160, 160))

def apply_affine_transform(src, srcTri, dstTri, size) :
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def morph_triangle(faceImg, img, t1, t2, t) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    # r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    # t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        # t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    # Get mask by filling triangle in rectangle
    mask = np.zeros((r[3], r[2]), dtype = np.uint8)
    cv2.fillConvexPoly(mask, np.int32(tRect), 1, 16, 0)

    # Apply warpImage to small rectangular patches
    faceImgRect = faceImg[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    # avgRect = avg[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    
    size = (r[2], r[3])
    try: imgRect = apply_affine_transform(faceImgRect, t1Rect, tRect, size)
    except: return

    # to morph with color as well uncomment the one below
    # imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    try: img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask
    except: pass

def generate_morph_frame(faceImg,points1,points2,tri_list, alpha):
    points = []
    for i in range(0, len(points1)):
        x1, x2 = points1[i][0], points2[i][0]
        x3 = 2*x1 - x2
        x = (1 - alpha) * x1 + alpha * x3
        y1, y2 = points1[i][1], points2[i][1]
        y3 = 2*y1 - y2
        y = (1 - alpha) * y1 + alpha * y3
        points.append((x,y))
    
    morphed_frame = np.zeros(faceImg.shape, dtype = np.uint8)

    for i in range(len(tri_list)):    
        x = int(tri_list[i][0])
        y = int(tri_list[i][1])
        z = int(tri_list[i][2])
        
        t1 = [mb(points1[x]), mb(points1[y]), mb(points1[z])]
        t2 = [mb(points2[x]), mb(points2[y]), mb(points2[z])]
        t =  [mb(points[x]), mb(points[y]), mb(points[z])]

        # Morph one triangle at a time.
        morph_triangle(faceImg, morphed_frame, t1, t2, t)
        
    return morphed_frame

def mb(t): #maintain bounds
    return min(max(t[0],0),159), min(max(t[1],0),159)