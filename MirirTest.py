import cv2
import numpy as np
import os
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from icecream import ic
import math
from skimage.morphology import skeletonize
from skimage.util import invert
import matplotlib.pyplot as plt



cap_start_msec = 323750
cap_end_msec = 324725
frameCount = 1
useEqualize = True
blurSize = 3
win_name = "FinderFish"

frame_end = (cap_end_msec - cap_start_msec) /20 

cap = cv2.VideoCapture('D:/MyCodeProjects/FishTailSpy/Data/Fish/outFishFFMPEG2.mp4')#GOPR6626

cap.set(cv2.CAP_PROP_POS_MSEC , cap_start_msec)
CannyThresh1 = 120
CannyThresh2 = 240
# Load camera parameters
data = np.load("camera_params.npz")
mtx = data['mtx']
dist = data['dist']


while True:
    ret, frame = cap.read()
    if not ret:
        print("Video can't open")
        break
    # Undistort frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    undistortFrame = cv2.undistort(frame, mtx, dist)
    # Extract region of interest
    #cv2.imwrite("test.png", undistortFrame)
    viewMirror = undistortFrame[420:1183, 525:870].copy() # y1:y2 x1:x2
    #cv2.imshow("sourseMirror", viewMirror)
    #cv2.waitKey(0)
    # Apply Gaussian blur if blur size is valid
    if blurSize >= 3:
        if blurSize % 2 == 0 and blurSize !=0 :
            blurSize += 1
        viewMirror = cv2.GaussianBlur(viewMirror, (blurSize, blurSize), 0)
    
    # Convert to grayscale and equalize histogram
    #viewMirror = cv2.cvtColor(viewMirror, cv2.COLOR_BGR2GRAY)
    if useEqualize:
        viewMirror = cv2.equalizeHist(viewMirror)
    cv2.imshow("Hist", viewMirror)
    mirrorEdges = cv2.Canny(viewMirror, CannyThresh1,CannyThresh2,None,3,False)
    cv2.imshow("Canny", mirrorEdges)
    mirrorPxEdges = np.argwhere(mirrorEdges == 255)
    if mirrorEdges.size == 0:
        print("Edges not found")
        break
    else:
        # Создаем копию для заполнения
        fillMirrorEdges = mirrorEdges.copy()
        # Структурный элемент для морфологических операций
        kernel = np.ones((5, 5), np.uint8)
        # Применяем морфологическое замыкание
        closedEdges = cv2.morphologyEx(fillMirrorEdges, cv2.MORPH_CLOSE, kernel)

        # Находим контуры на замкнутом изображении
        contours, _ = cv2.findContours(closedEdges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        max_contour2_flat = np.squeeze(max_contour)
        max_contour = [max_contour]
        # Создаем пустое изображение для заполнения
        filledImage = np.zeros_like(fillMirrorEdges)
        # Заполняем найденные контуры белым цветом
        cv2.fillPoly(filledImage, max_contour, 255)
        fillFish = filledImage

        # Вычислить моменты изображения
        moments = cv2.moments(fillFish)

        # Вычислить координаты центра масс (центроида)
        cx = (moments['m10'] / moments['m00'])
        cy = (moments['m01'] / moments['m00'])

        print(f"Центроид: ({cx}, {cy})")
        cv2.circle(filledImage, (int(cx), int(cy)), 1, (0,0,255),  )
        #cv2.imshow('Filled Contours', fillFish)
        cv2.imshow('Filled Contours2', filledImage)
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()