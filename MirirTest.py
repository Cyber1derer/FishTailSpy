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


x1_cut = 525
#cap_start_msec = 323750
#cap_end_msec = 324725



#cap_start_msec = (5*60 + 18) * 1000
#cap_end_msec = (5*60 + 22) * 1000

#=======================================================================================
#                           GOPR6626        
#======================================================================================
#5:23-5:24
#cap_start_msec = 323750
#cap_end_msec = 324725
#------------------------------
#cap_start_msec = 22*1000
#cap_end_msec = 24*1000

#cap_start_msec = (5*60 + 31) * 1000
#cap_end_msec = (5*60 + 32) * 1000

#Canny
#cap_start_msec = (5*60 + 54) * 1000
#cap_end_msec = (5*60 + 55) * 1000

#Bad
#cap_start_msec = (6*60 + 40) * 1000
#cap_end_msec = (6*60 + 41) * 1000


#=======================================================================================
#                           GP016626        
#======================================================================================
#Dark
#cap_start_msec = (1*60 + 36) * 1000
#cap_end_msec = (1*60 + 42) * 1000
#UpDirection
cap_start_msec = (5*60 + 47) * 1000
cap_end_msec = (5*60 + 49.2) * 1000

#=======================================================================================
#                           GP026626        
#======================================================================================
#UpDirection Done
#cap_start_msec = (5*60 + 41.7) * 1000
#cap_end_msec = (5*60 + 43.7) * 1000

#Wall Bad
#cap_start_msec = (3*60 + 49) * 1000
#cap_end_msec = (3*60 + 52) * 1000

#NeedCut Up dir
#cap_start_msec = (4*60 + 50.5) * 1000
#cap_end_msec = (4*60 + 54) * 1000

#cap_start_msec = (5*60 + 18) * 1000
#cap_end_msec = (5*60 + 22) * 1000






frameCount = 1
useEqualize = True
blurSize = 3
win_name = "FinderFish"

frame_end = (cap_end_msec - cap_start_msec) /20 

# Open video file
#cap = cv2.VideoCapture('D:/MyCodeProjects/FishTailSpy/Data/Fish/Fish1_3m.mp4')
#cap = cv2.VideoCapture('D:/MyCodeProjects/FishTailSpy/Data/Fish/outFishFFMPEG2.mp4')#GOPR6626
cap = cv2.VideoCapture('D:/MyCodeProjects/FishTailSpy/Data/Fish/FFMPEGm_Fish3.MP4')#GP016626
#cap = cv2.VideoCapture('D:/MyCodeProjects/FishTailSpy/Data/Fish/FFMPEGm_Fish4.MP4')#GP026626
#cap = cv2.VideoCapture('D:/MyCodeProjects/FishTailSpy/Data/Fish/FFMPEGm_Fish5.MP4')#GGP036626

cap.set(cv2.CAP_PROP_POS_MSEC , cap_start_msec)
CannyThresh1 = 120
CannyThresh2 = 240
# Load camera parameters
data = np.load("camera_params.npz")
mtx = data['mtx']
dist = data['dist']
fishHeight = []

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
    viewMirror = undistortFrame[420:1183, x1_cut:870].copy() # y1:y2 x1:x2
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
    #cv2.imshow("Hist", viewMirror)
    mirrorEdges = cv2.Canny(viewMirror, CannyThresh1,CannyThresh2,None,3,False)
    #cv2.imshow("Canny", mirrorEdges)
    mirrorPxEdges = np.argwhere(mirrorEdges == 255)
    if mirrorEdges.size == 0:
        print("Mirror edges not found")
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
        pxDistans = (cx + x1_cut) - 433
        RealHeight = (0.18/493) * pxDistans + 0.08527
        fishHeight.append(RealHeight)
        frameCount +=1

        cv2.circle(filledImage, (int(cx), int(cy)), 1, (0,0,255),  )
        #cv2.imshow('Filled Contours', fillFish)
        cv2.imshow('Filled Contours2', filledImage)
        cv2.waitKey(1)
    if frameCount > frame_end:
        np.savez('dataMirror.npz', fishHeight=fishHeight)
        ic (fishHeight)
        break
    if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()