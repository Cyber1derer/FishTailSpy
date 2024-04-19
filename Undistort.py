import cv2
import numpy as np
import os

def on_trackbar(val):
    global  CannyThesh1,CannyThesh2
    CannyThesh1 = cv2.getTrackbarPos('CannyTresh1', "Canny viewUp Video")
    CannyThesh2 = cv2.getTrackbarPos('CannyTresh2', "Canny viewUp Video")


#global useEqualize, blurSize, CannyThesh1,CannyThesh2
useEqualize = False
blurSize = 7
win_name = "FinderFish"
CannyThesh1 = 100
CannyThesh2 = 200


cv2.namedWindow("Canny viewUp Video")
cv2.createTrackbar("CannyTresh1", "Canny viewUp Video" , 0, 500, on_trackbar)
cv2.createTrackbar("CannyTresh2", "Canny viewUp Video" , 0, 500, on_trackbar)


cap = cv2.VideoCapture('D:\MyCodeProjects\FishTailSpy\Data\Fish\Fish1_3m.mp4')

data = np.load("camera_params.npz")

mtx = data['mtx']
dist = data['dist']
frameCount = 1
ret = True
cap.set(cv2.CAP_PROP_POS_MSEC , 34000)




while ret:
    ret, frame = cap.read()
    undistortFrame = cv2.undistort(frame, mtx, dist)

    viewUp = undistortFrame[306:1250, 1150:1440].copy() # y1:y2 x1:x2
    viewUpSource = undistortFrame [306:1250, 1150:1440].copy() # y1:y2 x1:x2


    if blurSize >= 3:
        viewUp = cv2.GaussianBlur(viewUp, (blurSize, blurSize), 0)

    viewUp = cv2.cvtColor(viewUp, cv2.COLOR_BGR2GRAY)
    viewUp = cv2.equalizeHist(viewUp)
    

    upEdges = cv2.Canny(viewUp, CannyThesh1,CannyThesh2)
    upPxEdges = np.argwhere(upEdges == 255)
    upPxEdges = upPxEdges[:, [1, 0]]

    #white_pixels = np.array(white_pixels, dtype=np.float32)
    # Отметка границ на изображении
    #for y, x in upPxEdges:
        #cv2.circle(viewUpSource, (x, y), 1, (0, 0, 255), -1)  # Рисуем красные точки
    
    ellipse = cv2.fitEllipse(upPxEdges)
    cv2.ellipse(viewUpSource,ellipse,(255,255,255),2)


    cv2.imshow('Canny viewUp Video', upEdges)
    cv2.imshow('viewUp Video', viewUp)
    cv2.imshow('Source Up', viewUpSource)

    #cv2.imwrite( "Data//UndistertImages//Undist_frame_" + str(frameCount) + ".png", undistortFrame )
    #cv2.imwrite( "Data//UndistertImages//Source_frame_" + str(frameCount) + ".png", frame )

    frameCount +=1


    #Circle video
    if frameCount > 400:
        cap.set(cv2.CAP_PROP_POS_MSEC , 34000)
        frameCount = 1
    #Exit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


