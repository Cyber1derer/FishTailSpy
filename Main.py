import cv2
import numpy as np
import os



#global useEqualize, blurSize, CannyThesh1,CannyThesh2
# Initialize default parameter values
useEqualize = True
blurSize = 3
win_name = "FinderFish"
CannyThresh1 = 120
CannyThresh2 = 240
frameCount = 1
ret = True

cap_start_msec = 21_000
cap_end_msec = 23_000
frame_end = (cap_end_msec - cap_start_msec) /20 


def on_trackbar(val):
    global  CannyThresh1,CannyThresh2, blurSize
    CannyThresh1 = cv2.getTrackbarPos('CannyTresh1', "Canny viewUp Video")
    CannyThresh2 = cv2.getTrackbarPos('CannyTresh2', "Canny viewUp Video")
    blurSize = cv2.getTrackbarPos('BlurGaus', "Canny viewUp Video")
# Create window and trackbars
cv2.namedWindow("Canny viewUp Video")
cv2.createTrackbar("CannyTresh1", "Canny viewUp Video" , CannyThresh1, 500, on_trackbar)
cv2.createTrackbar("CannyTresh2", "Canny viewUp Video" , CannyThresh2, 500, on_trackbar)
cv2.createTrackbar("BlurGaus", "Canny viewUp Video" , blurSize, 50, on_trackbar)

# Open video file
#cap = cv2.VideoCapture('D:/MyCodeProjects/FishTailSpy/Data/Fish/Fish1_3m.mp4')
cap = cv2.VideoCapture('D:/MyCodeProjects/FishTailSpy/Data/Fish/outFishFFMPEG2.mp4')#GOPR6626
#cap = cv2.VideoCapture('D:/MyCodeProjects/FishTailSpy/Data/Fish/FFMPEGm_Fish3.MP4')#GP016626
#cap = cv2.VideoCapture('D:/MyCodeProjects/FishTailSpy/Data/Fish/FFMPEGm_Fish4.MP4')#GP026626
#cap = cv2.VideoCapture('D:/MyCodeProjects/FishTailSpy/Data/Fish/FFMPEGm_Fish5.MP4')#GGP036626

cap.set(cv2.CAP_PROP_POS_MSEC , cap_start_msec)

# Load camera parameters
data = np.load("camera_params.npz")
mtx = data['mtx']
dist = data['dist']


while ret:
    ret, frame = cap.read()
    if not ret:
        print("Video can't open")
        break
    # Undistort frame
    undistortFrame = cv2.undistort(frame, mtx, dist)
    # Extract region of interest
    viewUp = undistortFrame[250:1400, 1050:1440].copy() # y1:y2 x1:x2
    viewUpSource = undistortFrame [250:1400, 1050:1440].copy() # y1:y2 x1:x2
    viewSource = undistortFrame.copy() # y1:y2 x1:x2

    # Apply Gaussian blur if blur size is valid
    if blurSize >= 3:
        if blurSize % 2 == 0 and blurSize !=0 :
            blurSize += 1
        viewUp = cv2.GaussianBlur(viewUp, (blurSize, blurSize), 0)
    
    # Convert to grayscale and equalize histogram
    viewUp = cv2.cvtColor(viewUp, cv2.COLOR_BGR2GRAY)
    if useEqualize:
        viewUp = cv2.equalizeHist(viewUp)
    
    #viewUp = cv2.morphologyEx(viewUp, cv2.MORPH_OPEN, (15,15))

    upEdges = cv2.Canny(viewUp, CannyThresh1,CannyThresh2,None,3,False)
    upPxEdges = np.argwhere(upEdges == 255)

    if upPxEdges.size == 0:
        print("Edges not found")
    else:
        upPxEdges = upPxEdges[:, [1, 0]]
        ellipse = cv2.fitEllipse(upPxEdges)
        (center, axes, angle) = ellipse
        [vx, vy, x, y] = cv2.fitLine(upPxEdges, cv2.DIST_L2, 0, 0.01, 0.01)
        k = vy / vx
        b = y - k * x

        # Вычисляем координаты концов линии для визуализации
        rows, cols = viewUpSource.shape[:2]
        left_x = 0
        right_x = cols - 1
        left_y = int(k * left_x + b)
        right_y = int(k * right_x + b)

        # Рисуем линию на изображении
        cv2.line(viewUpSource, (left_x, left_y), (right_x, right_y), (0, 255, 255), 2)

        cv2.circle(viewUpSource,(int(x), int(y)), 2,(0,0,255), -1)
        '''
        dirRect = cv2.minAreaRect(upPxEdges)
        (centerRect, size, orientation) = dirRect
        # Ориентация (угол поворота) прямоугольника
        orientation = orientation * 180.0 / np.pi
        major_axis_length_rect = max(size)

        x1r = int(centerRect[0] - major_axis_length_rect * np.cos(orientation))
        y1r = int(centerRect[1] - major_axis_length_rect * np.sin(orientation))
        x2r = int(centerRect[0] + major_axis_length_rect * np.cos(orientation))
        y2r = int(centerRect[1] + major_axis_length_rect * np.sin(orientation))
        # Вычислите координаты вершин прямоугольника
        box = cv2.boxPoints(dirRect)
        box = np.int0(box)

        # Нарисуйте ограничивающий прямоугольник
        cv2.polylines(viewUpSource, [box], True, (0, 0, 255), 2)
        '''
        # Преобразуйте углы из градусов в радианы
        angle = (angle * np.pi / 180.0) + 90

        # Вычислите концевые точки большой оси эллипса
        major_axis_len = max(axes)
        minor_axis_len = min(axes)
        x1 = int(center[0] - major_axis_len * np.cos(angle))
        y1 = int(center[1] - major_axis_len * np.sin(angle))
        x2 = int(center[0] + major_axis_len * np.cos(angle))
        y2 = int(center[1] + major_axis_len * np.sin(angle))

        # Нарисуйте большую ось эллипса
        cv2.line(viewUpSource, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.circle(viewUpSource,(int(center[0]), int(center[1])), 4,(0,0,255), -1)

       # cv2.line(viewUpSource, (x1r, y1r), (x2r, y2r), (255, 0, 0), 2)


        cv2.ellipse(viewUpSource,ellipse,(255,255,255),2)




    # Display images
    cv2.imshow('Canny viewUp Video', upEdges)
    cv2.imshow('viewUp Video', viewUp)
    cv2.imshow('Source Up', viewUpSource)

    #cv2.imwrite( "Data//UndistertImages//Undist_frame_" + str(frameCount) + ".png", undistortFrame )
    #cv2.imwrite( "Data//UndistertImages//Source_frame_" + str(frameCount) + ".png", frame )

    frameCount +=1

    # Reset video position after a certain frame count
    if frameCount > frame_end:
        cap.set(cv2.CAP_PROP_POS_MSEC , cap_start_msec)
        frameCount = 1
    #Exit loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


