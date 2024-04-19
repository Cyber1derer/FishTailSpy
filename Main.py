import cv2
import numpy as np
import os

def on_trackbar(val):
    global  CannyThresh1,CannyThresh2, blurSize
    CannyThresh1 = cv2.getTrackbarPos('CannyTresh1', "Canny viewUp Video")
    CannyThresh2 = cv2.getTrackbarPos('CannyTresh2', "Canny viewUp Video")
    blurSize = cv2.getTrackbarPos('BlurGaus', "Canny viewUp Video")

#global useEqualize, blurSize, CannyThesh1,CannyThesh2
# Initialize default parameter values
useEqualize = True
blurSize = 3
win_name = "FinderFish"
CannyThresh1 = 120
CannyThresh2 = 240
frameCount = 1
ret = True

# Create window and trackbars
cv2.namedWindow("Canny viewUp Video")
cv2.createTrackbar("CannyTresh1", "Canny viewUp Video" , CannyThresh1, 500, on_trackbar)
cv2.createTrackbar("CannyTresh2", "Canny viewUp Video" , CannyThresh2, 500, on_trackbar)
cv2.createTrackbar("BlurGaus", "Canny viewUp Video" , blurSize, 50, on_trackbar)

# Open video file
cap = cv2.VideoCapture('D:/MyCodeProjects/FishTailSpy/Data/Fish/Fish1_3m.mp4')
cap.set(cv2.CAP_PROP_POS_MSEC , 36000)

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
    viewUp = undistortFrame[450:1250, 1150:1440].copy() # y1:y2 x1:x2
    viewUpSource = undistortFrame [450:1250, 1150:1440].copy() # y1:y2 x1:x2

    # Apply Gaussian blur if blur size is valid
    if blurSize >= 3:
        if blurSize % 2 == 0 and blurSize !=0 :
            blurSize += 1
        viewUp = cv2.GaussianBlur(viewUp, (blurSize, blurSize), 0)
    
    # Convert to grayscale and equalize histogram
    viewUp = cv2.cvtColor(viewUp, cv2.COLOR_BGR2GRAY)
    if useEqualize:
        viewUp = cv2.equalizeHist(viewUp)
    

    upEdges = cv2.Canny(viewUp, CannyThresh1,CannyThresh2,None,3,False)
    upPxEdges = np.argwhere(upEdges == 255)

    if upPxEdges.size == 0:
        print("Edges not found")
    else:
        upPxEdges = upPxEdges[:, [1, 0]]
        ellipse = cv2.fitEllipse(upPxEdges)
        cv2.ellipse(viewUpSource,ellipse,(255,255,255),2)

    # Display images
    cv2.imshow('Canny viewUp Video', upEdges)
    cv2.imshow('viewUp Video', viewUp)
    cv2.imshow('Source Up', viewUpSource)

    #cv2.imwrite( "Data//UndistertImages//Undist_frame_" + str(frameCount) + ".png", undistortFrame )
    #cv2.imwrite( "Data//UndistertImages//Source_frame_" + str(frameCount) + ".png", frame )

    frameCount +=1

    # Reset video position after a certain frame count
    if frameCount > 400:
        cap.set(cv2.CAP_PROP_POS_MSEC , 36000)
        frameCount = 1
    #Exit loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


