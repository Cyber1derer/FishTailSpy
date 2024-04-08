import cv2
import numpy as np
import os

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#Take all files from folder
folder_path = r'Data\CalibData'  # Путь к папке с изображениями
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']  # Список допустимых расширений файлов

images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if any(f.endswith(ext) for ext in image_extensions)]



for fname in images:
    img = cv2.imread(fname)

    # Test image yet
    if img is None:
        print(f"Error: Could not open or find the image {fname}")
        exit()
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print("Camera matrix : \n", mtx)
print("Distortion coefficients : \n", dist)

#Save to file
np.savez('camera_params.npz', mtx=mtx, dist=dist)
