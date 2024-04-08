import cv2
import numpy as np
import os


cap = cv2.VideoCapture('Data/Fish/Fish1.MP4')

data = np.load("camera_params.npz")

mtx = data['mtx']
dist = data['dist']
frameN = 1
ret = True
#h,  w = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
#mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)

while ret:
    ret, frame = cap.read()
    #corrected = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    corrected = cv2.undistort(frame, mtx, dist)
    cv2.imshow('Corrected Video', corrected)
    cv2.imwrite( "Data//UndistertImages//Undist_frame_" + str(frameN) + ".png", corrected )
    cv2.imwrite( "Data//UndistertImages//Source_frame_" + str(frameN) + ".png", frame )

    frameN +=1
    cv2.waitKey(500)


'''
folder_path = r'Data\CalibData'  # Путь к папке с изображениями
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']  # Список допустимых расширений файлов


images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if any(f.endswith(ext) for ext in image_extensions)]


for fname in images:
'''


