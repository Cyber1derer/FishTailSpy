import cv2

import numpy as np
import os

import math

from icecream import ic




# Function to calculate Euclidean distance

def euclidean_distance(x1, y1, x2, y2):

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


chessUp = cv2.VideoCapture(r'D:\MyCodeProjects\FishTailSpy\Data\GOPR6629_1714036322353.MP4')

chessDown = cv2.VideoCapture(r'D:\MyCodeProjects\FishTailSpy\Data\GOPR6629_1714036520209.MP4')



# Load camera parameters

data = np.load("camera_params.npz")

mtx = data['mtx']

dist = data['dist']



total_pixel_distance = 0

total_metric_distance = 0

pair_count = 0

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 80, 0.001)



# Размер одной клетки шахматной доски в метрах

cell_size = 0.0085 # 8,5 мм 


imgpointsUp = []  # 2D точки в изображении

objpointsUp = []  # 3D точки в реальном мире


imgpointsDown = []  # 2D точки в изображении

objpointsDown = []  # 3D точки в реальном мире

def getPxSize(cap):

    while True:

        ret, frame = cap.read()

        if not ret:

            print("Video can't open")

            break

        # Undistort frame

        undistortFrame = cv2.undistort(frame, mtx, dist)

        undistortFrame =cv2.cvtColor(undistortFrame, cv2.COLOR_BGR2GRAY)

        undistortFrame = cv2.equalizeHist(undistortFrame)

        print( "Shape: ", undistortFrame.shape[::-1] )

        cv2.imshow("Source", undistortFrame)

        cv2.waitKey(10)

        #GoodChessUp = cv2.VideoCapture(r'D:\MyCodeProjects\FishTailSpy\Data\GOPR6629_1714036322353.MP4')


        # Отобразите изображение с углами

        #chessRet, corners = cv2.findChessboardCornersSB(undistortFrame, chessboard_size, flags=cv2.CALIB_CB_EXHAUSTIVE)

        chessRet, corners = cv2.findChessboardCorners(undistortFrame, chessboard_size, None)

        #corners2 = corners


        if chessRet:

            objpointsUp.append(objpUp)

            # Повышаем точность найденных углов

            corners2 = cv2.cornerSubPix(undistortFrame, corners, (11, 11), (-1, -1), criteria)

            # Нарисуйте углы на изображении (необязательно)

            #cv2.imwrite("testChess.png", undistortFrame)

            imgpointsUp.append(corners2)


            cv2.drawChessboardCorners(undistortFrame, chessboard_size, corners2, ret)

            cv2.imshow("Undist", undistortFrame)

            #print( "Cap frame count: " , chessUp.get(cv2.CAP_PROP_POS_FRAMES) )



            # Преобразуйте пиксельные координаты в метрические координаты

            # corners.reshape(-1, 2) упрощает работу с углами, т.к. делает их формата (кол-во_углов, 2)

            corners2 = corners2.squeeze()  # Убираем лишние измерения в массиве

            #ic(corners)

            pixel_coords = corners2.reshape(-1, 2)

            metric_coords = pixel_coords * cell_size

            #ic(pixel_coords)

            # Отобразите координаты


            for i, (pixel, metric) in enumerate(zip(pixel_coords, metric_coords)):

                px, py = pixel

                mx, my = metric

                #print(f"Corner {i+1}: Pixel Coordinates ({px:.2f}, {py:.2f}), Metric Coordinates ({mx:.2f}, {my:.2f})")




            corners2 = corners2.squeeze()  # Убираем лишние измерения в массиве

            '''

            distancesPx = []

            for i in range(chessboard_size[1]):  # цикла по вертикали (rows)

                for j in range(chessboard_size[0] - 1):  # цикла по горизонтали (columns)

                    p1 = corners[i * chessboard_size[0] + j][0]

                    p2 = corners[i * chessboard_size[0] + (j + 1)][0]

                    distancesPx.append(np.linalg.norm(p1 - p2))


            for i in range(chessboard_size[1] - 1):

                for j in range(chessboard_size[0]):

                    p1 = corners[i * chessboard_size[0] + j][0]

                    p2 = corners[(i + 1) * chessboard_size[0] + j][0]

                    distancesPx.append(np.linalg.norm(p1 - p2))


            ic(distancesPx)

            mean_distance = np.mean(distancesPx)

            print("Среднее расстояние между всеми соседними точками: {:.2f} пикселей".format(mean_distance))


            '''

            distancePxArr = []


            # Найдем расстояние между первой и второй точками

            for i in range(4):

                point1 = corners2[i]

                point2 = corners2[i+1]

                # Вычислим расстояние Евклидовой метрикой

                distancePx = np.linalg.norm(point1 - point2)

                distancePxArr.append(distancePx)
            

            for i in range(5, 9):

                point1 = corners2[i]

                point2 = corners2[i+1]

                # Вычислим расстояние Евклидовой метрикой

                distancePx = np.linalg.norm(point1 - point2)

                distancePxArr.append(distancePx)


            for i in range(10, 14):

                point1 = corners2[i]

                point2 = corners2[i+1]

                # Вычислим расстояние Евклидовой метрикой

                distancePx = np.linalg.norm(point1 - point2)

                distancePxArr.append(distancePx)



            for i in range(15, 19):

                point1 = corners2[i]

                point2 = corners2[i+1]

                # Вычислим расстояние Евклидовой метрикой

                distancePx = np.linalg.norm(point1 - point2)

                distancePxArr.append(distancePx)



            mean_distance = np.mean(distancePxArr)

            print("Среднее расстояние между всеми соседними точками: {:.2f} пикселей".format(mean_distance))


            PxSizeMetr = cell_size/mean_distance

            print("Размекр пикселя в метрах для верхнего положения", PxSizeMetr)


            ic(PxSizeMetr)
            

            cv2.waitKey(0)

            cv2.destroyAllWindows()

            break

        else:

            print("Шахматную доску не удалось найти на изображении.")

    return PxSizeMetr



def getPxSize2(cap):

    while True:

        ret, frame = cap.read()

        if not ret:

            print("Video can't open")

            break

        # Undistort frame

        undistortFrame = cv2.undistort(frame, mtx, dist)

        undistortFrame =cv2.cvtColor(undistortFrame, cv2.COLOR_BGR2GRAY)

        undistortFrame = cv2.equalizeHist(undistortFrame)


        undistortFrameCut = undistortFrame[410:742, 1145:1415]

        cv2.imshow("SourceCut", undistortFrameCut)

        #GoodChessUp = cv2.VideoCapture(r'D:\MyCodeProjects\FishTailSpy\Data\GOPR6629_1714036322353.MP4')


        # Отобразите изображение с углами

        #chessRet, corners = cv2.findChessboardCornersSB(undistortFrame, chessboard_size, flags=cv2.CALIB_CB_EXHAUSTIVE)

        #chessRet, corners = cv2.findChessboardCornersSB(undistortFrame, chessboard_size, flags=cv2.CALIB_CB_EXHAUSTIVE)

        chessRet, corners = cv2.findChessboardCorners(undistortFrameCut,chessboard_size, None)


        if chessRet:

            # Нарисуйте углы на изображении (необязательно)

            #cv2.imwrite("ChessUp.png", undistortFrame)

            objpointsDown.append(objpDown)


            corners2 = cv2.cornerSubPix(undistortFrameCut, corners, (11, 11), (-1, -1), criteria)

            corners2[:, 0, 0] += 1145

            corners2[:, 0, 1] += 410


            imgpointsDown.append(corners2)


            cv2.drawChessboardCorners(undistortFrame, chessboard_size, corners2, ret)

            cv2.imshow("Undist", undistortFrame)

            #print( "Cap frame count: " , chessUp.get(cv2.CAP_PROP_POS_FRAMES) )


            # Преобразуйте пиксельные координаты в метрические координаты

            # corners.reshape(-1, 2) упрощает работу с углами, т.к. делает их формата (кол-во_углов, 2)

            corners2 = corners2.squeeze()  # Убираем лишние измерения в массиве

            #ic(corners)

            pixel_coords = corners2.reshape(-1, 2)

            metric_coords = pixel_coords * cell_size

            #ic(pixel_coords)

            # Отобразите координаты

            for i, (pixel, metric) in enumerate(zip(pixel_coords, metric_coords)):

                px, py = pixel

                mx, my = metric

                #print(f"Corner {i+1}: Pixel Coordinates ({px:.2f}, {py:.2f}), Metric Coordinates ({mx:.2f}, {my:.2f})")




            corners2 = corners2.squeeze()  # Убираем лишние измерения в массиве

            '''

            distancesPx = []

            for i in range(chessboard_size[1]):  # цикла по вертикали (rows)

                for j in range(chessboard_size[0] - 1):  # цикла по горизонтали (columns)

                    p1 = corners[i * chessboard_size[0] + j][0]

                    p2 = corners[i * chessboard_size[0] + (j + 1)][0]

                    distancesPx.append(np.linalg.norm(p1 - p2))


            for i in range(chessboard_size[1] - 1):

                for j in range(chessboard_size[0]):

                    p1 = corners[i * chessboard_size[0] + j][0]

                    p2 = corners[(i + 1) * chessboard_size[0] + j][0]

                    distancesPx.append(np.linalg.norm(p1 - p2))


            ic(distancesPx)

            mean_distance = np.mean(distancesPx)

            print("Среднее расстояние между всеми соседними точками: {:.2f} пикселей".format(mean_distance))


            '''

            distancePxArr = []


            # Найдем расстояние между первой и второй точками

            for i in range(1):

                point1 = corners2[i]

                point2 = corners2[i+1]

                # Вычислим расстояние Евклидовой метрикой

                distancePx = np.linalg.norm(point1 - point2)

                distancePxArr.append(distancePx)
            

            for i in range(5, 6):

                point1 = corners2[i]

                point2 = corners2[i+1]

                # Вычислим расстояние Евклидовой метрикой

                distancePx = np.linalg.norm(point1 - point2)

                distancePxArr.append(distancePx)


            for i in range(10, 11):

                point1 = corners2[i]

                point2 = corners2[i+1]

                # Вычислим расстояние Евклидовой метрикой

                distancePx = np.linalg.norm(point1 - point2)

                distancePxArr.append(distancePx)



            for i in range(15, 16):

                point1 = corners2[i]

                point2 = corners2[i+1]

                # Вычислим расстояние Евклидовой метрикой

                distancePx = np.linalg.norm(point1 - point2)

                distancePxArr.append(distancePx)



            mean_distance = np.mean(distancePxArr)

            print("Среднее расстояние между всеми соседними точками: {:.2f} пикселей".format(mean_distance))


            PxSizeMetr = cell_size/mean_distance

            ic(PxSizeMetr)

            print("Размекр пикселя в метрах для нижнего положения {:.15f}", PxSizeMetr)

            #cv2.imwrite("BotChess.png", undistortFrame)


            cv2.waitKey(0)

            cv2.destroyAllWindows()

            break

        else:

            print("Шахматную доску не удалось найти на изображении.")

    return PxSizeMetr


chessboard_size = (5, 6) # Здесь предполагается доска 8x8, значит внутренние углы 7x7

chessUp.set(cv2.CAP_PROP_POS_FRAMES, 169) # On top



objpUp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)

objpUp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

objpUp *= cell_size


PxSizeTop = getPxSize(chessUp) #TODO Add crop uncrop



# Calibrate the camera

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpointsUp, imgpointsUp, (1920, 1440), None, None)


# Get the reprojection error to check the accuracy of calibration

mean_error = 0

for i in range(len(objpointsUp)):

    imgpoints2, _ = cv2.projectPoints(objpointsUp[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)

    error = cv2.norm(imgpointsUp[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)

    mean_error += error


mean_error /= len(objpointsUp)

print(f"Total reprojection error: {mean_error}")

print(f"Camera matrix:\n{camera_matrix}")

print(f"Distortion coefficients:\n{dist_coeffs}")

print("rvec: ",rvecs)

print("tvecs: ",tvecs)





chessboard_size = (6, 9) # Здесь предполагается доска 8x8, значит внутренние углы 7x7


objpDown = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)

objpDown[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

objpDown *= cell_size


#chessUp.set(cv2.CAP_PROP_POS_FRAMES, 1100) # On top

#getPxSize2(chessUp)

chessDown.set(cv2.CAP_PROP_POS_FRAMES, 20) # On botа

PxSizeBot = getPxSize2(chessDown)

print("Down proc ")

# Calibrate the camera

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpointsDown, imgpointsDown, (1920, 1440), None, None)


# Get the reprojection error to check the accuracy of calibration

mean_error = 0

for i in range(len(objpointsUp)):

    imgpoints2, _ = cv2.projectPoints(objpointsDown[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)

    error = cv2.norm(imgpointsDown[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)

    mean_error += error


mean_error /= len(objpointsUp)

print(f"Total reprojection error: {mean_error}")

print(f"Camera matrix:\n{camera_matrix}")

print(f"Distortion coefficients:\n{dist_coeffs}")

print("rvec: ",rvecs)

print("tvecs: ",tvecs)






cv2.destroyAllWindows()