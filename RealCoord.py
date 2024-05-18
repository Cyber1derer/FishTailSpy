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

chessboard_size = (5, 6) # Здесь предполагается доска 8x8, значит внутренние углы 7x7

total_pixel_distance = 0
total_metric_distance = 0
pair_count = 0

def getPxSize(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video can't open")
            break
        # Undistort frame
        undistortFrame = cv2.undistort(frame, mtx, dist)
        undistortFrame =cv2.cvtColor(undistortFrame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Source", undistortFrame)
        cv2.waitKey(10)
        #GoodChessUp = cv2.VideoCapture(r'D:\MyCodeProjects\FishTailSpy\Data\GOPR6629_1714036322353.MP4')

        # Отобразите изображение с углами
        chessRet, corners = cv2.findChessboardCornersSB(undistortFrame, chessboard_size, flags=cv2.CALIB_CB_EXHAUSTIVE)
        if chessRet:
            # Нарисуйте углы на изображении (необязательно)
            #cv2.imwrite("ChessUp.png", undistortFrame)

            cv2.drawChessboardCorners(undistortFrame, chessboard_size, corners, ret)
            cv2.imshow("Undist", undistortFrame)
            #print( "Cap frame count: " , chessUp.get(cv2.CAP_PROP_POS_FRAMES) )


            # Размер одной клетки шахматной доски в метрах
            cell_size = 0.0085 # 8,5 мм 
            # Преобразуйте пиксельные координаты в метрические координаты
            # corners.reshape(-1, 2) упрощает работу с углами, т.к. делает их формата (кол-во_углов, 2)
            corners = corners.squeeze()  # Убираем лишние измерения в массиве
            #ic(corners)
            pixel_coords = corners.reshape(-1, 2)
            metric_coords = pixel_coords * cell_size
            #ic(pixel_coords)
            # Отобразите координаты
            for i, (pixel, metric) in enumerate(zip(pixel_coords, metric_coords)):
                px, py = pixel
                mx, my = metric
                print(f"Corner {i+1}: Pixel Coordinates ({px:.2f}, {py:.2f}), Metric Coordinates ({mx:.2f}, {my:.2f})")



            corners = corners.squeeze()  # Убираем лишние измерения в массиве
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
                point1 = corners[i]
                point2 = corners[i+1]
                # Вычислим расстояние Евклидовой метрикой
                distancePx = np.linalg.norm(point1 - point2)
                distancePxArr.append(distancePx)
            
            for i in range(5, 9):
                point1 = corners[i]
                point2 = corners[i+1]
                # Вычислим расстояние Евклидовой метрикой
                distancePx = np.linalg.norm(point1 - point2)
                distancePxArr.append(distancePx)

            for i in range(10, 14):
                point1 = corners[i]
                point2 = corners[i+1]
                # Вычислим расстояние Евклидовой метрикой
                distancePx = np.linalg.norm(point1 - point2)
                distancePxArr.append(distancePx)


            for i in range(15, 19):
                point1 = corners[i]
                point2 = corners[i+1]
                # Вычислим расстояние Евклидовой метрикой
                distancePx = np.linalg.norm(point1 - point2)
                distancePxArr.append(distancePx)


            mean_distance = np.mean(distancePxArr)
            print("Среднее расстояние между всеми соседними точками: {:.2f} пикселей".format(mean_distance))

            PxSizeMetr = cell_size/mean_distance
            ic(PxSizeMetr)
            
            cv2.waitKey(0)
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
        cv2.imshow("Source", undistortFrame)
        cv2.waitKey(10)
        #GoodChessUp = cv2.VideoCapture(r'D:\MyCodeProjects\FishTailSpy\Data\GOPR6629_1714036322353.MP4')

        # Отобразите изображение с углами
        chessRet, corners = cv2.findChessboardCornersSB(undistortFrame, chessboard_size, flags=cv2.CALIB_CB_EXHAUSTIVE)
        if chessRet:
            # Нарисуйте углы на изображении (необязательно)
            #cv2.imwrite("ChessUp.png", undistortFrame)

            cv2.drawChessboardCorners(undistortFrame, chessboard_size, corners, ret)
            cv2.imshow("Undist", undistortFrame)
            #print( "Cap frame count: " , chessUp.get(cv2.CAP_PROP_POS_FRAMES) )


            # Размер одной клетки шахматной доски в метрах
            cell_size = 0.0085 # 8,5 мм 
            # Преобразуйте пиксельные координаты в метрические координаты
            # corners.reshape(-1, 2) упрощает работу с углами, т.к. делает их формата (кол-во_углов, 2)
            corners = corners.squeeze()  # Убираем лишние измерения в массиве
            #ic(corners)
            pixel_coords = corners.reshape(-1, 2)
            metric_coords = pixel_coords * cell_size
            #ic(pixel_coords)
            # Отобразите координаты
            for i, (pixel, metric) in enumerate(zip(pixel_coords, metric_coords)):
                px, py = pixel
                mx, my = metric
                print(f"Corner {i+1}: Pixel Coordinates ({px:.2f}, {py:.2f}), Metric Coordinates ({mx:.2f}, {my:.2f})")



            corners = corners.squeeze()  # Убираем лишние измерения в массиве
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
                point1 = corners[i]
                point2 = corners[i+1]
                # Вычислим расстояние Евклидовой метрикой
                distancePx = np.linalg.norm(point1 - point2)
                distancePxArr.append(distancePx)
            
            for i in range(5, 6):
                point1 = corners[i]
                point2 = corners[i+1]
                # Вычислим расстояние Евклидовой метрикой
                distancePx = np.linalg.norm(point1 - point2)
                distancePxArr.append(distancePx)

            for i in range(10, 11):
                point1 = corners[i]
                point2 = corners[i+1]
                # Вычислим расстояние Евклидовой метрикой
                distancePx = np.linalg.norm(point1 - point2)
                distancePxArr.append(distancePx)


            for i in range(15, 16):
                point1 = corners[i]
                point2 = corners[i+1]
                # Вычислим расстояние Евклидовой метрикой
                distancePx = np.linalg.norm(point1 - point2)
                distancePxArr.append(distancePx)


            mean_distance = np.mean(distancePxArr)
            print("Среднее расстояние между всеми соседними точками: {:.2f} пикселей".format(mean_distance))

            PxSizeMetr = cell_size/mean_distance
            ic(PxSizeMetr)
            cv2.imwrite("BotChess.png", undistortFrame)

            cv2.waitKey(0)
            break
        else:
            print("Шахматную доску не удалось найти на изображении.")
    return PxSizeMetr

chessUp.set(cv2.CAP_PROP_POS_FRAMES, 169) # On top
PxSizeTop = getPxSize(chessUp)
#chessUp.set(cv2.CAP_PROP_POS_FRAMES, 1100) # On top
chessboard_size = (5, 6) # Здесь предполагается доска 8x8, значит внутренние углы 7x7
#getPxSize(chessUp)
chessDown.set(cv2.CAP_PROP_POS_FRAMES, 20) # On bot
PxSizeBot = getPxSize2(chessDown)

cv2.destroyAllWindows()