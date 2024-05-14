import cv2
import numpy as np
import os
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from icecream import ic
import math


#TODO Зафиксировать гистограмму после 1 выранвнивания; Найти соединение по количеству точек хвоста

#global useEqualize, blurSize, CannyThesh1,CannyThesh2
# Initialize default parameter values
useEqualize = True
blurSize = 3
win_name = "FinderFish"
CannyThresh1 = 120
CannyThresh2 = 240
frameCount = 1
ret = True
prew_lengFish = 0
cap_start_msec = 21_000
cap_end_msec = 23_000
frame_end = (cap_end_msec - cap_start_msec) /20 
cdf = None
def find_farthest_point2(UpPx, x, y, vx, vy):
    max_distance = 0
    farthest_point = None

    for point in UpPx:
        # Вычислить расстояние между текущей точкой и заданной точкой (x, y)
        distance = math.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2)

        # Проекция точки на прямую vx, vy
        projection_x = (point[0] * vx + point[1] * vy) / (vx ** 2 + vy ** 2) * vx
        projection_y = (point[0] * vx + point[1] * vy) / (vx ** 2 + vy ** 2) * vy

        # Вычислить расстояние между проекцией и заданной точкой (x, y) вдоль прямой vx, vy
        projected_distance = abs(projection_x - x) if vx != 0 else abs(projection_y - y)

        if projected_distance > max_distance:
            max_distance = projected_distance
            farthest_point = point

    return farthest_point

def find_farthest_point(vx, vy, x, y, UpPx):    
    # Вычисляем направляющий вектор прямой
    direction = np.array([vx, vy])
    direction = direction / np.linalg.norm(direction)
    
# Преобразуем список точек в массив NumPy
    points = np.array(UpPx)
    
    # Вычисляем вектор от центра прямой до каждой точки
    vectors = points - [[x], [y]]
    
    # Вычисляем проекции векторов на прямую
    projections = np.dot(vectors, direction)
    
    
    # Найдем индекс точки с максимальной проекцией
    idx = np.argmax(np.abs(projections))
    
    # Возвращаем наиболее удаленную точку
    return points[idx] if idx < len(points) else None

def find_close_point_to_line(vx, vy, x, y, UpPx):    
    # Вычисляем направляющий вектор прямой
    direction = np.array([vx, vy])
    direction = direction / np.linalg.norm(direction)
    
# Преобразуем список точек в массив NumPy
    points = np.array(UpPx)
    
    # Вычисляем вектор от центра прямой до каждой точки
    vectors = points - [x,y]
    
    # Вычисляем проекции векторов на прямую
    projections = np.dot(vectors, direction)
    
    
    # Найдем индекс точки с максимальной проекцией
    idx = np.argmin(np.abs(projections))
    
    # Возвращаем наиболее удаленную точку
    return points[idx]



def find_closest_pairs(points1, points2):
    # Вычисляем матрицу расстояний между всеми парами точек
    dist_matrix = cdist(points1, points2)

    # Находим минимальное расстояние между точками
    min_dist = np.min(dist_matrix)

    # Находим индексы пар точек с минимальным расстоянием
    min_dist_indices = np.where(dist_matrix == min_dist)

    # Возвращаем пары точек с минимальным расстоянием
    point1_indices = min_dist_indices[0]
    point2_indices = min_dist_indices[1]
    closest_pairs = [(points1[i], points2[j]) for i, j in zip(point1_indices, point2_indices)]
    #ic(closest_pairs)
   #print("Пары точек с минимальным расстоянием:")
    #for pair in closest_pairs:
        #print(pair)
        #ic(pair)
    return closest_pairs

def find_midpoint(pairs):

    # Extract all x and y coordinates
    xs = [pair[0][0] for pair in pairs] + [pair[1][0] for pair in pairs]
    ys = [pair[0][1] for pair in pairs] + [pair[1][1] for pair in pairs]

    # Calculate the average x and y coordinates
    avg_x = np.mean(xs)
    avg_y = np.mean(ys)

    # Create the middle point
    middle_point = np.array([avg_x, avg_y], dtype=np.int64)

    return middle_point


def distance_to_line(line_params, point): 
  """
  Вычисляет расстояние от точки до прямой, заданной параметрами cv2.fitLine.
  Можно использовать для обрезки плавников

  Args:
      line_params: Массив [vx, vy, x, y], полученный с помощью cv2.fitLine.
      point: Двумерный массив, представляющий координаты точки (x, y).

  Returns:
      Расстояние от точки до прямой.
  """
  vx, vy, x0, y0 = line_params
  x1, y1 = point
  # Находим уравнение прямой в виде ax + by + c = 0
  a = -vy
  b = vx
  c = vy * x0 - vx * y0
  # Вычисляем расстояние до прямой
  distance = abs(a * x1 + b * y1 + c) / np.sqrt(a**2 + b**2)
  return distance

def find_closest_points_to_line2(PointsOfInterest, vx, vy, x, y):
    """
    Find all 2D points in UpPX that are at a minimum distance from the line defined by [vx, vy, x, y]

    Parameters:
    - UpPX: 2D array of shape (n, 2) containing the points to check
    - vx, vy, x, y: parameters of the line (from cv2.fitLine())

    Returns:
    - indices: array of indices of the points in UpPX that are at a minimum distance from the line
    """
    # Convert UpPX to a numpy array if it's not already
    PointsOfInterest = np.asarray(PointsOfInterest)

    # Calculate the distance from each point to the line
    distances = np.abs((vy * (PointsOfInterest[:, 0] - x) - vx * (PointsOfInterest[:, 1] - y)) / np.sqrt(vx**2 + vy**2))

    # Find the minimum distance
    min_distance = np.min(distances)

    # Find the indices of the points that are at a minimum distance from the line
    indices = np.where(distances < 0.8)[0]
    #ic(indices)
    return indices
def filter_points_between(white_pixels, start_point, end_point):
    """
    Фильтрует набор двумерных точек white_pixels, оставляя только те, которые
    лежат между точками start_point и end_point.
    Используется для разделения тела на голову и хвост
    Параметры:
    white_pixels (np.ndarray): Набор двумерных точек (x, y)
    start_point (tuple): Начальная точка (x, y)
    end_point (tuple): Конечная точка (x, y)
    
    Возвращает:
    np.ndarray: Отфильтрованный набор точек, лежащих между start_point и end_point
    """
    # Вычисляем вектор между start_point и end_point
    vector = np.array(end_point) - np.array(start_point)
    
    # Вычисляем проекцию каждой точки из white_pixels на этот вектор
    projections = np.dot(white_pixels - start_point, vector) / np.linalg.norm(vector)
    
    # Отбираем только те точки, проекция которых лежит между 0 и длиной вектора
    mask = (projections >= 0) & (projections <= np.linalg.norm(vector))
    return white_pixels[mask]

def calculate_angle(NoseFish, realInflectionPoint, TailPoint):
    # Vector representation of the segments
    vec1 = [realInflectionPoint[0] - NoseFish[0], realInflectionPoint[1] - NoseFish[1]]
    vec2 = [TailPoint[0] - realInflectionPoint[0], TailPoint[1] - realInflectionPoint[1]]

    # Calculate the dot product
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]

    # Calculate the magnitude of both vectors
    magnitude_vec1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
    magnitude_vec2 = math.sqrt(vec2[0]**2 + vec2[1]**2)

    # Calculate the cosine of the angle using the dot product and magnitudes
    cosine_angle = dot_product / (magnitude_vec1 * magnitude_vec2)

    # Calculate the angle in radians using the inverse cosine function
    angle_rad = math.acos(cosine_angle)

    # Convert the angle to degrees
    angle_deg = math.degrees(angle_rad)

    return angle_deg
def on_trackbar(val):
    global  CannyThresh1,CannyThresh2, blurSize
    CannyThresh1 = cv2.getTrackbarPos('CannyTresh1', "Canny viewUp Video")
    CannyThresh2 = cv2.getTrackbarPos('CannyTresh2', "Canny viewUp Video")
    blurSize = cv2.getTrackbarPos('BlurGaus', "Canny viewUp Video")



def HistMod(img,flag = False):
    global cdf
    if flag:
        cdf = np.load('cdf_values.npy')
        '''
        hist,bins = np.histogram(img.flatten(),256,[0,256])
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf,0)
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m,0).astype('uint8')
        np.save('cdf_values.npy', cdf)
        '''

    img2 = cdf[img]
    return img2


# Create window and trackbars
cv2.namedWindow("Source Up", cv2.WINDOW_NORMAL)
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

#cap.set(cv2.CAP_PROP_POS_FRAMES, 1167)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

CalHist = True

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
        viewUp = clahe.apply(viewUp)
        #viewUp = cv2.equalizeHist(viewUp)
        #viewUp = HistMod2(viewUp,CalHist)
        CalHist = False
        #cv2.imwrite(viewUp)
    
    #viewUp = cv2.morphologyEx(viewUp, cv2.MORPH_OPEN, (15,15))




  




    upEdges = cv2.Canny(viewUp, CannyThresh1,CannyThresh2,None,3,False)
    upPxEdges = np.argwhere(upEdges == 255)

    #KeyPoints
    sift = cv2.SIFT_create()
    kp = sift.detect(upEdges,None)
    img=cv2.drawKeypoints(viewUp,kp,viewUpSource)
    #cv2.imwrite('sift_keypoints.jpg',viewUpSource)


    if upPxEdges.size == 0:
        print("Edges not found")
    else:
        upPxEdges = upPxEdges[:, [1, 0]]
        
        #Find end's points===========
        hull = ConvexHull(upPxEdges)
        # Extract the points forming the hull
        hullpoints = upPxEdges[hull.vertices,:]
        # Naive way of finding the best pair in O(H^2) time if H is number of points on
        # hull
        hdist = cdist(hullpoints, hullpoints, metric='euclidean')
        # Get the farthest apart points
        bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
        #column_mean = np.mean(upPxEdges, axis=0, dtype=int) 
        #print("L: ", hdist.argmax())
        lengFish = np.linalg.norm(hullpoints[bestpair[0]] - hullpoints[bestpair[1]]) 
        
        #Проверка, что не попало ничего лишнего
        if (lengFish > prew_lengFish*1.2 or lengFish < prew_lengFish*0.8)  and prew_lengFish != 0:
            print("False corn")
        else:
            
            # Display images
            #cv2.imshow('Canny viewUp Video', upEdges)
            #cv2.imshow('viewUp Video', viewUp)
            #cv2.imshow('Source Up', viewUpSource)
            cv2.waitKey(1)
            prew_lengFish = lengFish
                    #Print them
            #print("max dist points ",  [hullpoints[bestpair[0]],hullpoints[bestpair[1]]])
            #Дальнейшие точки друг от друга
            #cv2.circle(viewUpSource,(hullpoints[bestpair[0]] ), 2,(255,0,255), -1)
            #cv2.circle(viewUpSource,(hullpoints[bestpair[1]] ), 2,(255,255,255), -1)
            #cv2.circle(viewUpSource,(column_mean[0], column_mean[1]), 8,(255,0,255), -1) 


            #=================================================================
            #ellipse = cv2.fitEllipse(upPxEdges)
            #(center, axes, angle) = ellipse
            [vx, vy, x, y] = cv2.fitLine(upPxEdges, cv2.DIST_L2, 0, 0.01, 0.01) #x,y - средние точки, vx vy параметры прямой
            k = vy[0] / vx[0]
            b = y[0] - k * x[0]
            inflectionPoint = np.zeros(2, dtype=int)
            # Вычисляем координаты концов линии для визуализации
            rows, cols = viewUpSource.shape[:2]
            left_x = 0
            right_x = cols - 1
            left_y = int(k * left_x + b)
            right_y = int(k * right_x + b)
            
            # Рисуем линию на изображении
            #cv2.line(viewUpSource, (left_x, left_y), (right_x, right_y), (0, 255, 255), 2) #Желтьая через всю рыбу

            #Смещение к хвосту ЗНАК ЗАВИСИТ ОТ ПОВОРОТА ПРЯМОЙ
            #ic(vx,vy)
            if vy > 0:
                inflectionPoint[0] = x[0] - lengFish/6 * vx[0] 
                inflectionPoint[1] = y[0] - lengFish/6 *vy[0]
                #TODO сделать ограничитель меньше чем длина всей рыбы            
                tailRange_x = x[0] - lengFish * vx[0] / math.sqrt(vx[0]**2 + vy[0]**2)
                tailRange_y = y[0] - lengFish * vy[0] / math.sqrt(vx[0]**2 + vy[0]**2)

                headRange_x = x[0] + ( lengFish * vx[0] / math.sqrt(vx[0]**2 + vy[0]**2) ) 
                headRange_y = y[0] + ( lengFish * vy[0] / math.sqrt(vx[0]**2 + vy[0]**2) )

            else:
                inflectionPoint[0] = x[0] + lengFish/6 * vx[0] 
                inflectionPoint[1] = y[0] + lengFish/6 *vy[0]

                tailRange_x = x[0] + lengFish * vx[0] / math.sqrt(vx[0]**2 + vy[0]**2)
                tailRange_y = y[0] + lengFish * vy[0] / math.sqrt(vx[0]**2 + vy[0]**2)

                headRange_x = x[0] - lengFish * vx[0] / math.sqrt(vx[0]**2 + vy[0]**2)
                headRange_y = y[0] - lengFish * vy[0] / math.sqrt(vx[0]**2 + vy[0]**2)


            #cv2.circle(viewUpSource,(int(x[0]), int(y[0])), 2,(0,0,255), -1)
            cv2.circle(viewUpSource,(int(inflectionPoint[0]), int(inflectionPoint[1])), 2,(0,0,255), -1) #Отрисовка поправленной точки стыковки хвоста
            #cv2.circle(viewUpSource, (int(tailRange_x), int(tailRange_y)), 1, (255, 0, 0), -1) # Отрисовка облассти хвоста

            tailFishContur = filter_points_between(upPxEdges, inflectionPoint, (int(tailRange_x), int(tailRange_y)) )
            headFishContur = filter_points_between(upPxEdges, inflectionPoint, (int(headRange_x), int(headRange_y)) )

            #NoseFish = find_farthest_point(vx, vy,x,y, headFishContur)

            
            closest_pairs = find_closest_pairs(headFishContur,tailFishContur )
            #ic (closest_pairs)
            realInflectionPoint = find_midpoint(closest_pairs)
            ic(cap.get(cv2.CAP_PROP_POS_FRAMES))
            NoseFish = find_farthest_point(vx, vy,x,y, headFishContur)
            #NoseFish = find_farthest_point2(headFishContur, vx, vy,x,y)
            if NoseFish is None:
                print("Bad Nose")
                continue
            #Разкоменть
            #cv2.circle(viewUpSource,NoseFish, 5,(0,0,255), -1) #Отрисовка поправленной точки стыковки хвоста



            #ic (realInflectionPoint)
            ''' Delete fin
            headFishContur = []
            for point_h in headFishContur_fh:
                if (distance_to_line([vx, vy, x, y], point_h ) < 25 ):
                    headFishContur.append(point_h)
            
            headFishContur = np.asarray(headFishContur)

            '''
            center_h_x = np.mean(headFishContur[:, 0])
            center_h_y = np.mean(headFishContur[:, 1])
            center = (int(center_h_x), int(center_h_y))

            '''PCA 
            headFishConturf = np.float32(headFishContur)
            # Вычисление главных компонент
            mean, eigenvectors = cv2.PCACompute(headFishConturf, mean=None)
            axis = eigenvectors[0]
            # Вычисление угла поворота
            angle = math.atan2(axis[1], axis[0])
            start_point_h = (int(center[0]), int(center[1]))
            end_point_h = (int(center[0] + axis[0]*100), int(center[1] + axis[1]*100))
            cv2.line(viewUpSource, start_point_h, end_point_h, (255, 0, 255), 2)
            '''

            '''Moment!!!
            moments = cv2.moments(headFishContur)

            angle = 0.5 * math.atan2(2 * moments["mu11"], (moments["mu20"] - moments["mu02"]))
            angle = math.degrees(angle)

            length = max(viewUpSource.shape[0], viewUpSource.shape[1]) // 4
            start = (int(center_h_x - length * math.cos(math.radians(angle))),
                    int(center_h_y - length * math.sin(math.radians(angle))))
            end = (int(center_h_x + length * math.cos(math.radians(angle))),
                int(center_h_y + length * math.sin(math.radians(angle))))
            cv2.line(viewUpSource, start, end, (0, 255, 0), 2)

            '''

            
            ''' Fit head
            
            [vx, vy, x, y] = cv2.fitLine(headFishContur, cv2.DIST_L2, 0, 0.01, 0.01) #x,y - средние точки, vx vy параметры прямой

            k = vy[0] / vx[0]
            b = y[0] - k * x[0]
            inflectionPoint = np.zeros(2, dtype=int)
            # Вычисляем координаты концов линии для визуализации
            rows, cols = viewUpSource.shape[:2]
            left_x = 0
            right_x = cols - 1
            left_y = int(k * left_x + b)
            right_y = int(k * right_x + b)
            cv2.line(viewUpSource, (left_x, left_y), (right_x, right_y), (255, 0, 255), 2)
            '''




            #"""XBOCT
            [vx, vy, x, y] = cv2.fitLine(tailFishContur, cv2.DIST_L2, 0, 0.01, 0.01) #x,y - средние точки, vx vy параметры прямой
            k = vy[0] / vx[0]
            b = y[0] - k * x[0]
            ceterFish = np.zeros(2, dtype=int)
            # Вычисляем координаты концов линии для визуализации
            rows, cols = viewUpSource.shape[:2]
            left_x = 0
            right_x = cols - 1
            left_y = int(k * left_x + b)
            right_y = int(k * right_x + b)
            #Разкоменть
            #cv2.line(viewUpSource, (left_x, left_y), (right_x, right_y), (255, 0, 255), 2) #Красная ось симметрии хвоста
            #"""

            TailPointIndx = find_closest_points_to_line2(tailFishContur,vx, vy,x,y )
            TailPoint = tailFishContur[TailPointIndx]
            TailPoint = TailPoint[0]

            #ic(TailPoint)

            #Раcкоменть ''
            '''
            for x_t, y_t in tailFishContur:
                cv2.circle(viewUpSource,( x_t, y_t), 1,(0,0,255), -1)

            for x_h, y_h in headFishContur:
                cv2.circle(viewUpSource,( x_h, y_h), 1,(255,0,0), -1)
            '''

            #cv2.circle(viewUpSource, realInflectionPoint , 2,(0,255,0), -1)
            #Разкоменть 3 строки
            #cv2.circle(viewUpSource, TailPoint , 1,(0,255,0), -1)
           # cv2.line(viewUpSource, NoseFish,realInflectionPoint,(0,255,255), 2 )
            #cv2.line(viewUpSource, realInflectionPoint,TailPoint,(0,255,0), 2 )
            tailAngle = calculate_angle(NoseFish, realInflectionPoint,TailPoint)
            print("Угол: ", tailAngle)
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
            #angle = (angle * np.pi / 180.0) + 90

            # Вычислите концевые точки большой оси эллипса
            #major_axis_len = max(axes)
            #minor_axis_len = min(axes)
            #x1 = int(center[0] - major_axis_len * np.cos(angle))
            #y1 = int(center[1] - major_axis_len * np.sin(angle))
            #x2 = int(center[0] + major_axis_len * np.cos(angle))
            #y2 = int(center[1] + major_axis_len * np.sin(angle))

            # Нарисуйте большую ось эллипса
            #cv2.line(viewUpSource, (x1, y1), (x2, y2), (0, 255, 0), 1)
            #cv2.circle(viewUpSource,(int(center[0]), int(center[1])), 4,(0,0,255), -1)

        # cv2.line(viewUpSource, (x1r, y1r), (x2r, y2r), (255, 0, 0), 2)


            #cv2.ellipse(viewUpSource,ellipse,(255,255,255),2)




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
    if cv2.waitKey(0) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
