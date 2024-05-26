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


#global useEqualize, blurSize, CannyThesh1,CannyThesh2
useEqualize = True
blurSize = 3
win_name = "FinderFish"
CannyThresh1 = 80
CannyThresh2 = 160
frameCount = 1
ret = True
prew_lengFish = 0
#cap_start_msec = 22_000
#cap_end_msec = 24_000
pxCenterFish = []


#=======================================================================================
#                           GOPR6626        
#======================================================================================
#5:23-5:24         Done
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
#UpDirection ----- 
cap_start_msec = (5*60 + 47) * 1000
cap_end_msec = (5*60 + 49.2) * 1000

#=======================================================================================
#                           GP026626        
#======================================================================================
#UpDirection  -- done
#cap_start_msec = (5*60 + 41.7) * 1000
#cap_end_msec = (5*60 + 43.7) * 1000

#Wall Bad
#cap_start_msec = (3*60 + 49) * 1000
#cap_end_msec = (3*60 + 52) * 1000

#NeedCut Up dir
#cap_start_msec = (4*60 + 50.5) * 1000
#cap_end_msec = (4*60 + 54) * 1000

# Done
#cap_start_msec = (5*60 + 18) * 1000   
#cap_end_msec = (5*60 + 22) * 1000

frame_end = (cap_end_msec - cap_start_msec) /20 


# Open video file
#cap = cv2.VideoCapture('D:/MyCodeProjects/FishTailSpy/Data/Fish/Fish1_3m.mp4')
#cap = cv2.VideoCapture('D:/MyCodeProjects/FishTailSpy/Data/Fish/outFishFFMPEG2.mp4')#GOPR6626
cap = cv2.VideoCapture('D:/MyCodeProjects/FishTailSpy/Data/Fish/FFMPEGm_Fish3.MP4')#GP016626
#cap = cv2.VideoCapture('D:/MyCodeProjects/FishTailSpy/Data/Fish/FFMPEGm_Fish4.MP4')#GP026626
#cap = cv2.VideoCapture('D:/MyCodeProjects/FishTailSpy/Data/Fish/FFMPEGm_Fish5.MP4')#GGP036626



def find_farthest_point(vx, vy, x, y, UpPx, botdir = True):    
    # Вычисляем направляющий вектор прямой
    #ic(vx ,vy, x, y)
    
    # Вычисление новых координат
    #x = x + 500 * vx
    #y = y + 500 * vy

    if vy < 0:
        #vy = -vy
        print("Костыль жесткий")
    direction = np.array([vx, vy])
    direction = direction / np.linalg.norm(direction)
    direction = -direction
# Преобразуем список точек в массив NumPy
    points = np.array(UpPx)

    # Вычисляем вектор от центра прямой до каждой точки
    vectors = points -  np.array([x, y]) #[[x], [y]]
    
    # Вычисляем проекции векторов на прямую
    projections = np.dot(vectors, direction)
    if botdir:
        # Найдем индекс точки с максимальной проекцией
        idx = np.argmax(projections)
    else:
        idx = np.argmin(projections)
    # Возвращаем наиболее удаленную точку
    return points[idx]

def find_farthest_point228(vx, vy, x, y, UpPx, botdir = True):
    # Вектор направляющий
    v = np.array([vx, vy])

    # Вычисление новых координат
    x = x + 50 * vx
    y = y + 50 * vy
    # Поиск точки, перпендикуляр из которой наиболее удален от (x, y)
    max_distance = 0
    P_max = None

    for (u_i, v_i) in UpPx:
        d_i = np.array([u_i - x, v_i - y])
        t_i = np.dot(d_i, v) / np.dot(v, v)
        p_i = np.array([x, y]) + t_i * v
        distance = np.linalg.norm(p_i - np.array([x, y]))
        # Определяем, находится ли точка "ниже" по направлению прямой
        vector_to_point = np.array([u_i, v_i]) - p_i
        if np.dot(vector_to_point, v) > 0:  # точка должна находиться в том же направлении, что и v
            if distance > max_distance:
                max_distance = distance
                P_max = (u_i, v_i)
    return P_max

    
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
    sorted_distances = np.dstack(np.unravel_index(np.argsort(dist_matrix.ravel()), dist_matrix.shape))[0]
    # Возвращаем пары точек с минимальным расстоянием
    point1_indices = min_dist_indices[0]
    point2_indices = min_dist_indices[1]
    closest_pairs = [(points1[i], points2[j]) for i, j in zip(point1_indices, point2_indices)]
   #print("Пары точек с минимальным расстоянием:")
    #for pair in closest_pairs:
        #print(pair)
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

    PointsOfInterest = np.asarray(PointsOfInterest)

    distances = np.abs((vy * (PointsOfInterest[:, 0] - x) - vx * (PointsOfInterest[:, 1] - y)) / np.sqrt(vx**2 + vy**2))

    min_distance = np.min(distances)

    indices = np.where(distances < 0.8)[0]
    return indices

def find_closest_point3(white_pixels, inflectionPoint):
    y_inflection, x_inflection = inflectionPoint
    min_distance = float('inf')
    closest_point = None

    for point in white_pixels:
        x, y = point
        # Вычисляем евклидово расстояние
        distance = math.sqrt((x - x_inflection) ** 2 + (y - y_inflection) ** 2)
        
        # Обновляем минимальное расстояние и ближайшую точку
        if distance < min_distance:
            min_distance = distance
            closest_point = point
    #ic(closest_point, min_distance)
    return closest_point

def filter_points_between(white_pixels, start_point, end_point,distmin = False):
    """
    Фильтрует набор двумерных точек white_pixels, оставляя только те, которые
    лежат между точками start_point и end_point.
    Используется для разделения тела на голову и хвост
  
    """
    # Вычисляем вектор между start_point и end_point
    vector = np.array(end_point) - np.array(start_point)
    if distmin:
        vector = distmin
    
    # Вычисляем проекцию каждой точки из white_pixels на этот вектор
    projections = np.dot(white_pixels - start_point, vector) / np.linalg.norm(vector)
    #ic (np.linalg.norm(vector) )

    # Шаг 1: Вычисление абсолютных значений
    abs_projections = np.abs(projections)

    # Шаг 2: Нахождение индекса минимального значения
    min_index = np.argmin(abs_projections)

    # Шаг 3: Получение значения из оригинального массива
    closest_to_zero = projections[min_index]

    #ic(closest_to_zero, min_index)
    #ic(np.linalg.norm(vector))

    mask = (projections >= 0) & (projections <= np.linalg.norm(vector) )
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

def reduce_binary_objects(up_px, source_img):
    """
    Reduces binary objects to 1 pixel wide representations.

    Args:
        up_px (list of tuples): Contour points of the object
        source_img (numpy array): Original image

    Returns:
        numpy array: Source image with the reduced line drawn
    """
    # Initialize an empty image with the same size as the source image
    reduced_img = np.zeros_like(source_img)

    # Iterate over each contour point
    for i in range(len(up_px) - 1):
        # Get the current and next points
        x1, y1 = up_px[i]
        x2, y2 = up_px[i + 1]

        # Calculate the slope and intercept of the line segment
        dx, dy = x2 - x1, y2 - y1
        slope = dy / dx if dx!= 0 else float('inf')
        intercept = y1 - slope * x1

        # Iterate over each pixel in the line segment
        for x in range(min(x1, x2), max(x1, x2) + 1):
            y = int(slope * x + intercept)
            reduced_img[y, x] = 255

    # Draw the reduced line on the source image using OpenCV
    cv2.line(source_img, tuple(up_px[0]), tuple(up_px[-1]), (0, 255, 0), 1)

    return source_img

# Функция для заполнения контуров белым цветом
def fill_contours(image):
    # Находим контуры
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)> 1: 
        contours = max(contours, key=cv2.contourArea)
    elif len(contours) <1:
        print("NoCounters")
        exit()
    #contours = sorted(contours, key=cv2.contourArea, reverse=True)
    #contours2 = contours[0]

    # Создаем маску с тем же размером, что и у входного изображения, но полностью черная
    mask = np.zeros(image.shape, np.uint8)
    
    # Заполняем контуры на маске белым цветом
    cv2.drawContours(mask, contours[0], -1, (255, 255, 255), cv2.FILLED)
    
    # Создаем белое изображение с тем же размером, что и у входного изображения
    white_image = np.full(image.shape, 255, dtype=np.uint8)
    
    # Используем маску для заполнения исходного изображения
    result = cv2.bitwise_and(white_image, white_image, mask=mask)
    
    return result
def on_trackbar(val):
    global  CannyThresh1,CannyThresh2, blurSize
    CannyThresh1 = cv2.getTrackbarPos('CannyTresh1', "Canny viewUp Video")
    CannyThresh2 = cv2.getTrackbarPos('CannyTresh2', "Canny viewUp Video")
    blurSize = cv2.getTrackbarPos('BlurGaus', "Canny viewUp Video")

def mirrorH ():
    pass

def interpolate_contour(contour, distance=1.0):
    """
    Interpolates points in the contour to fill large gaps.
    """
    interpolated_contour = []
    
    for i in range(len(contour)):
        point1 = contour[i][0]
        point2 = contour[(i + 1) % len(contour)][0]
        interpolated_contour.append(point1)
        
        # Compute the Euclidean distance between points
        dist = np.linalg.norm(point1 - point2)
        
        # Add intermediate points if the distance is larger than the threshold
        if dist > distance:
            num_points = int(dist // distance)
            for j in range(1, num_points):
                interpolated_point = (point1 + j * (point2 - point1) / num_points).astype(int)
                interpolated_contour.append(interpolated_point)
    
    return np.array(interpolated_contour).reshape(-1, 1, 2)


# Create window and trackbars
cv2.namedWindow("Canny viewUp Video")
cv2.createTrackbar("CannyTresh1", "Canny viewUp Video" , CannyThresh1, 500, on_trackbar)
cv2.createTrackbar("CannyTresh2", "Canny viewUp Video" , CannyThresh2, 500, on_trackbar)
cv2.createTrackbar("BlurGaus", "Canny viewUp Video" , blurSize, 50, on_trackbar)



cap.set(cv2.CAP_PROP_POS_MSEC , cap_start_msec)

# Load camera parameters
data = np.load("camera_params.npz")
mtx = data['mtx']
dist = data['dist']
TailAngleArray = []

while ret:
    ret, frame = cap.read()
    if not ret:
        print("Video can't open")
        break
    # Undistort frame
    undistortFrame = cv2.undistort(frame, mtx, dist)
    # Extract region of interest
    viewUp = undistortFrame[250:1400, 1050:1340].copy() # y1:y2 x1:x2
    viewUpSource = undistortFrame [250:1400, 1050:1340].copy() # y1:y2 x1:x2
    viewSource = undistortFrame.copy() # y1:y2 x1:x2

    viewMirror = undistortFrame[420:1183, 525:870].copy() # y1:y2 x1:x2

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

    #viewUp = cv2.morphologyEx(viewUp, cv2.MORPH_OPEN, (15,15))
     # Создаем копию для заполнения
    fillupEdges = upEdges.copy()
    # Структурный элемент для морфологических операций
    kernel = np.ones((5, 5), np.uint8)
    # Применяем морфологическое замыкание
    closedEdges = cv2.morphologyEx(fillupEdges, cv2.MORPH_CLOSE, kernel)
    #closedEdges = cv2.morphologyEx(fillupEdges, cv2.MORPH_CLOSE, kernel)

    # Находим контуры на замкнутом изображении
    contours, _ = cv2.findContours(closedEdges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    max_contour2_flat = np.squeeze(max_contour)
    max_contour = [max_contour]
    interpolated_contour = interpolate_contour(max_contour)
    # Создаем пустое изображение для заполнения
    filledImage = np.zeros_like(fillupEdges)
    # Заполняем найденные контуры белым цветом
    cv2.fillPoly(filledImage, max_contour, 255)
    fillFish = filledImage
    #cv2.imshow('Filled Contours', fillFish)
    cv2.imshow('Filled Contours2', filledImage)

    if upPxEdges.size == 0:
        print("Edges not found")
    else:
        upPxEdges = upPxEdges[:, [1, 0]]
        #Find end's points===========
        hull = ConvexHull(max_contour2_flat)
        # Extract the points forming the hull
        hullpoints = max_contour2_flat[hull.vertices,:]
        # Naive way of finding the best pair in O(H^2) time if H is number of points on
        # hull
        hdist = cdist(hullpoints, hullpoints, metric='euclidean')
        # Get the farthest apart points
        bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
        #column_mean = np.mean(upPxEdges, axis=0, dtype=int) 
        #print("L: ", hdist.argmax())
        lengFish = np.linalg.norm(hullpoints[bestpair[0]] - hullpoints[bestpair[1]]) 
        
        #Проверка, что не попало ничего лишнего
        if (lengFish > prew_lengFish*1.3 or lengFish < prew_lengFish*0.7)  and prew_lengFish != 0:
            print("False corn")
            cv2.waitKey(0)
            pxCenterFish.append(None)
            TailAngleArray.append(None)

        else:
            prew_lengFish = lengFish

           
            #Прямая по заполненой рыбе
            k, l = np.where(fillFish == 255)
            # Сохраните координаты в двумерный массив для использования с fitLine
            white_pixels_mask = np.column_stack((l, k))
            
            [vx, vy, x, y] = cv2.fitLine(white_pixels_mask, cv2.DIST_L2, 0, 0.01, 0.01) #x,y - средние точки, vx vy параметры прямой
            pxCenterFish.append( [x,y] )
            # Вычисление середины объекта
            #center_xtest = np.mean(white_pixels_mask[:, 0])
            #center_ytest = np.mean(white_pixels_mask[:, 1])

            #pxCenterFish2 = (center_xtest, center_ytest)
            k = vy[0] / vx[0]
            b = y[0] - k * x[0]
            # Вычисляем координаты концов линии для визуализации
            rows, cols = viewUpSource.shape[:2]
            left_x = 0
            right_x = cols - 1
            left_y = int(k * left_x + b)
            right_y = int(k * right_x + b)
            #cv2.line(viewUpSource, (left_x, left_y), (right_x, right_y), (255, 0, 255), 2) #Фиолетовая через всю рыбу

            inflectionPoint = np.zeros(2, dtype=int)

            botDir = False
            ic(vy)
            if vy > 0:
                botDir2 = True
            else:
                botDir2 = False
            try: #TODO Fix
                NoseFish = find_farthest_point(vx[0], vy[0],x[0],y[0], max_contour2_flat, botDir2) # Точка носа
            except IndexError:
                NoseFish = None  # Возвращаем None при возникновении ошибки
                print("Nose Error ")
                continue

            if not botDir:
                lengFish = - lengFish
            #Смещение к хвосту ЗНАК ЗАВИСИТ ОТ ПОВОРОТА ПРЯМОЙ
            #ic(vy)
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
            #cv2.circle(viewUpSource,(int(inflectionPoint[0]), int(inflectionPoint[1])), 5,(0,0,255), -1) #Отрисовка поправленной точки стыковки хвоста
            #cv2.circle(viewUpSource, (int(tailRange_x), int(tailRange_y)), 5, (255, 0, 0), -1) # Отрисовка облассти хвоста

            tailFishContur = filter_points_between(white_pixels_mask, inflectionPoint, (int(tailRange_x), int(tailRange_y)) )
            headFishContur = filter_points_between(white_pixels_mask, inflectionPoint, (int(headRange_x), int(headRange_y)) )

            skeleton = skeletonize(fillFish)
                # Находим координаты пикселей скелета
            skeletCurve = np.column_stack(np.where(skeleton))
            for Skpoint in skeletCurve:
                cv2.circle(viewUpSource, (Skpoint[1], Skpoint[0]), 1, (0, 0, 255), -1)
            #cv2.imshow('Skelet', skeleton)


            #closest_pairs = find_closest_pairs(headFishContur,tailFishContur ) 
            #realInflectionPoint = find_midpoint(closest_pairs)
            #min_distance = float('inf')
            #closest_point = None
            '''
            # Находим угловой коэффициент перпендикуляра
            p_k = -1 / k
            # Находим свободный коэффициент
            b = inflectionPoint[1] - p_k * inflectionPoint[0]
            for x_i, y_i in skeletCurve:
                distanceSkelet = abs(y_i - (p_k*x_i + b)) / np.sqrt(1 + p_k**2)
                    # Обновляем минимальное расстояние и ближайшую точку, если необходимо
                if distanceSkelet < min_distance:
                    min_distance = distanceSkelet
                    closest_point = (y_i,x_i)
            '''
            realInflectionPoint = find_closest_point3(skeletCurve, inflectionPoint)
                        
            save = realInflectionPoint[0]
            realInflectionPoint[0] = realInflectionPoint[1] # Точка перегиба
            realInflectionPoint[1] = save

            #realInflectionPoint = inflectionPoint
            #realInflectionPoint = inflectionPoint

            #------------------------------Work 50%
            '''
            realInflectionPoint = filter_points_between(skeletCurve,inflectionPoint,( int(inflectionPoint[0] + 0.5 * vx[0] ) , int(inflectionPoint[1] + 0.5 * vy[0] ) ) ) #TODO Подобрать вместо +1
            if len(realInflectionPoint) < 1:
                realInflectionPoint = filter_points_between(skeletCurve,inflectionPoint,( int(inflectionPoint[0] + 0.75 * vx) , int(inflectionPoint[1] + 0.75 * vy ) ) ) #TODO Подобрать вместо +1
            if len(realInflectionPoint) < 1:
                realInflectionPoint = filter_points_between(skeletCurve,inflectionPoint,( int(inflectionPoint[0] + 1) , int(inflectionPoint[1] + 1) ) ) #TODO Подобрать вместо +1
            if len(realInflectionPoint) > 0:
                realInflectionPoint = realInflectionPoint[0]
            else:
                print("realInflectionPoint not found")
            
            save = realInflectionPoint[0]
            realInflectionPoint[0] = realInflectionPoint[1] # Точка перегиба
            realInflectionPoint[1] = save
            '''
            #center_h_x = np.mean(headFishContur[:, 0])
            #center_h_y = np.mean(headFishContur[:, 1])
            #center = (int(center_h_x), int(center_h_y))

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
            #cv2.line(viewUpSource, (left_x, left_y), (right_x, right_y), (0, 0, 255), 2) #Красная ось симметрии хвоста
            #"""

            TailPointIndx = find_closest_points_to_line2(tailFishContur,vx[0], vy[0],x[0],y[0] )
            TailPoint = tailFishContur[TailPointIndx]
            if botDir == False:
                max_y_index = np.argmax(TailPoint[:, 1])
                TailPoint = TailPoint[max_y_index]
            else:
                min_y_index = np.argmin(TailPoint[:, 1])
                TailPoint = TailPoint[min_y_index]
                #TailPoint = TailPoint[0] #Точка хвоста

            #for x_t, y_t in tailFishContur:
            #    cv2.circle(viewUpSource,( x_t, y_t), 1,(0,0,255), -1)

            #for x_h, y_h in headFishContur:
            #    cv2.circle(viewUpSource,( x_h, y_h), 1,(255,0,0), -1)

            #PaintHullPoints
            #cv2.circle(viewUpSource,(hullpoints[bestpair[0]] ), 2,(255,0,255), -1)
            #cv2.circle(viewUpSource,(hullpoints[bestpair[1]] ), 2,(255,0,255), -1)
            cv2.circle(viewUpSource, inflectionPoint , 4,(255,0,0), -1)
            cv2.circle(viewUpSource, realInflectionPoint , 4,(0,255,255), -1)
            cv2.circle(viewUpSource, TailPoint , 1,(0,255,0), -1)
            cv2.line(viewUpSource, NoseFish,realInflectionPoint,(0,255,255), 1 )
            cv2.line(viewUpSource, realInflectionPoint,TailPoint,(0,255,0), 1 )


            tailAngle = calculate_angle(NoseFish, realInflectionPoint,TailPoint)
            TailAngleArray.append(tailAngle)
            print("Угол: ", tailAngle)

        

        cv2.imshow('Canny viewUp Video', upEdges)
        cv2.imshow('viewUp Video', viewUp)
        cv2.imshow('Source Up', viewUpSource)
        #cv2.imwrite('TestImgHist.png', viewUp)
        #cv2.imwrite( "Data//UndistertImages//Undist_frame_" + str(frameCount) + ".png", undistortFrame )
        #cv2.imwrite( "Data//UndistertImages//Source_frame_" + str(frameCount) + ".png", frame )

        frameCount +=1








    # Reset video position after a certain frame count
    if frameCount > frame_end:
        TailAngleArray = np.array(TailAngleArray)
        pxCenterFish = np.array(pxCenterFish)
        np.savez('dataMain.npz', TailAngleArray=TailAngleArray, pxCenterFish=pxCenterFish)
        print("Save Data")
        #cap.set(cv2.CAP_PROP_POS_MSEC , cap_start_msec)
        #frameCount = 1
        # Создаем график
        #plt.plot(TailAngle)
        #plt.show()  
        break
    #Exit loop if 'q' is pressed
    if cv2.waitKey(100) == ord('q'):
        break

#TailAngle спиоск углов. pxCenterFish координаты рыбы
cap.release()
cv2.destroyAllWindows()
