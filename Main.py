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
#cv2.namedWindow("Source Up", cv2.WINDOW_NORMAL)
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



# Create SIFT detector and descriptor
sift = cv2.SIFT_create()

object_boundary = [(314,528), (259,442), (233,393)]



# Initialize the previous frame and its keypoints
prev_frame = None
prev_kp = None


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
        #viewUp = clahe.apply(viewUp)
        viewUp = cv2.equalizeHist(viewUp)
        #viewUp = HistMod2(viewUp,CalHist)
        CalHist = False


    upEdges = cv2.Canny(viewUp, CannyThresh1,CannyThresh2,None,3,False)

    upPxEdges = np.argwhere(upEdges == 255)


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
            prew_lengFish = lengFish
            cv2.waitKey(1)

            # Detect keypoints in the current frame
            kp, des = sift.detectAndCompute(viewUpSource, None)

            # If this is not the first frame, track the keypoints
            if prev_kp is not None:
                # Match the keypoints between the previous and current frames
                matches = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE).match(des, prev_des)

                # Convert matches to list before sorting
                matches = list(matches)

                # Sort the matches by distance
                matches.sort(key=lambda x: x.distance)

                # Select the top 10 matches
                good_matches = matches[:10]

                # Draw the tracked keypoints
                for m in good_matches:
                    x1, y1 = kp[m.queryIdx].pt
                    x2, y2 = prev_kp[m.trainIdx].pt
                    cv2.line(viewUpSource, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Update the previous frame and its keypoints
            prev_frame = frame
            prev_kp = kp
            prev_des = des





            '''
            kp, des = sift.detectAndCompute(viewUp, None)

             # Find matching keypoints between current frame and object boundary
            matches = []
            for kp_obj in object_boundary:
                for kp_frame in kp:
                    if kp_obj.pt == kp_frame.pt:
                        matches.append((kp_obj, kp_frame))

            # Draw matched keypoints
            for match in matches:
                cv2.drawMarker(viewUpSource, match[0].pt, (0, 255, 0), cv2.MARKER_CROSS, 10)
                cv2.drawMarker(viewUpSource, match[1].pt, (0, 0, 255), cv2.MARKER_CROSS, 10)
            '''








        # Display images
        
        cv2.imshow('SIFT Tracker', viewUp)
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
