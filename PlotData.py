import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
import cv2
dt = 1/50
# Загрузка массивов из файла
dataMain = np.load('dataMain.npz')
dataMirror = np.load('dataMirror.npz')


dataMain = np.load('dataMain.npz')
dataMirror = np.load('dataMirror.npz')

dataMain = np.load('dataMain.npz')
dataMirror = np.load('dataMirror.npz')

dataMain = np.load('dataMain.npz')
dataMirror = np.load('dataMirror.npz')

dataMain = np.load('dataMain.npz')
dataMirror = np.load('dataMirror.npz')

dataMain = np.load('dataMain.npz')
dataMirror = np.load('dataMirror.npz')



# Load camera parameters
data = np.load("camera_params.npz")
# Извлечение массивов
TailAngleArray = dataMain['TailAngleArray']
pxCenterFish = dataMain['pxCenterFish']
fishHeight = dataMirror['fishHeight']

mtx = data['mtx']
dist = data['dist']

print("Массивы успешно загружены из файла")
# Убедимся, что pxCenterFish имеет две измерения
pxCenterFish = np.squeeze(pxCenterFish)
assert pxCenterFish.ndim == 2, "pxCenterFish должен быть двумерным массивом"
#вычислить координаты в нормализованной плоскости изображения:
# Извлекаем координаты x


# Извлечение координат x и y
x_coords = pxCenterFish[:, 0]
y_coords = pxCenterFish[:, 1]
#Crop 1025 and 250 



c_x = mtx[0,2] 
f_x = mtx[0,0]
c_y = mtx [1,2]
f_y = mtx[1,1]
ic(mtx, c_x, f_x,c_y,f_y)
# Выполняем операцию (x - c_x) * f_x
x_coords = (x_coords - c_x) / f_x
# Вставляем измененные координаты x обратно в массив
pxCenterFish[:, 0] = x_coords

# Выполняем операцию (y - c_y) * f_y
y_coords = (y_coords - c_y) / f_y
pxCenterFish[:, 1] = y_coords
# Создаем массив z, который во всех случаях равен 1
z_coords = np.ones((pxCenterFish.shape[0], 1))
# Добавляем координату z к массиву pxCenterFish
pxCenterFish_with_z = np.hstack((pxCenterFish, z_coords))

# Преобразование пиксельных координат в координаты камеры
#camera_coords = cv2.undistortPoints(pxCenterFish_with_z, mtx, None)

RealCenterFish = pxCenterFish_with_z *  fishHeight[:, np.newaxis] # В метрах
# Применяем центральную разность для вычисления скорости
#velocity = np.diff(RealCenterFish, axis=0) / dt # M/S
# Рассчитываем расстояние между точками
distances = np.sqrt(np.sum(np.diff(RealCenterFish, axis=0)**2, axis=1))
velocity = distances / dt

ic(velocity,distances)
# Создаем массив времени t, зная, что координаты записаны каждые 1/50 секунды
num_points = velocity.shape[0]
t = np.linspace(0, (num_points - 1) / 50, num_points)

'''
# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='b', label='Движение объекта')
plt.xlabel('Координата X')
plt.ylabel('Координата Y')
plt.title('Траектория движения объекта')
plt.legend()
plt.grid(True)

# Отображение графика
plt.show()
'''

# Построение графика x от t
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)  # создаем первый подграфик
plt.plot(t, velocity, marker='o', linestyle='-', color='r', label='x(t)')
plt.xlabel('Время (секунды)')
plt.ylabel('Скорость')
plt.title('Скорость от времени')
plt.legend()
plt.grid(True)

# Построение графика y от t
plt.subplot(2, 1, 2)  # создаем второй подграфик
plt.plot(t, TailAngleArray[1:], marker='o', linestyle='-', color='b', label='y(t)')
plt.xlabel('Время (секунды)')
plt.ylabel('Угол хвоста')
plt.title('Зависимость угла хвоста от времени')
plt.legend()
plt.grid(True)

# Отображение графика
plt.tight_layout()  # для корректного отображения графиков
plt.show()
