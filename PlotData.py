import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cv2

L = 5 #Длина хвоста в миллимитрах
dt = 1/50
# Загрузка массивов из файла
#dataMain = np.load('dataMain.npz')
#dataMirror = np.load('dataMirror.npz')

#=======================================================================================
#                           GOPR6626        
#======================================================================================

dataMain = np.load(r'.\ResData\GOPR6626\05_23_750__05_24_725\dataMain.npz')
dataMirror = np.load(r'.\ResData\GOPR6626\05_23_750__05_24_725\dataMirror.npz')
x_offset = 1025
y_offset = 250

#dataMain = np.load(r'.\ResData\GOPR6626\5_31p1___5_31p7\dataMain.npz')
#dataMirror = np.load(r'.\ResData\GOPR6626\5_31p1___5_31p7\dataMirror.npz')


#=======================================================================================
#                           GP016626        
#======================================================================================

#dataMain = np.load(r'.\ResData\GP016626\5_47___5_49p2\dataMain.npz')
#dataMirror = np.load(r'.\ResData\GP016626\5_47___5_49p2\dataMirror.npz')


#=======================================================================================
#                           GP026626        
#======================================================================================
#dataMain = np.load(r'.\ResData\GP026626\5_18__5_22\dataMain.npz')
#dataMirror = np.load(r'.\ResData\GP026626\5_18__5_22\dataMirror.npz')

#dataMain = np.load(r'.\ResData\GP026626\5_41p7____5_43p7\dataMain.npz')
#dataMirror = np.load(r'.\ResData\GP026626\5_41p7____5_43p7\dataMirror.npz')


#dataMain = np.load(r'.\ResData\GP026626\5_47____5_49\dataMain.npz')
#dataMirror = np.load(r'.\ResData\GP026626\5_47____5_49\dataMirror.npz')





#ic(dataMain, dataMirror)
'''
dataMain = np.load('dataMain.npz')
dataMirror = np.load('dataMirror.npz')

dataMain = np.load('dataMain.npz')
dataMirror = np.load('dataMirror.npz')

dataMain = np.load('dataMain.npz')
dataMirror = np.load('dataMirror.npz')

dataMain = np.load('dataMain.npz')
dataMirror = np.load('dataMirror.npz')
'''


# Load camera parameters
data = np.load("camera_params.npz")
mtx = data['mtx']
dist = data['dist']

def function1(dataMain, dataMirror, corPer = 0):
    # Извлечение массивов
    TailAngleArray = dataMain['TailAngleArray']
    pxCenterFish = dataMain['pxCenterFish']
    fishHeight = dataMirror['fishHeight']
    #ic(fishHeight.shape, TailAngleArray.shape )
    #print("Массивы успешно загружены из файла")
    # Убедимся, что pxCenterFish имеет две измерения
    pxCenterFish = np.squeeze(pxCenterFish)
    assert pxCenterFish.ndim == 2, "pxCenterFish должен быть двумерным массивом"
    #вычислить координаты в нормализованной плоскости изображения:
    # Извлекаем координаты x
    # Извлечение координат x и y
    x_coords = pxCenterFish[:, 0] + x_offset
    y_coords = pxCenterFish[:, 1]  + y_offset
    #Crop 1025 and 250 
    #CropMirror y 420   x 525
    c_x = mtx[0,2] - x_offset
    f_x = mtx[0,0] 
    c_y = mtx [1,2]- y_offset
    f_y = mtx[1,1]
    #ic(mtx, c_x, f_x,c_y,f_y)
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
    velocityMM = velocity * 1000

    mvelocity = np.mean( np.abs(velocity) )
    mvelocityMM = np.mean( np.abs(velocityMM) )
    #ic(velocity,TailAngleArray)
    # Создаем массив времени t, зная, что координаты записаны каждые 1/50 секунды
    num_points = velocity.shape[0]
    t = np.linspace(0, (num_points - 1) / 50, num_points)
    #ic(TailAngleArray)



    # Вычисляем производные (разности между соседними элементами)
    derivatives = np.diff(TailAngleArray)

    # Определяем точки изменения знака производной
    turning_points = np.where(np.diff(np.sign(derivatives)))[0] + 1

    # Добавляем первую и последнюю точки
    turning_points = np.insert(turning_points, 0, 0)
    turning_points = np.append(turning_points, len(data) - 1)

    # Получаем значения в точках изменения
    turning_values = TailAngleArray[turning_points]
    Acp  = np.abs(np.diff(turning_values) )
    Ampl = np.sin( np.mean(Acp) / (len(turning_values) /2 ) )  * L
    AmplPaint = np.sin( turning_values  )  * L
    ic(AmplPaint)
    Frenq = (len(turning_values) /2 ) / ( len(TailAngleArray) * 0.02 )  # В секунду
    ic(Ampl, Frenq, mvelocityMM)
    '''
    #*----------------------------------График два в ряд -------------------------------------------------------------------------------------
    # Построение графика x от t
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)  # создаем первый подграфик
    plt.plot(t, velocityMM, marker='o', linestyle='-', color='r' )
    plt.xlabel('Время (секунды)')
    plt.ylabel('Скорость (мм)')
    plt.title('Зависимость скорости от времени')
    plt.legend()
    plt.grid(True)
    #AmplPaintMM = AmplPaint * 1000
    num_points = AmplPaint.shape[0]
    t = np.linspace(0, (num_points - 1) / 50, num_points)
    # Построение графика y от t
    plt.subplot(2, 1, 2)  # создаем второй подграфик
    plt.plot(t, AmplPaint, marker='o', linestyle='-', color='b')
    plt.xlabel('Время (секунды)')
    plt.ylabel('Амлитуда (мм)')
    plt.title('Зависимость амлитуды от времени')
    plt.legend()
    plt.grid(True)

    # Отображение графика
    plt.tight_layout()  # для корректного отображения графиков
    plt.show()

    #*----------------------------------График два в ряд -------------------------------------------------------------------------------------
    '''

    dataResF = Ampl, Frenq, mvelocityMM
    return dataResF
#ic(turning_values.shape, len(turning_values))
#ic(turning_values)
#Ampl = np.mean(turning_values) / len(turning_values) 
#print("Точки изменения направления графика:")
#for i, index in enumerate(turning_points):

    #print(f"Точка {i+1}: индекс = {index}, значение = {TailAngleArray[index]}")
def linear_approximation(x_data, y_data):
    # Преобразование списков в массивы numpy для удобства работы
    x = np.array(x_data)
    y = np.array(y_data)
    
    # Вычисление коэффициентов прямой линии с использованием метода наименьших квадратов
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    
    # Создание набора точек для отображения аппроксимирующей прямой
    x_line = np.linspace(min(x), max(x), 100)
    y_line = m * x_line + c
    
    return m, c, x_line, y_line




dataRes = []

dataRes.append(function1(dataMain, dataMirror))

dataMain = np.load(r'.\ResData\GOPR6626\5_31p1___5_31p7\dataMain.npz')
dataMirror = np.load(r'.\ResData\GOPR6626\5_31p1___5_31p7\dataMirror.npz')

dataRes.append(function1(dataMain, dataMirror))

dataMain = np.load(r'.\ResData\GP016626\5_47___5_49p2\dataMain.npz')
dataMirror = np.load(r'.\ResData\GP016626\5_47___5_49p2\dataMirror.npz')

dataRes.append(function1(dataMain, dataMirror))

dataMain = np.load(r'.\ResData\GP026626\5_18__5_22\dataMain.npz')
dataMirror = np.load(r'.\ResData\GP026626\5_18__5_22\dataMirror.npz')

dataRes.append(function1(dataMain, dataMirror))

dataMain = np.load(r'.\ResData\GP026626\5_41p7____5_43p7\dataMain.npz')
dataMirror = np.load(r'.\ResData\GP026626\5_41p7____5_43p7\dataMirror.npz')

dataRes.append(function1(dataMain, dataMirror))

dataMain = np.load(r'.\ResData\GP026626\5_47____5_49\dataMain.npz')
dataMirror = np.load(r'.\ResData\GP026626\5_47____5_49\dataMirror.npz')
'''
ic (dataMain['pxCenterFish'][68][0][0] )
dataMain['pxCenterFish'][68][0][0] = dataMain['pxCenterFish'][68][0][0] + 200
ic (dataMain['pxCenterFish'][68][0] )
# Извлечение массива из данных
pxCenterFishkek = dataMain['pxCenterFish']
# Построение графика
plt.plot(pxCenterFishkek[:,:,0])
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.title('График значений из двумерного массива')
plt.grid(True)
plt.show()
'''
#ic(dataMain['pxCenterFish'], dataMain['TailAngleArray'])
#print(" dataMain['pxCenterFish'] ", dataMain['pxCenterFish'] )
dataRes.append(function1(dataMain, dataMirror))


dataMain = np.load(r'.\ResData\GOPR6626\0_22___0_24\dataMain.npz')
dataMirror = np.load(r'.\ResData\GOPR6626\0_22___0_24\dataMirror.npz')
'''
ic (dataMain )
# Извлечение массива из данных
pxCenterFishkek = dataMain['TailAngleArray']
# Построение графика
plt.plot(pxCenterFishkek, marker = "o")
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.title('График значений из двумерного массива')
plt.grid(True)
plt.show()
'''
dataRes.append(function1(dataMain, dataMirror))


dataMain = np.load(r'.\ResData\GOPR6626\5_54p5____5_55p5\dataMain.npz')
dataMirror = np.load(r'.\ResData\GOPR6626\5_54p5____5_55p5\dataMirror.npz')

dataRes.append(function1(dataMain, dataMirror))

#ic (dataRes)

# Разделение данных на отдельные списки
x_data = [AmplPoints[0] for AmplPoints in dataRes] #A
y_data = [FrenqPoints[1] for FrenqPoints in dataRes] #F
z_data = [velocitiPoints[2] for velocitiPoints in dataRes] #V



m, c, x_line, y_line = linear_approximation(x_data, y_data) # A F
AMean = np.mean(x_data)

x_OrangepointA = AMean #Средняя амплитуда
y_OrangepointA = m * x_OrangepointA + c # Средняя частота

#Amplit = 

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











# Построение двумерного графика 
plt.figure()
plt.scatter(x_data, y_data, marker='o', color='b')
plt.plot(x_line, y_line, color='g',linewidth =2 ) #label=f'Аппроксимация: y = {m:.2f}x + {c:.2f}'
plt.scatter(x_OrangepointA, y_OrangepointA, color='orange', s=50, edgecolor='black', zorder=3)
plt.xlabel('Средняя амплитуда')
plt.ylabel('Средняя частота')
plt.title('Зависимость средней амплитуды (мм) от средней частоты (колебаний/с)')
plt.grid(True)
plt.savefig('scatter_AF.png',  bbox_inches='tight')
plt.show()


m, c, x_line, y_line = linear_approximation(z_data, x_data) 

#x_OrangepointF=  m * y_OrangepointF + c
x_OrangepointV = (x_OrangepointA - c) / m #Средняя скорость
#y_OrangepointF = m * x_OrangepointF + c
#y_OrangepointF = x_OrangepointA

# Построение двумерного графика 
plt.figure()
plt.scatter(z_data, x_data, marker='o', color='b')
plt.plot(x_line, y_line, color='g',linewidth =2 ) #label=f'Аппроксимация: y = {m:.2f}x + {c:.2f}'
plt.scatter(x_OrangepointV, x_OrangepointA, color='orange', s=50, edgecolor='black', zorder=3)

plt.xlabel('Средняя линейная скорость ')
plt.ylabel('Средняя амплитуда')
plt.title('Зависимость средней амплитуды (мм) от средней линейной скорости (мм/с)')
plt.grid(True)
plt.savefig('scatter_VA.png',  bbox_inches='tight')
plt.show()

m, c, x_line, y_line = linear_approximation(z_data, y_data) 

y_OrangepointF = m * x_OrangepointV + c
# Построение двумерного графика 
plt.figure()
plt.scatter(z_data, y_data, marker='o', color='b')
plt.plot(x_line, y_line, color='g',linewidth =2 ) #label=f'Аппроксимация: y = {m:.2f}x + {c:.2f}'
plt.scatter(x_OrangepointV, y_OrangepointF, color='orange', s=50, edgecolor='black', zorder=3)

plt.xlabel('Средняя линейная скорость')
plt.ylabel('Средняя частота')
plt.title('Зависимость средней частоты (колебаний/с) от средней линейной скорости (мм/с)')
plt.grid(True)
plt.savefig('scatter_VF.png',  bbox_inches='tight')
plt.show()

#3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(velocity, TailAngleArray, t, cmap='viridis')
ax.scatter(z_data, x_data, y_data, c='b', marker='o')
ax.scatter(x_OrangepointV, x_OrangepointA, y_OrangepointF, color='orange', s=50, edgecolor='black', zorder=3)
ax.set_xlabel('Средняя линейная скорость')
ax.set_ylabel('Средняя амплитуда')
ax.set_zlabel('Средняя частота')
plt.savefig('scatter_VAF.png')
plt.show()