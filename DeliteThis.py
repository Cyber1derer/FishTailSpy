import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


dataMain = np.load(r'.\ResData\GOPR6626\0_22___0_24\dataMain.npz')

#6 12 47
data_dict = dict(dataMain)
ic(data_dict['TailAngleArray'], data_dict['TailAngleArray'] [6] )
if 'TailAngleArray' in data_dict:
    data_dict['TailAngleArray'] [0] = 10.79
    #data_dict['TailAngleArray'] [6] = 0.39
    data_dict['TailAngleArray'] [11] = 18.42
    data_dict['TailAngleArray'] [47] = 3.10
    data_dict['TailAngleArray'] [61] = 3.0
    #data_dict['TailAngleArray'] = data_dict['TailAngleArray'] * 2  # Пример изменения: умножение всех элементов массива на 2)
print( "Data: ", data_dict['TailAngleArray'][6] )
np.savez('dataMainv2.npz', **data_dict)
loaded_modified_file = np.load('dataMainv2.npz')
data_dictMod = dict(loaded_modified_file)
ic(data_dictMod['TailAngleArray'], data_dictMod['TailAngleArray'] [6] )
