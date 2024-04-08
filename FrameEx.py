import cv2

# Открываем видеопоток
video = cv2.VideoCapture('Data/CalibVideo/1CalibVideo.MP4')

# Получаем общее количество кадров в видео
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Список временных меток (в секундах), в которые нужно извлечь кадры

'''
#First video
FramesStamps = [34, 364,525, 737, 896, 923, 1057, 1376, 
                1521, 1819, 2204,2568, 2837,3094,3346, 3596,
                3694, 3796, 279, 412, 843,3568,3636,3750 ]  # Например, каждые 10 секунд
'''
#SecondVideo
FramesStamps = [1588,1903,2754, 3259, 533,1271,1536,1619,1775 ]
# Получаем частоту кадров видео
fps = video.get(cv2.CAP_PROP_FPS)
print("fps camera: ", fps)
# Извлекаем кадры в нужные моменты времени
#for timestamp in timestamps:
for frameN in FramesStamps:
    # Вычисляем номер кадра, соответствующий данной временной метке
    #frame_number = int(timestamp * fps)

    # Устанавливаем позицию в видео на нужный кадр
    #video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    video.set(cv2.CAP_PROP_POS_FRAMES, frameN)


    # Извлекаем кадр
    ret, frame = video.read()

    if ret:
        # Сохраняем извлеченный кадр (например, в формате png)
        cv2.imwrite(f'frame_{frameN}.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f'Кадр на {frameN} frame извлечен и сохранен.')
    else:
        print(f'Ошибка при извлечении кадра на {frameN} секунде.')

# Освобождаем ресурсы
video.release()
cv2.destroyAllWindows()