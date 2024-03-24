import cv2

# Открываем видеопоток
video = cv2.VideoCapture('path/to/video.mp4')

# Получаем общее количество кадров в видео
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Список временных меток (в секундах), в которые нужно извлечь кадры
timestamps = [10, 20, 30, 40, 50]  # Например, каждые 10 секунд

# Получаем частоту кадров видео
fps = video.get(cv2.CAP_PROP_FPS)

# Извлекаем кадры в нужные моменты времени
for timestamp in timestamps:
    # Вычисляем номер кадра, соответствующий данной временной метке
    frame_number = int(timestamp * fps)

    # Устанавливаем позицию в видео на нужный кадр
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Извлекаем кадр
    ret, frame = video.read()

    if ret:
        # Сохраняем извлеченный кадр (например, в формате JPEG)
        cv2.imwrite(f'frame_{timestamp}.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f'Кадр на {timestamp} секунде извлечен и сохранен.')
    else:
        print(f'Ошибка при извлечении кадра на {timestamp} секунде.')

# Освобождаем ресурсы
video.release()
cv2.destroyAllWindows()