# OpenCV solution
import cv2

cap = cv2.VideoCapture(r'Data/CalibVideo/1CalibVideo.MP4')
if cap.isOpened():
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = fourcc = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"Ширина кадра: {width} пикселей")
    print(f"Высота кадра: {height} пикселей")
    print(f"FourCC код: {fourcc}")
    print(f"Частота кадров: {fps} кадров/сек")

    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Не удалось открыть файл")
'''
# ffmeg for python solution
import ffmpeg

probe = ffmpeg.probe(r'path/')
video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
if video_stream is not None:
    pix_fmt = video_stream['pix_fmt']
    print(f'Формат кадров: {pix_fmt}')
'''