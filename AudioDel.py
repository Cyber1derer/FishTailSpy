'''
import cv2
from moviepy.editor import *

# Загрузка видеофайла
video = cv2.VideoCapture('D:\MyCodeProjects\FishTailSpy\Data\Fish\Source\Fish1.MP4')

# Получение информации о видео
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Создание временного файла без аудио
output_without_audio = 'output_without_audio.mp4'

# Создание объекта VideoFileClip
video_clip = VideoFileClip('D:\MyCodeProjects\FishTailSpy\Data\Fish\Source\Fish1.MP4')

# Удаление аудио из видеофайла
video_clip = video_clip.set_audio(None)

# Сохранение видео без аудио
video_clip.write_videofile(output_without_audio, codec='libx264', fps=fps)

# Закрытие видеофайла
video.release()

'''


import sys
import os

moviepy = False
ffmpeg = False
from moviepy.editor import VideoFileClip
if moviepy == True:
    # Загрузка видеофайла
    video = VideoFileClip("D:\MyCodeProjects\FishTailSpy\Data\Fish\Source\Fish2.MP4")
    # Извлечение видеопотока без аудио
    video_without_audio = video.set_audio(None)
    # Запись нового видеофайла без аудиодорожки
    video_without_audio.write_videofile('Data/Fish/outVideo.avi', codec="png", audio_codec="copy")
elif ffmpeg == True:
    # Need FFMPEG install to system
    input_file='D:\MyCodeProjects\FishTailSpy\Data\Fish\Source\Fish2.MP4'
    output_file='OutFFDelAu.MP4'
    cmd = f'ffmpeg -i "{input_file}" -c copy -an "{output_file}"'
    os.system(cmd)
    #ffmpeg -i $input_file -c copy -an $output_file
else:
    print("Metods do not check")