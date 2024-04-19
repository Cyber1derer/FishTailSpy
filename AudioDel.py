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
from moviepy.editor import VideoFileClip

# Загрузка видеофайла
video = VideoFileClip("D:\MyCodeProjects\FishTailSpy\Data\Fish\Source\Fish1.MP4")

# Извлечение видеопотока без аудио
video_without_audio = video.set_audio(None)

# Запись нового видеофайла без аудиодорожки
video_without_audio.write_videofile("output_video.mp4", codec="libx264", audio_codec="copy")