# OpenCV solution
import cv2

cap = cv2.VideoCapture(r'Data/CalibVideo/1CalibVideo.MP4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
print(f'Формат кадров (FourCC): {fourcc}')



'''
# ffmeg for python solution
import ffmpeg

probe = ffmpeg.probe(r'path/')
video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
if video_stream is not None:
    pix_fmt = video_stream['pix_fmt']
    print(f'Формат кадров: {pix_fmt}')
'''